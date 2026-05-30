package cache

import (
	"crypto/sha256"
	"encoding"
	"encoding/binary"
	"fmt"
	"log"
	"path/filepath"
	"sync"
	"sync/atomic"

	"predict/engine/pkg/eviction"
	"predict/engine/pkg/metadata"
)

// diskEngine implements Engine with Pebble metadata + LRU eviction.
type diskEngine struct {
	cfg  Config
	meta *metadata.Store
	pol  eviction.Policy

	mu sync.Mutex

	totalBytes        int64 // tracked by eviction policy
	blocksStored      atomic.Int64
	blocksEvicted     atomic.Int64
	blocksRetrieved   atomic.Int64
	chunksStored      atomic.Int64
	chunksRetrieved   atomic.Int64
	putRequests       atomic.Int64
	getRequests       atomic.Int64
	getHits           atomic.Int64
	getMisses         atomic.Int64
	matchRequests     atomic.Int64
	matchHits         atomic.Int64
	matchedTokens     atomic.Int64
	recordRequests    atomic.Int64
	recordBatchCalls  atomic.Int64
	chunkPutRequests  atomic.Int64
	chunkListRequests atomic.Int64
	chunkListHits     atomic.Int64
	removeRequests    atomic.Int64
	evictRequests     atomic.Int64
}

// New creates a new disk cache engine.
func New(cfg Config) (Engine, error) {
	meta, err := metadata.Open(cfg.MetadataPath)
	if err != nil {
		return nil, fmt.Errorf("open metadata: %w", err)
	}

	maxBytes := cfg.MaxSizeBytes
	pol := eviction.NewLRU(maxBytes)

	eng := &diskEngine{
		cfg:  cfg,
		meta: meta,
		pol:  pol,
	}

	// Rebuild eviction tracker from existing metadata
	if err := eng.rebuild(); err != nil {
		log.Printf("warn: rebuild eviction: %v", err)
	}

	log.Printf("disk-cache engine ready: path=%s max=%d", cfg.CachePath, cfg.MaxSizeBytes)
	return eng, nil
}

func (e *diskEngine) rebuild() error {
	return e.meta.IterateAll(func(m *metadata.BlockMeta) error {
		e.pol.Record(hexKey(m.Hash), m.Size)
		return nil
	})
}

// Put records a newly cached block.
// Python calls this after writing the file.
func (e *diskEngine) Put(hash uint64, filePath string, size int64) error {
	e.putRequests.Add(1)
	// Check space and evict if needed
	e.evictIfNeeded(size)

	if err := e.meta.Put(&metadata.BlockMeta{
		Hash:       hash,
		FilePath:   filePath,
		Size:       size,
		AccessTime: now(),
		CreateTime: now(),
	}); err != nil {
		return fmt.Errorf("meta put: %w", err)
	}

	e.pol.Record(hexKey(hash), size)
	e.blocksStored.Add(1)
	return nil
}

func (e *diskEngine) Get(hash uint64) (*BlockMeta, error) {
	e.getRequests.Add(1)
	m, err := e.meta.Get(hash)
	if err != nil {
		return nil, err
	}
	if m == nil {
		e.getMisses.Add(1)
		return nil, nil
	}

	// Update access time for LRU
	m.AccessTime = now()
	e.meta.Put(m)
	e.pol.Record(hexKey(hash), m.Size)
	e.blocksRetrieved.Add(1)
	e.getHits.Add(1)

	return &BlockMeta{
		Hash:       m.Hash,
		FilePath:   m.FilePath,
		Size:       m.Size,
		AccessTime: m.AccessTime,
		CreateTime: m.CreateTime,
	}, nil
}

// Remove deletes a block from metadata and eviction tracker.
func (e *diskEngine) Remove(hash uint64) error {
	e.removeRequests.Add(1)
	if err := e.meta.Delete(hash); err != nil {
		return err
	}
	e.pol.Remove(hexKey(hash))
	return nil
}

// Exists checks if a block is cached.
func (e *diskEngine) Exists(hash uint64) bool {
	m, err := e.meta.Get(hash)
	return err == nil && m != nil
}

// Evict returns blocks to delete to free targetBytes.
func (e *diskEngine) Evict(targetBytes int64) []BlockMeta {
	e.evictRequests.Add(1)
	keys := e.pol.Evict(targetBytes)
	var metas []BlockMeta
	for _, k := range keys {
		var hash uint64
		fmt.Sscanf(k, "%016x", &hash)
		if m, err := e.meta.Get(hash); err == nil && m != nil {
			metas = append(metas, BlockMeta{
				Hash:     m.Hash,
				FilePath: filepath.Join(e.cfg.CachePath, m.FilePath),
				Size:     m.Size,
			})
			e.meta.Delete(hash)
			e.blocksEvicted.Add(1)
		}
	}
	return metas
}

func (e *diskEngine) evictIfNeeded(needed int64) {
	if e.pol.TotalBytes()+needed <= e.cfg.MaxSizeBytes {
		return
	}
	e.Evict(needed)
}

// Stats returns engine statistics.
func (e *diskEngine) Stats() Stats {
	blockEntries, _ := e.meta.Count()
	sentinelEntries, _ := e.meta.CountSentinels()
	chunkEntries, _ := e.meta.CountChunks()

	return Stats{
		BlocksStored:      e.blocksStored.Load(),
		BlocksRetrieved:   e.blocksRetrieved.Load(),
		BlocksEvicted:     e.blocksEvicted.Load(),
		DiskUsedBytes:     e.pol.TotalBytes(),
		BlockEntries:      int64(blockEntries),
		SentinelEntries:   int64(sentinelEntries),
		ChunkEntries:      int64(chunkEntries),
		PutRequests:       e.putRequests.Load(),
		GetRequests:       e.getRequests.Load(),
		GetHits:           e.getHits.Load(),
		GetMisses:         e.getMisses.Load(),
		MatchRequests:     e.matchRequests.Load(),
		MatchHits:         e.matchHits.Load(),
		MatchedTokens:     e.matchedTokens.Load(),
		RecordRequests:    e.recordRequests.Load(),
		RecordBatchCalls:  e.recordBatchCalls.Load(),
		ChunkPutRequests:  e.chunkPutRequests.Load(),
		ChunkListRequests: e.chunkListRequests.Load(),
		ChunkListHits:     e.chunkListHits.Load(),
		ChunksStored:      e.chunksStored.Load(),
		ChunksRetrieved:   e.chunksRetrieved.Load(),
		RemoveRequests:    e.removeRequests.Load(),
		EvictRequests:     e.evictRequests.Load(),
	}
}

// ── Sentinel / Match ──

// RecordSentinel stores a single sentinel marker.
func (e *diskEngine) RecordSentinel(promptHash string, numTokens int) error {
	e.recordRequests.Add(1)
	return e.meta.RecordSentinel(promptHash, numTokens)
}

// ── Chunk storage ──

// PutChunk records a chunk file for a layer.
func (e *diskEngine) PutChunk(prefixKey, layerName string, chunkIndex, numTokens int) error {
	e.chunkPutRequests.Add(1)
	if err := e.meta.PutChunk(prefixKey, layerName, chunkIndex, numTokens); err != nil {
		return err
	}
	e.chunksStored.Add(1)
	return nil
}

// ListChunks returns chunk indices for a layer under a prefix key.
func (e *diskEngine) ListChunks(prefixKey, layerName string) ([]int, error) {
	e.chunkListRequests.Add(1)
	chunks, err := e.meta.ListChunks(prefixKey, layerName)
	if err != nil {
		return nil, err
	}
	if len(chunks) > 0 {
		e.chunkListHits.Add(1)
	}
	return chunks, nil
}

// RecordRetrieved records successful external cache chunk or block retrievals.
func (e *diskEngine) RecordRetrieved(count int64) {
	if count <= 0 {
		return
	}
	e.blocksRetrieved.Add(count)
	e.chunksRetrieved.Add(count)
}

// RecordAll computes all block-aligned cumulative hashes for a prompt
// and records them as sentinels in a single Pebble batch.
// numTokens is the actual number of KV tokens cached (aligned to block_size).
func (e *diskEngine) RecordAll(tokenIDs []int64, mmHashes []string, blockSize int) error {
	e.recordBatchCalls.Add(1)
	if blockSize <= 0 || len(tokenIDs) == 0 {
		return nil
	}
	numTokens := len(tokenIDs)
	numBlocks := numTokens / blockSize
	if numBlocks < 1 {
		return nil
	}

	// Incremental SHA-256: scan tokens once, snapshot at each block boundary.
	// For each snapshot, add mmHashes to finalize the cumulative hash.
	hashes := make([]string, numBlocks)
	h := sha256.New()
	buf := make([]byte, 4)

	for i := 0; i < numBlocks; i++ {
		start := i * blockSize
		end := start + blockSize
		if end > len(tokenIDs) {
			end = len(tokenIDs)
		}
		for _, tid := range tokenIDs[start:end] {
			binary.BigEndian.PutUint32(buf, uint32(tid))
			h.Write(buf)
		}
		// Clone state, add mmHashes, finalize
		clone := cloneHash(h)
		for _, mh := range mmHashes {
			clone.Write([]byte(mh))
		}
		hashes[i] = fmt.Sprintf("%x", clone.Sum(nil))[:32]
	}

	// Record all in batch (incremental: skip already-existing)
	for i, hash := range hashes {
		n := (i + 1) * blockSize
		if _, ok := e.meta.GetSentinel(hash); !ok {
			if err := e.meta.RecordSentinel(hash, n); err != nil {
				return err
			}
		}
	}
	return nil
}

// Match finds the largest cache hit:
//  1. Incremental SHA-256: scan token_ids once, collect cumulative hashes.
//  2. Binary search (since sentinels are monotonic: if hash[k] exists, hash[0..k-1] also exist).
//
// Returns matched token count and the corresponding prompt hash.
func (e *diskEngine) Match(tokenIDs []int64, mmHashes []string, blockSize int) MatchResult {
	e.matchRequests.Add(1)
	if len(tokenIDs) < 2 || blockSize < 1 {
		return MatchResult{}
	}

	numTokens := len(tokenIDs)
	numBlocks := numTokens / blockSize
	if numBlocks < 1 {
		return MatchResult{}
	}

	// 1. Incremental SHA-256: scan once, collect cumulative hashes
	hashes := make([]string, numBlocks)
	h := sha256.New()
	buf := make([]byte, 4)

	for i := 0; i < numBlocks; i++ {
		start := i * blockSize
		end := start + blockSize
		if end > len(tokenIDs) {
			end = len(tokenIDs)
		}
		for _, tid := range tokenIDs[start:end] {
			binary.BigEndian.PutUint32(buf, uint32(tid))
			h.Write(buf)
		}
		clone := cloneHash(h)
		for _, mh := range mmHashes {
			clone.Write([]byte(mh))
		}
		hashes[i] = fmt.Sprintf("%x", clone.Sum(nil))[:32]
	}

	// 2. Binary search
	lo, hi := 0, numBlocks-1
	for lo <= hi {
		mid := (lo + hi) / 2
		if _, ok := e.meta.GetSentinel(hashes[mid]); ok {
			lo = mid + 1
		} else {
			hi = mid - 1
		}
	}

	if hi < 0 {
		return MatchResult{}
	}
	matched := (hi + 1) * blockSize
	e.matchHits.Add(1)
	e.matchedTokens.Add(int64(matched))
	return MatchResult{MatchedTokens: matched, PromptHash: hashes[hi]}
}

func cloneHash(h interface{}) hashHash {
	// sha256.digest implements encoding.BinaryMarshaler/BinaryUnmarshaler
	marshaler, ok := h.(encoding.BinaryMarshaler)
	if !ok {
		// Fallback: can't clone, return new hash (wrong but safe at boundaries)
		return sha256.New()
	}
	state, err := marshaler.MarshalBinary()
	if err != nil {
		return sha256.New()
	}
	cl := sha256.New()
	unmarshaler := cl.(encoding.BinaryUnmarshaler)
	if err := unmarshaler.UnmarshalBinary(state); err != nil {
		return sha256.New()
	}
	return cl
}

// hashHash is the interface shared by sha256.digest.
type hashHash interface {
	Write(p []byte) (int, error)
	Sum(b []byte) []byte
}

func alignToBlockSize(numTokens, blockSize int) int {
	if numTokens < 1 {
		return 0
	}
	return (numTokens - 1) / blockSize * blockSize
}

func computePromptHash(tokenIDs []int64, mmHashes []string) string {
	h := sha256.New()
	buf := make([]byte, 4)
	for _, tid := range tokenIDs {
		binary.BigEndian.PutUint32(buf, uint32(tid))
		h.Write(buf)
	}
	for _, mh := range mmHashes {
		h.Write([]byte(mh))
	}
	return fmt.Sprintf("%x", h.Sum(nil))[:32]
}

// Close shuts down the engine.
func (e *diskEngine) Close() error {
	return e.meta.Close()
}

func hexKey(hash uint64) string {
	return fmt.Sprintf("%016x", hash)
}
