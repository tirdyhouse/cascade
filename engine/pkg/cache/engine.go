package cache

import (
	"context"
	"crypto/sha256"
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

	mu            sync.Mutex
	totalBytes    int64 // tracked by eviction policy
	blocksStored    atomic.Int64
	blocksEvicted   atomic.Int64
	blocksRetrieved atomic.Int64
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
		cfg: cfg,
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

// Get returns metadata for a cached block.
// Python calls this before reading.
func (e *diskEngine) Get(hash uint64) (*BlockMeta, error) {
	m, err := e.meta.Get(hash)
	if err != nil {
		return nil, err
	}
	if m == nil {
		return nil, nil
	}

	// Update access time for LRU
	m.AccessTime = now()
	e.meta.Put(m)
	e.pol.Record(hexKey(hash), m.Size)
	e.blocksRetrieved.Add(1)

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
	// Rough check: don't hold lock for eviction
	if e.pol.TotalBytes()+needed <= e.cfg.MaxSizeBytes {
		return
	}
	e.Evict(needed)
}

// Stats returns engine statistics.
func (e *diskEngine) Stats() Stats {
	return Stats{
		BlocksStored:    e.blocksStored.Load(),
		BlocksRetrieved: e.blocksRetrieved.Load(),
		BlocksEvicted:   e.blocksEvicted.Load(),
		DiskUsedBytes:   e.pol.TotalBytes(),
	}
}

// ── Sentinel / Match ──

// RecordSentinel stores a cache-complete marker in Pebble metadata.
// Python calls this after writing all layer files for a request.
func (e *diskEngine) RecordSentinel(promptHash string, numTokens int) error {
	return e.meta.RecordSentinel(promptHash, numTokens)
}

// Match finds the largest cache hit by trying all block-aligned lengths
// from largest to smallest. Uses parallel goroutines with context cancellation
// so the first hit stops all other checks.
func (e *diskEngine) Match(tokenIDs []int64, mmHashes []string, blockSize int) MatchResult {
	if len(tokenIDs) < 2 || blockSize < 1 {
		return MatchResult{}
	}

	// Align to block boundary: (len-1) // blockSize * blockSize
	numTokens := len(tokenIDs) - 1
	maxCheck := (numTokens / blockSize) * blockSize
	if maxCheck < blockSize {
		return MatchResult{}
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	type hit struct {
		tokens int
		hash   string
	}
	hitCh := make(chan hit, 1)

	// Concurrently check from largest to smallest.
	// Goroutines send to hitCh; first to send wins, cancel stops the rest.
	for check := maxCheck; check >= blockSize; check -= blockSize {
		go func(n int) {
			h := computePromptHash(tokenIDs[:n], mmHashes)
			if _, ok := e.meta.GetSentinel(h); ok {
				select {
				case hitCh <- hit{tokens: n, hash: h}:
				case <-ctx.Done():
				}
			}
		}(check)
	}

	select {
	case h := <-hitCh:
		cancel()
		return MatchResult{MatchedTokens: h.tokens, PromptHash: h.hash}
	case <-ctx.Done():
		return MatchResult{}
	}
}

// ── Hash helpers ──

func alignToBlockSize(numTokens, blockSize int) int {
	if numTokens < 1 {
		return 0
	}
	return (numTokens - 1) / blockSize * blockSize
}

func hashTokenCount(numTokens, blockSize int) int {
	aligned := alignToBlockSize(numTokens, blockSize)
	if aligned < 1 {
		return 1
	}
	return aligned
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
