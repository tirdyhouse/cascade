package metadata

import (
	"bytes"
	"encoding/binary"
	"encoding/gob"
	"os"
	"path/filepath"
	"sync"

	"github.com/cockroachdb/pebble"
)

// Store manages block metadata and sentinel markers using Pebble.
// Key layout:
//   - Block meta:  0x00 + 8-byte uint64 hash (big-endian), total 9 bytes
//   - Sentinel:    0x01 + string prompt_hash
type Store struct {
	db *pebble.DB
	mu sync.RWMutex
}

// BlockMeta stores metadata for a cached block on disk.
type BlockMeta struct {
	Hash       uint64
	FilePath   string
	Size       int64
	RefCount   int32
	AccessTime int64
	CreateTime int64
}

func Open(path string) (*Store, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil && !os.IsExist(err) {
		return nil, err
	}
	opts := &pebble.Options{
		Cache:        pebble.NewCache(64 << 20),
		MemTableSize: 4 << 20,
	}
	db, err := pebble.Open(path, opts)
	if err != nil {
		return nil, err
	}
	return &Store{db: db}, nil
}

// ── Block meta keys (prefix 0x00) ──

func blockKey(hash uint64) []byte {
	key := make([]byte, 9)
	key[0] = 0x00
	binary.BigEndian.PutUint64(key[1:], hash)
	return key
}

func (s *Store) Put(meta *BlockMeta) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(meta); err != nil {
		return err
	}
	return s.db.Set(blockKey(meta.Hash), buf.Bytes(), pebble.Sync)
}

func (s *Store) Get(hash uint64) (*BlockMeta, error) {
	val, closer, err := s.db.Get(blockKey(hash))
	if err == pebble.ErrNotFound {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer closer.Close()
	var meta BlockMeta
	if err := gob.NewDecoder(bytes.NewReader(val)).Decode(&meta); err != nil {
		return nil, err
	}
	return &meta, nil
}

func (s *Store) Delete(hash uint64) error {
	return s.db.Delete(blockKey(hash), pebble.Sync)
}

func (s *Store) Count() (int, error) {
	iter, err := s.db.NewIter(nil)
	if err != nil {
		return 0, err
	}
	defer iter.Close()
	count := 0
	for iter.First(); iter.Valid(); iter.Next() {
		key := iter.Key()
		if len(key) == 9 && key[0] == 0x00 {
			count++
		}
	}
	return count, iter.Error()
}

// IterateAll calls fn for each block metadata entry (skips sentinel keys).
func (s *Store) IterateAll(fn func(*BlockMeta) error) error {
	iter, err := s.db.NewIter(nil)
	if err != nil {
		return err
	}
	defer iter.Close()
	for iter.First(); iter.Valid(); iter.Next() {
		key := iter.Key()
		if len(key) != 9 || key[0] != 0x00 {
			continue
		}
		var meta BlockMeta
		if err := gob.NewDecoder(bytes.NewReader(iter.Value())).Decode(&meta); err != nil {
			continue
		}
		if err := fn(&meta); err != nil {
			return err
		}
	}
	return iter.Error()
}

// ── Sentinel keys (prefix 0x01) ──

func sentinelKey(hash string) []byte {
	key := make([]byte, 1+len(hash))
	key[0] = 0x01
	copy(key[1:], hash)
	return key
}

// RecordSentinel stores a cache-complete marker for a prompt hash.
// numTokens is the number of KV tokens cached (aligned to block_size).
func (s *Store) RecordSentinel(promptHash string, numTokens int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	buf := make([]byte, 8)
	binary.BigEndian.PutUint64(buf, uint64(numTokens))
	return s.db.Set(sentinelKey(promptHash), buf, pebble.Sync)
}

// GetSentinel returns the cached token count for a prompt hash, if recorded.
func (s *Store) GetSentinel(promptHash string) (int, bool) {
	val, closer, err := s.db.Get(sentinelKey(promptHash))
	if err == pebble.ErrNotFound {
		return 0, false
	}
	if err != nil {
		return 0, false
	}
	defer closer.Close()
	numTokens := int(binary.BigEndian.Uint64(val))
	return numTokens, true
}

// DeleteSentinel removes a sentinel marker.
func (s *Store) DeleteSentinel(promptHash string) error {
	return s.db.Delete(sentinelKey(promptHash), pebble.Sync)
}


// ── Chunk metadata (prefix 0x02) ──
// Key format: 0x02{prefix_key 32 hex}{layer_name}{chunk_index 8 bytes BE}
// Value: 8 bytes uint64 = num_tokens in this chunk

func chunkKey(prefixKey, layerName string, chunkIndex int) []byte {
	// 0x02 + 32 hex (prefix) + layer_name + 8 (index)
	key := make([]byte, 1+32+len(layerName)+8)
	key[0] = 0x02
	copy(key[1:], prefixKey)
	copy(key[1+32:], layerName)
	binary.BigEndian.PutUint64(key[1+32+len(layerName):], uint64(chunkIndex))
	return key
}

func (s *Store) PutChunk(prefixKey, layerName string, chunkIndex, numTokens int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	buf := make([]byte, 8)
	binary.BigEndian.PutUint64(buf, uint64(numTokens))
	return s.db.Set(chunkKey(prefixKey, layerName, chunkIndex), buf, pebble.Sync)
}

func (s *Store) ListChunks(prefixKey, layerName string) ([]int, error) {
	// Iterate over all keys matching prefix 0x02{prefix_key}{layer_name}
	prefix := make([]byte, 1+32+len(layerName))
	prefix[0] = 0x02
	copy(prefix[1:], prefixKey)
	copy(prefix[1+32:], layerName)

	iter, err := s.db.NewIter(&pebble.IterOptions{
		LowerBound: prefix,
		UpperBound: append(prefix, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF),
	})
	if err != nil {
		return nil, err
	}
	defer iter.Close()

	var indices []int
	for iter.First(); iter.Valid(); iter.Next() {
		key := iter.Key()
		// key format: 0x02{32}{layer}{8}=1+32+len+8 bytes
		if len(key) < 1+32+len(layerName)+8 {
			continue
		}
		idx := int(binary.BigEndian.Uint64(key[1+32+len(layerName):]))
		indices = append(indices, idx)
	}
	return indices, nil
}
func (s *Store) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.db.Close()
}
