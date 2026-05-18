// Package metadata provides persistent metadata storage using Pebble.
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

// Store manages block metadata in a Pebble database.
// Key: block hash (uint64 big-endian)
// Value: gob-encoded BlockMeta
type Store struct {
	db     *pebble.DB
	mu     sync.RWMutex
	sf     singleflight.Group
	wb     *pebble.Batch
	batchSize int
}

// BlockMeta is the persisted metadata for a single cached block.
type BlockMeta struct {
	Hash       uint64
	FilePath   string
	Size       int64
	Shape      []int
	Dtype      string
	RefCount   int32
	AccessTime int64
	CreateTime int64
}

// Open opens or creates a Pebble database at the given path.
func Open(path string) (*Store, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil && !os.IsExist(err) {
		return nil, err
	}

	opts := &pebble.Options{
		// Reduce memory usage for embedded use
		Cache:        pebble.NewCache(64 << 20), // 64 MB cache
		MemTableSize: 4 << 20,                    // 4 MB memtable
	}

	db, err := pebble.Open(path, opts)
	if err != nil {
		return nil, err
	}

	return &Store{
		db:     db,
		wb:     db.NewBatch(),
		batchSize: 0,
	}, nil
}

func hashKey(hash uint64) []byte {
	b := make([]byte, 8)
	binary.BigEndian.PutUint64(b, hash)
	return b
}

// Put stores a block's metadata.
func (s *Store) Put(meta *BlockMeta) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(meta); err != nil {
		return err
	}

	return s.db.Set(hashKey(meta.Hash), buf.Bytes(), pebble.Sync)
}

// Get retrieves a block's metadata by hash.
func (s *Store) Get(hash uint64) (*BlockMeta, error) {
	val, err := s.db.Get(hashKey(hash))
	if err == pebble.ErrNotFound {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer val.Close()

	var meta BlockMeta
	if err := gob.NewDecoder(bytes.NewReader(val.Data())).Decode(&meta); err != nil {
		return nil, err
	}
	return &meta, nil
}

// Delete removes a block's metadata.
func (s *Store) Delete(hash uint64) error {
	return s.db.Delete(hashKey(hash), pebble.Sync)
}

// IterateAll calls fn for every stored metadata entry.
// If fn returns an error, iteration stops.
func (s *Store) IterateAll(fn func(*BlockMeta) error) error {
	iter, err := s.db.NewIter(nil)
	if err != nil {
		return err
	}
	defer iter.Close()

	for iter.First(); iter.Valid(); iter.Next() {
		var meta BlockMeta
		if err := gob.NewDecoder(bytes.NewReader(iter.Value())).Decode(&meta); err != nil {
			continue // skip corrupt entries
		}
		if err := fn(&meta); err != nil {
			return err
		}
	}
	return iter.Error()
}

// Count returns the number of stored metadata entries.
func (s *Store) Count() (int, error) {
	iter, err := s.db.NewIter(nil)
	if err != nil {
		return 0, err
	}
	defer iter.Close()

	count := 0
	for iter.First(); iter.Valid(); iter.Next() {
		count++
	}
	return count, iter.Error()
}

// Close shuts down the Pebble database.
func (s *Store) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.db.Close()
}
