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

type Store struct {
	db *pebble.DB
	mu sync.RWMutex
}

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

func hashKey(hash uint64) []byte {
	b := make([]byte, 8)
	binary.BigEndian.PutUint64(b, hash)
	return b
}

func (s *Store) Put(meta *BlockMeta) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(meta); err != nil {
		return err
	}
	return s.db.Set(hashKey(meta.Hash), buf.Bytes(), pebble.Sync)
}

func (s *Store) Get(hash uint64) (*BlockMeta, error) {
	val, closer, err := s.db.Get(hashKey(hash))
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
	return s.db.Delete(hashKey(hash), pebble.Sync)
}

func (s *Store) IterateAll(fn func(*BlockMeta) error) error {
	iter, err := s.db.NewIter(nil)
	if err != nil {
		return err
	}
	defer iter.Close()
	for iter.First(); iter.Valid(); iter.Next() {
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

func (s *Store) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.db.Close()
}
