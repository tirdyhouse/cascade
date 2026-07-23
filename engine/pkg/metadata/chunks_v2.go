package metadata

import (
	"bytes"
	"encoding/binary"
	"encoding/gob"
	"errors"
	"fmt"

	"github.com/cockroachdb/pebble"
)

// ChunkObjectMeta is the stored metadata for a v2 chunk object.
// Fields mirror cache.ChunkObject but live in the metadata package.
type ChunkObjectMeta struct {
	Namespace   string
	Key         string
	Shard       string
	FilePath    string
	Index       int
	StartTokens int
	EndTokens   int
	Size        int64
}

// ── V2 chunk keys (prefix 0x03) ──
// Key: 0x03 + 2-byte BE len(namespace) + namespace + 2-byte BE len(key) + key + 2-byte BE len(shard) + shard
// This is a length-prefixed encoding with no separator dependency.

func v2ChunkKey(namespace, key, shard string) []byte {
	buf := make([]byte, 1+2+len(namespace)+2+len(key)+2+len(shard))
	buf[0] = 0x03
	off := 1
	binary.BigEndian.PutUint16(buf[off:], uint16(len(namespace)))
	off += 2
	copy(buf[off:], namespace)
	off += len(namespace)
	binary.BigEndian.PutUint16(buf[off:], uint16(len(key)))
	off += 2
	copy(buf[off:], key)
	off += len(key)
	binary.BigEndian.PutUint16(buf[off:], uint16(len(shard)))
	off += 2
	copy(buf[off:], shard)
	return buf
}

// v2ChunkPrefix returns the prefix for iterating all entries under (namespace, key).
func v2ChunkPrefix(namespace, key string) []byte {
	buf := make([]byte, 1+2+len(namespace)+2+len(key))
	buf[0] = 0x03
	off := 1
	binary.BigEndian.PutUint16(buf[off:], uint16(len(namespace)))
	off += 2
	copy(buf[off:], namespace)
	off += len(namespace)
	binary.BigEndian.PutUint16(buf[off:], uint16(len(key)))
	off += 2
	copy(buf[off:], key)
	return buf
}

// v2ChunkNamespacePrefix returns the prefix for all entries under a namespace.
func v2ChunkNamespacePrefix(namespace string) []byte {
	buf := make([]byte, 1+2+len(namespace))
	buf[0] = 0x03
	binary.BigEndian.PutUint16(buf[1:], uint16(len(namespace)))
	copy(buf[3:], namespace)
	return buf
}

// v2UpperBound returns an upper bound key for prefix iteration.
func v2UpperBound(prefix []byte) []byte {
	ub := make([]byte, len(prefix)+1)
	copy(ub, prefix)
	ub[len(prefix)] = 0xFF
	return ub
}

// CommitChunkObjects persists a batch of v2 chunk object metadata atomically with Sync.
// Returns the number of newly inserted entries (not counting idempotent re-commits).
// Idempotent re-commit (identical fields) is allowed and not double-counted.
// Conflict (same key/shard but different fields) returns an error.
func (s *Store) CommitChunkObjects(objects []ChunkObjectMeta) (int, error) {
	if len(objects) == 0 {
		return 0, nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	b := s.db.NewBatch()
	defer b.Close()

	newCount := 0
	for _, obj := range objects {
		key := v2ChunkKey(obj.Namespace, obj.Key, obj.Shard)

		// Check whether a metadata record already exists. Only ErrNotFound
		// permits insertion; I/O and corruption errors must not be mistaken for
		// an absent object.
		existingVal, closer, getErr := s.dbGet(key)
		if getErr != nil && !errors.Is(getErr, pebble.ErrNotFound) {
			return 0, fmt.Errorf(
				"read existing v2 chunk namespace=%q key=%q shard=%q: %w",
				obj.Namespace,
				obj.Key,
				obj.Shard,
				getErr,
			)
		}
		exists := getErr == nil
		if exists {
			var existing ChunkObjectMeta
			decErr := gob.NewDecoder(bytes.NewReader(existingVal)).Decode(&existing)
			closeErr := closer.Close()
			if decErr != nil {
				return 0, fmt.Errorf(
					"decode existing v2 chunk namespace=%q key=%q shard=%q: %w",
					obj.Namespace,
					obj.Key,
					obj.Shard,
					decErr,
				)
			}
			if closeErr != nil {
				return 0, fmt.Errorf("close existing v2 chunk value: %w", closeErr)
			}
			if existing.Namespace != obj.Namespace ||
				existing.Key != obj.Key ||
				existing.Shard != obj.Shard ||
				existing.FilePath != obj.FilePath ||
				existing.Index != obj.Index ||
				existing.StartTokens != obj.StartTokens ||
				existing.EndTokens != obj.EndTokens ||
				existing.Size != obj.Size {
				return 0, fmt.Errorf(
					"conflict: namespace=%q key=%q shard=%q already exists with different fields",
					obj.Namespace,
					obj.Key,
					obj.Shard,
				)
			}
		}

		var buf bytes.Buffer
		if err := gob.NewEncoder(&buf).Encode(obj); err != nil {
			return 0, fmt.Errorf("encode v2 chunk meta: %w", err)
		}
		if err := b.Set(key, buf.Bytes(), pebble.NoSync); err != nil {
			return 0, fmt.Errorf("batch set: %w", err)
		}

		if !exists {
			newCount++
		}
	}

	if err := b.Commit(pebble.Sync); err != nil {
		return 0, fmt.Errorf("batch commit: %w", err)
	}
	return newCount, nil
}

// GetV2Chunk retrieves a v2 chunk object metadata by namespace/key/shard.
func (s *Store) GetV2Chunk(namespace, key, shard string) (*ChunkObjectMeta, error) {
	val, closer, err := s.dbGet(v2ChunkKey(namespace, key, shard))
	if errors.Is(err, pebble.ErrNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer closer.Close()
	var meta ChunkObjectMeta
	if err := gob.NewDecoder(bytes.NewReader(val)).Decode(&meta); err != nil {
		return nil, err
	}
	return &meta, nil
}

// DeleteV2Chunk removes a v2 chunk object metadata.
func (s *Store) DeleteV2Chunk(namespace, key, shard string) error {
	return s.db.Delete(v2ChunkKey(namespace, key, shard), pebble.Sync)
}

// IterateV2Chunks iterates over v2 chunk entries with key prefix using namespace+key.
// If key is empty, iterates all entries under the namespace.
func (s *Store) IterateV2Chunks(namespace, key string, fn func(*ChunkObjectMeta) error) error {
	var prefix []byte
	if key != "" {
		prefix = v2ChunkPrefix(namespace, key)
	} else {
		prefix = v2ChunkNamespacePrefix(namespace)
	}

	iter, err := s.db.NewIter(&pebble.IterOptions{
		LowerBound: prefix,
		UpperBound: v2UpperBound(prefix),
	})
	if err != nil {
		return err
	}
	defer iter.Close()

	for iter.First(); iter.Valid(); iter.Next() {
		var meta ChunkObjectMeta
		if err := gob.NewDecoder(bytes.NewReader(iter.Value())).Decode(&meta); err != nil {
			return fmt.Errorf("decode v2 chunk metadata at key %x: %w", iter.Key(), err)
		}
		if err := fn(&meta); err != nil {
			return err
		}
	}
	return iter.Error()
}

// IterateAllV2 iterates over every v2 chunk entry (prefix 0x03) in the store.
func (s *Store) IterateAllV2(fn func(*ChunkObjectMeta) error) error {
	prefix := []byte{0x03}
	iter, err := s.db.NewIter(&pebble.IterOptions{
		LowerBound: prefix,
		UpperBound: []byte{0x04},
	})
	if err != nil {
		return err
	}
	defer iter.Close()

	for iter.First(); iter.Valid(); iter.Next() {
		var meta ChunkObjectMeta
		if err := gob.NewDecoder(bytes.NewReader(iter.Value())).Decode(&meta); err != nil {
			return fmt.Errorf("decode v2 chunk metadata at key %x: %w", iter.Key(), err)
		}
		if err := fn(&meta); err != nil {
			return err
		}
	}
	return iter.Error()
}

// CountV2 returns the number of v2 chunk entries (prefix 0x03).
func (s *Store) CountV2() (int, error) {
	return s.countByPrefix(0x03)
}
