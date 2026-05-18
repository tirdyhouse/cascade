// Package storage provides a local disk storage backend.
package storage

import (
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"sync/atomic"
)

// LocalDiskBackend stores KV cache data as individual files on local disk.
//
// Layout:
//
//	{root}/
//	  {hash[0:2]}/
//	    {hash[2:4]}/
//	      {hash}.bin
//
// Two-level sharding prevents any directory from holding too many entries.
type LocalDiskBackend struct {
	root     string
	capacity int64
	used     atomic.Int64
}

// NewLocalDisk creates a new local disk backend.
// root is the base directory for cache files.
// capacity is the maximum bytes to use (0 = unlimited).
func NewLocalDisk(root string, capacity int64) (*LocalDiskBackend, error) {
	if err := os.MkdirAll(root, 0755); err != nil {
		return nil, fmt.Errorf("mkdir cache root: %w", err)
	}
	return &LocalDiskBackend{
		root:     root,
		capacity: capacity,
	}, nil
}

// Write saves data to a file identified by key.
// Returns the relative file path and bytes written.
func (b *LocalDiskBackend) Write(key string, data []byte) (string, int64, error) {
	rel := b.keyToRelPath(key)
	path := filepath.Join(b.root, rel)

	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return "", 0, fmt.Errorf("mkdir: %w", err)
	}

	if b.capacity > 0 && b.used.Load()+int64(len(data)) > b.capacity {
		return "", 0, ErrDiskFull
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return "", 0, fmt.Errorf("write file: %w", err)
	}

	n := int64(len(data))
	b.used.Add(n)
	return rel, n, nil
}

// Read loads data from a file by its relative path.
func (b *LocalDiskBackend) Read(path string) ([]byte, error) {
	fullPath := filepath.Join(b.root, path)
	data, err := os.ReadFile(fullPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, ErrNotFound
		}
		return nil, fmt.Errorf("read file: %w", err)
	}
	return data, nil
}

// Delete removes a file by its relative path.
func (b *LocalDiskBackend) Delete(path string) error {
	fullPath := filepath.Join(b.root, path)
	if err := os.Remove(fullPath); err != nil {
		if os.IsNotExist(err) {
			return ErrNotFound
		}
		return fmt.Errorf("delete file: %w", err)
	}
	// Note: used count is not decremented here because the caller
	// is responsible for tracking usage via the eviction policy.
	return nil
}

// Capacity returns the maximum disk space in bytes.
func (b *LocalDiskBackend) Capacity() int64 { return b.capacity }

// Used returns the current usage in bytes.
func (b *LocalDiskBackend) Used() int64 { return b.used.Load() }

// Close is a no-op for the local disk backend.
func (b *LocalDiskBackend) Close() error { return nil }

// keyToRelPath converts a string key to a sharded relative path.
// Example: "abc123" → "ab/c1/ab-c1-23.bin"
func (b *LocalDiskBackend) keyToRelPath(key string) string {
	h := sha256.Sum256([]byte(key))
	hex := fmt.Sprintf("%x", h[:8]) // 16 hex chars
	return filepath.Join(hex[:2], hex[2:4], hex+".bin")
}
