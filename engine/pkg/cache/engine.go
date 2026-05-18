// Package cache implements the core disk cache engine.
package cache

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"

	"github.com/cici/disk-cache/engine/pkg/eviction"
	"github.com/cici/disk-cache/engine/pkg/metadata"
	"github.com/cici/disk-cache/engine/pkg/storage"
)

// diskEngine is the concrete implementation of Engine.
type diskEngine struct {
	cfg    Config
	store  *metadata.Store
	disk   storage.Backend
	policy eviction.Policy

	mu          sync.RWMutex
	blocksStored    atomic.Int64
	blocksRetrieved atomic.Int64
	blocksEvicted   atomic.Int64
	hitCount        atomic.Int64
	missCount       atomic.Int64
}

// New creates a new disk cache engine.
func New(cfg Config) (Engine, error) {
	// Ensure cache directory exists
	if err := os.MkdirAll(cfg.CachePath, 0755); err != nil {
		return nil, fmt.Errorf("create cache dir: %w", err)
	}

	// Open metadata store
	meta, err := metadata.Open(cfg.MetadataPath)
	if err != nil {
		return nil, fmt.Errorf("open metadata store: %w", err)
	}

	// Create local disk backend
	disk, err := storage.NewLocalDisk(cfg.CachePath, cfg.MaxSizeBytes)
	if err != nil {
		meta.Close()
		return nil, fmt.Errorf("create disk backend: %w", err)
	}

	// Create eviction policy
	pol := eviction.NewLRU(cfg.MaxSizeBytes)

	eng := &diskEngine{
		cfg:    cfg,
		store:  meta,
		disk:   disk,
		policy: pol,
	}

	// Rebuild eviction state from existing metadata on startup
	if err := eng.rebuildEviction(); err != nil {
		log.Printf("warning: rebuild eviction state: %v", err)
	}

	log.Printf("disk cache engine initialized: path=%s max_size=%d",
		cfg.CachePath, cfg.MaxSizeBytes)

	return eng, nil
}

// rebuildEviction scans existing metadata and loads it into the eviction policy.
func (e *diskEngine) rebuildEviction() error {
	return e.store.IterateAll(func(meta *metadata.BlockMeta) error {
		e.policy.Record(hashString(meta.Hash), meta.Size)
		return nil
	})
}

// Store saves a KV cache block to disk.
func (e *diskEngine) Store(key *CacheKey, gpuPtr uintptr, size int64) error {
	// For now, create a placeholder file. Real GDS path comes later.
	data := make([]byte, size) // FIXME: read from gpuPtr via cuda

	// Ensure space via eviction
	e.ensureSpace(size)

	// Write to disk
	skey := hashKey(key)
	relPath, n, err := e.disk.Write(skey, data)
	if err != nil {
		return fmt.Errorf("disk write: %w", err)
	}

	// Store metadata
	meta := &metadata.BlockMeta{
		Hash:       key.Hash,
		FilePath:   relPath,
		Size:       n,
		AccessTime: now(),
		CreateTime: now(),
	}
	if err := e.store.Put(meta); err != nil {
		return fmt.Errorf("metadata put: %w", err)
	}

	// Update eviction policy
	e.policy.Record(skey, n)

	e.blocksStored.Add(1)
	return nil
}

// Retrieve loads a KV cache block from disk.
func (e *diskEngine) Retrieve(key *CacheKey, gpuPtr uintptr) (int64, error) {
	skey := hashKey(key)

	// Check metadata
	meta, err := e.store.Get(key.Hash)
	if err != nil {
		return 0, fmt.Errorf("metadata get: %w", err)
	}
	if meta == nil {
		e.missCount.Add(1)
		return 0, storage.ErrNotFound
	}

	// Read from disk
	data, err := e.disk.Read(meta.FilePath)
	if err != nil {
		e.missCount.Add(1)
		return 0, err
	}

	// FIXME: copy data to gpuPtr via cuda
	_ = gpuPtr

	// Update access time and eviction policy
	meta.AccessTime = now()
	e.store.Put(meta)
	e.policy.Record(skey, meta.Size)

	e.hitCount.Add(1)
	e.blocksRetrieved.Add(1)
	return int64(len(data)), nil
}

// Remove deletes a cached block.
func (e *diskEngine) Remove(key *CacheKey) error {
	meta, err := e.store.Get(key.Hash)
	if err != nil {
		return err
	}
	if meta == nil {
		return nil
	}

	// Delete data file
	if err := e.disk.Delete(meta.FilePath); err != nil && err != storage.ErrNotFound {
		return fmt.Errorf("disk delete: %w", err)
	}

	// Delete metadata
	if err := e.store.Delete(key.Hash); err != nil {
		return fmt.Errorf("metadata delete: %w", err)
	}

	// Update eviction policy
	e.policy.Remove(hashKey(key))

	return nil
}

// Exists checks if a block is cached.
func (e *diskEngine) Exists(hash uint64) bool {
	meta, err := e.store.Get(hash)
	return err == nil && meta != nil
}

// Stats returns engine statistics.
func (e *diskEngine) Stats() Stats {
	return Stats{
		BlocksStored:    e.blocksStored.Load(),
		BlocksRetrieved: e.blocksRetrieved.Load(),
		BlocksEvicted:   e.blocksEvicted.Load(),
		DiskUsedBytes:   e.disk.Used(),
		DiskFreeBytes:   e.disk.Capacity() - e.disk.Used(),
		HitCount:        e.hitCount.Load(),
		MissCount:       e.missCount.Load(),
	}
}

// Close shuts down the engine.
func (e *diskEngine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if err := e.store.Close(); err != nil {
		return err
	}
	return e.disk.Close()
}

// ensureSpace evicts blocks until there is enough free space.
func (e *diskEngine) ensureSpace(needed int64) {
	available := e.disk.Capacity() - e.disk.Used()
	if available >= needed {
		return
	}

	target := needed - available
	keys := e.policy.Evict(target)

	for _, k := range keys {
		// Parse hash from string key
		var hash uint64
		fmt.Sscanf(k, "%016x", &hash)
		e.Remove(&CacheKey{Hash: hash})
		e.blocksEvicted.Add(1)
	}
}

// hashKey converts a CacheKey to a string key for storage/eviction.
func hashKey(key *CacheKey) string {
	return fmt.Sprintf("%016x", key.Hash)
}

// hashString parses a hex string back to uint64.
func hashString(s string) int64 {
	var h uint64
	fmt.Sscanf(s, "%016x", &h)
	return int64(h)
}
