// Package cache provides the core disk cache engine types.
package cache

import (
	"time"
)

// CacheKey uniquely identifies a KV cache block.
type CacheKey struct {
	Hash    uint64 // Content hash (vLLM block hash)
	GroupID uint32 // Request group ID
	BlockID uint32 // Block sequence number
	Size    int64  // Data size in bytes (for eviction)
}

// BlockMeta stores metadata for a cached block.
type BlockMeta struct {
	Key        CacheKey
	FilePath   string // Absolute path to disk file
	Offset     int64  // Byte offset within file
	Size       int64  // Data size
	Shape      []int  // Tensor shape (for reconstruction)
	Dtype      string // Data type string e.g. "float16"
	RefCount   int32  // Reference count (pin/unpin)
	AccessTime int64  // Unix timestamp of last access
	CreateTime int64  // Unix timestamp of creation
}

// Stats exposes cache engine metrics.
type Stats struct {
	BlocksStored    int64
	BlocksRetrieved int64
	BlocksEvicted   int64
	DiskUsedBytes   int64
	DiskFreeBytes   int64
	HitCount        int64
	MissCount       int64
}

// Engine is the core disk cache interface exposed to Python via cgo.
type Engine interface {
	// Store writes a KV cache block to disk.
	// gpuPtr is the CUDA pointer to GPU tensor data.
	Store(key *CacheKey, gpuPtr uintptr, size int64) error

	// Retrieve reads a KV cache block from disk into GPU memory.
	// gpuPtr is the pre-allocated CUDA buffer.
	Retrieve(key *CacheKey, gpuPtr uintptr) (int64, error)

	// Remove deletes a cached block.
	Remove(key *CacheKey) error

	// Exists checks if a block is cached locally.
	Exists(hash uint64) bool

	// Stats returns current engine statistics.
	Stats() Stats

	// Close shuts down the engine and releases resources.
	Close() error
}

// Config holds engine configuration.
type Config struct {
	// CachePath is the root directory for cached files.
	CachePath string

	// MaxSizeBytes is the maximum disk space for cache (0 = unlimited).
	MaxSizeBytes int64

	// EvictionPolicy selects the eviction algorithm: "lru", "lfu", "fifo".
	EvictionPolicy string

	// MetadataPath is the path for the Pebble database.
	MetadataPath string
}

// DefaultConfig returns a default engine configuration.
func DefaultConfig() Config {
	return Config{
		CachePath:      "/tmp/disk-cache",
		MaxSizeBytes:   100 * 1024 * 1024 * 1024, // 100 GB
		EvictionPolicy: "lru",
		MetadataPath:   "/tmp/disk-cache-meta",
	}
}

// now is a helper for timestamp.
func now() int64 {
	return time.Now().UnixNano()
}
