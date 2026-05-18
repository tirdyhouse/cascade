// Package cache provides the core disk cache engine.
// Python handles file I/O, Go manages metadata + eviction.
package cache

import (
	"time"
)

// BlockMeta stores metadata for a cached block on disk.
type BlockMeta struct {
	Hash       uint64 // Block hash (lookup key)
	FilePath   string // Relative path under cache root
	Size       int64  // Data size in bytes (for eviction)
	AccessTime int64  // Unix timestamp of last access
	CreateTime int64  // Unix timestamp of creation
}

// Stats exposes cache engine metrics.
type Stats struct {
	BlocksStored    int64
	BlocksRetrieved int64
	BlocksEvicted   int64
	DiskUsedBytes   int64
}

// Engine is the metadata + eviction engine.
// Python calls these after/before file I/O.
type Engine interface {
	Put(hash uint64, filePath string, size int64) error
	Get(hash uint64) (*BlockMeta, error)
	Remove(hash uint64) error
	Exists(hash uint64) bool
	Evict(targetBytes int64) []BlockMeta
	Stats() Stats
	Close() error
}

// Config holds engine configuration.
type Config struct {
	CachePath      string
	MaxSizeBytes   int64
	EvictionPolicy string
	MetadataPath   string
}

func DefaultConfig() Config {
	return Config{
		CachePath:      "/tmp/disk-cache",
		MaxSizeBytes:   100 * 1024 * 1024 * 1024,
		EvictionPolicy: "lru",
		MetadataPath:   "/tmp/disk-cache-meta",
	}
}

func now() int64 { return time.Now().UnixNano() }
