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
	// Legacy counters kept for compatibility with existing agents and dashboards.
	BlocksStored    int64
	BlocksRetrieved int64
	BlocksEvicted   int64
	DiskUsedBytes   int64

	// Current metadata inventory.
	BlockEntries    int64
	SentinelEntries int64
	ChunkEntries    int64

	// Event counters for finer-grained observability.
	PutRequests       int64
	GetRequests       int64
	GetHits           int64
	GetMisses         int64
	MatchRequests     int64
	MatchHits         int64
	MatchedTokens     int64
	RecordRequests    int64
	RecordBatchCalls  int64
	ChunkPutRequests  int64
	ChunkListRequests int64
	ChunkListHits     int64
	ChunksStored      int64
	ChunksRetrieved   int64
	RemoveRequests    int64
	EvictRequests     int64
}

// MatchResult is the result of a cache match query.
type MatchResult struct {
	MatchedTokens int    `json:"matched_tokens"`
	PromptHash    string `json:"prompt_hash"`
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
	Match(tokenIDs []int64, mmHashes []string, blockSize int) MatchResult
	RecordSentinel(promptHash string, numTokens int) error
	RecordAll(tokenIDs []int64, mmHashes []string, blockSize int) error
	PutChunk(prefixKey, layerName string, chunkIndex, numTokens int) error
	ListChunks(prefixKey, layerName string) ([]int, error)
	RecordRetrieved(count int64)
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
