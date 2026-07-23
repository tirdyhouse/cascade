package cache

import (
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/cockroachdb/pebble"

	"predict/engine/pkg/metadata"
)

func newTestEngine(t *testing.T, maxSize int64) Engine {
	t.Helper()

	root := t.TempDir()
	eng, err := New(Config{
		CachePath:      filepath.Join(root, "blocks"),
		MetadataPath:   filepath.Join(root, "meta"),
		MaxSizeBytes:   maxSize,
		EvictionPolicy: "lru",
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	t.Cleanup(func() {
		if err := eng.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
	})
	return eng
}

func TestPutGetStatsAndRemove(t *testing.T) {
	eng := newTestEngine(t, 1024)

	if got := eng.Exists(0xabc); got {
		t.Fatalf("Exists() before Put = %v, want false", got)
	}
	if err := eng.Put(0xabc, "ab/cd/block.bin", 128); err != nil {
		t.Fatalf("Put() error = %v", err)
	}
	if got := eng.Exists(0xabc); !got {
		t.Fatalf("Exists() after Put = %v, want true", got)
	}

	meta, err := eng.Get(0xabc)
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if meta == nil {
		t.Fatal("Get() = nil, want metadata")
	}
	if meta.Hash != 0xabc || meta.FilePath != "ab/cd/block.bin" || meta.Size != 128 {
		t.Fatalf("Get() = %+v, want hash/path/size preserved", meta)
	}

	stats := eng.Stats()
	if stats.BlocksStored != 1 || stats.BlocksRetrieved != 1 || stats.DiskUsedBytes != 128 {
		t.Fatalf("Stats() = %+v, want stored=1 retrieved=1 used=128", stats)
	}
	if stats.PutRequests != 1 || stats.GetRequests != 1 || stats.GetHits != 1 || stats.GetMisses != 0 || stats.BlockEntries != 1 {
		t.Fatalf("Stats() = %+v, want put/get counters and one block entry", stats)
	}

	if missing, err := eng.Get(0xdef); err != nil || missing != nil {
		t.Fatalf("Get(missing) = (%+v, %v), want nil, nil", missing, err)
	}
	if stats := eng.Stats(); stats.GetRequests != 2 || stats.GetMisses != 1 {
		t.Fatalf("Stats() after missing get = %+v, want get requests=2 misses=1", stats)
	}

	if err := eng.Remove(0xabc); err != nil {
		t.Fatalf("Remove() error = %v", err)
	}
	if got := eng.Exists(0xabc); got {
		t.Fatalf("Exists() after Remove = %v, want false", got)
	}
}

func TestRestartRestoresMetadataAndCounts(t *testing.T) {
	root := t.TempDir()
	cfg := Config{
		CachePath:      filepath.Join(root, "blocks"),
		MetadataPath:   filepath.Join(root, "meta"),
		MaxSizeBytes:   1024,
		EvictionPolicy: "lru",
	}
	eng1, err := New(cfg)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if err := eng1.Put(0x111, "one.bin", 10); err != nil {
		t.Fatalf("Put() error = %v", err)
	}
	tokens := []int64{1, 2, 3, 4, 5, 6}
	if err := eng1.RecordAll(tokens, nil, 2); err != nil {
		t.Fatalf("RecordAll() error = %v", err)
	}
	if err := eng1.PutChunk("prompt-a", "layer.0", 0, 3); err != nil {
		t.Fatalf("PutChunk() error = %v", err)
	}
	if err := eng1.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	eng2, err := New(cfg)
	if err != nil {
		t.Fatalf("New() reopen error = %v", err)
	}
	t.Cleanup(func() {
		if err := eng2.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
	})

	stats := eng2.Stats()
	if stats.BlockEntries != 1 || stats.ChunkEntries != 1 || stats.SentinelEntries != 3 {
		t.Fatalf("Stats() after reopen = %+v, want block/chunk entries=1 and 3 sentinels", stats)
	}
	if got := eng2.Exists(0x111); !got {
		t.Fatal("Exists() after reopen = false, want true")
	}
	match := eng2.Match(tokens, nil, 2)
	if match.MatchedTokens != 6 || match.PromptHash == "" {
		t.Fatalf("Match() after reopen = %+v, want matched tokens=6 with prompt hash", match)
	}
	chunks, err := eng2.ListChunks("prompt-a", "layer.0")
	if err != nil {
		t.Fatalf("ListChunks() after reopen error = %v", err)
	}
	if !reflect.DeepEqual(chunks, []int{0}) {
		t.Fatalf("ListChunks() after reopen = %v, want [0]", chunks)
	}
}

func TestMissingChunkDoesNotChangeRetrievedCount(t *testing.T) {
	eng := newTestEngine(t, 1024)
	if err := eng.RecordAll([]int64{1, 2, 3, 4, 5, 6}, nil, 2); err != nil {
		t.Fatalf("RecordAll() error = %v", err)
	}
	if err := eng.PutChunk("prompt-a", "layer.0", 0, 3); err != nil {
		t.Fatalf("PutChunk() error = %v", err)
	}
	before := eng.Stats().BlocksRetrieved
	chunks, err := eng.ListChunks("prompt-a", "layer.0")
	if err != nil {
		t.Fatalf("ListChunks() error = %v", err)
	}
	if got := eng.Stats().BlocksRetrieved - before; got != 0 {
		t.Fatalf("ListChunks() retrieved delta = %d, want 0", got)
	}
	if len(chunks) != 1 || chunks[0] != 0 {
		t.Fatalf("ListChunks() = %v, want [0]", chunks)
	}
}

func TestConcurrentRequestsKeepMetadataAndStatsConsistent(t *testing.T) {
	eng := newTestEngine(t, 1<<30)
	const (
		workers         = 8
		blocksPerWorker = 25
	)

	var wg sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		worker := worker
		wg.Add(1)
		go func() {
			defer wg.Done()
			for index := 0; index < blocksPerWorker; index++ {
				hash := uint64(worker*blocksPerWorker + index + 1)
				if err := eng.Put(hash, fmt.Sprintf("worker-%d/block-%d.bin", worker, index), 64); err != nil {
					t.Errorf("Put(%d, %d) error = %v", worker, index, err)
					return
				}
				if meta, err := eng.Get(hash); err != nil || meta == nil {
					t.Errorf("Get(%d, %d) = (%+v, %v), want metadata", worker, index, meta, err)
					return
				}

				tokens := []int64{int64(worker), int64(index), int64(worker + index), int64(worker*index + 1)}
				if err := eng.RecordAll(tokens, nil, 2); err != nil {
					t.Errorf("RecordAll(%d, %d) error = %v", worker, index, err)
					return
				}
				match := eng.Match(tokens, nil, 2)
				if match.MatchedTokens != 4 || match.PromptHash == "" {
					t.Errorf("Match(%d, %d) = %+v, want full hit", worker, index, match)
					return
				}
				if err := eng.PutChunk(match.PromptHash, "layer.0", index, 2); err != nil {
					t.Errorf("PutChunk(%d, %d) error = %v", worker, index, err)
					return
				}
				chunks, err := eng.ListChunks(match.PromptHash, "layer.0")
				if err != nil {
					t.Errorf("ListChunks(%d, %d) error = %v", worker, index, err)
					return
				}
				if len(chunks) != 1 || chunks[0] != index {
					t.Errorf("ListChunks(%d, %d) = %v, want [%d]", worker, index, chunks, index)
					return
				}
				eng.RecordRetrieved(1)
			}
		}()
	}
	wg.Wait()

	expected := int64(workers * blocksPerWorker)
	stats := eng.Stats()
	if stats.BlockEntries != expected || stats.ChunkEntries != expected || stats.SentinelEntries != expected*2 {
		t.Fatalf("Stats() inventory = %+v, want block/chunk entries=%d sentinel entries=%d", stats, expected, expected*2)
	}
	if stats.PutRequests != expected || stats.GetRequests != expected || stats.GetHits != expected || stats.GetMisses != 0 {
		t.Fatalf("Stats() block counters = %+v, want put/get/getHits=%d getMisses=0", stats, expected)
	}
	if stats.RecordBatchCalls != expected || stats.MatchRequests != expected || stats.MatchHits != expected || stats.MatchedTokens != expected*4 {
		t.Fatalf("Stats() match counters = %+v, want record/match=%d matchedTokens=%d", stats, expected, expected*4)
	}
	if stats.ChunkPutRequests != expected || stats.ChunksStored != expected || stats.ChunkListRequests != expected || stats.ChunkListHits != expected {
		t.Fatalf("Stats() chunk counters = %+v, want chunk put/list=%d", stats, expected)
	}
	if stats.ChunksRetrieved != expected {
		t.Fatalf("ChunksRetrieved = %d, want %d", stats.ChunksRetrieved, expected)
	}
	if stats.BlocksRetrieved != expected*2 {
		t.Fatalf("BlocksRetrieved = %d, want %d (Get hits + explicit retrieved records)", stats.BlocksRetrieved, expected*2)
	}
}

func TestEvictRemovesLeastRecentlyUsedBlocks(t *testing.T) {
	eng := newTestEngine(t, 200)

	if err := eng.Put(0x1, "one.bin", 100); err != nil {
		t.Fatalf("Put(1) error = %v", err)
	}
	if err := eng.Put(0x2, "two.bin", 100); err != nil {
		t.Fatalf("Put(2) error = %v", err)
	}
	if _, err := eng.Get(0x1); err != nil {
		t.Fatalf("Get(1) error = %v", err)
	}

	evicted := eng.Evict(100)
	if len(evicted) != 1 {
		t.Fatalf("Evict() returned %d blocks, want 1: %+v", len(evicted), evicted)
	}
	if evicted[0].Hash != 0x2 {
		t.Fatalf("Evict() removed hash %#x, want least-recently-used hash 0x2", evicted[0].Hash)
	}
	if eng.Exists(0x2) {
		t.Fatal("evicted block still exists in metadata")
	}
	if !eng.Exists(0x1) {
		t.Fatal("recently accessed block was evicted")
	}

	stats := eng.Stats()
	if stats.BlocksEvicted != 1 || stats.DiskUsedBytes != 100 || stats.EvictRequests != 1 {
		t.Fatalf("Stats() = %+v, want evicted=1 used=100 evictRequests=1", stats)
	}
}

func TestRecordAllMatchAndChunks(t *testing.T) {
	eng := newTestEngine(t, 1024)
	tokens := []int64{11, 22, 33, 44, 55, 66}
	mmHashes := []string{"image-a"}

	if err := eng.RecordAll(tokens, mmHashes, 2); err != nil {
		t.Fatalf("RecordAll() error = %v", err)
	}
	match := eng.Match(tokens, mmHashes, 2)
	if match.MatchedTokens != 6 {
		t.Fatalf("Match().MatchedTokens = %d, want 6", match.MatchedTokens)
	}
	if match.PromptHash == "" {
		t.Fatal("Match().PromptHash is empty")
	}
	stats := eng.Stats()
	if stats.RecordBatchCalls != 1 || stats.MatchRequests != 1 || stats.MatchHits != 1 || stats.MatchedTokens != 6 || stats.SentinelEntries != 3 {
		t.Fatalf("Stats() after match = %+v, want record/match counters and 3 sentinel entries", stats)
	}

	partial := eng.Match(tokens[:5], mmHashes, 2)
	if partial.MatchedTokens != 4 {
		t.Fatalf("partial Match().MatchedTokens = %d, want 4", partial.MatchedTokens)
	}

	miss := eng.Match(tokens, []string{"different-mm"}, 2)
	if miss.MatchedTokens != 0 || miss.PromptHash != "" {
		t.Fatalf("miss Match() = %+v, want zero value", miss)
	}

	if err := eng.PutChunk(match.PromptHash, "layer.0", 0, 2); err != nil {
		t.Fatalf("PutChunk(0) error = %v", err)
	}
	if err := eng.PutChunk(match.PromptHash, "layer.0", 1, 2); err != nil {
		t.Fatalf("PutChunk(1) error = %v", err)
	}
	before := eng.Stats().BlocksRetrieved
	chunks, err := eng.ListChunks(match.PromptHash, "layer.0")
	if err != nil {
		t.Fatalf("ListChunks() error = %v", err)
	}
	if !reflect.DeepEqual(chunks, []int{0, 1}) {
		t.Fatalf("ListChunks() = %v, want [0 1]", chunks)
	}
	stats = eng.Stats()
	if stats.ChunkPutRequests != 2 || stats.ChunksStored != 2 || stats.ChunkEntries != 2 || stats.ChunkListRequests != 1 || stats.ChunkListHits != 1 {
		t.Fatalf("Stats() after chunk ops = %+v, want chunk counters", stats)
	}
	if got := stats.BlocksRetrieved - before; got != 0 {
		t.Fatalf("ListChunks() retrieval stats delta = %d, want 0", got)
	}

	eng.RecordRetrieved(int64(len(chunks)))
	stats = eng.Stats()
	if got := stats.BlocksRetrieved - before; got != int64(len(chunks)) {
		t.Fatalf("RecordRetrieved() stats delta = %d, want %d", got, len(chunks))
	}
	if stats.ChunksRetrieved != int64(len(chunks)) {
		t.Fatalf("ChunksRetrieved = %d, want %d", stats.ChunksRetrieved, len(chunks))
	}
}

// ── V2 chunk API tests ──

func obj(namespace, key, shard, filePath string, index, start, end int, size int64) ChunkObject {
	return ChunkObject{
		Namespace:   namespace,
		Key:         key,
		Shard:       shard,
		FilePath:    filePath,
		Index:       index,
		StartTokens: start,
		EndTokens:   end,
		Size:        size,
	}
}

func cand(key string, endTokens int) ChunkCandidate {
	return ChunkCandidate{Key: key, EndTokens: endTokens}
}

func TestV2CommitChunksBasic(t *testing.T) {
	eng := newTestEngine(t, 1<<20)

	objects := []ChunkObject{
		obj("ns1", "chunk-a", "shard1", "path/a.bin", 0, 0, 100, 50),
		obj("ns1", "chunk-a", "shard2", "path/a.bin", 0, 0, 100, 50),
		obj("ns1", "chunk-b", "shard1", "path/b.bin", 1, 100, 200, 60),
	}
	if err := eng.CommitChunks(objects); err != nil {
		t.Fatalf("CommitChunks() error = %v", err)
	}

	stats := eng.Stats()
	if stats.ChunksStored != 3 {
		t.Fatalf("ChunksStored = %d, want 3", stats.ChunksStored)
	}
	if stats.ChunkEntries != 3 {
		t.Fatalf("ChunkEntries (v1+v2) = %d, want 3", stats.ChunkEntries)
	}
}

func TestV2CommitChunksValidation(t *testing.T) {
	tests := []struct {
		name string
		obj  ChunkObject
	}{
		{"empty namespace", obj("", "k", "s", "f", 0, 0, 10, 1)},
		{"empty key", obj("ns", "", "s", "f", 0, 0, 10, 1)},
		{"empty shard", obj("ns", "k", "", "f", 0, 0, 10, 1)},
		{"empty file_path", obj("ns", "k", "s", "", 0, 0, 10, 1)},
		{"negative start_tokens", obj("ns", "k", "s", "f", 0, -1, 10, 1)},
		{"end <= start", obj("ns", "k", "s", "f", 0, 5, 5, 1)},
		{"end < start", obj("ns", "k", "s", "f", 0, 10, 5, 1)},
		{"negative size", obj("ns", "k", "s", "f", 0, 0, 10, -1)},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			eng := newTestEngine(t, 1<<20)
			err := eng.CommitChunks([]ChunkObject{tc.obj})
			if err == nil {
				t.Fatal("expected validation error")
			}
		})
	}
}

func TestV2CommitChunksIdempotent(t *testing.T) {
	eng := newTestEngine(t, 1<<20)

	objects := []ChunkObject{
		obj("ns1", "chunk-a", "shard1", "path/a.bin", 0, 0, 100, 50),
	}
	// First commit
	if err := eng.CommitChunks(objects); err != nil {
		t.Fatalf("first CommitChunks() error = %v", err)
	}
	before := eng.Stats()

	// Idempotent re-commit
	if err := eng.CommitChunks(objects); err != nil {
		t.Fatalf("second CommitChunks() error = %v", err)
	}
	after := eng.Stats()

	if after.ChunksStored != before.ChunksStored {
		t.Fatalf("ChunksStored changed from %d to %d after idempotent commit", before.ChunksStored, after.ChunksStored)
	}
	if after.ChunkEntries != before.ChunkEntries {
		t.Fatalf("ChunkEntries changed after idempotent commit: %d -> %d", before.ChunkEntries, after.ChunkEntries)
	}
	if after.DiskUsedBytes != before.DiskUsedBytes {
		t.Fatalf("DiskUsedBytes changed after idempotent commit: %d -> %d", before.DiskUsedBytes, after.DiskUsedBytes)
	}
}

func TestV2MatchChunksFullHit(t *testing.T) {
	eng := newTestEngine(t, 1<<20)

	// Commit two chunks with two shards each
	if err := eng.CommitChunks([]ChunkObject{
		obj("ns1", "chunk-a", "shard1", "p/a.bin", 0, 0, 100, 50),
		obj("ns1", "chunk-a", "shard2", "p/a.bin", 0, 0, 100, 50),
		obj("ns1", "chunk-b", "shard1", "p/b.bin", 1, 100, 200, 60),
		obj("ns1", "chunk-b", "shard2", "p/b.bin", 1, 100, 200, 60),
	}); err != nil {
		t.Fatalf("CommitChunks() error = %v", err)
	}

	result, err := eng.MatchChunks("ns1", []ChunkCandidate{
		cand("chunk-a", 100),
		cand("chunk-b", 200),
	}, []string{"shard1", "shard2"})
	if err != nil {
		t.Fatalf("MatchChunks() error = %v", err)
	}
	if result.MatchedChunks != 2 {
		t.Fatalf("MatchedChunks = %d, want 2", result.MatchedChunks)
	}
	if result.MatchedTokens != 200 {
		t.Fatalf("MatchedTokens = %d, want 200", result.MatchedTokens)
	}
	if !reflect.DeepEqual(result.MatchedKeys, []string{"chunk-a", "chunk-b"}) {
		t.Fatalf("MatchedKeys = %v, want [chunk-a chunk-b]", result.MatchedKeys)
	}
}

func TestV2PolicyKeyRoundTrip(t *testing.T) {
	cases := []struct {
		namespace string
		key       string
		shard     string
	}{
		{namespace: "model/ns:1", key: "key:with:colons", shard: "层:0"},
		{namespace: "ns", key: "key", shard: "shard"},
	}
	for _, tc := range cases {
		t.Run(tc.namespace+"/"+tc.key+"/"+tc.shard, func(t *testing.T) {
			namespace, key, shard, ok := parseV2PolicyKey(
				v2PolicyKey(tc.namespace, tc.key, tc.shard),
			)
			if !ok {
				t.Fatal("parseV2PolicyKey() rejected a valid key")
			}
			if namespace != tc.namespace || key != tc.key || shard != tc.shard {
				t.Fatalf("parsed identity = (%q, %q, %q), want (%q, %q, %q)", namespace, key, shard, tc.namespace, tc.key, tc.shard)
			}
		})
	}

	for _, malformed := range []string{"v2:0002:ns:0003:key", "v2:zzzz:ns:0003:key:0001:s"} {
		if _, _, _, ok := parseV2PolicyKey(malformed); ok {
			t.Fatalf("parseV2PolicyKey(%q) accepted malformed input", malformed)
		}
	}
}

func TestV2MatchChunksFirstMissing(t *testing.T) {
	eng := newTestEngine(t, 1<<20)

	if err := eng.CommitChunks([]ChunkObject{
		obj("ns1", "chunk-a", "shard1", "p/a.bin", 0, 0, 100, 50),
		// missing shard2 for chunk-a
	}); err != nil {
		t.Fatalf("CommitChunks() error = %v", err)
	}

	result, err := eng.MatchChunks("ns1", []ChunkCandidate{
		cand("chunk-a", 100),
	}, []string{"shard1", "shard2"})
	if err != nil {
		t.Fatalf("MatchChunks() error = %v", err)
	}
	if result.MatchedChunks != 0 {
		t.Fatalf("MatchedChunks = %d, want 0 (first candidate missing shard)", result.MatchedChunks)
	}
	if result.MatchedKeys != nil {
		t.Fatalf("MatchedKeys = %v, want nil on a complete miss", result.MatchedKeys)
	}
}

func TestV2MatchChunksMiddleMissing(t *testing.T) {
	eng := newTestEngine(t, 1<<20)

	if err := eng.CommitChunks([]ChunkObject{
		obj("ns1", "chunk-a", "shard1", "p/a.bin", 0, 0, 100, 50),
		obj("ns1", "chunk-a", "shard2", "p/a.bin", 0, 0, 100, 50),
		// chunk-b missing shard2
		obj("ns1", "chunk-b", "shard1", "p/b.bin", 1, 100, 200, 60),
	}); err != nil {
		t.Fatalf("CommitChunks() error = %v", err)
	}

	result, err := eng.MatchChunks("ns1", []ChunkCandidate{
		cand("chunk-a", 100),
		cand("chunk-b", 200),
	}, []string{"shard1", "shard2"})
	if err != nil {
		t.Fatalf("MatchChunks() error = %v", err)
	}
	if result.MatchedChunks != 1 {
		t.Fatalf("MatchedChunks = %d, want 1 (stops at second candidate)", result.MatchedChunks)
	}
	if result.MatchedTokens != 100 {
		t.Fatalf("MatchedTokens = %d, want 100", result.MatchedTokens)
	}
	if !reflect.DeepEqual(result.MatchedKeys, []string{"chunk-a"}) {
		t.Fatalf("MatchedKeys = %v, want [chunk-a]", result.MatchedKeys)
	}
}

func TestV2MatchChunksMissingShard(t *testing.T) {
	eng := newTestEngine(t, 1<<20)

	if err := eng.CommitChunks([]ChunkObject{
		obj("ns1", "chunk-a", "shard1", "p/a.bin", 0, 0, 100, 50),
		obj("ns1", "chunk-a", "shard2", "p/a.bin", 0, 0, 100, 50),
	}); err != nil {
		t.Fatalf("CommitChunks() error = %v", err)
	}

	// Required shard "shard3" does not exist
	result, err := eng.MatchChunks("ns1", []ChunkCandidate{
		cand("chunk-a", 100),
	}, []string{"shard1", "shard2", "shard3"})
	if err != nil {
		t.Fatalf("MatchChunks() error = %v", err)
	}
	if result.MatchedChunks != 0 {
		t.Fatalf("MatchedChunks = %d, want 0 (missing required shard3)", result.MatchedChunks)
	}
}

func TestV2MatchChunksEndTokensMismatch(t *testing.T) {
	eng := newTestEngine(t, 1<<20)

	if err := eng.CommitChunks([]ChunkObject{
		obj("ns1", "chunk-a", "shard1", "p/a.bin", 0, 0, 100, 50),
		obj("ns1", "chunk-a", "shard2", "p/a.bin", 0, 0, 200, 50), // mismatched EndTokens
	}); err != nil {
		t.Fatalf("CommitChunks() error = %v", err)
	}

	result, err := eng.MatchChunks("ns1", []ChunkCandidate{
		cand("chunk-a", 100),
	}, []string{"shard1", "shard2"})
	if err != nil {
		t.Fatalf("MatchChunks() error = %v", err)
	}
	if result.MatchedChunks != 0 {
		t.Fatalf("MatchedChunks = %d, want 0 (EndTokens mismatch between shards)", result.MatchedChunks)
	}
}

func TestV2MatchChunksEmptyRequiredShards(t *testing.T) {
	eng := newTestEngine(t, 1<<20)
	_, err := eng.MatchChunks("ns1", []ChunkCandidate{cand("a", 100)}, nil)
	if err == nil {
		t.Fatal("expected error for empty required_shards")
	}
}

func TestV2MatchChunksEmptyCandidates(t *testing.T) {
	eng := newTestEngine(t, 1<<20)
	result, err := eng.MatchChunks("ns1", nil, []string{"shard1"})
	if err != nil {
		t.Fatalf("MatchChunks() error = %v", err)
	}
	if result.MatchedChunks != 0 {
		t.Fatalf("MatchedChunks = %d, want 0", result.MatchedChunks)
	}
}

func TestV2MatchChunksNonIncreasingEndTokens(t *testing.T) {
	eng := newTestEngine(t, 1<<20)
	_, err := eng.MatchChunks("ns1", []ChunkCandidate{
		cand("a", 100),
		cand("b", 50), // not strictly increasing
	}, []string{"shard1"})
	if err == nil {
		t.Fatal("expected error for non-increasing end_tokens")
	}
}

func TestV2MatchChunksDifferentNamespace(t *testing.T) {
	eng := newTestEngine(t, 1<<20)

	if err := eng.CommitChunks([]ChunkObject{
		obj("ns1", "chunk-a", "shard1", "p/a.bin", 0, 0, 100, 50),
	}); err != nil {
		t.Fatalf("CommitChunks() error = %v", err)
	}

	// Match in a different namespace should not find anything
	result, err := eng.MatchChunks("ns2", []ChunkCandidate{
		cand("chunk-a", 100),
	}, []string{"shard1"})
	if err != nil {
		t.Fatalf("MatchChunks() error = %v", err)
	}
	if result.MatchedChunks != 0 {
		t.Fatalf("MatchedChunks = %d, want 0 (different namespace)", result.MatchedChunks)
	}
}

func TestV2ResolveChunksOrdered(t *testing.T) {
	eng := newTestEngine(t, 1<<20)

	if err := eng.CommitChunks([]ChunkObject{
		obj("ns1", "key-b", "shard1", "path/b.bin", 1, 100, 200, 60),
		obj("ns1", "key-a", "shard1", "path/a.bin", 0, 0, 100, 50),
	}); err != nil {
		t.Fatalf("CommitChunks() error = %v", err)
	}

	results, err := eng.ResolveChunks("ns1", []string{"key-a", "key-b"}, "shard1")
	if err != nil {
		t.Fatalf("ResolveChunks() error = %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("got %d results, want 2", len(results))
	}
	// Verify order is preserved
	if results[0].Key != "key-a" || results[1].Key != "key-b" {
		t.Fatalf("results order = %v, want [key-a key-b]", []string{results[0].Key, results[1].Key})
	}
}

func TestV2ResolveChunksMissing(t *testing.T) {
	eng := newTestEngine(t, 1<<20)

	if err := eng.CommitChunks([]ChunkObject{
		obj("ns1", "key-a", "shard1", "path/a.bin", 0, 0, 100, 50),
	}); err != nil {
		t.Fatalf("CommitChunks() error = %v", err)
	}

	// key-b does not exist
	_, err := eng.ResolveChunks("ns1", []string{"key-a", "key-b"}, "shard1")
	if err == nil {
		t.Fatal("expected error for missing key")
	}
}

func TestV2V1Isolation(t *testing.T) {
	eng := newTestEngine(t, 1<<20)

	// Commit v2 chunk
	if err := eng.CommitChunks([]ChunkObject{
		obj("ns1", "chunk-a", "shard1", "p/a.bin", 0, 0, 100, 50),
	}); err != nil {
		t.Fatalf("CommitChunks() error = %v", err)
	}

	// Put v1 block
	if err := eng.Put(0x42, "block.bin", 100); err != nil {
		t.Fatalf("Put() error = %v", err)
	}

	// Put v1 chunk
	if err := eng.PutChunk("prefix1234567890123456789012345678", "layer.0", 0, 3); err != nil {
		t.Fatalf("PutChunk() error = %v", err)
	}

	stats := eng.Stats()
	if stats.BlockEntries != 1 {
		t.Fatalf("BlockEntries = %d, want 1", stats.BlockEntries)
	}
	if stats.ChunkEntries != 2 { // 1 v1 chunk + 1 v2 object
		t.Fatalf("ChunkEntries (v1+v2) = %d, want 2", stats.ChunkEntries)
	}
	if stats.ChunksStored != 2 { // 1 v1 + 1 v2
		t.Fatalf("ChunksStored = %d, want 2", stats.ChunksStored)
	}
}

func TestV2Eviction(t *testing.T) {
	// Small cache to trigger eviction at commit time
	eng := newTestEngine(t, 200)

	// Commit first v2 chunk (150 bytes)
	if err := eng.CommitChunks([]ChunkObject{
		obj("ns1", "chunk-a", "shard1", "p/a.bin", 0, 0, 100, 150),
	}); err != nil {
		t.Fatalf("CommitChunks() first error = %v", err)
	}
	if got := eng.Stats().DiskUsedBytes; got != 150 {
		t.Fatalf("DiskUsedBytes after first commit = %d, want 150", got)
	}

	// Commit second chunk (100 bytes) — total would be 250 > 200
	// Auto-eviction should evict chunk-a during this commit
	if err := eng.CommitChunks([]ChunkObject{
		obj("ns1", "chunk-b", "shard1", "p/b.bin", 1, 100, 200, 100),
	}); err != nil {
		t.Fatalf("CommitChunks() second error = %v", err)
	}

	// After auto-eviction, LRU should only track chunk-b (100 bytes)
	if got := eng.Stats().DiskUsedBytes; got != 100 {
		t.Fatalf("DiskUsedBytes after auto-eviction = %d, want 100", got)
	}

	// chunk-a should be gone from metadata
	_, err := eng.ResolveChunks("ns1", []string{"chunk-a"}, "shard1")
	if err == nil {
		t.Fatal("expected error for chunk-a after auto-eviction")
	}

	// chunk-b should still be resolvable
	resolved, err := eng.ResolveChunks("ns1", []string{"chunk-b"}, "shard1")
	if err != nil {
		t.Fatalf("ResolveChunks(chunk-b) error = %v", err)
	}
	if resolved[0].Key != "chunk-b" {
		t.Fatalf("resolved key = %q, want chunk-b", resolved[0].Key)
	}

	// Public Evict should still work and evict chunk-b
	evicted := eng.Evict(100)
	if len(evicted) != 1 {
		t.Fatalf("Evict() returned %d entries, want 1: %+v", len(evicted), evicted)
	}
	if !strings.Contains(evicted[0].FilePath, "p/b.bin") {
		t.Fatalf("expected chunk-b to be evicted, got: %+v", evicted)
	}

	// chunk-b should now be gone too
	_, err = eng.ResolveChunks("ns1", []string{"chunk-b"}, "shard1")
	if err == nil {
		t.Fatal("expected error for chunk-b after Evict")
	}

	// LRU should be empty
	if got := eng.Stats().DiskUsedBytes; got != 0 {
		t.Fatalf("DiskUsedBytes after full eviction = %d, want 0", got)
	}
}
func TestNewRejectsCorruptV2MetadataAndReleasesDatabaseLock(t *testing.T) {
	root := t.TempDir()
	cfg := Config{
		CachePath:      filepath.Join(root, "blocks"),
		MetadataPath:   filepath.Join(root, "meta"),
		MaxSizeBytes:   1 << 20,
		EvictionPolicy: "lru",
	}

	store, err := metadata.Open(cfg.MetadataPath)
	if err != nil {
		t.Fatalf("metadata.Open() error = %v", err)
	}
	key := rawV2KeyForTest("ns", "key", "shard")
	if err := store.Close(); err != nil {
		t.Fatalf("close initial metadata store: %v", err)
	}

	rawDB, err := pebble.Open(cfg.MetadataPath, &pebble.Options{})
	if err != nil {
		t.Fatalf("open raw metadata DB: %v", err)
	}
	if err := rawDB.Set(key, []byte("not-a-gob"), pebble.Sync); err != nil {
		rawDB.Close()
		t.Fatalf("inject corrupt v2 metadata: %v", err)
	}
	if err := rawDB.Close(); err != nil {
		t.Fatalf("close raw metadata DB: %v", err)
	}

	eng, err := New(cfg)
	if err == nil || eng != nil {
		t.Fatalf("New() = (%v, %v), want nil engine and rebuild error", eng, err)
	}
	if !strings.Contains(err.Error(), "rebuild eviction tracker") {
		t.Fatalf("New() error = %v, want rebuild context", err)
	}

	reopened, err := metadata.Open(cfg.MetadataPath)
	if err != nil {
		t.Fatalf("metadata lock was not released after failed New(): %v", err)
	}
	if err := reopened.Close(); err != nil {
		t.Fatalf("close reopened metadata store: %v", err)
	}
}

func rawV2KeyForTest(namespace, key, shard string) []byte {
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

func TestV2RestartRestores(t *testing.T) {
	root := t.TempDir()
	cfg := Config{
		CachePath:      filepath.Join(root, "blocks"),
		MetadataPath:   filepath.Join(root, "meta"),
		MaxSizeBytes:   1 << 20,
		EvictionPolicy: "lru",
	}

	eng1, err := New(cfg)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if err := eng1.CommitChunks([]ChunkObject{
		obj("ns1", "chunk-a", "shard1", "p/a.bin", 0, 0, 100, 50),
		obj("ns1", "chunk-a", "shard2", "p/a.bin", 0, 0, 100, 50),
	}); err != nil {
		t.Fatalf("CommitChunks() error = %v", err)
	}
	beforeStats := eng1.Stats()
	if err := eng1.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	eng2, err := New(cfg)
	if err != nil {
		t.Fatalf("New() reopen error = %v", err)
	}
	t.Cleanup(func() {
		eng2.Close()
	})

	afterStats := eng2.Stats()
	if afterStats.ChunkEntries != beforeStats.ChunkEntries {
		t.Fatalf("ChunkEntries after restart = %d, want %d", afterStats.ChunkEntries, beforeStats.ChunkEntries)
	}
	_ = beforeStats.ChunksStored

	result, err := eng2.MatchChunks("ns1", []ChunkCandidate{
		cand("chunk-a", 100),
	}, []string{"shard1", "shard2"})
	if err != nil {
		t.Fatalf("MatchChunks() after restart error = %v", err)
	}
	if result.MatchedChunks != 1 {
		t.Fatalf("MatchedChunks after restart = %d, want 1", result.MatchedChunks)
	}

	resolved, err := eng2.ResolveChunks("ns1", []string{"chunk-a"}, "shard1")
	if err != nil {
		t.Fatalf("ResolveChunks() after restart error = %v", err)
	}
	if len(resolved) != 1 || resolved[0].Key != "chunk-a" {
		t.Fatalf("ResolveChunks() after restart = %+v, want [chunk-a]", resolved)
	}
}

func TestV2ConcurrentCommitAndMatch(t *testing.T) {
	eng := newTestEngine(t, 1<<30)
	const (
		workers    = 8
		chunksEach = 20
	)

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		w := w
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < chunksEach; i++ {
				key := fmt.Sprintf("chunk-%d-%d", w, i)
				objects := []ChunkObject{
					obj("conc-ns", key, "shard1", fmt.Sprintf("p/%s.bin", key), i, i*100, (i+1)*100, 50),
					obj("conc-ns", key, "shard2", fmt.Sprintf("p/%s.bin", key), i, i*100, (i+1)*100, 50),
				}
				if err := eng.CommitChunks(objects); err != nil {
					t.Errorf("CommitChunks error: %v", err)
					return
				}

				result, err := eng.MatchChunks("conc-ns", []ChunkCandidate{
					cand(key, (i+1)*100),
				}, []string{"shard1", "shard2"})
				if err != nil {
					t.Errorf("MatchChunks error: %v", err)
					return
				}
				if result.MatchedChunks != 1 {
					t.Errorf("MatchChunks: got %d matched, want 1", result.MatchedChunks)
				}
			}
		}()
	}
	wg.Wait()

	stats := eng.Stats()
	expected := int64(workers * chunksEach)
	if stats.ChunksStored != expected*2 {
		t.Fatalf("ChunksStored = %d, want %d", stats.ChunksStored, expected*2)
	}
}

func TestV2MatchDeduplicatesRequiredShards(t *testing.T) {
	eng := newTestEngine(t, 1<<20)

	if err := eng.CommitChunks([]ChunkObject{
		obj("ns1", "chunk-a", "shard1", "p/a.bin", 0, 0, 100, 50),
	}); err != nil {
		t.Fatalf("CommitChunks() error = %v", err)
	}

	// Duplicate shard1 in required_shards should still match
	result, err := eng.MatchChunks("ns1", []ChunkCandidate{
		cand("chunk-a", 100),
	}, []string{"shard1", "shard1"})
	if err != nil {
		t.Fatalf("MatchChunks() error = %v", err)
	}
	if result.MatchedChunks != 1 {
		t.Fatalf("MatchedChunks = %d, want 1 (deduplicated)", result.MatchedChunks)
	}
}

func TestV2StatsCounters(t *testing.T) {
	eng := newTestEngine(t, 1<<20)

	// Commit v2 objects
	if err := eng.CommitChunks([]ChunkObject{
		obj("ns1", "a", "s1", "f", 0, 0, 100, 50),
	}); err != nil {
		t.Fatalf("CommitChunks() error = %v", err)
	}

	// Match
	result, err := eng.MatchChunks("ns1", []ChunkCandidate{cand("a", 100)}, []string{"s1"})
	if err != nil {
		t.Fatalf("MatchChunks() error = %v", err)
	}
	if result.MatchedChunks != 1 {
		t.Fatalf("MatchedChunks = %d, want 1", result.MatchedChunks)
	}

	// Resolve
	resolved, err := eng.ResolveChunks("ns1", []string{"a"}, "s1")
	if err != nil {
		t.Fatalf("ResolveChunks() error = %v", err)
	}
	if len(resolved) != 1 {
		t.Fatalf("ResolveChunks() = %d results, want 1", len(resolved))
	}

	// Record retrieved
	eng.RecordRetrieved(5)

	stats := eng.Stats()
	if stats.MatchRequests <= 0 {
		t.Fatal("MatchRequests should be > 0")
	}
	if stats.MatchHits <= 0 {
		t.Fatal("MatchHits should be > 0")
	}
	if stats.ChunksStored != 1 {
		t.Fatalf("ChunksStored = %d, want 1", stats.ChunksStored)
	}
	if stats.ChunksRetrieved != 5 {
		t.Fatalf("ChunksRetrieved = %d, want 5", stats.ChunksRetrieved)
	}
	if stats.BlocksRetrieved != 5 {
		t.Fatalf("BlocksRetrieved = %d, want 5", stats.BlocksRetrieved)
	}
}

func TestV2InvalidateChunkRemovesMetadataAndAccounting(t *testing.T) {
	eng := newTestEngine(t, 1<<20)
	if err := eng.CommitChunks([]ChunkObject{
		obj("ns", "chunk-a", "shard-0", "p/a.cobj", 0, 0, 100, 64),
	}); err != nil {
		t.Fatalf("CommitChunks() error = %v", err)
	}
	if got := eng.Stats().DiskUsedBytes; got != 64 {
		t.Fatalf("DiskUsedBytes before invalidation = %d, want 64", got)
	}

	if err := eng.InvalidateChunk("ns", "chunk-a", "shard-0"); err != nil {
		t.Fatalf("InvalidateChunk() error = %v", err)
	}
	if err := eng.InvalidateChunk("ns", "chunk-a", "shard-0"); err != nil {
		t.Fatalf("idempotent InvalidateChunk() error = %v", err)
	}

	match, err := eng.MatchChunks(
		"ns",
		[]ChunkCandidate{cand("chunk-a", 100)},
		[]string{"shard-0"},
	)
	if err != nil {
		t.Fatalf("MatchChunks() after invalidation error = %v", err)
	}
	if match.MatchedChunks != 0 || match.MatchedTokens != 0 {
		t.Fatalf("MatchChunks() after invalidation = %+v, want miss", match)
	}
	if _, err := eng.ResolveChunks("ns", []string{"chunk-a"}, "shard-0"); err == nil {
		t.Fatal("ResolveChunks() after invalidation succeeded, want missing error")
	}
	stats := eng.Stats()
	if stats.DiskUsedBytes != 0 || stats.ChunkEntries != 0 {
		t.Fatalf("stats after invalidation = %+v, want zero inventory", stats)
	}
}

func TestV2InvalidateChunkValidatesIdentity(t *testing.T) {
	eng := newTestEngine(t, 1<<20)
	for _, tc := range []struct {
		name      string
		namespace string
		key       string
		shard     string
	}{
		{name: "namespace", key: "key", shard: "shard"},
		{name: "key", namespace: "ns", shard: "shard"},
		{name: "shard", namespace: "ns", key: "key"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := eng.InvalidateChunk(tc.namespace, tc.key, tc.shard); !errors.Is(err, ErrInvalidArgument) {
				t.Fatalf("InvalidateChunk() error = %v, want ErrInvalidArgument", err)
			}
		})
	}
}

func TestV2CommitChunksConflict(t *testing.T) {
	eng := newTestEngine(t, 1<<20)

	// First commit
	if err := eng.CommitChunks([]ChunkObject{
		obj("ns1", "chunk-a", "shard1", "path/a.bin", 0, 0, 100, 50),
	}); err != nil {
		t.Fatalf("first CommitChunks() error = %v", err)
	}

	// Same identity but different file_path — should be rejected
	err := eng.CommitChunks([]ChunkObject{
		obj("ns1", "chunk-a", "shard1", "path/different.bin", 0, 0, 100, 50),
	})
	if err == nil {
		t.Fatal("expected conflict error for different file_path")
	}

	// Same identity but different index — should be rejected
	err = eng.CommitChunks([]ChunkObject{
		obj("ns1", "chunk-a", "shard1", "path/a.bin", 99, 0, 100, 50),
	})
	if err == nil {
		t.Fatal("expected conflict error for different index")
	}

	// Same identity but different size — should be rejected
	err = eng.CommitChunks([]ChunkObject{
		obj("ns1", "chunk-a", "shard1", "path/a.bin", 0, 0, 100, 999),
	})
	if err == nil {
		t.Fatal("expected conflict error for different size")
	}

	// Same identity but different end_tokens — should be rejected
	err = eng.CommitChunks([]ChunkObject{
		obj("ns1", "chunk-a", "shard1", "path/a.bin", 0, 0, 200, 50),
	})
	if err == nil {
		t.Fatal("expected conflict error for different end_tokens")
	}

	// Same identity but different start_tokens — should be rejected
	err = eng.CommitChunks([]ChunkObject{
		obj("ns1", "chunk-a", "shard1", "path/a.bin", 0, 50, 100, 50),
	})
	if err == nil {
		t.Fatal("expected conflict error for different start_tokens")
	}

	// Exact same fields — idempotent (no error)
	if err := eng.CommitChunks([]ChunkObject{
		obj("ns1", "chunk-a", "shard1", "path/a.bin", 0, 0, 100, 50),
	}); err != nil {
		t.Fatalf("idempotent re-commit error = %v", err)
	}

	// ChunksStored should not double-count idempotent re-commits
	stats := eng.Stats()
	if stats.ChunksStored != 1 {
		t.Fatalf("ChunksStored = %d, want 1 (idempotent re-commit should not double-count)", stats.ChunksStored)
	}
}

func TestV2MatchChunksEndTokensFinalNotSum(t *testing.T) {
	eng := newTestEngine(t, 1<<20)

	if err := eng.CommitChunks([]ChunkObject{
		obj("ns1", "chunk-a", "shard1", "p/a.bin", 0, 0, 100, 50),
		obj("ns1", "chunk-a", "shard2", "p/a.bin", 0, 0, 100, 50),
		obj("ns1", "chunk-b", "shard1", "p/b.bin", 1, 100, 300, 60),
		obj("ns1", "chunk-b", "shard2", "p/b.bin", 1, 100, 300, 60),
	}); err != nil {
		t.Fatalf("CommitChunks() error = %v", err)
	}

	// Match should return final end_tokens (300), not sum (100+300=400)
	result, err := eng.MatchChunks("ns1", []ChunkCandidate{
		cand("chunk-a", 100),
		cand("chunk-b", 300),
	}, []string{"shard1", "shard2"})
	if err != nil {
		t.Fatalf("MatchChunks() error = %v", err)
	}
	if result.MatchedTokens != 300 {
		t.Fatalf("MatchedTokens = %d, want 300 (final end_tokens, not sum)", result.MatchedTokens)
	}
	if result.MatchedChunks != 2 {
		t.Fatalf("MatchedChunks = %d, want 2", result.MatchedChunks)
	}
}

func TestV2MatchChunksRefreshLRU(t *testing.T) {
	eng := newTestEngine(t, 1<<20)

	if err := eng.CommitChunks([]ChunkObject{
		obj("ns1", "chunk-a", "shard1", "p/a.bin", 0, 0, 100, 50),
		obj("ns1", "chunk-b", "shard1", "p/b.bin", 1, 100, 200, 60),
	}); err != nil {
		t.Fatalf("CommitChunks() error = %v", err)
	}

	// Match chunk-b (most recent) - refreshes LRU
	result, err := eng.MatchChunks("ns1", []ChunkCandidate{
		cand("chunk-a", 100),
		cand("chunk-b", 200),
	}, []string{"shard1"})
	if err != nil {
		t.Fatalf("MatchChunks() error = %v", err)
	}
	if result.MatchedChunks != 2 {
		t.Fatalf("MatchedChunks = %d, want 2", result.MatchedChunks)
	}

	// After match, the LRU order should be chunk-b, chunk-a (chunk-b was refreshed last)
	// Evict 60 bytes — should evict chunk-a (least recently used)
	evicted := eng.Evict(60)
	if len(evicted) == 0 {
		t.Fatal("expected at least 1 evicted entry after match")
	}
	// chunk-a should be evicted first because it wasn't refreshed
	foundChunkA := false
	for _, e := range evicted {
		if e.FilePath != "" && strings.Contains(e.FilePath, "p/a.bin") {
			foundChunkA = true
		}
	}
	if !foundChunkA {
		t.Fatalf("expected chunk-a to be evicted first after match refresh, got: %+v", evicted)
	}
}

func TestV2CommitChunksEvictsPhysicalFileAtCapacity(t *testing.T) {
	eng := newTestEngine(t, 200)
	cacheRoot := eng.(*diskEngine).cfg.CachePath
	for _, name := range []string{"a.cobj", "b.cobj", "c.cobj"} {
		path := filepath.Join(cacheRoot, "p", name)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(name), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	if err := eng.CommitChunks([]ChunkObject{
		obj("ns", "a", "s", "p/a.cobj", 0, 0, 10, 100),
		obj("ns", "b", "s", "p/b.cobj", 1, 10, 20, 100),
	}); err != nil {
		t.Fatalf("initial CommitChunks() error = %v", err)
	}
	if err := eng.CommitChunks([]ChunkObject{
		obj("ns", "c", "s", "p/c.cobj", 2, 20, 30, 100),
	}); err != nil {
		t.Fatalf("overflow CommitChunks() error = %v", err)
	}

	if _, err := os.Stat(filepath.Join(cacheRoot, "p", "a.cobj")); !os.IsNotExist(err) {
		t.Fatalf("evicted file still exists, stat err = %v", err)
	}
	if _, err := eng.ResolveChunks("ns", []string{"a"}, "s"); err == nil {
		t.Fatal("expected evicted chunk metadata to be absent")
	}
	stats := eng.Stats()
	if stats.DiskUsedBytes > 200 {
		t.Fatalf("DiskUsedBytes = %d, want <= 200", stats.DiskUsedBytes)
	}
}

func TestV2CommitChunksRejectsOverCapacityBatchWithoutEviction(t *testing.T) {
	eng := newTestEngine(t, 200)
	if err := eng.CommitChunks([]ChunkObject{
		obj("ns", "a", "s", "p/a.cobj", 0, 0, 10, 150),
	}); err != nil {
		t.Fatalf("initial CommitChunks() error = %v", err)
	}
	if err := eng.CommitChunks([]ChunkObject{
		obj("ns", "b", "s", "p/b.cobj", 1, 10, 20, 150),
		obj("ns", "c", "s", "p/c.cobj", 2, 20, 30, 100),
	}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("CommitChunks() error = %v, want ErrInvalidArgument", err)
	}
	if _, err := eng.ResolveChunks("ns", []string{"a"}, "s"); err != nil {
		t.Fatalf("existing entry was evicted by rejected batch: %v", err)
	}
}

func TestChunkPathsRejectTraversal(t *testing.T) {
	eng := newTestEngine(t, 1024)
	for _, path := range []string{"../escape", "/tmp/escape", "p/../../escape"} {
		err := eng.CommitChunks([]ChunkObject{
			obj("ns", "key"+strings.ReplaceAll(path, "/", "_"), "s", path, 0, 0, 1, 1),
		})
		if !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("CommitChunks(%q) error = %v, want ErrInvalidArgument", path, err)
		}
	}
	if err := eng.Put(1, "../escape", 1); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("Put traversal error = %v, want ErrInvalidArgument", err)
	}
}

func TestV2ResolveChunksRefreshLRU(t *testing.T) {
	eng := newTestEngine(t, 300)

	if err := eng.CommitChunks([]ChunkObject{
		obj("ns1", "chunk-a", "shard1", "p/a.bin", 0, 0, 100, 150),
		obj("ns1", "chunk-b", "shard1", "p/b.bin", 1, 100, 200, 100),
	}); err != nil {
		t.Fatalf("CommitChunks() error = %v", err)
	}

	// Resolve chunk-a — refreshes LRU
	_, err := eng.ResolveChunks("ns1", []string{"chunk-a"}, "shard1")
	if err != nil {
		t.Fatalf("ResolveChunks() error = %v", err)
	}

	// After resolve, chunk-a is most recently used.
	// Evict 150 bytes — should evict chunk-b (not chunk-a)
	evicted := eng.Evict(150)
	if len(evicted) == 0 {
		t.Fatal("expected at least 1 evicted entry")
	}
	foundChunkB := false
	for _, e := range evicted {
		if strings.Contains(e.FilePath, "p/b.bin") {
			foundChunkB = true
		}
	}
	if !foundChunkB {
		t.Fatalf("expected chunk-b to be evicted first (chunk-a was refreshed by Resolve), got: %+v", evicted)
	}
}

func TestV2ValidateStringLengthOverflow(t *testing.T) {
	longNS := strings.Repeat("x", 70000)
	longKey := strings.Repeat("y", 70000)

	// Test validation directly (not through CommitChunks which uses Pebble internally)
	err := validateChunkObjects([]ChunkObject{
		obj(longNS, "k", "s", "f", 0, 0, 10, 1),
	})
	if err == nil {
		t.Fatal("expected error for namespace > 65535 bytes")
	}

	err = validateChunkObjects([]ChunkObject{
		obj("ns", longKey, "s", "f", 0, 0, 10, 1),
	})
	if err == nil {
		t.Fatal("expected error for key > 65535 bytes")
	}

	err = validateChunkObjects([]ChunkObject{
		obj("ns", "k", longKey, "f", 0, 0, 10, 1),
	})
	if err == nil {
		t.Fatal("expected error for shard > 65535 bytes")
	}
}

func TestV2MatchResolveInvalidNamespaceKeyShard(t *testing.T) {
	eng := newTestEngine(t, 1<<20)

	// Empty namespace
	_, err := eng.MatchChunks("", []ChunkCandidate{cand("a", 100)}, []string{"s1"})
	if err == nil {
		t.Fatal("expected error for empty namespace in MatchChunks")
	}

	_, err = eng.ResolveChunks("", []string{"a"}, "s1")
	if err == nil {
		t.Fatal("expected error for empty namespace in ResolveChunks")
	}

	// Empty keys
	_, err = eng.ResolveChunks("ns1", []string{}, "s1")
	if err == nil {
		t.Fatal("expected error for empty keys in ResolveChunks")
	}

	// Empty shard in Resolve
	_, err = eng.ResolveChunks("ns1", []string{"a"}, "")
	if err == nil {
		t.Fatal("expected error for empty shard in ResolveChunks")
	}

	// Empty required_shards in Match
	_, err = eng.MatchChunks("ns1", []ChunkCandidate{cand("a", 100)}, nil)
	if err == nil {
		t.Fatal("expected error for empty required_shards")
	}
}

func TestV2DuplicateShardInObjects(t *testing.T) {
	// Duplicate namespace/key/shard should be caught by validation
	err := validateChunkObjects([]ChunkObject{
		obj("ns1", "chunk-a", "shard1", "p/a.bin", 0, 0, 100, 50),
		obj("ns1", "chunk-a", "shard1", "p/a.bin", 0, 0, 100, 50),
	})
	if err == nil {
		t.Fatal("expected error for duplicate namespace/key/shard")
	}
}

func TestV2CommitChunksNonPositiveSize(t *testing.T) {
	eng := newTestEngine(t, 1<<20)

	err := eng.CommitChunks([]ChunkObject{
		obj("ns1", "a", "s1", "f", 0, 0, 10, 0),
	})
	if err == nil {
		t.Fatal("expected error for zero size")
	}

	err = eng.CommitChunks([]ChunkObject{
		obj("ns1", "a", "s1", "f", 0, 0, 10, -1),
	})
	if err == nil {
		t.Fatal("expected error for negative size")
	}
}
