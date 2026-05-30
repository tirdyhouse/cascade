package cache

import (
	"path/filepath"
	"reflect"
	"testing"
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
