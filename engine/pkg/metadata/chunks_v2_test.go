package metadata

import (
	"bytes"
	"strings"
	"testing"

	"github.com/cockroachdb/pebble"
)

func openTestStore(t *testing.T) *Store {
	t.Helper()
	store, err := Open(t.TempDir())
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
	})
	return store
}

func putCorruptV2(t *testing.T, store *Store, namespace, key, shard string) {
	t.Helper()
	if err := store.db.Set(
		v2ChunkKey(namespace, key, shard),
		[]byte("not-a-gob"),
		pebble.Sync,
	); err != nil {
		t.Fatalf("write corrupt v2 value: %v", err)
	}
}

func TestCommitChunkObjectsRejectsCorruptExistingValue(t *testing.T) {
	store := openTestStore(t)
	putCorruptV2(t, store, "ns", "chunk", "shard")

	_, err := store.CommitChunkObjects([]ChunkObjectMeta{{
		Namespace:   "ns",
		Key:         "chunk",
		Shard:       "shard",
		FilePath:    "path/chunk.cobj",
		Index:       0,
		StartTokens: 0,
		EndTokens:   8,
		Size:        128,
	}})
	if err == nil || !strings.Contains(err.Error(), "decode existing v2 chunk") {
		t.Fatalf("CommitChunkObjects() error = %v, want decode error", err)
	}

	value, closer, getErr := store.db.Get(v2ChunkKey("ns", "chunk", "shard"))
	if getErr != nil {
		t.Fatalf("Get() corrupt value error = %v", getErr)
	}
	defer closer.Close()
	if !bytes.Equal(value, []byte("not-a-gob")) {
		t.Fatalf("corrupt value was overwritten: %q", value)
	}
}

func TestIterateV2ChunksReturnsDecodeError(t *testing.T) {
	store := openTestStore(t)
	putCorruptV2(t, store, "ns", "chunk", "shard")

	err := store.IterateV2Chunks("ns", "chunk", func(*ChunkObjectMeta) error {
		t.Fatal("callback must not run for corrupt metadata")
		return nil
	})
	if err == nil || !strings.Contains(err.Error(), "decode v2 chunk metadata") {
		t.Fatalf("IterateV2Chunks() error = %v, want decode error", err)
	}
}

func TestIterateAllV2ReturnsDecodeError(t *testing.T) {
	store := openTestStore(t)
	putCorruptV2(t, store, "ns", "chunk", "shard")

	err := store.IterateAllV2(func(*ChunkObjectMeta) error {
		t.Fatal("callback must not run for corrupt metadata")
		return nil
	})
	if err == nil || !strings.Contains(err.Error(), "decode v2 chunk metadata") {
		t.Fatalf("IterateAllV2() error = %v, want decode error", err)
	}
}

func TestCommitChunkObjectsPropagatesClosedStoreError(t *testing.T) {
	store, err := Open(t.TempDir())
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	_, err = store.CommitChunkObjects([]ChunkObjectMeta{{
		Namespace:   "ns",
		Key:         "chunk",
		Shard:       "shard",
		FilePath:    "path/chunk.cobj",
		Index:       0,
		StartTokens: 0,
		EndTokens:   8,
		Size:        128,
	}})
	if err == nil || !strings.Contains(err.Error(), "read existing v2 chunk") {
		t.Fatalf("CommitChunkObjects() error = %v, want store read error", err)
	}
}
