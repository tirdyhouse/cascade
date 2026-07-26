package cache

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func sharedTestObject(path string, size int64) ChunkObject {
	return ChunkObject{
		Namespace:   "namespace",
		Key:         "chunk-key",
		Shard:       "tp0-pp0",
		FilePath:    path,
		Index:       0,
		StartTokens: 0,
		EndTokens:   16,
		Size:        size,
	}
}

func TestValidatePublishedChunkObjectsAcceptsVisibleObject(t *testing.T) {
	root := t.TempDir()
	relativePath := filepath.Join("v2", "object.cobj")
	fullPath := filepath.Join(root, relativePath)
	if err := os.MkdirAll(filepath.Dir(fullPath), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(fullPath, []byte("cache"), 0644); err != nil {
		t.Fatal(err)
	}

	if err := ValidatePublishedChunkObjects(root, []ChunkObject{
		sharedTestObject(relativePath, 5),
	}); err != nil {
		t.Fatalf("ValidatePublishedChunkObjects() error = %v", err)
	}
}

func TestEnsureSharedCacheMarkerBindsRootToOneCacheID(t *testing.T) {
	root := t.TempDir()
	marker, err := EnsureSharedCacheMarker(root, "nvfile-prod")
	if err != nil {
		t.Fatalf("EnsureSharedCacheMarker() error = %v", err)
	}
	if marker.FormatVersion != SharedCacheFormatVersion || marker.SharedCacheID != "nvfile-prod" {
		t.Fatalf("marker = %+v", marker)
	}
	if _, err := ReadSharedCacheMarker(root); err != nil {
		t.Fatalf("ReadSharedCacheMarker() error = %v", err)
	}
	if _, err := EnsureSharedCacheMarker(root, "other-cache"); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("mismatched marker error = %v, want ErrInvalidArgument", err)
	}
}

func TestValidatePublishedChunkObjectsRejectsMissingAndEscapingPaths(t *testing.T) {
	root := t.TempDir()
	missingErr := ValidatePublishedChunkObjects(root, []ChunkObject{
		sharedTestObject("v2/missing.cobj", 1),
	})
	if missingErr == nil || errors.Is(missingErr, ErrInvalidArgument) {
		t.Fatalf("missing object error = %v, want visibility error", missingErr)
	}

	escapeErr := ValidatePublishedChunkObjects(root, []ChunkObject{
		sharedTestObject("../outside.cobj", 1),
	})
	if !errors.Is(escapeErr, ErrInvalidArgument) {
		t.Fatalf("escape error = %v, want ErrInvalidArgument", escapeErr)
	}
}

func TestDisableEvictionAllowsSharedMetadataToOutliveLocalCapacity(t *testing.T) {
	root := t.TempDir()
	eng, err := New(Config{
		CachePath:       filepath.Join(root, "cache"),
		MetadataPath:    filepath.Join(root, "metadata"),
		MaxSizeBytes:    1,
		EvictionPolicy:  "lru",
		DisableEviction: true,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	t.Cleanup(func() {
		if closeErr := eng.Close(); closeErr != nil {
			t.Fatalf("Close() error = %v", closeErr)
		}
	})

	if err := eng.CommitChunks([]ChunkObject{
		sharedTestObject("v2/object.cobj", 2),
	}); err != nil {
		t.Fatalf("CommitChunks() error = %v, want no capacity eviction in shared mode", err)
	}
	if evicted := eng.Evict(2); len(evicted) != 0 {
		t.Fatalf("Evict() = %+v, want disabled eviction", evicted)
	}
}
