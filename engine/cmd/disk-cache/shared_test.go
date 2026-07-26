package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func withSharedMetadataService(t *testing.T, cacheRoot string, fn func()) {
	t.Helper()
	previous := metadataService
	metadataService = metadataServiceConfig{
		mode:          "shared",
		sharedCacheID: "test-nvfile",
		cacheRoot:     cacheRoot,
	}
	t.Cleanup(func() {
		metadataService = previous
	})
	fn()
}

func TestSharedMetadataPublishesOnlyVisibleObjects(t *testing.T) {
	withTestEngine(t, func() {
		cacheRoot := t.TempDir()
		relativePath := filepath.Join("v2", "object.cobj")
		fullPath := filepath.Join(cacheRoot, relativePath)
		if err := os.MkdirAll(filepath.Dir(fullPath), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(fullPath, []byte("cache"), 0644); err != nil {
			t.Fatal(err)
		}

		withSharedMetadataService(t, cacheRoot, func() {
			infoRecorder := httptest.NewRecorder()
			handleV2ClusterInfo(infoRecorder, httptest.NewRequest(http.MethodGet, "/v2/cluster/info", nil))
			if infoRecorder.Code != http.StatusOK {
				t.Fatalf("cluster info status = %d, body=%s", infoRecorder.Code, infoRecorder.Body.String())
			}
			var info clusterInfoResponse
			if err := json.NewDecoder(infoRecorder.Body).Decode(&info); err != nil {
				t.Fatal(err)
			}
			if info.MetadataMode != "shared" || info.SharedCacheID != "test-nvfile" ||
				!info.PublishedObjectVerified || info.EvictionEnabled {
				t.Fatalf("cluster info = %+v", info)
			}

			body := `{"objects":[{"namespace":"ns1","key":"chunk-a","shard":"tp0-pp0","file_path":"v2/object.cobj","index":0,"start_tokens":0,"end_tokens":16,"size":5}]}`
			recorder := httptest.NewRecorder()
			handleV2CommitChunks(recorder, httptest.NewRequest(http.MethodPost, "/v2/chunks/commit", strings.NewReader(body)))
			if recorder.Code != http.StatusOK {
				t.Fatalf("visible shared commit status = %d, body=%s", recorder.Code, recorder.Body.String())
			}

			missing := httptest.NewRecorder()
			missingBody := `{"objects":[{"namespace":"ns1","key":"chunk-b","shard":"tp0-pp0","file_path":"v2/missing.cobj","index":1,"start_tokens":16,"end_tokens":32,"size":5}]}`
			handleV2CommitChunks(missing, httptest.NewRequest(http.MethodPost, "/v2/chunks/commit", strings.NewReader(missingBody)))
			if missing.Code != http.StatusConflict {
				t.Fatalf("missing shared commit status = %d, want 409, body=%s", missing.Code, missing.Body.String())
			}

			evict := httptest.NewRecorder()
			handleEvict(evict, httptest.NewRequest(http.MethodPost, "/evict", strings.NewReader(`{"target_bytes":1}`)))
			if evict.Code != http.StatusConflict {
				t.Fatalf("shared evict status = %d, want 409", evict.Code)
			}
		})
	})
}

func TestConfigureMetadataServiceRequiresSharedIdentityAndMountedRoot(t *testing.T) {
	root := t.TempDir()
	if _, err := configureMetadataService("shared", "", root); err == nil {
		t.Fatal("missing shared cache id was accepted")
	}
	if _, err := configureMetadataService("shared", "id", "relative"); err == nil {
		t.Fatal("relative shared cache path was accepted")
	}
	config, err := configureMetadataService("shared", "id", root)
	if err != nil {
		t.Fatalf("configureMetadataService() error = %v", err)
	}
	if !config.shared() || config.cacheRoot != root || config.sharedCacheID != "id" {
		t.Fatalf("config = %+v", config)
	}
}
