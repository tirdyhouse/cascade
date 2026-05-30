package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"predict/engine/pkg/cache"
)

func withTestEngine(t *testing.T, fn func()) {
	t.Helper()

	old := eng
	tmp := t.TempDir()
	var err error
	eng, err = cache.New(cache.Config{
		CachePath:      tmp + "/blocks",
		MetadataPath:   tmp + "/meta",
		MaxSizeBytes:   1 << 20,
		EvictionPolicy: "lru",
	})
	if err != nil {
		t.Fatalf("cache.New() error = %v", err)
	}
	t.Cleanup(func() {
		if err := eng.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
		eng = old
	})

	fn()
}

func statsResponse(t *testing.T) map[string]int64 {
	t.Helper()

	rr := httptest.NewRecorder()
	handleStats(rr, httptest.NewRequest(http.MethodGet, "/stats", nil))
	if rr.Code != http.StatusOK {
		t.Fatalf("handleStats status = %d, body=%s", rr.Code, rr.Body.String())
	}
	var stats map[string]int64
	if err := json.NewDecoder(rr.Body).Decode(&stats); err != nil {
		t.Fatalf("decode stats: %v", err)
	}
	return stats
}

func TestChunkListDoesNotCountAsRetrieved(t *testing.T) {
	withTestEngine(t, func() {
		body := `{"prefix_key":"0123456789abcdef0123456789abcdef","layer_name":"layer.0","chunk_index":0,"num_tokens":16}`
		rr := httptest.NewRecorder()
		handleChunkPut(rr, httptest.NewRequest(http.MethodPost, "/chunk_put", strings.NewReader(body)))
		if rr.Code != http.StatusOK {
			t.Fatalf("handleChunkPut status = %d, body=%s", rr.Code, rr.Body.String())
		}

		rr = httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/chunk_list?prefix_key=0123456789abcdef0123456789abcdef&layer_name=layer.0", nil)
		handleChunkList(rr, req)
		if rr.Code != http.StatusOK {
			t.Fatalf("handleChunkList status = %d, body=%s", rr.Code, rr.Body.String())
		}
		if got := statsResponse(t)["BlocksRetrieved"]; got != 0 {
			t.Fatalf("BlocksRetrieved after chunk_list = %d, want 0", got)
		}
	})
}

func TestRetrievedEndpointCountsSuccessfulLoads(t *testing.T) {
	withTestEngine(t, func() {
		rr := httptest.NewRecorder()
		handleRetrieved(rr, httptest.NewRequest(http.MethodPost, "/retrieved", strings.NewReader(`{"count":3}`)))
		if rr.Code != http.StatusOK {
			t.Fatalf("handleRetrieved status = %d, body=%s", rr.Code, rr.Body.String())
		}
		if got := statsResponse(t)["BlocksRetrieved"]; got != 3 {
			t.Fatalf("BlocksRetrieved after /retrieved = %d, want 3", got)
		}

		rr = httptest.NewRecorder()
		handleRetrieved(rr, httptest.NewRequest(http.MethodPost, "/retrieved", strings.NewReader(`{"count":0}`)))
		if rr.Code != http.StatusOK {
			t.Fatalf("handleRetrieved zero status = %d, body=%s", rr.Code, rr.Body.String())
		}
		if got := statsResponse(t)["BlocksRetrieved"]; got != 3 {
			t.Fatalf("BlocksRetrieved after zero /retrieved = %d, want 3", got)
		}
	})
}
