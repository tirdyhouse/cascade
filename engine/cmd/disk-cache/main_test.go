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

func TestRetrievedEndpointCountsSuccessfulLoads(t *testing.T) {
	withTestEngine(t, func() {
		rr := httptest.NewRecorder()
		handleRetrieved(rr, httptest.NewRequest(http.MethodPost, "/retrieved", strings.NewReader(`{"count":3}`)))
		if rr.Code != http.StatusOK {
			t.Fatalf("handleRetrieved status = %d, body=%s", rr.Code, rr.Body.String())
		}
		stats := statsResponse(t)
		if got := stats["BlocksRetrieved"]; got != 3 {
			t.Fatalf("BlocksRetrieved after /retrieved = %d, want 3", got)
		}
		if got := stats["ChunksRetrieved"]; got != 3 {
			t.Fatalf("ChunksRetrieved after /retrieved = %d, want 3", got)
		}

		rr = httptest.NewRecorder()
		handleRetrieved(rr, httptest.NewRequest(http.MethodPost, "/retrieved", strings.NewReader(`{"count":0}`)))
		if rr.Code != http.StatusOK {
			t.Fatalf("handleRetrieved zero status = %d, body=%s", rr.Code, rr.Body.String())
		}
		stats = statsResponse(t)
		if got := stats["BlocksRetrieved"]; got != 3 {
			t.Fatalf("BlocksRetrieved after zero /retrieved = %d, want 3", got)
		}
		if got := stats["ChunksRetrieved"]; got != 3 {
			t.Fatalf("ChunksRetrieved after zero /retrieved = %d, want 3", got)
		}
	})
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
		stats := statsResponse(t)
		if got := stats["BlocksRetrieved"]; got != 0 {
			t.Fatalf("BlocksRetrieved after chunk_list = %d, want 0", got)
		}
		if stats["ChunkPutRequests"] != 1 || stats["ChunksStored"] != 1 || stats["ChunkEntries"] != 1 || stats["ChunkListRequests"] != 1 || stats["ChunkListHits"] != 1 {
			t.Fatalf("stats after chunk_list = %+v, want chunk counters", stats)
		}
	})
}

func TestV2CommitChunksHTTP(t *testing.T) {
	withTestEngine(t, func() {
		body := `{"objects":[{"namespace":"ns1","key":"chunk-a","shard":"shard1","file_path":"p/a.bin","index":0,"start_tokens":0,"end_tokens":100,"size":50}]}`
		rr := httptest.NewRecorder()
		handleV2CommitChunks(rr, httptest.NewRequest(http.MethodPost, "/v2/chunks/commit", strings.NewReader(body)))
		if rr.Code != http.StatusOK {
			t.Fatalf("handleV2CommitChunks status = %d, body=%s", rr.Code, rr.Body.String())
		}
		stats := statsResponse(t)
		if stats["ChunksStored"] != 1 {
			t.Fatalf("ChunksStored = %d, want 1", stats["ChunksStored"])
		}
	})
}

func TestV2MatchChunksHTTP(t *testing.T) {
	withTestEngine(t, func() {
		// First commit some data
		commitBody := `{"objects":[{"namespace":"ns1","key":"chunk-a","shard":"shard1","file_path":"p/a.bin","index":0,"start_tokens":0,"end_tokens":100,"size":50}]}`
		rr := httptest.NewRecorder()
		handleV2CommitChunks(rr, httptest.NewRequest(http.MethodPost, "/v2/chunks/commit", strings.NewReader(commitBody)))
		if rr.Code != http.StatusOK {
			t.Fatalf("commit status = %d", rr.Code)
		}

		// Match
		matchBody := `{"namespace":"ns1","candidates":[{"key":"chunk-a","end_tokens":100}],"required_shards":["shard1"]}`
		rr = httptest.NewRecorder()
		handleV2MatchChunks(rr, httptest.NewRequest(http.MethodPost, "/v2/chunks/match", strings.NewReader(matchBody)))
		if rr.Code != http.StatusOK {
			t.Fatalf("handleV2MatchChunks status = %d, body=%s", rr.Code, rr.Body.String())
		}
		var result cache.ChunkMatchResult
		if err := json.NewDecoder(rr.Body).Decode(&result); err != nil {
			t.Fatalf("decode match result: %v", err)
		}
		if result.MatchedChunks != 1 {
			t.Fatalf("MatchedChunks = %d, want 1", result.MatchedChunks)
		}
	})
}

func TestV2ResolveChunksHTTP(t *testing.T) {
	withTestEngine(t, func() {
		commitBody := `{"objects":[{"namespace":"ns1","key":"chunk-a","shard":"shard1","file_path":"p/a.bin","index":0,"start_tokens":0,"end_tokens":100,"size":50}]}`
		rr := httptest.NewRecorder()
		handleV2CommitChunks(rr, httptest.NewRequest(http.MethodPost, "/v2/chunks/commit", strings.NewReader(commitBody)))
		if rr.Code != http.StatusOK {
			t.Fatalf("commit status = %d", rr.Code)
		}

		resolveBody := `{"namespace":"ns1","keys":["chunk-a"],"shard":"shard1"}`
		rr = httptest.NewRecorder()
		handleV2ResolveChunks(rr, httptest.NewRequest(http.MethodPost, "/v2/chunks/resolve", strings.NewReader(resolveBody)))
		if rr.Code != http.StatusOK {
			t.Fatalf("handleV2ResolveChunks status = %d, body=%s", rr.Code, rr.Body.String())
		}
		var results []cache.ChunkObject
		if err := json.NewDecoder(rr.Body).Decode(&results); err != nil {
			t.Fatalf("decode resolve result: %v", err)
		}
		if len(results) != 1 || results[0].Key != "chunk-a" {
			t.Fatalf("resolve result = %+v, want [chunk-a]", results)
		}
	})
}

func TestV2RetrievedHTTP(t *testing.T) {
	withTestEngine(t, func() {
		rr := httptest.NewRecorder()
		handleV2Retrieved(rr, httptest.NewRequest(http.MethodPost, "/v2/chunks/retrieved", strings.NewReader(`{"count":5}`)))
		if rr.Code != http.StatusOK {
			t.Fatalf("handleV2Retrieved status = %d", rr.Code)
		}
		stats := statsResponse(t)
		if got := stats["BlocksRetrieved"]; got != 5 {
			t.Fatalf("BlocksRetrieved = %d, want 5", got)
		}
		if got := stats["ChunksRetrieved"]; got != 5 {
			t.Fatalf("ChunksRetrieved = %d, want 5", got)
		}
	})
}

func TestV2InvalidateHTTP(t *testing.T) {
	withTestEngine(t, func() {
		commitBody := `{"objects":[{"namespace":"ns1","key":"a","shard":"s1","file_path":"p/a.bin","index":0,"start_tokens":0,"end_tokens":100,"size":50}]}`
		rr := httptest.NewRecorder()
		handleV2CommitChunks(rr, httptest.NewRequest(http.MethodPost, "/v2/chunks/commit", strings.NewReader(commitBody)))
		if rr.Code != http.StatusOK {
			t.Fatalf("commit status = %d, body=%s", rr.Code, rr.Body.String())
		}

		body := `{"namespace":"ns1","key":"a","shard":"s1"}`
		for attempt := 0; attempt < 2; attempt++ {
			rr = httptest.NewRecorder()
			handleV2InvalidateChunk(rr, httptest.NewRequest(http.MethodPost, "/v2/chunks/invalidate", strings.NewReader(body)))
			if rr.Code != http.StatusOK {
				t.Fatalf("invalidate attempt %d status = %d, body=%s", attempt+1, rr.Code, rr.Body.String())
			}
		}

		matchBody := `{"namespace":"ns1","candidates":[{"key":"a","end_tokens":100}],"required_shards":["s1"]}`
		rr = httptest.NewRecorder()
		handleV2MatchChunks(rr, httptest.NewRequest(http.MethodPost, "/v2/chunks/match", strings.NewReader(matchBody)))
		var result cache.ChunkMatchResult
		if err := json.NewDecoder(rr.Body).Decode(&result); err != nil {
			t.Fatalf("decode match result: %v", err)
		}
		if result.MatchedChunks != 0 {
			t.Fatalf("MatchedChunks after invalidation = %d, want 0", result.MatchedChunks)
		}
	})
}

func TestV2InvalidateHTTPValidation(t *testing.T) {
	withTestEngine(t, func() {
		for _, tc := range []struct {
			name   string
			method string
			body   string
		}{
			{name: "method", method: http.MethodGet, body: `{}`},
			{name: "namespace", method: http.MethodPost, body: `{"key":"a","shard":"s1"}`},
			{name: "key", method: http.MethodPost, body: `{"namespace":"ns1","shard":"s1"}`},
			{name: "shard", method: http.MethodPost, body: `{"namespace":"ns1","key":"a"}`},
		} {
			t.Run(tc.name, func(t *testing.T) {
				rr := httptest.NewRecorder()
				handleV2InvalidateChunk(rr, httptest.NewRequest(tc.method, "/v2/chunks/invalidate", strings.NewReader(tc.body)))
				if rr.Code != http.StatusBadRequest {
					t.Fatalf("status = %d, want 400, body=%s", rr.Code, rr.Body.String())
				}
			})
		}
	})
}

func TestV2HTTPBadMethod(t *testing.T) {
	withTestEngine(t, func() {
		for name, handler := range map[string]func(http.ResponseWriter, *http.Request){
			"commit":   handleV2CommitChunks,
			"match":    handleV2MatchChunks,
			"resolve":  handleV2ResolveChunks,
			"retrieve": handleV2Retrieved,
		} {
			t.Run(name, func(t *testing.T) {
				rr := httptest.NewRecorder()
				handler(rr, httptest.NewRequest(http.MethodGet, "/v2/chunks/"+name, nil))
				if rr.Code != 400 {
					t.Fatalf("expected 400 for GET, got %d", rr.Code)
				}
			})
		}
	})
}

func TestV2HTTPBadJSON(t *testing.T) {
	withTestEngine(t, func() {
		for name, handler := range map[string]func(http.ResponseWriter, *http.Request){
			"commit":  handleV2CommitChunks,
			"match":   handleV2MatchChunks,
			"resolve": handleV2ResolveChunks,
		} {
			t.Run(name, func(t *testing.T) {
				rr := httptest.NewRecorder()
				handler(rr, httptest.NewRequest(http.MethodPost, "/v2/chunks/"+name, strings.NewReader(`not json`)))
				if rr.Code != 400 {
					t.Fatalf("expected 400 for bad JSON, got %d, body=%s", rr.Code, rr.Body.String())
				}
			})
		}
	})
}

func TestV2ResolveChunksHTTPMissing(t *testing.T) {
	withTestEngine(t, func() {
		body := `{"namespace":"ns1","keys":["nonexistent"],"shard":"shard1"}`
		rr := httptest.NewRecorder()
		handleV2ResolveChunks(rr, httptest.NewRequest(http.MethodPost, "/v2/chunks/resolve", strings.NewReader(body)))
		if rr.Code != 500 {
			t.Fatalf("expected 500 for missing resolve, got %d, body=%s", rr.Code, rr.Body.String())
		}
	})
}

func TestV2HTTPResponseJSON(t *testing.T) {
	withTestEngine(t, func() {
		// Commit two objects and match with full hit — verify JSON output
		commitBody := `{"objects":[
			{"namespace":"ns1","key":"a","shard":"s1","file_path":"p/a.bin","index":0,"start_tokens":0,"end_tokens":100,"size":50},
			{"namespace":"ns1","key":"a","shard":"s2","file_path":"p/a.bin","index":0,"start_tokens":0,"end_tokens":100,"size":50}
		]}`
		rr := httptest.NewRecorder()
		handleV2CommitChunks(rr, httptest.NewRequest(http.MethodPost, "/v2/chunks/commit", strings.NewReader(commitBody)))
		if rr.Code != http.StatusOK {
			t.Fatalf("commit status = %d", rr.Code)
		}

		matchBody := `{"namespace":"ns1","candidates":[{"key":"a","end_tokens":100}],"required_shards":["s1","s2"]}`
		rr = httptest.NewRecorder()
		handleV2MatchChunks(rr, httptest.NewRequest(http.MethodPost, "/v2/chunks/match", strings.NewReader(matchBody)))
		if rr.Code != http.StatusOK {
			t.Fatalf("match status = %d, body=%s", rr.Code, rr.Body.String())
		}
		var result cache.ChunkMatchResult
		if err := json.NewDecoder(rr.Body).Decode(&result); err != nil {
			t.Fatalf("decode: %v", err)
		}
		if result.MatchedChunks != 1 || result.MatchedTokens != 100 || len(result.MatchedKeys) != 1 || result.MatchedKeys[0] != "a" {
			t.Fatalf("unexpected match result: %+v", result)
		}
	})
}

func TestV2HTTPHandlerValidation(t *testing.T) {
	withTestEngine(t, func() {
		tests := []struct {
			name       string
			method     string
			path       string
			body       string
			wantStatus int
			handler    func(http.ResponseWriter, *http.Request)
		}{
			// Bad method tests
			{"commit GET", "GET", "/v2/chunks/commit", "", 400, handleV2CommitChunks},
			{"match GET", "GET", "/v2/chunks/match", "", 400, handleV2MatchChunks},
			{"resolve GET", "GET", "/v2/chunks/resolve", "", 400, handleV2ResolveChunks},
			{"retrieved GET", "GET", "/v2/chunks/retrieved", "", 400, handleV2Retrieved},

			// Match validation
			{"match empty namespace", "POST", "/v2/chunks/match", `{"namespace":"","candidates":[{"key":"a","end_tokens":100}],"required_shards":["s1"]}`, 400, handleV2MatchChunks},
			{"match negative end_tokens", "POST", "/v2/chunks/match", `{"namespace":"ns1","candidates":[{"key":"a","end_tokens":-1}],"required_shards":["s1"]}`, 400, handleV2MatchChunks},
			{"match zero end_tokens", "POST", "/v2/chunks/match", `{"namespace":"ns1","candidates":[{"key":"a","end_tokens":0}],"required_shards":["s1"]}`, 400, handleV2MatchChunks},
			{"match empty candidates", "POST", "/v2/chunks/match", `{"namespace":"ns1","candidates":[],"required_shards":["s1"]}`, 400, handleV2MatchChunks},
			{"match empty required_shards", "POST", "/v2/chunks/match", `{"namespace":"ns1","candidates":[{"key":"a","end_tokens":100}],"required_shards":[]}`, 400, handleV2MatchChunks},

			// Resolve validation
			{"resolve empty namespace", "POST", "/v2/chunks/resolve", `{"namespace":"","keys":["a"],"shard":"s1"}`, 400, handleV2ResolveChunks},
			{"resolve empty keys", "POST", "/v2/chunks/resolve", `{"namespace":"ns1","keys":[],"shard":"s1"}`, 400, handleV2ResolveChunks},
			{"resolve empty shard", "POST", "/v2/chunks/resolve", `{"namespace":"ns1","keys":["a"],"shard":""}`, 400, handleV2ResolveChunks},

			// Retrieved validation
			{"retrieved negative count", "POST", "/v2/chunks/retrieved", `{"count":-1}`, 400, handleV2Retrieved},

			// Commit validation
			{"commit empty objects", "POST", "/v2/chunks/commit", `{"objects":[]}`, 400, handleV2CommitChunks},
			{"commit negative size", "POST", "/v2/chunks/commit", `{"objects":[{"namespace":"ns1","key":"a","shard":"s1","file_path":"f","index":0,"start_tokens":0,"end_tokens":10,"size":-1}]}`, 400, handleV2CommitChunks},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				rr := httptest.NewRecorder()
				req := httptest.NewRequest(tc.method, tc.path, strings.NewReader(tc.body))
				tc.handler(rr, req)
				if rr.Code != tc.wantStatus {
					t.Errorf("status = %d, want %d, body=%s", rr.Code, tc.wantStatus, rr.Body.String())
				}
			})
		}
	})
}

func TestV2MatchResponseContentType(t *testing.T) {
	withTestEngine(t, func() {
		// Commit first
		commitBody := `{"objects":[{"namespace":"ns1","key":"a","shard":"s1","file_path":"p/a.bin","index":0,"start_tokens":0,"end_tokens":100,"size":50}]}`
		rr := httptest.NewRecorder()
		handleV2CommitChunks(rr, httptest.NewRequest(http.MethodPost, "/v2/chunks/commit", strings.NewReader(commitBody)))
		if rr.Code != http.StatusOK {
			t.Fatalf("commit status = %d", rr.Code)
		}

		// Match and verify Content-Type
		matchBody := `{"namespace":"ns1","candidates":[{"key":"a","end_tokens":100}],"required_shards":["s1"]}`
		rr = httptest.NewRecorder()
		handleV2MatchChunks(rr, httptest.NewRequest(http.MethodPost, "/v2/chunks/match", strings.NewReader(matchBody)))
		if rr.Code != http.StatusOK {
			t.Fatalf("match status = %d", rr.Code)
		}
		ct := rr.Header().Get("Content-Type")
		if ct != "application/json" {
			t.Fatalf("Content-Type = %q, want application/json", ct)
		}
	})
}

func TestV2ResolveResponseContentType(t *testing.T) {
	withTestEngine(t, func() {
		commitBody := `{"objects":[{"namespace":"ns1","key":"a","shard":"s1","file_path":"p/a.bin","index":0,"start_tokens":0,"end_tokens":100,"size":50}]}`
		rr := httptest.NewRecorder()
		handleV2CommitChunks(rr, httptest.NewRequest(http.MethodPost, "/v2/chunks/commit", strings.NewReader(commitBody)))
		if rr.Code != http.StatusOK {
			t.Fatalf("commit status = %d", rr.Code)
		}

		resolveBody := `{"namespace":"ns1","keys":["a"],"shard":"s1"}`
		rr = httptest.NewRecorder()
		handleV2ResolveChunks(rr, httptest.NewRequest(http.MethodPost, "/v2/chunks/resolve", strings.NewReader(resolveBody)))
		if rr.Code != http.StatusOK {
			t.Fatalf("resolve status = %d", rr.Code)
		}
		ct := rr.Header().Get("Content-Type")
		if ct != "application/json" {
			t.Fatalf("Content-Type = %q, want application/json", ct)
		}
	})
}

func TestV2CommitResponseJSON(t *testing.T) {
	withTestEngine(t, func() {
		// Verify commit returns JSON with status: ok
		body := `{"objects":[{"namespace":"ns1","key":"a","shard":"s1","file_path":"p/a.bin","index":0,"start_tokens":0,"end_tokens":100,"size":50}]}`
		rr := httptest.NewRecorder()
		handleV2CommitChunks(rr, httptest.NewRequest(http.MethodPost, "/v2/chunks/commit", strings.NewReader(body)))
		if rr.Code != http.StatusOK {
			t.Fatalf("commit status = %d", rr.Code)
		}
		ct := rr.Header().Get("Content-Type")
		if ct != "application/json" {
			t.Fatalf("Content-Type = %q, want application/json", ct)
		}
		var result map[string]string
		if err := json.NewDecoder(rr.Body).Decode(&result); err != nil {
			t.Fatalf("decode response: %v", err)
		}
		if result["status"] != "ok" {
			t.Fatalf("status = %q, want ok", result["status"])
		}
	})
}

func TestV2RetrievedResponseJSON(t *testing.T) {
	withTestEngine(t, func() {
		body := `{"count":5}`
		rr := httptest.NewRecorder()
		handleV2Retrieved(rr, httptest.NewRequest(http.MethodPost, "/v2/chunks/retrieved", strings.NewReader(body)))
		if rr.Code != http.StatusOK {
			t.Fatalf("status = %d", rr.Code)
		}
		ct := rr.Header().Get("Content-Type")
		if ct != "application/json" {
			t.Fatalf("Content-Type = %q, want application/json", ct)
		}
		var result map[string]string
		if err := json.NewDecoder(rr.Body).Decode(&result); err != nil {
			t.Fatalf("decode response: %v", err)
		}
		if result["status"] != "ok" {
			t.Fatalf("status = %q, want ok", result["status"])
		}
	})
}

func TestV2ResolveArrayFormat(t *testing.T) {
	withTestEngine(t, func() {
		commitBody := `{"objects":[{"namespace":"ns1","key":"a","shard":"s1","file_path":"p/a.bin","index":0,"start_tokens":0,"end_tokens":100,"size":50}]}`
		rr := httptest.NewRecorder()
		handleV2CommitChunks(rr, httptest.NewRequest(http.MethodPost, "/v2/chunks/commit", strings.NewReader(commitBody)))
		if rr.Code != http.StatusOK {
			t.Fatalf("commit status = %d", rr.Code)
		}

		resolveBody := `{"namespace":"ns1","keys":["a"],"shard":"s1"}`
		rr = httptest.NewRecorder()
		handleV2ResolveChunks(rr, httptest.NewRequest(http.MethodPost, "/v2/chunks/resolve", strings.NewReader(resolveBody)))
		if rr.Code != http.StatusOK {
			t.Fatalf("resolve status = %d", rr.Code)
		}

		// Verify response is a JSON array
		var results []cache.ChunkObject
		if err := json.NewDecoder(rr.Body).Decode(&results); err != nil {
			t.Fatalf("decode array: %v", err)
		}
		if len(results) != 1 || results[0].Key != "a" {
			t.Fatalf("results = %+v, want [{Key:a}]", results)
		}
	})
}
