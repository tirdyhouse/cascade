package agent

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestCacheProxyStatsUsesV2EntryAndMatchCounters(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/stats" {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte(`{
			"BlocksStored": 2,
			"BlocksRetrieved": 9,
			"ChunksStored": 5,
			"MatchRequests": 4,
			"MatchHits": 3
		}`))
	}))
	defer server.Close()

	stats := NewCacheProxy(&Config{CachePath: server.URL}).Stats()
	if stats == nil {
		t.Fatal("Stats() = nil")
	}
	if got, want := stats.EntryCount(), int64(7); got != want {
		t.Fatalf("EntryCount() = %d, want %d", got, want)
	}
	if got, want := stats.HitRate, 75.0; got != want {
		t.Fatalf("HitRate = %v, want %v", got, want)
	}
}
