package agent

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// CacheProxy communicates with the local disk-cache engine over HTTP.
type CacheProxy struct {
	cfg     *Config
	baseURL string
	client  *http.Client
}

// CacheStats is the response from the local disk-cache /stats endpoint.
type CacheStats struct {
	BlocksStored    int64 `json:"BlocksStored"`
	BlocksRetrieved int64 `json:"BlocksRetrieved"`
	BlocksEvicted   int64 `json:"BlocksEvicted"`
	DiskUsedBytes   int64 `json:"DiskUsedBytes"`
	ChunksStored    int64 `json:"ChunksStored"`
	ChunksRetrieved int64 `json:"ChunksRetrieved"`
	MatchRequests   int64 `json:"MatchRequests"`
	MatchHits       int64 `json:"MatchHits"`
	MatchedTokens   int64 `json:"MatchedTokens"`
	// HitRate is computed from retrieved / (retrieved + stored) during Stats().
	HitRate float64 `json:"-"`
}

// NewCacheProxy creates a CacheProxy.
func NewCacheProxy(cfg *Config) *CacheProxy {
	return &CacheProxy{
		cfg:     cfg,
		baseURL: cfg.CachePath,
		client: &http.Client{
			Timeout: 5 * time.Second,
		},
	}
}

// Stats fetches cache statistics from the local disk-cache engine.
func (cp *CacheProxy) Stats() *CacheStats {
	if cp.baseURL == "" {
		return nil
	}

	resp, err := cp.client.Get(cp.baseURL + "/stats")
	if err != nil {
		// Engine not reachable — not necessarily an error (vLLM may not be running)
		return nil
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil
	}

	var stats CacheStats
	if err := json.Unmarshal(body, &stats); err != nil {
		return nil
	}
	// Compute hit rate
	total := stats.BlocksRetrieved + stats.BlocksStored
	if total > 0 {
		if stats.BlocksRetrieved >= stats.BlocksStored {
			// More retrievals than stored blocks — use retrieved as denominator
			stats.HitRate = float64(stats.BlocksRetrieved) / float64(stats.BlocksRetrieved+stats.BlocksEvicted) * 100.0
		} else {
			// Otherwise use total stored as estimate
			stats.HitRate = float64(stats.BlocksRetrieved) / float64(total) * 100.0
		}
	}

	return &stats
}

// Health checks if the local disk-cache engine is reachable.
func (cp *CacheProxy) Health() error {
	resp, err := cp.client.Get(cp.baseURL + "/stats")
	if err != nil {
		return fmt.Errorf("cache engine unreachable: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("cache engine status: %d", resp.StatusCode)
	}
	return nil
}
