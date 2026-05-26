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
	BlocksStored    int64   `json:"blocks_stored"`
	BlocksRetrieved int64   `json:"blocks_retrieved"`
	BlocksEvicted   int64   `json:"blocks_evicted"`
	DiskUsedBytes   int64   `json:"disk_used_bytes"`
	HitRate         float64 `json:"-"`
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
