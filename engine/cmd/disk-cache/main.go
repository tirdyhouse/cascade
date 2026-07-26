package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"predict/engine/pkg/cache"
	"strconv"
	"strings"
)

var (
	cachePath        = flag.String("cache-path", "/tmp/disk-cache", "Cache file directory")
	metadataPath     = flag.String("metadata-path", "/tmp/disk-cache-meta", "Local Pebble metadata database directory")
	maxSize          = flag.String("max-size", "100GB", "Maximum cache size (e.g. 100GB, 1TB)")
	listenAddr       = flag.String("listen", ":9100", "HTTP API listen address")
	metadataModeFlag = flag.String("metadata-mode", "local", "Metadata mode: local | shared")
	sharedCacheID    = flag.String("shared-cache-id", "", "Required stable cache identity when metadata-mode=shared")
)

type metadataServiceConfig struct {
	mode          string
	sharedCacheID string
	cacheRoot     string
}

func (c metadataServiceConfig) shared() bool {
	return c.mode == "shared"
}

// metadataService is initialized once at process startup. Tests replace it
// temporarily to exercise the shared HTTP contract.
var metadataService = metadataServiceConfig{mode: "local"}

type clusterInfoResponse struct {
	APIVersion              int    `json:"api_version"`
	MetadataMode            string `json:"metadata_mode"`
	SharedCacheID           string `json:"shared_cache_id,omitempty"`
	PublishedObjectVerified bool   `json:"published_object_verified"`
	EvictionEnabled         bool   `json:"eviction_enabled"`
}

// MatchReq is the JSON body for POST /match.
type MatchReq struct {
	TokenIDs  []int64  `json:"token_ids"`
	MMHashes  []string `json:"mm_hashes"`
	BlockSize int      `json:"block_size"`
}

// RecordReq is the JSON body for POST /record.
type RecordReq struct {
	PromptHash string `json:"prompt_hash"`
	NumTokens  int    `json:"num_tokens"`
}

func parseSize(s string) (int64, error) {
	if len(s) < 3 {
		return strconv.ParseInt(s, 10, 64)
	}
	unit := s[len(s)-2:]
	val := s[:len(s)-2]
	n, err := strconv.ParseInt(val, 10, 64)
	if err != nil {
		return 0, err
	}
	switch unit {
	case "GB":
		return n * 1024 * 1024 * 1024, nil
	case "TB":
		return n * 1024 * 1024 * 1024 * 1024, nil
	case "MB":
		return n * 1024 * 1024, nil
	default:
		return 0, fmt.Errorf("unknown unit: %s", unit)
	}
}

var eng cache.Engine

func main() {
	flag.Parse()

	size, err := parseSize(*maxSize)
	if err != nil {
		log.Fatalf("invalid max-size: %v", err)
	}

	metadataService, err = configureMetadataService(
		*metadataModeFlag,
		*sharedCacheID,
		*cachePath,
	)
	if err != nil {
		log.Fatalf("invalid metadata service configuration: %v", err)
	}

	cfg := cache.DefaultConfig()
	cfg.CachePath = *cachePath
	cfg.MetadataPath = *metadataPath
	cfg.MaxSizeBytes = size
	cfg.DisableEviction = metadataService.shared()

	eng, err = cache.New(cfg)
	if err != nil {
		log.Fatalf("engine init: %v", err)
	}
	defer eng.Close()

	if metadataService.shared() {
		log.Printf(
			"shared metadata service started on %s cache_id=%s cache_root=%s eviction=disabled",
			*listenAddr,
			metadataService.sharedCacheID,
			metadataService.cacheRoot,
		)
	} else {
		log.Printf("disk-cache engine started on %s", *listenAddr)
	}

	if err := http.ListenAndServe(*listenAddr, newAPIHandler()); err != nil {
		log.Fatalf("server error: %v", err)
	}
}

func configureMetadataService(mode, cacheID, cacheRoot string) (metadataServiceConfig, error) {
	mode = strings.ToLower(strings.TrimSpace(mode))
	switch mode {
	case "local":
		return metadataServiceConfig{mode: mode}, nil
	case "shared":
		if strings.TrimSpace(cacheID) == "" {
			return metadataServiceConfig{}, errors.New("shared-cache-id is required in shared mode")
		}
		if !filepath.IsAbs(cacheRoot) {
			return metadataServiceConfig{}, errors.New("cache-path must be absolute in shared mode")
		}
		root, err := filepath.Abs(cacheRoot)
		if err != nil {
			return metadataServiceConfig{}, fmt.Errorf("resolve cache-path: %w", err)
		}
		info, err := os.Stat(root)
		if err != nil {
			return metadataServiceConfig{}, fmt.Errorf("shared cache-path %q is not mounted or accessible: %w", root, err)
		}
		if !info.IsDir() {
			return metadataServiceConfig{}, fmt.Errorf("shared cache-path %q is not a directory", root)
		}
		if _, err := cache.EnsureSharedCacheMarker(root, cacheID); err != nil {
			return metadataServiceConfig{}, fmt.Errorf("initialize shared cache marker: %w", err)
		}
		return metadataServiceConfig{
			mode:          mode,
			sharedCacheID: cacheID,
			cacheRoot:     root,
		}, nil
	default:
		return metadataServiceConfig{}, fmt.Errorf("unknown metadata-mode %q", mode)
	}
}

func newAPIHandler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/put", handlePut)
	mux.HandleFunc("/get", handleGet)
	mux.HandleFunc("/remove", handleRemove)
	mux.HandleFunc("/exists", handleExists)
	mux.HandleFunc("/evict", handleEvict)
	mux.HandleFunc("/stats", handleStats)
	mux.HandleFunc("/match", handleMatch)
	mux.HandleFunc("/record", handleRecord)
	mux.HandleFunc("/record_batch", handleRecordBatch)
	mux.HandleFunc("/retrieved", handleRetrieved)
	mux.HandleFunc("/chunk_put", handleChunkPut)
	mux.HandleFunc("/chunk_list", handleChunkList)
	mux.HandleFunc("/batch_load", handleBatchLoad)
	mux.HandleFunc("/batch_retrieved", handleBatchRetrieved)

	mux.HandleFunc("/v2/chunks/commit", handleV2CommitChunks)
	mux.HandleFunc("/v2/chunks/match", handleV2MatchChunks)
	mux.HandleFunc("/v2/chunks/resolve", handleV2ResolveChunks)
	mux.HandleFunc("/v2/chunks/invalidate", handleV2InvalidateChunk)
	mux.HandleFunc("/v2/chunks/retrieved", handleV2Retrieved)
	mux.HandleFunc("/v2/cluster/info", handleV2ClusterInfo)
	return mux
}

func handlePut(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "POST required", 400)
		return
	}
	var req struct {
		Hash     uint64 `json:"hash"`
		FilePath string `json:"file_path"`
		Size     int64  `json:"size"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	if err := eng.Put(req.Hash, req.FilePath, req.Size); err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	w.WriteHeader(200)
}

func handleGet(w http.ResponseWriter, r *http.Request) {
	hash, _ := strconv.ParseUint(r.URL.Query().Get("hash"), 16, 64)
	meta, err := eng.Get(hash)
	if err != nil || meta == nil {
		http.NotFound(w, r)
		return
	}
	json.NewEncoder(w).Encode(meta)
}

func handleRemove(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Hash uint64 `json:"hash"`
	}
	json.NewDecoder(r.Body).Decode(&req)
	eng.Remove(req.Hash)
	w.WriteHeader(200)
}

func handleExists(w http.ResponseWriter, r *http.Request) {
	hash, _ := strconv.ParseUint(r.URL.Query().Get("hash"), 16, 64)
	json.NewEncoder(w).Encode(map[string]bool{"exists": eng.Exists(hash)})
}

func handleEvict(w http.ResponseWriter, r *http.Request) {
	if metadataService.shared() {
		http.Error(w, "eviction is disabled in shared metadata mode until distributed read leases are available", http.StatusConflict)
		return
	}
	var req struct {
		TargetBytes int64 `json:"target_bytes"`
	}
	json.NewDecoder(r.Body).Decode(&req)
	metas := eng.Evict(req.TargetBytes)
	json.NewEncoder(w).Encode(metas)
}

// ── /record_batch: records all sub-block sentinels ──

// RecordBatchReq records sentinels for all block-aligned prefixes.
// The Go engine computes cumulative hashes incrementally and records
// sentinel markers for each block boundary (16, 32, 48, ...).
// Python calls this after writing all layer files.
type RecordBatchReq struct {
	TokenIDs  []int64  `json:"token_ids"`
	MMHashes  []string `json:"mm_hashes"`
	BlockSize int      `json:"block_size"`
}

func handleRecordBatch(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "POST required", 400)
		return
	}
	var req RecordBatchReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	if err := eng.RecordAll(req.TokenIDs, req.MMHashes, req.BlockSize); err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	w.WriteHeader(200)
}
func handleStats(w http.ResponseWriter, r *http.Request) {
	json.NewEncoder(w).Encode(eng.Stats())
}

// RetrievedReq records successful external cache retrievals.
type RetrievedReq struct {
	Count int64 `json:"count"`
}

func handleRetrieved(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "POST required", 400)
		return
	}
	var req RetrievedReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	eng.RecordRetrieved(req.Count)
	w.WriteHeader(200)
}

// handleMatch receives token IDs and returns the best cache hit.
// The Go engine computes hashes for all block-aligned lengths in parallel
// and returns the largest match.
func handleMatch(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "POST required", 400)
		return
	}
	var req MatchReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	result := eng.Match(req.TokenIDs, req.MMHashes, req.BlockSize)
	json.NewEncoder(w).Encode(result)
}

// handleRecord stores a sentinel marker for a completed cache entry.
func handleRecord(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "POST required", 400)
		return
	}
	var req RecordReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	if err := eng.RecordSentinel(req.PromptHash, req.NumTokens); err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	w.WriteHeader(200)
}

// ── Chunk handlers ──

type ChunkPutReq struct {
	PrefixKey  string `json:"prefix_key"`
	LayerName  string `json:"layer_name"`
	ChunkIndex int    `json:"chunk_index"`
	NumTokens  int    `json:"num_tokens"`
}

// ── Batch API for reduced HTTP round-trips ──

// BatchLoadReq requests chunk lists for multiple layers in one call.
// This reduces HTTP round-trips from 28 (per layer) to 1.
type BatchLoadReq struct {
	PrefixKey string   `json:"prefix_key"`
	Layers    []string `json:"layers"`
}

// BatchLoadResp returns chunk lists for all requested layers.
type BatchLoadResp struct {
	Results map[string][]int `json:"results"` // layer_name -> chunk_indices
}

func handleBatchLoad(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "POST required", 400)
		return
	}
	var req BatchLoadReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	resp := BatchLoadResp{Results: make(map[string][]int)}
	for _, layer := range req.Layers {
		indices, err := eng.ListChunks(req.PrefixKey, layer)
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		if indices == nil {
			indices = []int{}
		}
		resp.Results[layer] = indices
	}
	json.NewEncoder(w).Encode(resp)
}

// BatchRetrievedReq records retrieved counts for multiple layers at once.
type BatchRetrievedReq struct {
	Counts map[string]int64 `json:"counts"` // layer_name -> count
}

func handleBatchRetrieved(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "POST required", 400)
		return
	}
	var req BatchRetrievedReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	var total int64
	for _, count := range req.Counts {
		total += count
	}
	eng.RecordRetrieved(total)
	w.WriteHeader(200)
}

func handleChunkPut(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "POST required", 400)
		return
	}
	var req ChunkPutReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	if err := eng.PutChunk(req.PrefixKey, req.LayerName, req.ChunkIndex, req.NumTokens); err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	w.WriteHeader(200)
}

func handleChunkList(w http.ResponseWriter, r *http.Request) {
	prefix := r.URL.Query().Get("prefix_key")
	layer := r.URL.Query().Get("layer_name")
	if prefix == "" || layer == "" {
		http.Error(w, "prefix_key and layer_name required", 400)
		return
	}
	indices, err := eng.ListChunks(prefix, layer)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	if indices == nil {
		indices = []int{}
	}
	json.NewEncoder(w).Encode(map[string]interface{}{
		"chunks": indices,
	})
}

// ── V2 chunk handlers ──

func writeV2EngineError(w http.ResponseWriter, err error) {
	status := http.StatusInternalServerError
	if errors.Is(err, cache.ErrInvalidArgument) {
		status = http.StatusBadRequest
	}
	http.Error(w, err.Error(), status)
}

type V2CommitReq struct {
	Objects []cache.ChunkObject `json:"objects"`
}

type V2MatchReq struct {
	Namespace      string                 `json:"namespace"`
	Candidates     []cache.ChunkCandidate `json:"candidates"`
	RequiredShards []string               `json:"required_shards"`
	ResolveShard   string                 `json:"resolve_shard,omitempty"`
}

type V2ResolveReq struct {
	Namespace string   `json:"namespace"`
	Keys      []string `json:"keys"`
	Shard     string   `json:"shard"`
}

type V2RetrievedReq struct {
	Count int64 `json:"count"`
}

func handleV2ClusterInfo(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "GET required", http.StatusBadRequest)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(clusterInfoResponse{
		APIVersion:              1,
		MetadataMode:            metadataService.mode,
		SharedCacheID:           metadataService.sharedCacheID,
		PublishedObjectVerified: metadataService.shared(),
		EvictionEnabled:         !metadataService.shared(),
	})
}

func handleV2CommitChunks(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "POST required", 400)
		return
	}
	var req V2CommitReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	if len(req.Objects) == 0 {
		http.Error(w, "empty objects", 400)
		return
	}
	for i, obj := range req.Objects {
		if obj.Size <= 0 {
			http.Error(w, fmt.Sprintf("objects[%d]: non-positive size (%d)", i, obj.Size), 400)
			return
		}
	}
	if metadataService.shared() {
		if err := cache.ValidatePublishedChunkObjects(metadataService.cacheRoot, req.Objects); err != nil {
			status := http.StatusConflict
			if errors.Is(err, cache.ErrInvalidArgument) {
				status = http.StatusBadRequest
			}
			http.Error(w, fmt.Sprintf("shared object is not ready for publication: %v", err), status)
			return
		}
	}
	if err := eng.CommitChunks(req.Objects); err != nil {
		writeV2EngineError(w, err)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(200)
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func handleV2MatchChunks(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "POST required", 400)
		return
	}
	var req V2MatchReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	if req.Namespace == "" {
		http.Error(w, "empty namespace", 400)
		return
	}
	if len(req.Candidates) == 0 {
		http.Error(w, "empty candidates", 400)
		return
	}
	for i, c := range req.Candidates {
		if c.EndTokens <= 0 {
			http.Error(w, fmt.Sprintf("candidates[%d]: non-positive end_tokens (%d)", i, c.EndTokens), 400)
			return
		}
	}
	if len(req.RequiredShards) == 0 {
		http.Error(w, "empty required_shards", 400)
		return
	}
	result, err := eng.MatchChunks(req.Namespace, req.Candidates, req.RequiredShards)
	if err != nil {
		writeV2EngineError(w, err)
		return
	}
	if req.ResolveShard != "" && result.MatchedChunks > 0 {
		allowed := false
		for _, shard := range req.RequiredShards {
			if shard == req.ResolveShard {
				allowed = true
				break
			}
		}
		if !allowed {
			http.Error(w, "resolve_shard must be present in required_shards", 400)
			return
		}
		objects, resolveErr := eng.ResolveChunks(
			req.Namespace,
			result.MatchedKeys,
			req.ResolveShard,
		)
		if resolveErr != nil {
			writeV2EngineError(w, resolveErr)
			return
		}
		result.MatchedObjects = objects
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

func handleV2ResolveChunks(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "POST required", 400)
		return
	}
	var req V2ResolveReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	if req.Namespace == "" {
		http.Error(w, "empty namespace", 400)
		return
	}
	if len(req.Keys) == 0 {
		http.Error(w, "empty keys", 400)
		return
	}
	if req.Shard == "" {
		http.Error(w, "empty shard", 400)
		return
	}
	result, err := eng.ResolveChunks(req.Namespace, req.Keys, req.Shard)
	if err != nil {
		writeV2EngineError(w, err)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

type V2InvalidateReq struct {
	Namespace string `json:"namespace"`
	Key       string `json:"key"`
	Shard     string `json:"shard"`
}

func handleV2InvalidateChunk(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST required", http.StatusBadRequest)
		return
	}
	var req V2InvalidateReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if err := eng.InvalidateChunk(req.Namespace, req.Key, req.Shard); err != nil {
		writeV2EngineError(w, err)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func handleV2Retrieved(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "POST required", 400)
		return
	}
	var req V2RetrievedReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	if req.Count < 0 {
		http.Error(w, "negative count", 400)
		return
	}
	eng.RecordRetrieved(req.Count)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(200)
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}
