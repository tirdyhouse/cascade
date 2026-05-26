package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"strconv"

	"predict/engine/pkg/cache"
)

var (
	cachePath    = flag.String("cache-path", "/tmp/disk-cache", "Cache file directory")
	metadataPath = flag.String("metadata-path", "/tmp/disk-cache-meta", "Metadata database directory")
	maxSize      = flag.String("max-size", "100GB", "Maximum cache size (e.g. 100GB, 1TB)")
	listenAddr   = flag.String("listen", ":9100", "HTTP API listen address")
)

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

	cfg := cache.DefaultConfig()
	cfg.CachePath = *cachePath
	cfg.MetadataPath = *metadataPath
	cfg.MaxSizeBytes = size

	eng, err = cache.New(cfg)
	if err != nil {
		log.Fatalf("engine init: %v", err)
	}
	defer eng.Close()

	log.Printf("disk-cache engine started on %s", *listenAddr)

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
	mux.HandleFunc("/record", handleRecord)    // POST prompt_hash, num_tokens
	mux.HandleFunc("/record_batch", handleRecordBatch) // POST token_ids, mm_hashes, block_size
	mux.HandleFunc("/match", handleMatch)      // POST token_ids, mm_hashes, block_size
	mux.HandleFunc("/record", handleRecord)    // POST prompt_hash, num_tokens

	if err := http.ListenAndServe(*listenAddr, mux); err != nil {
		log.Fatalf("server error: %v", err)
	}
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
	var req struct{ Hash uint64 `json:"hash"` }
	json.NewDecoder(r.Body).Decode(&req)
	eng.Remove(req.Hash)
	w.WriteHeader(200)
}

func handleExists(w http.ResponseWriter, r *http.Request) {
	hash, _ := strconv.ParseUint(r.URL.Query().Get("hash"), 16, 64)
	json.NewEncoder(w).Encode(map[string]bool{"exists": eng.Exists(hash)})
}

func handleEvict(w http.ResponseWriter, r *http.Request) {
	var req struct{ TargetBytes int64 `json:"target_bytes"` }
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
