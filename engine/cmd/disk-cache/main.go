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
	mux.HandleFunc("/put", handlePut)       // POST hash, file_path, size
	mux.HandleFunc("/get", handleGet)        // GET ?hash=
	mux.HandleFunc("/remove", handleRemove)  // POST hash
	mux.HandleFunc("/exists", handleExists)  // GET ?hash=
	mux.HandleFunc("/evict", handleEvict)    // POST target_bytes
	mux.HandleFunc("/stats", handleStats)

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

func handleStats(w http.ResponseWriter, r *http.Request) {
	json.NewEncoder(w).Encode(eng.Stats())
}
