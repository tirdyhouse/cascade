// Command disk-cache starts the disk cache engine as a standalone server.
//
// Usage:
//
//	disk-cache -cache-path /mnt/nvme/kv-cache -max-size 100GB
//
// The server exposes a simple HTTP API for integration with the Python connector.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"strconv"

	"github.com/cici/disk-cache/engine/pkg/cache"
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

	eng, err := cache.New(cfg)
	if err != nil {
		log.Fatalf("failed to create engine: %v", err)
	}
	defer eng.Close()

	log.Printf("disk-cache engine started on %s", *listenAddr)
	log.Printf("  cache path: %s", cfg.CachePath)
	log.Printf("  max size:   %d bytes", cfg.MaxSizeBytes)

	// HTTP API for Python connector
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	http.HandleFunc("/stats", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(eng.Stats())
	})

	if err := http.ListenAndServe(*listenAddr, nil); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
