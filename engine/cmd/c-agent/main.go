package main

import (
	"flag"
	"log"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"

	"predict/engine/pkg/agent"
	"predict/engine/pkg/cluster"
)

var (
	serverAddr = flag.String("server", "127.0.0.1:9000", "S端 rpcx address")
	nodeID     = flag.String("node-id", "", "Node ID (default: hostname)")
	rpcxPort   = flag.Int("rpcx-port", 9001, "Local rpcx port for bidirectional")
	cacheMode  = flag.String("cache-mode", "local_nvme", "Cache mode: local_nvme | shared_pool")
	cachePath  = flag.String("cache-path", "http://127.0.0.1:9100", "Local disk-cache HTTP API base URL")
	workDir    = flag.String("work-dir", "/root/cascade/agent", "Working directory for models, logs, and cache")

	// GPU
	gpuType  = flag.String("gpu-type", "", "GPU type (e.g. H100)")
	gpuMemMB = flag.Int64("gpu-mem", 0, "GPU memory in MB")
	gpuCount = flag.Int("gpu-count", 1, "Number of GPUs")

	// Disks
	disksRaw = flag.String("disks", "", "Comma-separated disk paths and sizes: /mnt/nvme0:3500,/mnt/nvme1:3500")
)

func main() {
	flag.Parse()

	cfg := agent.DefaultConfig()
	cfg.ServerAddr = *serverAddr
	cfg.RPCPort = *rpcxPort
	cfg.CachePath = *cachePath
	cfg.WorkDir = *workDir


	// Node ID
	if *nodeID != "" {
		cfg.NodeID = *nodeID
	} else {
		hostname, _ := os.Hostname()
		cfg.NodeID = hostname
	}

	// GPU
	cfg.GPUType = *gpuType
	cfg.GPUMemMB = *gpuMemMB
	cfg.GPUCount = *gpuCount

	// Disks
	if *disksRaw != "" {
		for _, part := range strings.Split(*disksRaw, ",") {
			part = strings.TrimSpace(part)
			if part == "" {
				continue
			}
			pieces := strings.Split(part, ":")
			path := pieces[0]
			sizeGB := int64(0)
			if len(pieces) > 1 {
				if v, err := strconv.ParseInt(pieces[1], 10, 64); err == nil {
					sizeGB = v
				}
			}
			cfg.Disks = append(cfg.Disks, cluster.DiskInfo{
				Path:    path,
				TotalGB: sizeGB,
				FreeGB:  sizeGB,
			})
		}
	}

	agt := agent.New(cfg)

	// Handle graceful shutdown
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		sig := <-sigCh
		log.Printf("[main] received %v, shutting down", sig)
		agt.Stop()
	}()

	log.Printf("=== C端 Agent ===")
	log.Printf("node=%s server=%s cache_mode=%s", cfg.NodeID, cfg.ServerAddr, cfg.CacheMode)
	log.Printf("disks: %+v", cfg.Disks)

	if err := agt.Start(); err != nil {
		log.Fatalf("agent error: %v", err)
	}
}
