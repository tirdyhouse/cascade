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
	serverAddr      = flag.String("server", "127.0.0.1:9000", "S端 rpcx address")
	nodeID          = flag.String("node-id", "", "Node ID (default: hostname)")
	rpcxPort        = flag.Int("rpcx-port", 9001, "Local rpcx port for bidirectional")
	cacheMode       = flag.String("cache-mode", "local_nvme", "Cache mode: local_nvme | shared_pool")
	cachePath       = flag.String("cache-path", "http://127.0.0.1:9100", "Disk-cache metadata HTTP API base URL")
	sharedCacheRoot = flag.String("shared-cache-root", "", "Shared cache filesystem root; required for cache-mode=shared_pool")
	sharedCacheID   = flag.String("shared-cache-id", "", "Stable shared cache identity; required for cache-mode=shared_pool")
	workDir         = flag.String("work-dir", "/root/cascade/agent", "Working directory for models, logs, and cache")
	vllmHost        = flag.String("vllm-host", "0.0.0.0", "vLLM listen host; must be reachable by the cluster gateway")
	vllmPort        = flag.Int("vllm-port", 8000, "vLLM HTTP port advertised to the cluster gateway")
	advertiseHost   = flag.String("advertise-host", "", "Gateway-reachable host or IP for this Agent; defaults to auto-detected outbound IP")
	vllmPath        = flag.String("vllm-path", "", "Path to the vLLM executable; defaults to Cascade venv or PATH")
	diagnosticsHost = flag.String("diagnostics-host", "0.0.0.0", "Read-only Agent diagnostics listen host")
	diagnosticsPort = flag.Int("diagnostics-port", 9002, "Read-only Agent diagnostics port; <=0 disables it")
	diagnosticsURL  = flag.String("diagnostics-url", "", "Externally reachable Agent diagnostics URL override")

	// GPU
	gpuType  = flag.String("gpu-type", "", "GPU type (e.g. H100)")
	gpuMemMB = flag.Int64("gpu-mem", 0, "GPU memory in MB")
	gpuCount = flag.Int("gpu-count", 1, "Number of GPUs")

	// Disks
	disksRaw = flag.String("disks", "", "Comma-separated filesystem paths; optional legacy :capacity suffix is ignored at runtime")
)

func main() {
	flag.Parse()

	cfg := agent.DefaultConfig()
	cfg.ServerAddr = *serverAddr
	cfg.RPCPort = *rpcxPort
	cfg.CachePath = *cachePath
	cfg.WorkDir = *workDir
	cfg.VLLMHost = strings.TrimSpace(*vllmHost)
	cfg.VLLMPort = *vllmPort
	cfg.AdvertiseHost = strings.TrimSpace(*advertiseHost)
	cfg.VLLMPath = strings.TrimSpace(*vllmPath)
	cfg.DiagnosticsHost = strings.TrimSpace(*diagnosticsHost)
	cfg.DiagnosticsPort = *diagnosticsPort
	cfg.DiagnosticsURL = strings.TrimSpace(*diagnosticsURL)
	if cfg.VLLMHost == "" {
		log.Fatal("vllm-host cannot be empty")
	}
	if cfg.VLLMPort < 1 || cfg.VLLMPort > 65535 {
		log.Fatalf("invalid vllm-port %d", cfg.VLLMPort)
	}
	if cfg.DiagnosticsPort > 65535 {
		log.Fatalf("invalid diagnostics-port %d", cfg.DiagnosticsPort)
	}

	mode := cluster.CacheMode(strings.ToLower(strings.TrimSpace(*cacheMode)))
	if mode != cluster.CacheModeLocalNVMe && mode != cluster.CacheModeSharedPool {
		log.Fatalf("invalid cache-mode %q; expected local_nvme or shared_pool", *cacheMode)
	}
	cfg.CacheMode = mode
	cfg.SharedCacheRoot = *sharedCacheRoot
	cfg.SharedCacheID = *sharedCacheID
	if cfg.CacheMode == cluster.CacheModeSharedPool &&
		(strings.TrimSpace(cfg.SharedCacheRoot) == "" || strings.TrimSpace(cfg.SharedCacheID) == "") {
		log.Fatal("shared-cache-root and shared-cache-id are required for cache-mode=shared_pool")
	}

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
				Path: path,
				// Runtime capacity is collected with statfs. Keep accepting the
				// legacy suffix for CLI compatibility, but never publish it as a
				// live disk metric.
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
	log.Printf("vLLM listen=%s:%d", cfg.VLLMHost, cfg.VLLMPort)
	if cfg.DiagnosticsPort > 0 {
		log.Printf("agent diagnostics listen=%s:%d", cfg.DiagnosticsHost, cfg.DiagnosticsPort)
	}
	log.Printf("disks: %+v", cfg.Disks)

	if err := agt.Start(); err != nil {
		log.Fatalf("agent error: %v", err)
	}
}
