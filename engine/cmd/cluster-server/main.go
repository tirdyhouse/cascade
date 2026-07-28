package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"

	"predict/engine/pkg/server"
)

var (
	rpcxPort               = flag.Int("rpcx-port", 9000, "rpcx server port for C端 communication")
	httpPort               = flag.Int("http-port", 8080, "HTTP port for REST API + Web UI")
	modelsFile             = flag.String("models-file", "", "Path to models.json (optional)")
	modelsDir              = flag.String("models-dir", "", "Directory to auto-scan for models (optional)")
	publicURL              = flag.String("public-url", "", "Public URL for model download links (optional)")
	stateDir               = flag.String("state-dir", "/var/lib/cascade/control-plane", "Directory for durable command history and pending command state")
	gatewayMaxInFlight     = flag.Int("gateway-max-inflight-per-node", 16, "Maximum active gateway requests per vLLM node")
	gatewayMaxRequestBytes = flag.Int64("gateway-max-request-bytes", 64<<20, "Maximum OpenAI gateway request body size in bytes")
)

func main() {
	flag.Parse()

	cfg := server.DefaultConfig()
	cfg.RPCPort = *rpcxPort
	cfg.HTTPPort = *httpPort
	cfg.ModelsFile = *modelsFile
	cfg.ModelsDir = *modelsDir
	cfg.PublicURL = *publicURL
	cfg.StateDir = *stateDir
	cfg.GatewayMaxInFlightPerNode = *gatewayMaxInFlight
	cfg.GatewayMaxRequestBytes = *gatewayMaxRequestBytes
	if cfg.GatewayMaxInFlightPerNode <= 0 {
		log.Fatal("gateway-max-inflight-per-node must be greater than zero")
	}
	if cfg.GatewayMaxRequestBytes <= 0 {
		log.Fatal("gateway-max-request-bytes must be greater than zero")
	}

	srv := server.New(cfg)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle graceful shutdown
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		sig := <-sigCh
		log.Printf("[main] received signal %v, shutting down...", sig)
		srv.Stop()
		cancel()
	}()

	log.Printf("=== S端 Cluster Server ===")
	log.Printf("rpcx :%d  ← C端 agents connect here", cfg.RPCPort)
	log.Printf("HTTP :%d  ← Web UI: http://localhost:%d", cfg.HTTPPort, cfg.HTTPPort)
	log.Printf("Gateway :%d  ← OpenAI API: http://localhost:%d/v1", cfg.HTTPPort, cfg.HTTPPort)

	if err := srv.Start(ctx); err != nil {
		log.Fatalf("server start error: %v", err)
	}

	<-ctx.Done()
	log.Println("[main] goodbye")
}
