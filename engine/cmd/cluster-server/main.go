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
	rpcxPort    = flag.Int("rpcx-port", 9000, "rpcx server port for C端 communication")
	httpPort    = flag.Int("http-port", 8080, "HTTP port for REST API + Web UI")
	modelsFile  = flag.String("models-file", "", "Path to models.json (optional)")
	modelsDir   = flag.String("models-dir", "", "Directory to auto-scan for models (optional)")
	publicURL   = flag.String("public-url", "", "Public URL for model download links (optional)")
)

func main() {
	flag.Parse()

	cfg := server.DefaultConfig()
	cfg.RPCPort = *rpcxPort
	cfg.HTTPPort = *httpPort
	cfg.ModelsFile = *modelsFile
	cfg.ModelsDir = *modelsDir
	cfg.PublicURL = *publicURL



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

	if err := srv.Start(ctx); err != nil {
		log.Fatalf("server start error: %v", err)
	}

	<-ctx.Done()
	log.Println("[main] goodbye")
}
