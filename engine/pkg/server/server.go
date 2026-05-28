package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"predict/engine/pkg/cluster"

	"github.com/smallnest/rpcx/server"
)

// Server is the S端 main server.
// It manages:
//   - rpcx server (for C端 communication)
//   - Node registry (tracks all C端 agents)
//   - Command dispatcher (sends commands to C端)
//   - Disk tracker (tracks per-node per-disk usage)
//   - Cache router (routes KV cache lookups)
//   - REST API + Web UI (for human operators)
type Server struct {
	// Core components
	registry    *Registry
	dispatcher  *Dispatcher
	diskTracker *DiskTracker
	router      *Router

	// rpcx
	rpcxServer *server.Server
	xaddrs     []string // known C端 addresses for push (optional)

	// HTTP
	httpServer *http.Server
	httpClient *http.Client

	// Models
	models *ModelRegistry

	// Config
	config *Config
}
// Config holds S端 configuration.
type Config struct {
	RPCPort     int    // rpcx server port (C端 connect here)
	HTTPPort    int    // REST API + Web UI port
	MetadataDir string // metadata directory for existing disk-cache engine
	ModelsFile  string // path to models.json (optional)
	ModelsDir   string // directory to auto-scan for models (optional)
	PublicURL   string // public URL for model download links (optional)
}

// DefaultConfig returns sensible defaults.
func DefaultConfig() *Config {
	return &Config{
		RPCPort:  9000,
		HTTPPort: 8080,
	}
}

// New creates a new S端 Server.
func New(cfg *Config) *Server {
	reg := NewRegistry()
	dt := NewDiskTracker()

	srv := &Server{
		registry:    reg,
		dispatcher:  NewDispatcher(reg),
		diskTracker: dt,
		config:      cfg,
		httpClient: &http.Client{
			Timeout: 120 * time.Second,
		},
	}

	// Router with nil meta backend for now — will be connected to disk-cache engine later
	srv.router = NewRouter(reg, dt, nil)
	// Initialize model registry with directory scanning
	baseURL := cfg.PublicURL
	if baseURL == "" && cfg.HTTPPort > 0 {
		baseURL = fmt.Sprintf("http://localhost:%d/models/", cfg.HTTPPort)
	}
	srv.models = NewModelRegistry(cfg.ModelsDir, baseURL)
	if cfg.ModelsFile != "" {
		if err := srv.models.LoadFromFile(cfg.ModelsFile); err != nil {
			log.Printf("[server] warn: load models from %s: %v", cfg.ModelsFile, err)
		}
	}
	srv.models.Scan()

	// Periodic rescan (every 30s) — will be stopped when Start's context cancels
	// Scan is called here; Start() adds the periodic goroutine with its context

	log.Printf("[server] models: %d available (dir=%s file=%s)",
		len(srv.models.List()), cfg.ModelsDir, cfg.ModelsFile)
	return srv
}

// SetCacheMeta sets the cache metadata backend (the existing disk-cache engine).
// Called after the engine is initialized.
func (s *Server) SetCacheMeta(meta CacheMetaBackend) {
	s.router.cacheMeta = meta
}

// Start launches the S端 server.
func (s *Server) Start(ctx context.Context) error {
	// 1. Start rpcx server
	if err := s.startRPCX(ctx); err != nil {
		return err
	}

	// 2. Start GC loop
	go s.registry.GC(ctx)

	// 2b. Start periodic model scan
	go func() {
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				s.models.Scan()
			case <-ctx.Done():
				return
			}
		}
	}()

	// 3. Start REST API + Web UI
	if err := s.startHTTP(ctx); err != nil {
		return err
	}

	return nil
}

// ── rpcx ────────────────────────────────────────────────────────────────

func (s *Server) startRPCX(ctx context.Context) error {
	s.rpcxServer = server.NewServer()
	addr := s.rpcxAddr()

	// Register services
	s.rpcxServer.RegisterName("ClusterService", &clusterSvc{s.registry}, "")
	s.rpcxServer.RegisterName("CommandService", &cmdSvc{s.registry}, "")
	s.rpcxServer.RegisterName("QueryService", &querySvc{s.registry, s.router}, "")
	s.rpcxServer.RegisterName("AdminService", &adminSvc{s.dispatcher, s.registry}, "")

	log.Printf("[server] rpcx listening on %s", addr)
	go func() {
		if err := s.rpcxServer.Serve("tcp", addr); err != nil {
			log.Fatalf("[server] rpcx serve error: %v", err)
		}
	}()
	return nil
}

func (s *Server) rpcxAddr() string {
	return fmt.Sprintf(":%d", s.config.RPCPort)
}

// ── HTTP ────────────────────────────────────────────────────────────────

func (s *Server) startHTTP(ctx context.Context) error {
	// REST API mux (uses Go 1.22+ enhanced patterns for route params)
	apiMux := http.NewServeMux()
	s.registerAPI(apiMux)

	// Static file server
	cwd, _ := os.Getwd()
	staticDir := cwd + "/engine/pkg/server/web/static"

	// Single entry point: API routes → apiMux, everything else → static files
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path := r.URL.Path
		// Serve model files for distribution (e.g. /models/Qwen2.5-7B-Instruct/)
		if len(path) >= 8 && path[:8] == "/models/" && s.config.ModelsDir != "" {
			http.StripPrefix("/models/", http.FileServer(http.Dir(s.config.ModelsDir))).ServeHTTP(w, r)
			return
		}

		if len(path) >= 8 && path[:8] == "/api/v1/" {
			apiMux.ServeHTTP(w, r)
			return
		}
		// Everything else → serve static file
		fullPath := staticDir + "/index.html"
		if path != "/" && path != "" {
			fullPath = staticDir + path
		}
		http.ServeFile(w, r, fullPath)
	})

	addr := fmt.Sprintf(":%d", s.config.HTTPPort)
	s.httpServer = &http.Server{
		Addr:    addr,
		Handler: handler,
	}

	log.Printf("[server] HTTP + Web UI on %s", addr)
	go func() {
		if err := s.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("[server] HTTP serve error: %v", err)
		}
	}()
	return nil
}

// Stop gracefully shuts down the server.
func (s *Server) Stop() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if s.httpServer != nil {
		s.httpServer.Shutdown(ctx)
	}
	if s.rpcxServer != nil {
		s.rpcxServer.Shutdown(ctx)
	}
	log.Println("[server] stopped")
}

// ── REST API ────────────────────────────────────────────────────────────

func (s *Server) registerAPI(mux *http.ServeMux) {
	mux.HandleFunc("GET /api/v1/cluster/status", s.apiClusterStatus)
	mux.HandleFunc("GET /api/v1/nodes", s.apiListNodes)
	mux.HandleFunc("GET /api/v1/nodes/{id}", s.apiNodeDetail)
	mux.HandleFunc("GET /api/v1/nodes/{id}/logs", s.apiNodeLogs)
	mux.HandleFunc("POST /api/v1/command", s.apiDispatchCommand)
	mux.HandleFunc("GET /api/v1/commands", s.apiCommandHistory)
	mux.HandleFunc("GET /api/v1/models", s.apiModels)
	mux.HandleFunc("POST /api/v1/nodes/{id}/vllm/chat", s.apiNodeVLLMChat)
	mux.HandleFunc("GET /api/v1/nodes/{id}/vllm/models", s.apiNodeVLLMModels)
	mux.HandleFunc("GET /api/v1/nodes/{id}/cache/stats", s.apiNodeCacheStats)
	mux.HandleFunc("GET /api/v1/nodes/{id}/vllm/metrics", s.apiNodeVLLMMetrics)
}

func jsonResp(w http.ResponseWriter, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(data)
}

func (s *Server) apiClusterStatus(w http.ResponseWriter, r *http.Request) {
	summary := s.registry.Summary()
	jsonResp(w, summary)
}

func (s *Server) apiListNodes(w http.ResponseWriter, r *http.Request) {
	summary := s.registry.Summary()
	jsonResp(w, summary.Nodes)
}

func (s *Server) apiNodeDetail(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	detail := s.registry.NodeDetail(id)
	if detail == nil {
		http.Error(w, "node not found", 404)
		return
	}
	jsonResp(w, detail)
}

func (s *Server) apiDispatchCommand(w http.ResponseWriter, r *http.Request) {
	var req cluster.DispatchReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	ok := s.dispatcher.Dispatch(&req)
	jsonResp(w, ok)
}

func (s *Server) apiCommandHistory(w http.ResponseWriter, r *http.Request) {
	jsonResp(w, s.registry.CommandHistory())
}

func (s *Server) apiModels(w http.ResponseWriter, r *http.Request) {
	jsonResp(w, s.models.List())
}

// fetchVLLMMetric parses a single float64 value from vLLM's Prometheus /metrics.
func (s *Server) fetchVLLMMetric(nodeIP, metricName string) float64 {
	target := fmt.Sprintf("http://%s:8000/metrics", nodeIP)
	resp, err := s.httpClient.Get(target)
	if err != nil {
		return 0
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0
	}
	prefix := metricName + "{"
	for _, line := range strings.Split(string(body), "\n") {
		if strings.HasPrefix(line, "#") || line == "" {
			continue
		}
		if strings.HasPrefix(line, metricName+" ") || strings.HasPrefix(line, prefix) {
			parts := strings.Fields(line)
			if len(parts) >= 2 {
				if v, err := strconv.ParseFloat(parts[len(parts)-1], 64); err == nil {
					return v
				}
			}
		}
	}
	return 0
}


// fetchGoEngineBlocksRetrieved queries the node's Go disk-cache engine /stats for BlocksRetrieved.
func (s *Server) fetchGoEngineBlocksRetrieved(nodeIP string) int64 {
	target := fmt.Sprintf("http://%s:9100/stats", nodeIP)
	resp, err := s.httpClient.Get(target)
	if err != nil {
		return 0
	}
	defer resp.Body.Close()
	var stats struct {
		BlocksRetrieved int64 `json:"BlocksRetrieved"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&stats); err != nil {
		return 0
	}
	return stats.BlocksRetrieved
}

// fetchGoEngineBlocksStored queries the node's Go disk-cache engine /stats for BlocksStored.
func (s *Server) fetchGoEngineBlocksStored(nodeIP string) int64 {
	target := fmt.Sprintf("http://%s:9100/stats", nodeIP)
	resp, err := s.httpClient.Get(target)
	if err != nil {
		return 0
	}
	defer resp.Body.Close()
	var stats struct {
		BlocksStored int64 `json:"BlocksStored"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&stats); err != nil {
		return 0
	}
	return stats.BlocksStored
}


// apiNodeVLLMChat proxies a chat completion request to the node's vLLM instance,
// returning cache hit info alongside the vLLM response.
func (s *Server) apiNodeVLLMChat(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	detail := s.registry.NodeDetail(id)
	if detail == nil || detail.Info == nil {
		http.Error(w, `{"error":"node not found"}`, 404)
		return
	}

	// Read vLLM prefix cache counters BEFORE the request
	hitsBefore := s.fetchVLLMMetric(detail.Info.IP, "vllm:prefix_cache_hits_total")
	queriesBefore := s.fetchVLLMMetric(detail.Info.IP, "vllm:prefix_cache_queries_total")

	// Read the request body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, `{"error":"read body: `+err.Error()+`"}`, 400)
		return
	}

	// Proxy to vLLM (always port 8000 on the node)
	target := fmt.Sprintf("http://%s:8000/v1/chat/completions", detail.Info.IP)
	req, err := http.NewRequestWithContext(r.Context(), "POST", target, bytes.NewReader(body))
	if err != nil {
		http.Error(w, `{"error":"create request: `+err.Error()+`"}`, 500)
		return
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.httpClient.Do(req)
	if err != nil {
		http.Error(w, `{"error":"vLLM proxy: `+err.Error()+`"}`, 502)
		return
	}
	defer resp.Body.Close()

	// Read the vLLM response body
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		http.Error(w, `{"error":"read vLLM response: `+err.Error()+`"}`, 502)
		return
	}

	// Read vLLM prefix cache counters AFTER the request
	hitsAfter := s.fetchVLLMMetric(detail.Info.IP, "vllm:prefix_cache_hits_total")
	queriesAfter := s.fetchVLLMMetric(detail.Info.IP, "vllm:prefix_cache_queries_total")
	// Read disk engine stats (cumulative BlocksRetrieved)
	diskRetrieved := s.fetchGoEngineBlocksRetrieved(detail.Info.IP)

	hitTokens := int64(hitsAfter - hitsBefore)
	if hitTokens < 0 {
		hitTokens = 0
	}
	queriedTokens := int64(queriesAfter - queriesBefore)
	if queriedTokens < 0 {
		queriedTokens = 0
	}

	// Try to embed cache info into the response JSON
	var responseMap map[string]interface{}
	if err := json.Unmarshal(respBody, &responseMap); err == nil {
		responseMap["_cache"] = map[string]interface{}{
			"hit_tokens":          hitTokens,
			"queried_tokens":      queriedTokens,
			"disk_blocks_retrieved": diskRetrieved,
			"disk_blocks_stored":   s.fetchGoEngineBlocksStored(detail.Info.IP),
		}
		// Also add cache info to usage.prompt_tokens_details if present
		if usage, ok := responseMap["usage"].(map[string]interface{}); ok {
			usage["cached_tokens"] = hitTokens
		}
		respBody, _ = json.Marshal(responseMap)
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	w.Write(respBody)
}

// apiNodeVLLMModels proxies a model list request to the node's vLLM instance.
func (s *Server) apiNodeVLLMModels(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	detail := s.registry.NodeDetail(id)
	if detail == nil || detail.Info == nil {
		http.Error(w, `{"error":"node not found"}`, 404)
		return
	}

	target := fmt.Sprintf("http://%s:8000/v1/models", detail.Info.IP)
	resp, err := s.httpClient.Get(target)
	if err != nil {
		http.Error(w, `{"error":"vLLM proxy: `+err.Error()+`"}`, 502)
		return
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		http.Error(w, `{"error":"read response: `+err.Error()+`"}`, 502)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	w.Write(body)
}

// apiNodeCacheStats returns the node's current cache stats (from latest heartbeat).
func (s *Server) apiNodeCacheStats(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	detail := s.registry.NodeDetail(id)
	if detail == nil || detail.Status == nil {
		http.Error(w, `{"error":"node not found or no status"}`, 404)
		return
	}
	jsonResp(w, map[string]interface{}{
		"cache_blocks":    detail.Status.CacheBlocks,
		"cache_bytes":     detail.Status.CacheBytes,
		"cache_retrieved": detail.Status.CacheRetrieved,
		"cache_evicted":   detail.Status.CacheEvicted,
		"cache_hit_rate":  detail.Status.CacheHitRate,
	})
}

// apiNodeVLLMMetrics proxies to vLLM's /metrics and returns cache-related values.
func (s *Server) apiNodeVLLMMetrics(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	detail := s.registry.NodeDetail(id)
	if detail == nil || detail.Info == nil {
		http.Error(w, `{"error":"node not found"}`, 404)
		return
	}

	target := fmt.Sprintf("http://%s:8000/metrics", detail.Info.IP)
	resp, err := s.httpClient.Get(target)
	if err != nil {
		http.Error(w, `{"error":"vLLM metrics proxy: `+err.Error()+`"}`, 502)
		return
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		http.Error(w, `{"error":"read metrics: `+err.Error()+`"}`, 502)
		return
	}

	// Parse key cache metrics from Prometheus text format
	metrics := map[string]float64{
		"prefix_cache_queries_total":      0,
		"prefix_cache_hits_total":         0,
		"external_prefix_cache_queries_total": 0,
		"kv_cache_usage_perc":            0,
	}
	lines := strings.Split(string(body), "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "#") || line == "" {
			continue
		}
		for key := range metrics {
			prefix := key + "{"
			if strings.HasPrefix(line, key+" ") || strings.HasPrefix(line, prefix) {
				parts := strings.Fields(line)
				if len(parts) >= 2 {
					if v, err := strconv.ParseFloat(parts[len(parts)-1], 64); err == nil {
						metrics[key] = v
					}
				}
				break
			}
		}
	}

	jsonResp(w, metrics)
}

func (s *Server) apiNodeLogs(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	offsetStr := r.URL.Query().Get("offset")
	linesStr := r.URL.Query().Get("lines")
	offset, _ := strconv.Atoi(offsetStr)
	maxLines, _ := strconv.Atoi(linesStr)
	if maxLines <= 0 {
		maxLines = 50
	}

	// Try to find the latest vLLM log file for this node
	logDir := fmt.Sprintf("/root/cascade/agent/logs")
	entries, err := os.ReadDir(logDir)
	if err != nil {
		jsonResp(w, cluster.LogChunk{NodeID: id, Lines: "", EOF: true})
		return
	}

	// Find the most recent vllm-*.log file
	var latest string
	var latestMod time.Time
	for _, e := range entries {
		if !e.IsDir() && strings.HasPrefix(e.Name(), "vllm-") && strings.HasSuffix(e.Name(), ".log") {
			fi, err := e.Info()
			if err == nil && fi.ModTime().After(latestMod) {
				latest = filepath.Join(logDir, e.Name())
				latestMod = fi.ModTime()
			}
		}
	}
	if latest == "" {
		jsonResp(w, cluster.LogChunk{NodeID: id, Lines: "", EOF: true})
		return
	}

	// Read from offset, return lines after offset
	data, err := os.ReadFile(latest)
	if err != nil {
		http.Error(w, "read log: "+err.Error(), 500)
		return
	}
	total := len(data)
	if offset >= total {
		jsonResp(w, cluster.LogChunk{NodeID: id, Lines: "", Offset: int64(total), EOF: true})
		return
	}
	// Return content from offset (max maxLines lines)
	chunk := data[offset:]
	lines := strings.Split(string(chunk), "\n")
	end := len(lines)
	if end > maxLines {
		end = maxLines
	}
	out := strings.Join(lines[:end], "\n")
	newOffset := offset
	for i := 0; i < end; i++ {
		newOffset += len(lines[i]) + 1
	}
	if newOffset > total {
		newOffset = total
	}
	jsonResp(w, cluster.LogChunk{
		NodeID:   id,
		Lines:    out,
		Offset:   int64(newOffset),
		EOF:      newOffset >= total,
	})
}

// ============================================================================
// rpcx Service Implementations (internal)
// ============================================================================

// clusterSvc implements cluster.ClusterService.
type clusterSvc struct{ reg *Registry }

func (s *clusterSvc) Register(ctx context.Context, args *cluster.NodeInfo, reply *cluster.RegisterReply) error {
	*reply = *s.reg.Register(args)
	return nil
}

func (s *clusterSvc) Heartbeat(ctx context.Context, args *cluster.MachineStatus, reply *cluster.HeartbeatReply) error {
	*reply = *s.reg.Heartbeat(args)
	return nil
}

// cmdSvc implements cluster.CommandService.
type cmdSvc struct{ reg *Registry }

func (s *cmdSvc) FetchCommands(ctx context.Context, nodeID string, reply *[]*cluster.Command) error {
	state := s.reg.GetNode(nodeID)
	if state == nil {
		return nil
	}
	*reply = state.commands
	state.commands = nil
	return nil
}

func (s *cmdSvc) ReportResult(ctx context.Context, args *cluster.CmdResult, reply *cluster.OK) error {
	s.reg.RecordResult(args)
	*reply = cluster.OK{OK: true}
	return nil
}

// querySvc implements cluster.QueryService.
type querySvc struct {
	reg    *Registry
	router *Router
}

func (s *querySvc) CacheLookup(ctx context.Context, hash uint64, reply *cluster.CacheLocation) error {
	loc, err := s.router.Lookup(hash)
	if err != nil {
		return err
	}
	if loc != nil {
		*reply = *loc
	}
	return nil
}

func (s *querySvc) ClusterStatus(ctx context.Context, _ *cluster.Empty, reply *cluster.ClusterSummary) error {
	*reply = *s.reg.Summary()
	return nil
}

func (s *querySvc) NodeDetail(ctx context.Context, nodeID string, reply *cluster.NodeDetail) error {
	detail := s.reg.NodeDetail(nodeID)
	if detail != nil {
		*reply = *detail
	}
	return nil
}

// adminSvc implements cluster.AdminService.
type adminSvc struct {
	disp *Dispatcher
	reg  *Registry
}

func (s *adminSvc) DispatchCommand(ctx context.Context, args *cluster.DispatchReq, reply *cluster.OK) error {
	*reply = *s.disp.Dispatch(args)
	return nil
}

func (s *adminSvc) CommandHistory(ctx context.Context, _ *cluster.Empty, reply *[]*cluster.CmdResult) error {
	*reply = s.reg.CommandHistory()
	return nil
}

// Ensure compiler compliance
var (
	_ cluster.ClusterService  = (*clusterSvc)(nil)
	_ cluster.CommandService  = (*cmdSvc)(nil)
	_ cluster.QueryService    = (*querySvc)(nil)
	_ cluster.AdminService    = (*adminSvc)(nil)
)
