package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
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
	httpServer    *http.Server
	httpClient    *http.Client
	gatewayClient *http.Client
	scheduler     *GatewayScheduler

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
	StateDir    string // durable command history and pending command state (optional)

	// GatewayMaxInFlightPerNode bounds requests accepted by the global OpenAI
	// gateway before the agent heartbeat has observed them.
	GatewayMaxInFlightPerNode int
	GatewayMaxRequestBytes    int64
}

// DefaultConfig returns sensible defaults.
func DefaultConfig() *Config {
	return &Config{
		RPCPort:                   9000,
		HTTPPort:                  8080,
		GatewayMaxInFlightPerNode: 16,
		GatewayMaxRequestBytes:    64 << 20,
	}
}

// New creates a new S端 Server.
func New(cfg *Config) *Server {
	if cfg == nil {
		cfg = DefaultConfig()
	}
	reg, err := NewPersistentRegistry(cfg.StateDir)
	if err != nil {
		log.Printf("[server] command persistence unavailable (%v); starting with empty in-memory state", err)
		reg = NewRegistry()
	}
	dt := NewDiskTracker()
	gatewayTransport := http.DefaultTransport.(*http.Transport).Clone()

	srv := &Server{
		registry:    reg,
		dispatcher:  NewDispatcher(reg),
		diskTracker: dt,
		config:      cfg,
		httpClient: &http.Client{
			Timeout: 120 * time.Second,
		},
		gatewayClient: &http.Client{Transport: gatewayTransport},
		scheduler:     NewGatewayScheduler(reg, cfg.GatewayMaxInFlightPerNode),
	}

	// Router with nil meta backend for now — will be connected to disk-cache engine later
	srv.router = NewRouter(reg, dt, nil)
	// Initialize model registry with directory scanning
	baseURL := strings.TrimSpace(cfg.PublicURL)
	if baseURL == "" && cfg.ModelsDir != "" {
		log.Printf("[server] model distribution disabled: --public-url must be an Agent-reachable /models/ URL")
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
	s.rpcxServer.RegisterName("AdminService", &adminSvc{disp: s.dispatcher, reg: s.registry, enrich: s.enrichDispatchRequest}, "")

	log.Printf("[server] rpcx listening on %s", addr)
	go func() {
		if err := s.rpcxServer.Serve("tcp", addr); err != nil {
			// Shutdown causes rpcx Serve to return. It must not turn a graceful
			// server stop into a process-wide fatal exit.
			log.Printf("[server] rpcx serve stopped: %v", err)
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
	staticDir, err := resolveStaticDir()
	if err != nil {
		return err
	}

	// Single entry point: API routes → apiMux, everything else → static files
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path := r.URL.Path
		// Serve model files for distribution (e.g. /models/Qwen2.5-7B-Instruct/)
		if len(path) >= 8 && path[:8] == "/models/" && s.config.ModelsDir != "" {
			http.StripPrefix("/models/", http.FileServer(http.Dir(s.config.ModelsDir))).ServeHTTP(w, r)
			return
		}

		if strings.HasPrefix(path, "/api/v1/") || strings.HasPrefix(path, "/v1/") || path == "/health" {
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

// resolveStaticDir locates the checked-in web bundle in both supported
// development entry points: the repository root and engine/. It also supports
// a binary placed in the repository's bin/ directory.
func resolveStaticDir() (string, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("get working directory for web assets: %w", err)
	}

	roots := []string{cwd}
	if executable, err := os.Executable(); err == nil {
		executableDir := filepath.Dir(executable)
		roots = append(roots, executableDir, filepath.Dir(executableDir))
	}

	seen := make(map[string]struct{})
	for _, root := range roots {
		for _, relative := range []string{
			filepath.Join("engine", "pkg", "server", "web", "static"),
			filepath.Join("pkg", "server", "web", "static"),
		} {
			candidate := filepath.Join(root, relative)
			if _, checked := seen[candidate]; checked {
				continue
			}
			seen[candidate] = struct{}{}
			index, err := os.Stat(filepath.Join(candidate, "index.html"))
			if err == nil && !index.IsDir() {
				return candidate, nil
			}
		}
	}

	return "", fmt.Errorf("web static assets not found from working directory %q", cwd)
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
	// Cluster-wide OpenAI-compatible data plane. These routes select a healthy
	// vLLM replica; callers must not pin a node ID.
	mux.HandleFunc("GET /health", s.apiGatewayHealth)
	mux.HandleFunc("GET /v1/models", s.apiGatewayModels)
	mux.HandleFunc("POST /v1/chat/completions", s.apiGatewayInference)
	mux.HandleFunc("POST /v1/completions", s.apiGatewayInference)
	mux.HandleFunc("POST /v1/embeddings", s.apiGatewayInference)

	mux.HandleFunc("GET /api/v1/cluster/status", s.apiClusterStatus)
	mux.HandleFunc("GET /api/v1/gateway/status", s.apiGatewayStatus)
	mux.HandleFunc("GET /api/v1/nodes", s.apiListNodes)
	mux.HandleFunc("GET /api/v1/nodes/{id}", s.apiNodeDetail)
	mux.HandleFunc("GET /api/v1/nodes/{id}/logs", s.apiNodeLogs)
	mux.HandleFunc("POST /api/v1/command", s.apiDispatchCommand)
	mux.HandleFunc("GET /api/v1/commands", s.apiCommandHistory)
	mux.HandleFunc("GET /api/v1/models", s.apiModels)
	mux.HandleFunc("GET /api/v1/models/{name}/manifest", s.apiModelManifest)
	mux.HandleFunc("POST /api/v1/nodes/{id}/vllm/chat", s.apiNodeVLLMChat)
	mux.HandleFunc("GET /api/v1/nodes/{id}/vllm/models", s.apiNodeVLLMModels)
	mux.HandleFunc("GET /api/v1/nodes/{id}/cache/stats", s.apiNodeCacheStats)
	mux.HandleFunc("GET /api/v1/nodes/{id}/vllm/metrics", s.apiNodeVLLMMetrics)
}

func jsonResp(w http.ResponseWriter, data interface{}) {
	jsonRespStatus(w, http.StatusOK, data)
}

func jsonRespStatus(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func (s *Server) apiClusterStatus(w http.ResponseWriter, r *http.Request) {
	summary := s.registry.Summary()
	jsonResp(w, summary)
}

func (s *Server) apiGatewayStatus(w http.ResponseWriter, r *http.Request) {
	jsonResp(w, s.gatewayStatus())
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
	if err := s.enrichDispatchRequest(&req); err != nil {
		jsonRespStatus(w, http.StatusBadRequest, &cluster.OK{OK: false, Err: err.Error()})
		return
	}
	ok := s.dispatcher.Dispatch(&req)
	if !ok.OK {
		jsonRespStatus(w, http.StatusBadRequest, ok)
		return
	}
	jsonResp(w, ok)
}

func (s *Server) apiCommandHistory(w http.ResponseWriter, r *http.Request) {
	jsonResp(w, s.registry.CommandHistory())
}

func (s *Server) apiModels(w http.ResponseWriter, r *http.Request) {
	jsonResp(w, s.models.List())
}

func (s *Server) apiModelManifest(w http.ResponseWriter, r *http.Request) {
	if s.models == nil {
		http.Error(w, `{"error":"model registry unavailable"}`, http.StatusServiceUnavailable)
		return
	}
	manifest, err := s.models.Manifest(r.PathValue("name"))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			http.Error(w, `{"error":"published model not found"}`, http.StatusNotFound)
			return
		}
		jsonRespStatus(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	jsonResp(w, manifest)
}

// enrichDispatchRequest resolves model distribution through the server's
// catalog. The browser does not get to supply an arbitrary URL for an Agent to
// fetch; that would be both unsafe and impossible to verify operationally.
func (s *Server) enrichDispatchRequest(req *cluster.DispatchReq) error {
	if req == nil || req.Action != cluster.CmdDownloadModel {
		return nil
	}
	if s.models == nil {
		return errors.New("model registry is unavailable")
	}
	modelName := strings.TrimSpace(req.Params["model"])
	model := s.models.Find(modelName)
	if model == nil {
		return fmt.Errorf("model %q is not published by this control server", modelName)
	}
	if !model.DistributionReady {
		return fmt.Errorf("model %q is not ready for distribution; configure a matching manifest or a Hugging Face repository source", modelName)
	}
	if supplied := strings.TrimSpace(req.Params["download_url"]); supplied != "" && supplied != model.DownloadURL {
		return fmt.Errorf("model distribution source is controlled by the catalog and cannot be overridden")
	}
	if supplied := strings.TrimSpace(req.Params["manifest_url"]); supplied != "" && supplied != model.ManifestURL {
		return fmt.Errorf("model manifest is controlled by the catalog and cannot be overridden")
	}
	params := cloneParams(req.Params)
	if params == nil {
		params = make(map[string]string)
	}
	params["download_url"] = model.DownloadURL
	if model.ManifestURL != "" {
		params["manifest_url"] = model.ManifestURL
	}
	req.Params = params
	return nil
}

// fetchVLLMMetric parses a single float64 value from vLLM's Prometheus /metrics.
func (s *Server) fetchVLLMMetric(nodeIP string, vllmPort int, metricName string) float64 {
	target, err := vllmEndpointURL(nodeIP, vllmPort, "/metrics", "")
	if err != nil {
		return 0
	}
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

// cacheStatsSnapshot mirrors the metadata service's legacy and v2 counters.
// It stays local to the control server so the server does not import the Agent
// package (which would blur the control-plane ownership boundary).
type cacheStatsSnapshot struct {
	BlocksStored    int64 `json:"BlocksStored"`
	BlocksRetrieved int64 `json:"BlocksRetrieved"`
	BlocksEvicted   int64 `json:"BlocksEvicted"`
	DiskUsedBytes   int64 `json:"DiskUsedBytes"`
	ChunksStored    int64 `json:"ChunksStored"`
	ChunksRetrieved int64 `json:"ChunksRetrieved"`
	MatchRequests   int64 `json:"MatchRequests"`
	MatchHits       int64 `json:"MatchHits"`
	MatchedTokens   int64 `json:"MatchedTokens"`
}

func (stats cacheStatsSnapshot) entryCount() int64 {
	return stats.BlocksStored + stats.ChunksStored
}

func (stats cacheStatsSnapshot) retrievedCount() int64 {
	return stats.BlocksRetrieved + stats.ChunksRetrieved
}

// fetchCacheStats follows the cache metadata endpoint explicitly advertised by
// an Agent. Shared-pool deployments have one central service, which is often
// not on the Agent IP or the historical hard-coded port 9100.
func (s *Server) fetchCacheStats(metadataURL string) (cacheStatsSnapshot, error) {
	target, err := cacheStatsURL(metadataURL)
	if err != nil {
		return cacheStatsSnapshot{}, err
	}
	client := s.httpClient
	if client == nil {
		client = http.DefaultClient
	}
	response, err := client.Get(target)
	if err != nil {
		return cacheStatsSnapshot{}, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return cacheStatsSnapshot{}, fmt.Errorf("cache metadata status %d", response.StatusCode)
	}
	var stats cacheStatsSnapshot
	if err := json.NewDecoder(response.Body).Decode(&stats); err != nil {
		return cacheStatsSnapshot{}, err
	}
	return stats, nil
}

func cacheStatsURL(raw string) (string, error) {
	endpoint, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || endpoint.Host == "" || (endpoint.Scheme != "http" && endpoint.Scheme != "https") {
		return "", fmt.Errorf("invalid cache metadata URL %q", raw)
	}
	endpoint.Path = strings.TrimRight(endpoint.Path, "/") + "/stats"
	endpoint.RawQuery = ""
	return endpoint.String(), nil
}

// apiNodeVLLMChat proxies a chat completion request to the node's vLLM instance,
func (s *Server) apiNodeVLLMChat(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	detail := s.registry.NodeDetail(id)
	if detail == nil || detail.Info == nil {
		http.Error(w, `{"error":"node not found"}`, 404)
		return
	}

	// Read cache metadata before the request. This is best-effort diagnostics;
	// the inference request itself must not fail merely because telemetry is
	// temporarily unavailable.
	cacheBefore, _ := s.fetchCacheStats(detail.Info.CacheMetadataURL)

	// Read the request body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, `{"error":"read body: `+err.Error()+`"}`, 400)
		return
	}

	// Proxy to the explicitly advertised vLLM endpoint.
	target, err := vllmEndpointURL(detail.Info.IP, detail.Info.VLLMPort, "/v1/chat/completions", r.URL.RawQuery)
	if err != nil {
		http.Error(w, `{"error":"invalid vLLM endpoint: `+err.Error()+`"}`, http.StatusBadGateway)
		return
	}
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

	// Read stats after the request. v2 chunks are the active cache objects, so
	// include them alongside legacy blocks in every diagnostic total.
	cacheAfter, _ := s.fetchCacheStats(detail.Info.CacheMetadataURL)
	retrievedObjects := cacheAfter.retrievedCount() - cacheBefore.retrievedCount()
	if retrievedObjects < 0 {
		retrievedObjects = 0
	}

	// Only inject _cache for successful responses
	hitTokens := int64(0)
	var responseMap map[string]interface{}
	if resp.StatusCode == http.StatusOK && json.Unmarshal(respBody, &responseMap) == nil {
		if usage, ok := responseMap["usage"].(map[string]interface{}); ok {
			if details, ok := usage["prompt_tokens_details"].(map[string]interface{}); ok {
				if ct, ok := details["cached_tokens"].(float64); ok {
					hitTokens = int64(ct)
				}
			}
		}
		responseMap["_cache"] = map[string]interface{}{
			"hit_tokens":              hitTokens,
			"retrieved_objects":       retrievedObjects,
			"retrieved_objects_total": cacheAfter.retrievedCount(),
			"stored_objects":          cacheAfter.entryCount(),
			"match_requests_total":    cacheAfter.MatchRequests,
			"match_hits_total":        cacheAfter.MatchHits,
			"matched_tokens_total":    cacheAfter.MatchedTokens,
			"disk_blocks":             retrievedObjects, // legacy compatibility
			"disk_blocks_total":       cacheAfter.retrievedCount(),
			"disk_blocks_stored":      cacheAfter.entryCount(),
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

	target, err := vllmEndpointURL(detail.Info.IP, detail.Info.VLLMPort, "/v1/models", r.URL.RawQuery)
	if err != nil {
		http.Error(w, `{"error":"invalid vLLM endpoint: `+err.Error()+`"}`, http.StatusBadGateway)
		return
	}
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
		"cache_objects":        detail.Status.CacheBlocks,
		"cache_blocks":         detail.Status.CacheBlocks,
		"cache_bytes":          detail.Status.CacheBytes,
		"cache_retrieved":      detail.Status.CacheRetrieved,
		"cache_evicted":        detail.Status.CacheEvicted,
		"cache_hit_rate":       detail.Status.CacheHitRate,
		"cache_match_requests": detail.Status.CacheMatchRequests,
		"cache_match_hits":     detail.Status.CacheMatchHits,
		"cache_matched_tokens": detail.Status.CacheMatchedTokens,
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

	target, err := vllmEndpointURL(detail.Info.IP, detail.Info.VLLMPort, "/metrics", r.URL.RawQuery)
	if err != nil {
		http.Error(w, `{"error":"invalid vLLM endpoint: `+err.Error()+`"}`, http.StatusBadGateway)
		return
	}
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
		"prefix_cache_queries_total":          0,
		"prefix_cache_hits_total":             0,
		"external_prefix_cache_queries_total": 0,
		"kv_cache_usage_perc":                 0,
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
	detail := s.registry.NodeDetail(id)
	if detail == nil || detail.Info == nil {
		http.Error(w, `{"error":"node not found"}`, http.StatusNotFound)
		return
	}
	target, err := diagnosticsLogURL(detail.Info.DiagnosticsURL, r.URL.Query())
	if err != nil {
		jsonRespStatus(w, http.StatusServiceUnavailable, map[string]string{"error": err.Error()})
		return
	}
	client := s.httpClient
	if client == nil {
		client = http.DefaultClient
	}
	response, err := client.Get(target)
	if err != nil {
		jsonRespStatus(w, http.StatusBadGateway, map[string]string{"error": "agent diagnostics unavailable: " + err.Error()})
		return
	}
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	if err != nil {
		jsonRespStatus(w, http.StatusBadGateway, map[string]string{"error": "read agent diagnostics: " + err.Error()})
		return
	}
	contentType := response.Header.Get("Content-Type")
	if contentType == "" {
		contentType = "application/json"
	}
	w.Header().Set("Content-Type", contentType)
	w.WriteHeader(response.StatusCode)
	_, _ = w.Write(body)
}

func diagnosticsLogURL(raw string, query url.Values) (string, error) {
	endpoint, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || endpoint.Host == "" || (endpoint.Scheme != "http" && endpoint.Scheme != "https") {
		return "", errors.New("Agent does not advertise a usable diagnostics endpoint")
	}
	endpoint.Path = strings.TrimRight(endpoint.Path, "/") + "/v1/logs"
	endpoint.RawQuery = query.Encode()
	return endpoint.String(), nil
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
	*reply = s.reg.FetchCommands(nodeID)
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
	disp   *Dispatcher
	reg    *Registry
	enrich func(*cluster.DispatchReq) error
}

func (s *adminSvc) DispatchCommand(ctx context.Context, args *cluster.DispatchReq, reply *cluster.OK) error {
	if args == nil {
		*reply = cluster.OK{OK: false, Err: "dispatch request is required"}
		return nil
	}
	request := *args
	request.Params = cloneParams(args.Params)
	if s.enrich != nil {
		if err := s.enrich(&request); err != nil {
			*reply = cluster.OK{OK: false, Err: err.Error()}
			return nil
		}
	}
	*reply = *s.disp.Dispatch(&request)
	return nil
}

func (s *adminSvc) CommandHistory(ctx context.Context, _ *cluster.Empty, reply *[]*cluster.CmdResult) error {
	*reply = s.reg.CommandHistory()
	return nil
}

// Ensure compiler compliance
var (
	_ cluster.ClusterService = (*clusterSvc)(nil)
	_ cluster.CommandService = (*cmdSvc)(nil)
	_ cluster.QueryService   = (*querySvc)(nil)
	_ cluster.AdminService   = (*adminSvc)(nil)
)
