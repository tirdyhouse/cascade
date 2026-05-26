package server

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
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

	// REST
	httpServer *http.Server

	// Config
	config *Config
}

// Config holds S端 configuration.
type Config struct {
	RPCPort     int    // rpcx server port (C端 connect here)
	HTTPPort    int    // REST API + Web UI port
	MetadataDir string // metadata directory for existing disk-cache engine
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
	}

	// Router with nil meta backend for now — will be connected to disk-cache engine later
	srv.router = NewRouter(reg, dt, nil)

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
	mux := http.NewServeMux()

	// REST API
	s.registerAPI(mux)

	// Web UI static files
	fs := http.FileServer(http.Dir("pkg/server/web/static"))
	mux.Handle("/", fs)

	addr := fmt.Sprintf(":%d", s.config.HTTPPort)
	s.httpServer = &http.Server{
		Addr:    addr,
		Handler: mux,
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
	mux.HandleFunc("POST /api/v1/command", s.apiDispatchCommand)
	mux.HandleFunc("GET /api/v1/commands", s.apiCommandHistory)
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
