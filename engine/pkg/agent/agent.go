package agent

import (
	"context"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"predict/engine/pkg/cluster"
)

// Agent is the C端 agent that runs on each GPU machine.
// It connects to S端, reports status periodically, and executes commands.
type Agent struct {
	config *Config

	// Component state
	statusSeq int64

	// Sub-components
	collector *Collector
	process   *ProcessManager
	cache     *CacheProxy

	// rpcx client
	client *RPCClient

	// Lifecycle
	mu     sync.Mutex
	ctx    context.Context
	cancel context.CancelFunc

	// Command result replay protects an operation's visible lifecycle across a
	// control-server restart. Results are retained only until the server accepts
	// them, and duplicate command delivery is answered without running a second
	// process operation.
	commandMu     sync.Mutex
	commandStates map[string]*cluster.CmdResult
	unreported    map[string]*cluster.CmdResult
	statusRefresh chan struct{}
	pollErrorMu   sync.Mutex
	pollError     string
	pollErrorAt   time.Time

	diagnosticsURL string
}

const commandPollErrorLogInterval = 30 * time.Second

// Config holds C端 agent configuration.
type Config struct {
	NodeID          string
	ServerAddr      string // S端 rpcx address
	RPCPort         int    // local rpcx port for bidirectional (optional)
	CacheMode       cluster.CacheMode
	CachePath       string // disk-cache metadata API base URL
	SharedCacheRoot string // shared filesystem root when CacheModeSharedPool
	SharedCacheID   string // stable shared cache identity when CacheModeSharedPool
	WorkDir         string // working directory for models, logs, cache
	VLLMHost        string // host passed to vLLM; must be reachable by cluster-server
	VLLMPort        int    // HTTP port passed to vLLM and advertised to cluster-server
	AdvertiseHost   string // optional host/IP reachable from the control-plane gateway
	VLLMPath        string // optional explicit vLLM executable path
	DiagnosticsHost string // read-only agent diagnostics listen host
	DiagnosticsPort int    // read-only agent diagnostics port; <=0 disables it explicitly
	DiagnosticsURL  string // externally reachable override advertised to cluster-server

	// Hardware
	GPUType  string
	GPUMemMB int64
	GPUCount int

	// Disks
	Disks []cluster.DiskInfo
}

// DefaultConfig returns a configuration with sensible defaults.
func DefaultConfig() *Config {
	return &Config{
		RPCPort:         9001,
		CacheMode:       cluster.CacheModeLocalNVMe,
		VLLMHost:        defaultVLLMHost,
		VLLMPort:        defaultVLLMPort,
		DiagnosticsHost: defaultDiagnosticsHost,
		DiagnosticsPort: defaultDiagnosticsPort,
	}
}

// New creates a new C端 Agent.
func New(cfg *Config) *Agent {
	ctx, cancel := context.WithCancel(context.Background())
	return &Agent{
		config:        cfg,
		collector:     NewCollector(cfg),
		process:       NewProcessManager(cfg),
		cache:         NewCacheProxy(cfg),
		client:        NewRPCClient(cfg.ServerAddr),
		ctx:           ctx,
		cancel:        cancel,
		commandStates: make(map[string]*cluster.CmdResult),
		unreported:    make(map[string]*cluster.CmdResult),
		statusRefresh: make(chan struct{}, 1),
	}
}

// Start begins the agent main loop.
func (a *Agent) Start() error {
	log.Printf("[agent] starting C端 agent: node=%s server=%s mode=%s",
		a.config.NodeID, a.config.ServerAddr, a.config.CacheMode)

	// The agent owns its diagnostics endpoint. It is started before registration
	// so the advertised URL is immediately usable by the control plane.
	diagnostics, err := startDiagnosticsServer(a.config)
	if err != nil {
		return err
	}
	if diagnostics != nil {
		a.mu.Lock()
		a.diagnosticsURL = a.advertisedDiagnosticsURL(diagnostics.port())
		a.mu.Unlock()
		defer diagnostics.Stop()
		log.Printf("[agent] diagnostics available at %s", a.advertisedDiagnosticsURL(diagnostics.port()))
	}

	// 1. Connect to S端
	if err := a.client.Connect(); err != nil {
		return err
	}
	defer a.client.Stop()

	// 2. Register with S端
	if err := a.register(); err != nil {
		return err
	}
	a.replayUnreported()

	log.Printf("[agent] registered as %s", a.config.NodeID)

	// 3. Start a single serial command worker. Heartbeats must stay independent
	// of long operations such as a model download or vLLM warm-up.
	cmdCh := make(chan *cluster.Command, 16)
	go a.commandLoop(cmdCh)
	// Publish the first status and collect pending work immediately. Waiting for
	// the first periodic tick used to make a newly registered Agent appear idle
	// for up to five seconds and made an operator-issued command feel lost.
	if err := a.heartbeat(cmdCh); err != nil {
		log.Printf("[agent] initial heartbeat error: %v", err)
		a.reconnectAndRegister()
	}

	// 4. Heartbeats report full telemetry every five seconds. Commands are
	// polled independently once per second so a user action is not delayed by
	// the telemetry cadence.
	heartbeatTicker := time.NewTicker(5 * time.Second)
	defer heartbeatTicker.Stop()
	commandTicker := time.NewTicker(time.Second)
	defer commandTicker.Stop()

	for {
		select {
		case <-heartbeatTicker.C:
			if err := a.heartbeat(cmdCh); err != nil {
				log.Printf("[agent] heartbeat error: %v", err)
				a.reconnectAndRegister()
			}
		case <-commandTicker.C:
			a.fetchPendingCommands(cmdCh)
		case <-a.statusRefresh:
			if err := a.heartbeat(cmdCh); err != nil {
				log.Printf("[agent] status refresh error: %v", err)
				a.reconnectAndRegister()
			}
		case <-a.ctx.Done():
			log.Println("[agent] stopping")
			return nil
		}
	}
}

// Stop gracefully stops the agent.
func (a *Agent) Stop() {
	a.cancel()
	if a.process != nil {
		a.process.Stop()
	}
}

// ── Registration ──────────────────────────────────────────────────────

func (a *Agent) register() error {
	hostname, err := os.Hostname()
	if err != nil || strings.TrimSpace(hostname) == "" {
		hostname = a.config.NodeID
	}
	info := &cluster.NodeInfo{
		NodeID:           a.config.NodeID,
		Hostname:         hostname,
		IP:               a.advertisedHost(),
		RPCPort:          a.config.RPCPort,
		VLLMPort:         a.vllmPort(),
		DiagnosticsURL:   a.currentDiagnosticsURL(),
		CacheMode:        a.config.CacheMode,
		SharedCacheID:    a.config.SharedCacheID,
		CacheMetadataURL: a.config.CachePath,
		GPUType:          a.config.GPUType,
		GPUMemMB:         a.config.GPUMemMB,
		GPUCount:         a.config.GPUCount,
		Disks:            a.collector.DiskInfo(),
	}
	reply, err := a.client.Register(info)
	if err != nil {
		return err
	}
	if !reply.Accepted {
		return fmt.Errorf("cluster registration rejected: %s", reply.Reason)
	}
	log.Printf("[agent] register reply: accepted=%v cluster_size=%d mode=%s",
		reply.Accepted, reply.ClusterSize, reply.CacheMode)
	return nil
}

// advertisedHost is the gateway-facing endpoint for this Agent's managed
// vLLM service. Auto-detection is convenient on simple hosts, but an explicit
// value is required for machines with separate management, storage, and
// inference networks.
func (a *Agent) advertisedHost() string {
	if configured := strings.TrimSpace(a.config.AdvertiseHost); configured != "" {
		return configured
	}
	return getOutboundIP()
}

func (a *Agent) reconnectAndRegister() {
	if err := a.client.Reconnect(); err != nil {
		log.Printf("[agent] reconnect failed: %v", err)
		return
	}
	if err := a.register(); err != nil {
		log.Printf("[agent] re-register failed: %v", err)
		return
	}
	log.Printf("[agent] re-registered after control-plane reconnect")
	a.replayUnreported()
}

func (a *Agent) advertisedDiagnosticsURL(port int) string {
	if configured := strings.TrimSpace(a.config.DiagnosticsURL); configured != "" {
		return strings.TrimRight(configured, "/")
	}
	if port <= 0 {
		return ""
	}
	return "http://" + net.JoinHostPort(a.advertisedHost(), strconv.Itoa(port))
}

func (a *Agent) currentDiagnosticsURL() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.diagnosticsURL
}

// ── Heartbeat ─────────────────────────────────────────────────────────

func (a *Agent) heartbeat(cmdCh chan<- *cluster.Command) error {
	a.mu.Lock()
	a.statusSeq++
	seq := a.statusSeq
	a.mu.Unlock()

	// Build status
	status := a.collector.Collect(seq)

	// Add KV cache stats from local disk-cache
	if cacheStats := a.cache.Stats(); cacheStats != nil {
		status.CacheBlocks = cacheStats.EntryCount()
		status.CacheBytes = cacheStats.DiskUsedBytes
		status.CacheHitRate = cacheStats.HitRate
		status.CacheRetrieved = cacheStats.BlocksRetrieved + cacheStats.ChunksRetrieved
		status.CacheEvicted = cacheStats.BlocksEvicted
		status.CacheMatchRequests = cacheStats.MatchRequests
		status.CacheMatchHits = cacheStats.MatchHits
		status.CacheMatchedTokens = cacheStats.MatchedTokens
	}
	status.VLLMStatus = a.process.Status()
	// Available models
	status.AvailableModels = a.collector.GetAvailableModels()

	// vLLM health check: transition loading→running / running→error
	switch status.VLLMStatus {
	case "running":
		if !a.vllmHealthy() {
			// Keep the ProcessManager in sync with telemetry. The process can
			// still be alive after its HTTP endpoint fails, and operators must be
			// able to restart or stop that managed process rather than queueing a
			// second start that will inevitably fail.
			a.process.MarkError()
			status.VLLMStatus = a.process.Status()
		}
	case "loading":
		if a.vllmHealthy() {
			status.VLLMStatus = "running"
			a.process.MarkRunning()
		}
	}
	if status.VLLMStatus == "running" {
		status.QueueLen = a.vllmQueueLen()
	}
	status.ModelName = a.process.ModelName()

	reply, err := a.client.Heartbeat(status)
	if err != nil {
		return err
	}
	if reply == nil || !reply.OK {
		return fmt.Errorf("control server no longer recognizes node %q", a.config.NodeID)
	}

	a.enqueueCommands(cmdCh, reply.PendingCmds)

	return nil
}

func (a *Agent) fetchPendingCommands(cmdCh chan<- *cluster.Command) {
	commands, err := a.client.FetchCommands(a.config.NodeID)
	if err != nil {
		// The next telemetry heartbeat owns reconnect and registration. A failed
		// low-latency poll must not interrupt a healthy long-running operation.
		if a.shouldLogCommandPollError(time.Now(), err) {
			log.Printf("[agent] command poll error: %v", err)
		}
		return
	}
	a.resetCommandPollError()
	a.enqueueCommands(cmdCh, commands)
}

func (a *Agent) shouldLogCommandPollError(now time.Time, err error) bool {
	if err == nil {
		return false
	}
	message := err.Error()
	a.pollErrorMu.Lock()
	defer a.pollErrorMu.Unlock()
	if message == a.pollError && now.Sub(a.pollErrorAt) < commandPollErrorLogInterval {
		return false
	}
	a.pollError = message
	a.pollErrorAt = now
	return true
}

func (a *Agent) resetCommandPollError() {
	a.pollErrorMu.Lock()
	a.pollError = ""
	a.pollErrorAt = time.Time{}
	a.pollErrorMu.Unlock()
}

func (a *Agent) enqueueCommands(cmdCh chan<- *cluster.Command, commands []*cluster.Command) {
	for _, cmd := range commands {
		if cmd == nil {
			continue
		}
		log.Printf("[agent] received pending cmd: %s action=%s", cmd.CmdID, cmd.Action)
		select {
		case cmdCh <- cmd:
		case <-a.ctx.Done():
			return
		}
	}
}

// ── Command Execution ────────────────────────────────────────────────

func (a *Agent) executeCommand(cmd *cluster.Command) {
	if cmd == nil || strings.TrimSpace(cmd.CmdID) == "" {
		log.Printf("[agent] ignoring malformed command")
		return
	}
	if previous := a.commandState(cmd.CmdID); previous != nil {
		// A server restart can re-deliver a command that this Agent already
		// accepted. Re-report its latest state instead of issuing a second start,
		// stop, or download operation.
		log.Printf("[agent] replaying existing cmd state: %s status=%s", cmd.CmdID, previous.Status)
		a.reportResult(previous)
		return
	}
	log.Printf("[agent] executing cmd: %s action=%s", cmd.CmdID, cmd.Action)
	a.reportCommand(cmd, "running", 0, "accepted by agent", "")

	switch cmd.Action {
	case cluster.CmdStartVLLM:
		a.executeStartVLLM(cmd)
	case cluster.CmdStopVLLM:
		a.executeStopVLLM(cmd)
	case cluster.CmdRestartVLLM:
		a.executeRestartVLLM(cmd)
	case cluster.CmdDownloadModel:
		a.executeDownloadModel(cmd)
	default:
		a.reportCommand(cmd, "failed", 100, "", "action is not supported by this Agent: "+string(cmd.Action))
	}
}

func (a *Agent) executeStartVLLM(cmd *cluster.Command) {
	// Guard: don't start if vLLM is already running
	if st := a.process.Status(); st == "running" || st == "loading" {
		log.Printf("[agent] vLLM already %s, ignoring start command", st)
		a.reportCommand(cmd, "failed", 100, "", "vLLM already "+st)
		return
	}

	if raw := strings.TrimSpace(cmd.Params["raw_args"]); raw != "" {
		a.reportCommand(cmd, "failed", 100, "", "raw_args is not supported; use the validated structured start parameters")
		return
	}

	model := strings.TrimSpace(cmd.Params["model"])
	if model == "" {
		a.reportCommand(cmd, "failed", 100, "", "model parameter required")
		return
	}
	gpuUtil := cmd.Params["gpu_util"]
	if gpuUtil == "" {
		gpuUtil = "0.9"
	}
	enablePrefix := cmd.Params["enable_prefix_caching"]
	enableDiskCache := cmd.Params["enable_disk_cache"]

	output, err := a.process.Start(&StartOptions{
		Model:           model,
		GPUUtil:         gpuUtil,
		PrefixCaching:   enablePrefix == "true",
		DiskCache:       enableDiskCache == "true",
		WorkDir:         a.config.WorkDir,
		Quantization:    cmd.Params["quantization"],
		CacheMode:       a.config.CacheMode,
		CacheEngineAddr: a.config.CachePath,
		SharedCacheRoot: a.config.SharedCacheRoot,
		SharedCacheID:   a.config.SharedCacheID,
		VLLMHost:        a.config.VLLMHost,
		VLLMPort:        a.vllmPort(),
		VLLMPath:        a.config.VLLMPath,
	})
	if err != nil {
		a.reportCommand(cmd, "failed", 100, output, err.Error())
		return
	}
	a.reportCommand(cmd, "running", 10, output+"; waiting for vLLM health check", "")
	a.waitForVLLMReady(cmd, output)
}

func (a *Agent) executeDownloadModel(cmd *cluster.Command) {
	model := strings.TrimSpace(cmd.Params["model"])
	url := strings.TrimSpace(cmd.Params["download_url"])
	if url == "" {
		a.reportCommand(cmd, "failed", 100, "", "download_url required")
		return
	}
	if model == "" {
		a.reportCommand(cmd, "failed", 100, "", "model parameter required")
		return
	}
	log.Printf("[agent] downloading model %s from %s", model, url)

	a.reportCommand(cmd, "running", 0, "downloading "+model, "")

	// Keep node operations serial. Heartbeats run on their own loop, so there is
	// no reason to start a second lifecycle command while this download is still
	// verifying its files. In particular, a start command must not race the
	// atomic publication of the model directory.
	timeout := commandTimeout(cmd, time.Hour)
	ctx, cancel := context.WithTimeout(a.ctx, timeout)
	defer cancel()
	output, err := a.process.DownloadModel(ctx, model, url, cmd.Params["manifest_url"], a.config.WorkDir, func(progress int32, message string) {
		a.reportCommand(cmd, "running", progress, message, "")
	})
	if err != nil {
		a.reportCommand(cmd, "failed", 100, output, err.Error())
		return
	}
	a.reportCommand(cmd, "success", 100, output, "")
}

func (a *Agent) executeStopVLLM(cmd *cluster.Command) {
	output, err := a.process.Stop()
	if err != nil {
		a.reportCommand(cmd, "failed", 100, output, err.Error())
	} else {
		a.reportCommand(cmd, "success", 100, output, "")
	}
}

func (a *Agent) executeRestartVLLM(cmd *cluster.Command) {
	output, err := a.process.Stop()
	if err != nil {
		a.reportCommand(cmd, "failed", 100, output, "restart could not stop vLLM: "+err.Error())
		return
	}
	a.reportCommand(cmd, "running", 25, output+"; starting replacement", "")
	a.executeStartVLLM(cmd)
}

func (a *Agent) waitForVLLMReady(cmd *cluster.Command, startOutput string) {
	timeout := commandTimeout(cmd, 10*time.Minute)
	deadline := time.NewTimer(timeout)
	defer deadline.Stop()
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()

	for {
		if a.vllmHealthy() {
			a.process.MarkRunning()
			a.reportCommand(cmd, "success", 100, startOutput+"; vLLM health check passed", "")
			return
		}
		switch a.process.Status() {
		case "error", "stopped":
			a.reportCommand(cmd, "failed", 100, startOutput, "vLLM exited before becoming healthy")
			return
		}
		select {
		case <-a.ctx.Done():
			a.reportCommand(cmd, "failed", 100, startOutput, "agent stopped while waiting for vLLM readiness")
			return
		case <-deadline.C:
			stopOutput, _ := a.process.Stop()
			a.reportCommand(cmd, "failed", 100, startOutput+"; "+stopOutput, fmt.Sprintf("vLLM did not become healthy within %s", timeout))
			return
		case <-ticker.C:
			a.reportCommand(cmd, "running", 50, "vLLM process is starting; waiting for /health", "")
		}
	}
}

func commandTimeout(cmd *cluster.Command, fallback time.Duration) time.Duration {
	if cmd != nil && cmd.Timeout > 0 {
		return time.Duration(cmd.Timeout) * time.Second
	}
	return fallback
}

func (a *Agent) reportCommand(cmd *cluster.Command, status string, progress int32, output, resultErr string) {
	if cmd == nil {
		return
	}
	a.reportResult(&cluster.CmdResult{
		CmdID:    cmd.CmdID,
		NodeID:   a.config.NodeID,
		Action:   cmd.Action,
		Status:   status,
		Progress: progress,
		Output:   output,
		Error:    resultErr,
	})
}

func (a *Agent) commandState(cmdID string) *cluster.CmdResult {
	a.commandMu.Lock()
	defer a.commandMu.Unlock()
	result := a.commandStates[cmdID]
	if result == nil {
		return nil
	}
	copy := *result
	return &copy
}

func (a *Agent) reportResult(result *cluster.CmdResult) {
	if result == nil || strings.TrimSpace(result.CmdID) == "" {
		return
	}
	copy := *result
	if copy.NodeID == "" {
		copy.NodeID = a.config.NodeID
	}
	if copy.Timestamp == 0 {
		copy.Timestamp = time.Now().UnixNano()
	}
	a.commandMu.Lock()
	a.commandStates[copy.CmdID] = &copy
	a.unreported[copy.CmdID] = &copy
	a.commandMu.Unlock()

	log.Printf("[agent] cmd result: %s status=%s progress=%d", copy.CmdID, copy.Status, copy.Progress)
	if err := a.client.ReportResult(&copy); err != nil {
		log.Printf("[agent] report result error: %v", err)
	} else {
		a.acknowledgeResult(&copy)
	}
	if isTerminalResultStatus(copy.Status) {
		a.requestStatusRefresh()
	}
}

func (a *Agent) requestStatusRefresh() {
	select {
	case a.statusRefresh <- struct{}{}:
	default:
	}
}

func isTerminalResultStatus(status string) bool {
	switch status {
	case "success", "failed", "timeout":
		return true
	default:
		return false
	}
}

func (a *Agent) acknowledgeResult(result *cluster.CmdResult) {
	if result == nil {
		return
	}
	a.commandMu.Lock()
	defer a.commandMu.Unlock()
	if pending := a.unreported[result.CmdID]; pending != nil && pending.Timestamp == result.Timestamp {
		delete(a.unreported, result.CmdID)
	}
}

func (a *Agent) replayUnreported() {
	a.commandMu.Lock()
	pending := make([]*cluster.CmdResult, 0, len(a.unreported))
	for _, result := range a.unreported {
		copy := *result
		pending = append(pending, &copy)
	}
	a.commandMu.Unlock()
	for _, result := range pending {
		if err := a.client.ReportResult(result); err != nil {
			log.Printf("[agent] replay result %s failed: %v", result.CmdID, err)
			continue
		}
		a.acknowledgeResult(result)
	}
}

// ── Command polling loop ─────────────────────────────────────────────

func (a *Agent) commandLoop(cmdCh <-chan *cluster.Command) {
	for {
		select {
		case cmd, ok := <-cmdCh:
			if !ok {
				return
			}
			a.executeCommand(cmd)
		case <-a.ctx.Done():
			return
		}
	}
}

// vllmHealthy checks if the vLLM HTTP API is reachable.
func (a *Agent) vllmHealthy() bool {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(a.vllmLocalURL("/health"))
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == 200
}

func (a *Agent) vllmPort() int {
	if a.config.VLLMPort > 0 {
		return a.config.VLLMPort
	}
	return defaultVLLMPort
}

func (a *Agent) vllmLocalURL(path string) string {
	return fmt.Sprintf("http://127.0.0.1:%d%s", a.vllmPort(), path)
}

// vllmQueueLen reads the gauges exposed by current and older vLLM metric
// names. The gateway also tracks its own in-flight work between heartbeats.
func (a *Agent) vllmQueueLen() int32 {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(a.vllmLocalURL("/metrics"))
	if err != nil {
		return 0
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0
	}
	return parseVLLMQueueLen(string(body))
}

func parseVLLMQueueLen(metrics string) int32 {
	metricNames := map[string]struct{}{
		"vllm:num_requests_running": {},
		"vllm:num_requests_waiting": {},
		"vllm_num_requests_running": {},
		"vllm_num_requests_waiting": {},
	}
	var total int64
	for _, line := range strings.Split(metrics, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		name := strings.SplitN(fields[0], "{", 2)[0]
		if _, ok := metricNames[name]; !ok {
			continue
		}
		value, err := strconv.ParseFloat(fields[1], 64)
		if err != nil || value <= 0 {
			continue
		}
		count := int64(value)
		if value > float64(count) {
			count++
		}
		total += count
		if total >= int64(^uint32(0)>>1) {
			return int32(^uint32(0) >> 1)
		}
	}
	return int32(total)
}
