package agent

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
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
}

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
		RPCPort:   9001,
		CacheMode: cluster.CacheModeLocalNVMe,
		VLLMHost:  defaultVLLMHost,
		VLLMPort:  defaultVLLMPort,
	}
}

// New creates a new C端 Agent.
func New(cfg *Config) *Agent {
	ctx, cancel := context.WithCancel(context.Background())
	return &Agent{
		config:    cfg,
		collector: NewCollector(cfg),
		process:   NewProcessManager(cfg),
		cache:     NewCacheProxy(cfg),
		client:    NewRPCClient(cfg.ServerAddr),
		ctx:       ctx,
		cancel:    cancel,
	}
}

// Start begins the agent main loop.
func (a *Agent) Start() error {
	log.Printf("[agent] starting C端 agent: node=%s server=%s mode=%s",
		a.config.NodeID, a.config.ServerAddr, a.config.CacheMode)

	// 1. Connect to S端
	if err := a.client.Connect(); err != nil {
		return err
	}
	defer a.client.Stop()

	// 2. Register with S端
	if err := a.register(); err != nil {
		return err
	}

	log.Printf("[agent] registered as %s", a.config.NodeID)

	// 3. Start command handler (pull-based)
	cmdCh := make(chan *cluster.Command, 16)
	go a.commandLoop(cmdCh)

	// 4. Main heartbeat loop
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if err := a.heartbeat(cmdCh); err != nil {
				log.Printf("[agent] heartbeat error: %v", err)
				// Reconnect on failure
				if reconnectErr := a.client.Reconnect(); reconnectErr != nil {
					log.Printf("[agent] reconnect failed: %v", reconnectErr)
				}
			}
		case cmd := <-cmdCh:
			a.executeCommand(cmd)
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
	info := &cluster.NodeInfo{
		NodeID:           a.config.NodeID,
		Hostname:         a.config.NodeID, // simplified; could use os.Hostname()
		IP:               getOutboundIP(),
		RPCPort:          a.config.RPCPort,
		VLLMPort:         a.vllmPort(),
		CacheMode:        a.config.CacheMode,
		SharedCacheID:    a.config.SharedCacheID,
		CacheMetadataURL: a.config.CachePath,
		GPUType:          a.config.GPUType,
		GPUMemMB:         a.config.GPUMemMB,
		GPUCount:         a.config.GPUCount,
		Disks:            a.config.Disks,
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
		status.CacheRetrieved = cacheStats.BlocksRetrieved
		status.CacheEvicted = cacheStats.BlocksEvicted
	}
	status.VLLMStatus = a.process.Status()
	// Available models
	status.AvailableModels = a.collector.GetAvailableModels()

	// vLLM health check: transition loading→running / running→error
	switch status.VLLMStatus {
	case "running":
		if !a.vllmHealthy() {
			status.VLLMStatus = "error"
		}
	case "loading":
		if a.vllmHealthy() {
			status.VLLMStatus = "running"
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

	// Process any pending commands from heartbeat reply
	if reply != nil && len(reply.PendingCmds) > 0 {
		for _, cmd := range reply.PendingCmds {
			log.Printf("[agent] received pending cmd: %s action=%s", cmd.CmdID, cmd.Action)
			cmdCh <- cmd
		}
	}

	return nil
}

// ── Command Execution ────────────────────────────────────────────────

func (a *Agent) executeCommand(cmd *cluster.Command) {
	log.Printf("[agent] executing cmd: %s action=%s", cmd.CmdID, cmd.Action)

	// Report initial "running" status
	a.reportResult(&cluster.CmdResult{
		CmdID:  cmd.CmdID,
		NodeID: a.config.NodeID,
		Status: "running",
	})

	switch cmd.Action {
	case cluster.CmdStartVLLM:
		a.executeStartVLLM(cmd)
	case cluster.CmdStopVLLM:
		a.executeStopVLLM(cmd)
	case cluster.CmdRestartVLLM:
		a.executeStopVLLM(cmd)
		a.executeStartVLLM(cmd)
	case cluster.CmdLoadModel:
		a.executeLoadModel(cmd)
	case cluster.CmdUnloadModel:
		a.executeUnloadModel(cmd)
	case cluster.CmdExecShell:
		a.executeShell(cmd)
	case cluster.CmdDownloadModel:
		a.executeDownloadModel(cmd)
	default:
		a.reportResult(&cluster.CmdResult{
			CmdID:  cmd.CmdID,
			NodeID: a.config.NodeID,
			Status: "failed",
			Error:  "unknown action: " + string(cmd.Action),
		})
	}
}

func (a *Agent) executeStartVLLM(cmd *cluster.Command) {
	// Guard: don't start if vLLM is already running
	if st := a.process.Status(); st == "running" || st == "loading" {
		log.Printf("[agent] vLLM already %s, ignoring start command", st)
		a.reportResult(&cluster.CmdResult{
			CmdID: cmd.CmdID, NodeID: a.config.NodeID,
			Status: "failed", Error: "vLLM already " + st,
		})
		return
	}

	// If raw_args is provided, use it directly as the full command line
	if raw := cmd.Params["raw_args"]; raw != "" {
		output, err := a.process.StartRaw(raw, a.config.WorkDir)
		result := &cluster.CmdResult{CmdID: cmd.CmdID, NodeID: a.config.NodeID}
		if err != nil {
			result.Status = "failed"
			result.Error = err.Error()
			result.Output = output
		} else {
			result.Status = "success"
			result.Output = output
		}
		a.reportResult(result)
		return
	}

	model := strings.TrimSpace(cmd.Params["model"])
	if model == "" {
		a.reportResult(&cluster.CmdResult{
			CmdID:  cmd.CmdID,
			NodeID: a.config.NodeID,
			Status: "failed",
			Error:  "model parameter required",
		})
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
	})
	result := &cluster.CmdResult{
		CmdID:  cmd.CmdID,
		NodeID: a.config.NodeID,
	}
	if err != nil {
		result.Status = "failed"
		result.Error = err.Error()
		result.Output = output
	} else {
		result.Status = "success"
		result.Output = output
	}
	a.reportResult(result)
}

func (a *Agent) executeDownloadModel(cmd *cluster.Command) {
	model := cmd.Params["model"]
	url := cmd.Params["download_url"]
	if url == "" {
		a.reportResult(&cluster.CmdResult{
			CmdID: cmd.CmdID, NodeID: a.config.NodeID,
			Status: "failed", Error: "download_url required",
		})
		return
	}
	log.Printf("[agent] downloading model %s from %s", model, url)

	// Acknowledge immediately, run download in background
	a.reportResult(&cluster.CmdResult{
		CmdID: cmd.CmdID, NodeID: a.config.NodeID,
		Status: "running", Output: "downloading " + model,
	})

	go func() {
		output, err := a.process.DownloadModel(model, url, a.config.WorkDir)
		result := &cluster.CmdResult{CmdID: cmd.CmdID, NodeID: a.config.NodeID}
		if err != nil {
			result.Status = "failed"
			result.Error = err.Error()
			result.Output = output
		} else {
			result.Status = "success"
			result.Output = output
		}
		a.reportResult(result)
	}()
}

func (a *Agent) executeStopVLLM(cmd *cluster.Command) {
	output, err := a.process.Stop()
	result := &cluster.CmdResult{
		CmdID:  cmd.CmdID,
		NodeID: a.config.NodeID,
	}
	if err != nil {
		result.Status = "failed"
		result.Error = err.Error()
		result.Output = output
	} else {
		result.Status = "success"
		result.Output = output
	}
	a.reportResult(result)
}

func (a *Agent) executeLoadModel(cmd *cluster.Command) {
	model := cmd.Params["model"]
	if model == "" {
		a.reportResult(&cluster.CmdResult{
			CmdID:  cmd.CmdID,
			NodeID: a.config.NodeID,
			Status: "failed",
			Error:  "model parameter required",
		})
		return
	}

	// Load model via vLLM (this is async in practice)
	output, err := a.process.LoadModel(model)
	result := &cluster.CmdResult{
		CmdID:  cmd.CmdID,
		NodeID: a.config.NodeID,
	}
	if err != nil {
		result.Status = "failed"
		result.Error = err.Error()
		result.Output = output
	} else {
		result.Status = "success"
		result.Output = output
	}
	a.reportResult(result)
}

func (a *Agent) executeUnloadModel(cmd *cluster.Command) {
	output, err := a.process.UnloadModel()
	result := &cluster.CmdResult{
		CmdID:  cmd.CmdID,
		NodeID: a.config.NodeID,
	}
	if err != nil {
		result.Status = "failed"
		result.Error = err.Error()
		result.Output = output
	} else {
		result.Status = "success"
		result.Output = output
	}
	a.reportResult(result)
}

func (a *Agent) executeShell(cmd *cluster.Command) {
	output, err := runShell(cmd.Params)
	result := &cluster.CmdResult{
		CmdID:  cmd.CmdID,
		NodeID: a.config.NodeID,
	}
	if err != nil {
		result.Status = "failed"
		result.Error = err.Error()
		result.Output = output
	} else {
		result.Status = "success"
		result.Output = output
	}
	a.reportResult(result)
}

func (a *Agent) reportResult(result *cluster.CmdResult) {
	log.Printf("[agent] cmd result: %s status=%s", result.CmdID, result.Status)
	if err := a.client.ReportResult(result); err != nil {
		log.Printf("[agent] report result error: %v", err)
	}
}

// ── Command polling loop ─────────────────────────────────────────────

func (a *Agent) commandLoop(cmdCh <-chan *cluster.Command) {
	// Process commands as they come
	for cmd := range cmdCh {
		a.executeCommand(cmd)
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
