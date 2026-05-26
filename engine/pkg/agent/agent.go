package agent

import (
	"context"
	"log"
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
	NodeID      string
	ServerAddr  string // S端 rpcx address
	RPCPort     int    // local rpcx port for bidirectional (optional)
	CacheMode   cluster.CacheMode
	CachePath   string // path to local disk-cache API

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
	defer a.client.Close()

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
		NodeID:    a.config.NodeID,
		Hostname:  a.config.NodeID, // simplified; could use os.Hostname()
		IP:        getOutboundIP(),
		RPCPort:   a.config.RPCPort,
		CacheMode: a.config.CacheMode,
		GPUType:   a.config.GPUType,
		GPUMemMB:  a.config.GPUMemMB,
		GPUCount:  a.config.GPUCount,
		Disks:     a.config.Disks,
	}
	reply, err := a.client.Register(info)
	if err != nil {
		return err
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
		status.CacheBlocks = cacheStats.BlocksStored
		status.CacheBytes = cacheStats.DiskUsedBytes
	}
	status.VLLMStatus = a.process.Status()
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
	model := cmd.Params["model"]
	gpuUtil := cmd.Params["gpu_memory_utilization"]
	if gpuUtil == "" {
		gpuUtil = "0.9"
	}

	output, err := a.process.Start(model, gpuUtil)
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
