package server

import (
	"context"
	"log"
	"sync"
	"time"

	"predict/engine/pkg/cluster"
)

const (
	// HeartbeatTimeout is how long without heartbeat before marking a node offline.
	HeartbeatTimeout = 15 * time.Second
)

// NodeState holds the full in-memory state of a connected C端 agent.
type NodeState struct {
	Info   *cluster.NodeInfo
	Status *cluster.MachineStatus // latest heartbeat

	lastSeen time.Time
	commands []*cluster.Command // pending commands
}

// Registry manages all registered C端 nodes.
// Thread-safe; all methods are safe for concurrent use.
type Registry struct {
	mu    sync.RWMutex
	nodes map[string]*NodeState

	seq     int64 // monotonic command ID counter
	history []*cluster.CmdResult // recent command results (ring buffer)
}

// NewRegistry creates a new Registry.
func NewRegistry() *Registry {
	return &Registry{
		nodes:   make(map[string]*NodeState),
		history: make([]*cluster.CmdResult, 0, 1000),
	}
}

// Register adds or updates a node. Returns the assigned node ID.
func (r *Registry) Register(info *cluster.NodeInfo) *cluster.RegisterReply {
	r.mu.Lock()
	defer r.mu.Unlock()

	state, exists := r.nodes[info.NodeID]
	if !exists {
		state = &NodeState{}
		r.nodes[info.NodeID] = state
		log.Printf("[registry] node registered: %s (%s) mode=%s gpu=%s disks=%d",
			info.NodeID, info.IP, info.CacheMode, info.GPUType, len(info.Disks))
	}

	state.Info = info
	state.lastSeen = time.Now()

	// If status was nil, init with baseline
	if state.Status == nil {
		state.Status = cluster.NewMachineStatus(info.NodeID, 0)
		state.Status.Disks = make([]cluster.DiskUsage, len(info.Disks))
		for i, d := range info.Disks {
			state.Status.Disks[i] = cluster.DiskUsage{
				Path:   d.Path,
				FreeGB: d.FreeGB,
				UsedGB: d.UsedGB,
			}
		}
	}

	return &cluster.RegisterReply{
		NodeID:      info.NodeID,
		ClusterSize: len(r.nodes),
		CacheMode:   info.CacheMode,
		Accepted:    true,
	}
}

// Heartbeat updates node status, returns pending commands.
func (r *Registry) Heartbeat(status *cluster.MachineStatus) *cluster.HeartbeatReply {
	r.mu.Lock()
	defer r.mu.Unlock()

	state, ok := r.nodes[status.NodeID]
	if !ok {
		return &cluster.HeartbeatReply{OK: false}
	}

	// Update status
	state.Status = status
	state.lastSeen = time.Now()

	// Collect pending commands
	pending := state.commands
	state.commands = nil

	// Filter out expired commands
	var valid []*cluster.Command
	for _, cmd := range pending {
		if cmd.CreatedAt > 0 && time.Since(time.Unix(0, cmd.CreatedAt)) > time.Duration(cmd.Timeout)*time.Second {
			log.Printf("[registry] command %s expired (timeout=%ds)", cmd.CmdID, cmd.Timeout)
			continue
		}
		valid = append(valid, cmd)
	}

	return &cluster.HeartbeatReply{
		OK:          true,
		PendingCmds: valid,
	}
}

// GetNode returns a node's state.
func (r *Registry) GetNode(nodeID string) *NodeState {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.nodes[nodeID]
}

// EnqueueCommand adds a command to a node's pending queue.
func (r *Registry) EnqueueCommand(cmd *cluster.Command) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if cmd.Target == "*" {
		// Broadcast to all nodes
		for _, state := range r.nodes {
			state.commands = append(state.commands, cmd)
		}
		log.Printf("[registry] broadcast command %s to %d nodes", cmd.CmdID, len(r.nodes))
	} else {
		state, ok := r.nodes[cmd.Target]
		if !ok {
			log.Printf("[registry] cannot enqueue command %s: node %s not found", cmd.CmdID, cmd.Target)
			return
		}
		state.commands = append(state.commands, cmd)
		log.Printf("[registry] enqueued command %s for node %s", cmd.CmdID, cmd.Target)
	}
}

// RecordResult stores a command result and appends to history.
func (r *Registry) RecordResult(result *cluster.CmdResult) {
	r.mu.Lock()
	defer r.mu.Unlock()

	log.Printf("[registry] command result: %s node=%s status=%s", result.CmdID, result.NodeID, result.Status)

	// Ring buffer: keep last 1000
	r.history = append(r.history, result)
	if len(r.history) > 1000 {
		r.history = r.history[len(r.history)-1000:]
	}
}

// CommandHistory returns recent command results.
func (r *Registry) CommandHistory() []*cluster.CmdResult {
	r.mu.RLock()
	defer r.mu.RUnlock()
	result := make([]*cluster.CmdResult, len(r.history))
	copy(result, r.history)
	return result
}

// NextSeq generates a unique command ID.
func (r *Registry) NextSeq() int64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.seq++
	return r.seq
}

// Summary builds a ClusterSummary from current state.
func (r *Registry) Summary() *cluster.ClusterSummary {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var summary cluster.ClusterSummary
	summary.Nodes = make([]cluster.NodeSummary, 0)
	var cacheMode cluster.CacheMode
	var totalBlocks, totalBytes int64
	var hitRateSum float64
	var hitRateCount int

	for _, state := range r.nodes {
		ns := cluster.NodeSummary{
			NodeID:      state.Info.NodeID,
			IP:          state.Info.IP,
			Status:      cluster.NodeOnline,
			GPUUtil:     0,
			CacheBlocks: 0,
			HitRate:     0,
			LastSeen:    state.lastSeen.UnixNano(),
		}

		if time.Since(state.lastSeen) > HeartbeatTimeout {
			ns.Status = cluster.NodeOffline
		} else {
			summary.OnlineNodes++
		}

		if state.Status != nil {
			ns.GPUUtil = state.Status.GPUUtil
			ns.GPUMemUsed = state.Status.GPUMemUsedMB
			ns.ModelName = state.Status.ModelName
			ns.VLLMStatus = state.Status.VLLMStatus
			ns.QueueLen = state.Status.QueueLen
			ns.LoadingPct = state.Status.LoadingPct
			ns.CacheBlocks = state.Status.CacheBlocks
			ns.HitRate = state.Status.CacheHitRate
			ns.Disks = state.Status.Disks

			totalBlocks += state.Status.CacheBlocks
			totalBytes += state.Status.CacheBytes
			if state.Status.CacheHitRate > 0 {
				hitRateSum += state.Status.CacheHitRate
				hitRateCount++
			}
		}

		if state.Info != nil && cacheMode == "" {
			cacheMode = state.Info.CacheMode
		}

		summary.Nodes = append(summary.Nodes, ns)
	}

	summary.CacheMode = cacheMode
	summary.TotalNodes = len(r.nodes)
	summary.TotalBlocks = totalBlocks
	summary.TotalBytes = totalBytes
	if hitRateCount > 0 {
		summary.HitRate = hitRateSum / float64(hitRateCount)
	}

	return &summary
}

// NodeDetail returns full detail for a single node.
func (r *Registry) NodeDetail(nodeID string) *cluster.NodeDetail {
	r.mu.RLock()
	defer r.mu.RUnlock()

	state, ok := r.nodes[nodeID]
	if !ok {
		return nil
	}

	// Collect recent command results for this node
	var recent []*cluster.CmdResult
	for _, h := range r.history {
		if h.NodeID == nodeID {
			recent = append(recent, h)
		}
	}

	return &cluster.NodeDetail{
		Info:   state.Info,
		Status: state.Status,
		Recent: recent,
	}
}

// OnlineCount returns the number of nodes that have heartbeated recently.
func (r *Registry) OnlineCount() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	count := 0
	for _, state := range r.nodes {
		if time.Since(state.lastSeen) <= HeartbeatTimeout {
			count++
		}
	}
	return count
}

// GC marks nodes as offline if they have timed out, and cleans up stale state.
// Call this periodically (every 30s).
func (r *Registry) GC(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			r.mu.Lock()
			for id, state := range r.nodes {
				if time.Since(state.lastSeen) > HeartbeatTimeout {
					// Don't delete — keep offline marker for visibility
					log.Printf("[registry] node %s heartbeat timeout — marking offline", id)
				}
			}
			r.mu.Unlock()

		case <-ctx.Done():
			return
		}
	}
}
