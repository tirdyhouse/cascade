package server

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
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

	lastSeen  time.Time
	commands  []*cluster.Command // queued or in-flight commands for this node
	delivered map[string]bool    // command IDs already handed to this Agent
}

// Registry manages all registered C端 nodes.
// Thread-safe; all methods are safe for concurrent use.
type Registry struct {
	mu    sync.RWMutex
	nodes map[string]*NodeState

	seq      int64                // monotonic command ID counter
	history  []*cluster.CmdResult // recent command results (ring buffer)
	commands map[string]*cluster.Command

	// deferred holds durable commands for nodes which have not re-registered
	// after a control-plane restart.
	deferred  map[string][]persistedPending
	statePath string
}

type persistedPending struct {
	Command   *cluster.Command `json:"command"`
	Delivered bool             `json:"delivered"`
}

type persistedRegistry struct {
	Seq     int64                         `json:"seq"`
	History []*cluster.CmdResult          `json:"history"`
	Pending map[string][]persistedPending `json:"pending"`
	SavedAt int64                         `json:"saved_at"`
}

// NewRegistry creates a new Registry.
func NewRegistry() *Registry {
	return &Registry{
		nodes:    make(map[string]*NodeState),
		history:  make([]*cluster.CmdResult, 0, 1000),
		commands: make(map[string]*cluster.Command),
		deferred: make(map[string][]persistedPending),
	}
}

// NewPersistentRegistry restores command history and unfinished work from disk.
// Node membership remains heartbeat-driven, so no stale node is ever treated
// as online simply because it appeared in an older snapshot.
func NewPersistentRegistry(stateDir string) (*Registry, error) {
	registry := NewRegistry()
	stateDir = strings.TrimSpace(stateDir)
	if stateDir == "" {
		return registry, nil
	}
	if err := os.MkdirAll(stateDir, 0750); err != nil {
		return nil, fmt.Errorf("create control-plane state directory: %w", err)
	}
	registry.statePath = filepath.Join(stateDir, "command-state.json")
	if err := registry.loadState(); err != nil {
		return nil, err
	}
	return registry, nil
}

func (r *Registry) loadState() error {
	if r.statePath == "" {
		return nil
	}
	data, err := os.ReadFile(r.statePath)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("read command state: %w", err)
	}
	var snapshot persistedRegistry
	if err := json.Unmarshal(data, &snapshot); err != nil {
		return fmt.Errorf("decode command state: %w", err)
	}
	r.seq = snapshot.Seq
	if len(snapshot.History) > 1000 {
		snapshot.History = snapshot.History[len(snapshot.History)-1000:]
	}
	r.history = copyResults(snapshot.History)
	for nodeID, pending := range snapshot.Pending {
		for _, entry := range pending {
			if entry.Command == nil || entry.Command.CmdID == "" {
				continue
			}
			copy := cloneCommand(entry.Command)
			// A delivery bit only prevents duplicate heartbeat replies during one
			// server lifetime. After a restart, the response that carried a
			// command may have been lost before the Agent received it. Re-deliver
			// unfinished work; Agents deduplicate by command ID and re-report their
			// latest state instead of executing a second operation.
			r.deferred[nodeID] = append(r.deferred[nodeID], persistedPending{Command: copy, Delivered: false})
			r.commands[copy.CmdID] = copy
		}
	}
	if len(r.history) > 0 || len(r.deferred) > 0 {
		log.Printf("[registry] restored %d command events and pending work for %d node(s)", len(r.history), len(r.deferred))
	}
	return nil
}

// Register adds or updates a node. Returns the assigned node ID.
func (r *Registry) Register(info *cluster.NodeInfo) *cluster.RegisterReply {
	r.mu.Lock()
	defer r.mu.Unlock()

	if info == nil {
		return &cluster.RegisterReply{
			Accepted: false,
			Reason:   "missing node information",
		}
	}
	if info.CacheMode == cluster.CacheModeSharedPool {
		info.SharedCacheID = strings.TrimSpace(info.SharedCacheID)
		if info.SharedCacheID == "" {
			return &cluster.RegisterReply{
				NodeID:      info.NodeID,
				ClusterSize: len(r.nodes),
				CacheMode:   info.CacheMode,
				Accepted:    false,
				Reason:      "shared_pool nodes must provide shared_cache_id",
			}
		}
		for existingID, existing := range r.nodes {
			if existingID == info.NodeID || existing.Info == nil ||
				existing.Info.CacheMode != cluster.CacheModeSharedPool {
				continue
			}
			if existing.Info.SharedCacheID != info.SharedCacheID {
				return &cluster.RegisterReply{
					NodeID:      info.NodeID,
					ClusterSize: len(r.nodes),
					CacheMode:   info.CacheMode,
					Accepted:    false,
					Reason:      "shared_cache_id differs from registered shared_pool nodes",
				}
			}
		}
	}

	state, exists := r.nodes[info.NodeID]
	if !exists {
		state = &NodeState{delivered: make(map[string]bool)}
		r.nodes[info.NodeID] = state
		if pending := r.deferred[info.NodeID]; len(pending) > 0 {
			for _, entry := range pending {
				if entry.Command == nil {
					continue
				}
				state.commands = append(state.commands, cloneCommand(entry.Command))
				state.delivered[entry.Command.CmdID] = entry.Delivered
			}
			delete(r.deferred, info.NodeID)
		}
		log.Printf("[registry] node registered: %s (%s) mode=%s gpu=%s disks=%d",
			info.NodeID, info.IP, info.CacheMode, info.GPUType, len(info.Disks))
	}
	if state.delivered == nil {
		state.delivered = make(map[string]bool)
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

	// Hand every queued command to an Agent once during this server lifetime.
	// The durable delivery bit prevents duplicate work on the next heartbeat;
	// it is intentionally reset on a server restart (see loadState).
	valid, changed := r.pendingForDeliveryLocked(state)
	if changed {
		r.persistLocked()
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

// ValidateOperation rejects commands that cannot be performed from the
// Agent's latest reported lifecycle state. Queueing a Stop against a stopped
// process used to create a misleading "queued" event followed by a predictable
// failure; the dispatcher now returns a useful synchronous error instead.
func (r *Registry) ValidateOperation(action cluster.CommandAction, target string) error {
	r.mu.RLock()
	defer r.mu.RUnlock()

	targets := make([]string, 0, len(r.nodes))
	if target == "*" {
		for nodeID := range r.nodes {
			targets = append(targets, nodeID)
		}
		sort.Strings(targets)
		if len(targets) == 0 {
			return fmt.Errorf("no registered nodes available for broadcast")
		}
	} else {
		if _, exists := r.nodes[target]; !exists {
			return fmt.Errorf("node %q not found", target)
		}
		targets = append(targets, target)
	}

	for _, nodeID := range targets {
		state := r.nodes[nodeID]
		if state == nil || state.Info == nil || state.lastSeen.IsZero() || time.Since(state.lastSeen) > HeartbeatTimeout {
			return fmt.Errorf("node %q is offline", nodeID)
		}
		if active := activeNodeCommand(state, time.Now()); active != nil {
			return fmt.Errorf("node %q already has %s command %q in progress", nodeID, active.Action, active.CmdID)
		}
		vllmStatus := "stopped"
		if state.Status != nil && strings.TrimSpace(state.Status.VLLMStatus) != "" {
			vllmStatus = strings.ToLower(strings.TrimSpace(state.Status.VLLMStatus))
		}
		switch action {
		case cluster.CmdStartVLLM:
			if vllmStatus != "stopped" && vllmStatus != "error" {
				return fmt.Errorf("node %q cannot start vLLM while it is %s", nodeID, vllmStatus)
			}
			if vllmStatus == "error" && state.Status != nil && strings.TrimSpace(state.Status.ModelName) != "" {
				return fmt.Errorf("node %q has an unhealthy managed vLLM process; use restart or stop", nodeID)
			}
		case cluster.CmdRestartVLLM:
			if vllmStatus != "running" && vllmStatus != "loading" && vllmStatus != "error" {
				return fmt.Errorf("node %q cannot restart vLLM while it is %s; use start instead", nodeID, vllmStatus)
			}
			if vllmStatus == "error" && (state.Status == nil || strings.TrimSpace(state.Status.ModelName) == "") {
				return fmt.Errorf("node %q has no managed vLLM process to restart; use start instead", nodeID)
			}
		case cluster.CmdStopVLLM:
			if vllmStatus != "running" && vllmStatus != "loading" && vllmStatus != "error" {
				return fmt.Errorf("node %q cannot stop vLLM while it is %s", nodeID, vllmStatus)
			}
			if vllmStatus == "error" && (state.Status == nil || strings.TrimSpace(state.Status.ModelName) == "") {
				return fmt.Errorf("node %q has no managed vLLM process to stop; use start instead", nodeID)
			}
		}
	}
	return nil
}

func activeNodeCommand(state *NodeState, now time.Time) *cluster.Command {
	if state == nil {
		return nil
	}
	for _, command := range state.commands {
		if command == nil || command.CmdID == "" || commandExpired(command, now) {
			continue
		}
		return command
	}
	return nil
}

// ValidateModelReady ensures an operator starts only a model that the selected
// Agent has verified and published in a fresh heartbeat. This catches a failed
// or partial distribution before a vLLM process is launched.
func (r *Registry) ValidateModelReady(target, model string) error {
	r.mu.RLock()
	defer r.mu.RUnlock()

	targets := make([]string, 0, len(r.nodes))
	if target == "*" {
		for nodeID := range r.nodes {
			targets = append(targets, nodeID)
		}
		sort.Strings(targets)
	} else {
		targets = append(targets, target)
	}
	for _, nodeID := range targets {
		state := r.nodes[nodeID]
		ready := false
		if state != nil && state.Status != nil {
			for _, local := range state.Status.AvailableModels {
				if local.Name == model && local.Status == "ready" {
					ready = true
					break
				}
			}
		}
		if !ready {
			return fmt.Errorf("model %q is not verified and ready on node %q", model, nodeID)
		}
	}
	return nil
}

// EnqueueCommand adds a command to a node's pending queue and records a real
// queued event for every affected node. It returns an error rather than
// accepting a command for a nonexistent target.
func (r *Registry) EnqueueCommand(cmd *cluster.Command) (int, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if cmd == nil {
		return 0, fmt.Errorf("command is required")
	}
	if cmd.CmdID == "" {
		return 0, fmt.Errorf("command ID is required")
	}

	targets := make([]string, 0, len(r.nodes))
	if cmd.Target == "*" {
		if len(r.nodes) == 0 {
			return 0, fmt.Errorf("no registered nodes available for broadcast")
		}
		for nodeID, state := range r.nodes {
			state.commands = append(state.commands, cloneCommand(cmd))
			if state.delivered == nil {
				state.delivered = make(map[string]bool)
			}
			state.delivered[cmd.CmdID] = false
			targets = append(targets, nodeID)
		}
		log.Printf("[registry] broadcast command %s to %d nodes", cmd.CmdID, len(targets))
	} else {
		state, ok := r.nodes[cmd.Target]
		if !ok {
			return 0, fmt.Errorf("node %q not found", cmd.Target)
		}
		state.commands = append(state.commands, cloneCommand(cmd))
		if state.delivered == nil {
			state.delivered = make(map[string]bool)
		}
		state.delivered[cmd.CmdID] = false
		targets = append(targets, cmd.Target)
		log.Printf("[registry] enqueued command %s for node %s", cmd.CmdID, cmd.Target)
	}
	r.commands[cmd.CmdID] = cmd
	queuedAt := time.Now().UnixNano()
	for _, nodeID := range targets {
		r.appendHistoryLocked(&cluster.CmdResult{
			CmdID:     cmd.CmdID,
			NodeID:    nodeID,
			Action:    cmd.Action,
			CreatedAt: cmd.CreatedAt,
			Status:    "queued",
			Timestamp: queuedAt,
		})
	}
	r.persistLocked()
	return len(targets), nil
}

// RecordResult stores a command result and appends to history.
func (r *Registry) RecordResult(result *cluster.CmdResult) {
	if result == nil {
		return
	}
	resultCopy := *result
	result = &resultCopy
	r.mu.Lock()
	defer r.mu.Unlock()

	if result.Action == "" {
		if cmd, ok := r.commands[result.CmdID]; ok {
			result.Action = cmd.Action
			if result.CreatedAt == 0 {
				result.CreatedAt = cmd.CreatedAt
			}
		}
	}
	if result.CreatedAt == 0 {
		if cmd, ok := r.commands[result.CmdID]; ok {
			result.CreatedAt = cmd.CreatedAt
		}
	}
	if result.Timestamp == 0 {
		result.Timestamp = time.Now().UnixNano()
	}
	if r.hasRecordedResultLocked(result) {
		if isTerminalCommandStatus(result.Status) {
			r.removeNodeCommandLocked(result.NodeID, result.CmdID)
			r.persistLocked()
		}
		return
	}
	log.Printf("[registry] command result: %s node=%s status=%s", result.CmdID, result.NodeID, result.Status)
	r.appendHistoryLocked(result)
	if isTerminalCommandStatus(result.Status) {
		r.removeNodeCommandLocked(result.NodeID, result.CmdID)
	}
	r.persistLocked()
}

// hasRecordedResultLocked detects a replay after an RPC response was lost.
// Agent results retain their timestamp, so this is intentionally narrower than
// comparing only command status: a late outcome after a server-side timeout is
// still useful evidence and remains visible to the operator.
func (r *Registry) hasRecordedResultLocked(result *cluster.CmdResult) bool {
	if result == nil || result.CmdID == "" || result.NodeID == "" || result.Timestamp == 0 {
		return false
	}
	for index := len(r.history) - 1; index >= 0; index-- {
		previous := r.history[index]
		if previous == nil {
			continue
		}
		if previous.CmdID == result.CmdID && previous.NodeID == result.NodeID &&
			previous.Status == result.Status && previous.Timestamp == result.Timestamp {
			return true
		}
	}
	return false
}

func (r *Registry) appendHistoryLocked(result *cluster.CmdResult) {
	copy := *result
	if copy.Status == "running" {
		for index := len(r.history) - 1; index >= 0; index-- {
			previous := r.history[index]
			if previous.CmdID == copy.CmdID && previous.NodeID == copy.NodeID && previous.Status == "running" {
				r.history[index] = &copy
				return
			}
			if previous.CmdID == copy.CmdID && previous.NodeID == copy.NodeID && isTerminalCommandStatus(previous.Status) {
				break
			}
		}
	}
	r.history = append(r.history, &copy)
	if len(r.history) > 1000 {
		r.history = r.history[len(r.history)-1000:]
	}
}

// CommandHistory returns recent command results.
func (r *Registry) CommandHistory() []*cluster.CmdResult {
	r.mu.RLock()
	defer r.mu.RUnlock()
	result := make([]*cluster.CmdResult, len(r.history))
	for i, entry := range r.history {
		copy := *entry
		result[i] = &copy
	}
	return result
}

// FetchCommands returns work not yet delivered during this server lifetime.
// Terminal results remove it from the durable queue; a control-plane restart
// resets the delivery bit so an Agent can safely deduplicate a re-delivery.
func (r *Registry) FetchCommands(nodeID string) []*cluster.Command {
	r.mu.Lock()
	defer r.mu.Unlock()
	state, ok := r.nodes[nodeID]
	if !ok || len(state.commands) == 0 {
		return nil
	}
	pending, changed := r.pendingForDeliveryLocked(state)
	if changed {
		r.persistLocked()
	}
	return pending
}

func (r *Registry) pendingForDeliveryLocked(state *NodeState) ([]*cluster.Command, bool) {
	if state == nil {
		return nil, false
	}
	if state.delivered == nil {
		state.delivered = make(map[string]bool)
	}
	pending := make([]*cluster.Command, 0, len(state.commands))
	changed := r.expireCommandsLocked(state, time.Now())
	for _, command := range state.commands {
		if command == nil || command.CmdID == "" {
			continue
		}
		if !state.delivered[command.CmdID] {
			state.delivered[command.CmdID] = true
			pending = append(pending, cloneCommand(command))
			changed = true
		}
	}
	return pending, changed
}

func (r *Registry) expireCommandsLocked(state *NodeState, now time.Time) bool {
	if state == nil || len(state.commands) == 0 {
		return false
	}
	nodeID := ""
	if state.Info != nil {
		nodeID = state.Info.NodeID
	}
	kept := state.commands[:0]
	changed := false
	for _, command := range state.commands {
		if command == nil || command.CmdID == "" {
			changed = true
			continue
		}
		if !commandExpired(command, now) {
			kept = append(kept, command)
			continue
		}
		log.Printf("[registry] command %s expired (timeout=%ds)", command.CmdID, command.Timeout)
		r.appendHistoryLocked(&cluster.CmdResult{
			CmdID:     command.CmdID,
			NodeID:    nodeID,
			Action:    command.Action,
			CreatedAt: command.CreatedAt,
			Status:    "timeout",
			Error:     "command timed out before completion",
			Timestamp: now.UnixNano(),
		})
		delete(state.delivered, command.CmdID)
		changed = true
	}
	state.commands = kept
	return changed
}

func (r *Registry) removeNodeCommandLocked(nodeID, cmdID string) {
	state := r.nodes[nodeID]
	if state == nil {
		return
	}
	kept := state.commands[:0]
	for _, command := range state.commands {
		if command != nil && command.CmdID == cmdID {
			continue
		}
		kept = append(kept, command)
	}
	state.commands = kept
	delete(state.delivered, cmdID)
}

func commandExpired(command *cluster.Command, now time.Time) bool {
	if command == nil || command.CreatedAt <= 0 || command.Timeout <= 0 {
		return false
	}
	return now.Sub(time.Unix(0, command.CreatedAt)) > time.Duration(command.Timeout)*time.Second
}

func isTerminalCommandStatus(status string) bool {
	switch status {
	case "success", "failed", "timeout":
		return true
	default:
		return false
	}
}

func (r *Registry) persistLocked() {
	if r.statePath == "" {
		return
	}
	snapshot := persistedRegistry{
		Seq:     r.seq,
		History: copyResults(r.history),
		Pending: make(map[string][]persistedPending),
		SavedAt: time.Now().UnixNano(),
	}
	for nodeID, state := range r.nodes {
		if state == nil || len(state.commands) == 0 {
			continue
		}
		entries := make([]persistedPending, 0, len(state.commands))
		for _, command := range state.commands {
			if command == nil || command.CmdID == "" {
				continue
			}
			entries = append(entries, persistedPending{Command: cloneCommand(command), Delivered: state.delivered[command.CmdID]})
		}
		if len(entries) > 0 {
			snapshot.Pending[nodeID] = entries
		}
	}
	for nodeID, entries := range r.deferred {
		if _, registered := snapshot.Pending[nodeID]; registered {
			continue
		}
		copied := make([]persistedPending, 0, len(entries))
		for _, entry := range entries {
			if entry.Command == nil || entry.Command.CmdID == "" {
				continue
			}
			copied = append(copied, persistedPending{Command: cloneCommand(entry.Command), Delivered: entry.Delivered})
		}
		if len(copied) > 0 {
			snapshot.Pending[nodeID] = copied
		}
	}

	data, err := json.Marshal(snapshot)
	if err != nil {
		log.Printf("[registry] encode command state: %v", err)
		return
	}
	temporary := r.statePath + ".tmp"
	if err := os.WriteFile(temporary, data, 0600); err != nil {
		log.Printf("[registry] write command state: %v", err)
		return
	}
	if err := os.Rename(temporary, r.statePath); err != nil {
		log.Printf("[registry] publish command state: %v", err)
	}
}

func cloneCommand(command *cluster.Command) *cluster.Command {
	if command == nil {
		return nil
	}
	copy := *command
	if command.Params != nil {
		copy.Params = make(map[string]string, len(command.Params))
		for key, value := range command.Params {
			copy.Params[key] = value
		}
	}
	return &copy
}

func copyResults(results []*cluster.CmdResult) []*cluster.CmdResult {
	copy := make([]*cluster.CmdResult, 0, len(results))
	for _, result := range results {
		if result == nil {
			continue
		}
		entry := *result
		copy = append(copy, &entry)
	}
	return copy
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
	seenSharedCacheStats := make(map[string]struct{})

	for _, state := range r.nodes {
		ns := cluster.NodeSummary{
			NodeID:        state.Info.NodeID,
			IP:            state.Info.IP,
			VLLMPort:      state.Info.VLLMPort,
			CacheMode:     state.Info.CacheMode,
			SharedCacheID: state.Info.SharedCacheID,
			Status:        cluster.NodeOnline,
			GPUUtil:       0,
			CacheBlocks:   0,
			HitRate:       0,
			LastSeen:      state.lastSeen.UnixNano(),
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
			ns.CacheRetrieved = state.Status.CacheRetrieved
			ns.CacheEvicted = state.Status.CacheEvicted
			ns.Disks = state.Status.Disks
			ns.AvailableModels = state.Status.AvailableModels
			includeCacheStats := true
			if state.Info.CacheMode == cluster.CacheModeSharedPool &&
				state.Info.SharedCacheID != "" {
				if _, seen := seenSharedCacheStats[state.Info.SharedCacheID]; seen {
					includeCacheStats = false
				} else {
					seenSharedCacheStats[state.Info.SharedCacheID] = struct{}{}
				}
			}
			if includeCacheStats {
				totalBlocks += state.Status.CacheBlocks
				totalBytes += state.Status.CacheBytes
				if state.Status.CacheHitRate > 0 {
					hitRateSum += state.Status.CacheHitRate
					hitRateCount++
				}
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

// gatewayCandidates returns immutable snapshots of nodes which can currently
// serve a particular model. The scheduler owns its own transient in-flight
// state, so the registry only exposes agent-reported facts here.
func (r *Registry) gatewayCandidates(model string, now time.Time) []gatewayCandidate {
	r.mu.RLock()
	defer r.mu.RUnlock()

	candidates := make([]gatewayCandidate, 0, len(r.nodes))
	for _, state := range r.nodes {
		if state == nil || state.Info == nil || state.Status == nil || state.lastSeen.IsZero() {
			continue
		}
		if now.Sub(state.lastSeen) > HeartbeatTimeout ||
			!strings.EqualFold(strings.TrimSpace(state.Status.VLLMStatus), "running") ||
			!modelsEquivalent(model, state.Status.ModelName) {
			continue
		}

		ip := strings.TrimSpace(state.Info.IP)
		if ip == "" {
			continue
		}
		port := state.Info.VLLMPort
		if port == 0 {
			port = defaultVLLMPort
		}
		if port < 1 || port > 65535 {
			continue
		}
		candidates = append(candidates, gatewayCandidate{
			NodeID:       state.Info.NodeID,
			IP:           ip,
			VLLMPort:     port,
			ModelName:    state.Status.ModelName,
			GPUUtil:      state.Status.GPUUtil,
			QueueLen:     state.Status.QueueLen,
			CacheHitRate: state.Status.CacheHitRate,
			CacheMode:    state.Info.CacheMode,
		})
	}
	return candidates
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
			changed := false
			for id, state := range r.nodes {
				if time.Since(state.lastSeen) > HeartbeatTimeout {
					// Don't delete — keep offline marker for visibility
					log.Printf("[registry] node %s heartbeat timeout — marking offline", id)
				}
				if r.expireCommandsLocked(state, time.Now()) {
					changed = true
				}
			}
			if changed {
				r.persistLocked()
			}
			r.mu.Unlock()

		case <-ctx.Done():
			return
		}
	}
}
