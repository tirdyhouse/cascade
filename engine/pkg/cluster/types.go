// Package cluster provides shared data structures and rpcx service definitions
// for the C/S (Client/Server) architecture of the KV Cache cluster management system.
package cluster

import (
	"fmt"
	"time"
)

// ============================================================================
// Disk
// ============================================================================

// DiskInfo is the static information of a disk/NVMe device.
// Used in node registration to report what disks are available.
type DiskInfo struct {
	Path    string `json:"path"`     // "/mnt/nvme0"
	TotalGB int64  `json:"total_gb"` // total capacity in GB
	UsedGB  int64  `json:"used_gb"`  // used space in GB
	FreeGB  int64  `json:"free_gb"`  // free space in GB
}

// DiskUsage is the runtime usage snapshot of a disk.
// Included in periodic heartbeat.
type DiskUsage struct {
	Path   string `json:"path"`
	FreeGB int64  `json:"free_gb"`
	UsedGB int64  `json:"used_gb"`
}

// ============================================================================
// Cache Mode
// ============================================================================

type CacheMode string

const (
	CacheModeLocalNVMe CacheMode = "local_nvme"  // standalone: cache on local NVMe, route by ip+disk
	CacheModeSharedPool CacheMode = "shared_pool" // pooled: cache on shared storage, any node can serve
)

// ============================================================================
// Node
// ============================================================================

// NodeInfo is the static information a C端 agent reports when registering.
type NodeInfo struct {
	NodeID    string     `json:"node_id"`
	Hostname  string     `json:"hostname"`
	IP        string     `json:"ip"`         // management / data IP
	RPCPort   int        `json:"rpc_port"`   // port this agent listens on for rpcx bidirectional
	CacheMode CacheMode  `json:"cache_mode"` // "local_nvme" | "shared_pool"

	// Hardware
	GPUType   string     `json:"gpu_type"`   // "H100"
	GPUMemMB  int64      `json:"gpu_mem_mb"`
	GPUCount  int        `json:"gpu_count"`

	// Disks — critical for local_nvme mode (multi-disk awareness)
	Disks     []DiskInfo `json:"disks"`
}

// NodeStatus represents the current health of a node.
type NodeStatus string

const (
	NodeOnline  NodeStatus = "online"
	NodeOffline NodeStatus = "offline"
	NodeLoading NodeStatus = "loading"
	NodeError   NodeStatus = "error"
)

// MachineStatus is the runtime status snapshot sent periodically by C端.
type MachineStatus struct {
	NodeID    string `json:"node_id"`
	Timestamp int64  `json:"timestamp"`
	Seq       int64  `json:"seq"` // monotonically increasing sequence, for dedup

	// Machine metrics
	GPUUtil      float64 `json:"gpu_util"`       // 0.0-1.0
	GPUMemUsedMB int64   `json:"gpu_mem_used_mb"`
	MemUsedMB    int64   `json:"mem_used_mb"`
	CPULoad      float64 `json:"cpu_load"`

	// Per-disk usage
	Disks []DiskUsage `json:"disks"`

	// vLLM process
	VLLMStatus string `json:"vllm_status"` // "running"|"loading"|"stopped"|"error"
	ModelName  string `json:"model_name"`
	QueueLen   int32  `json:"queue_len"`
	LoadingPct int32  `json:"loading_pct"` // 0-100

	// KV Cache stats (from local disk-cache engine)
	CacheBlocks  int64   `json:"cache_blocks"`
	CacheBytes   int64   `json:"cache_bytes"`
	CacheHitRate float64 `json:"cache_hit_rate"`
}

func NewMachineStatus(nodeID string, seq int64) *MachineStatus {
	return &MachineStatus{
		NodeID:    nodeID,
		Timestamp: time.Now().UnixNano(),
		Seq:       seq,
	}
}

// ============================================================================
// Commands
// ============================================================================

type CommandAction string

const (
	CmdStartVLLM     CommandAction = "start_vllm"
	CmdStopVLLM      CommandAction = "stop_vllm"
	CmdRestartVLLM   CommandAction = "restart_vllm"
	CmdLoadModel     CommandAction = "load_model"
	CmdUnloadModel   CommandAction = "unload_model"
	CmdUpdateConfig  CommandAction = "update_config"
	CmdExecShell     CommandAction = "exec_shell"
)

// Command is an operation dispatched from S端 to C端.
type Command struct {
	CmdID     string            `json:"cmd_id"`
	Action    CommandAction     `json:"action"`
	Params    map[string]string `json:"params"`
	Target    string            `json:"target"`     // target node_id, "" = broadcast
	CreatedAt int64             `json:"created_at"`
	Timeout   int               `json:"timeout"`    // seconds
}

// CmdResult is the execution result reported back to S端.
type CmdResult struct {
	CmdID    string `json:"cmd_id"`
	NodeID   string `json:"node_id"`
	Status   string `json:"status"`   // "running"|"success"|"failed"|"timeout"
	Output   string `json:"output"`
	Error    string `json:"error"`
	Progress int32  `json:"progress"` // 0-100
}

// ============================================================================
// Cache Routing
// ============================================================================

// CacheLocation is the result of a cache lookup.
// Fields populated depend on cache mode.
type CacheLocation struct {
	Hash     uint64 `json:"hash"`
	Size     int64  `json:"size"`

	// Mode 1 (local_nvme): IP + exact disk path
	NodeID   string `json:"node_id,omitempty"`
	IP       string `json:"ip,omitempty"`
	DiskPath string `json:"disk_path,omitempty"` // "/mnt/nvme0"
	FilePath string `json:"file_path,omitempty"` // "kv/blk_a1b2"

	// Mode 2 (shared_pool)
	SharedPath string `json:"shared_path,omitempty"`
}

// ============================================================================
// Query / Summary
// ============================================================================

// NodeSummary is a compact view of a node for listing.
type NodeSummary struct {
	NodeID      string      `json:"node_id"`
	IP          string      `json:"ip"`
	Status      NodeStatus  `json:"status"`
	GPUUtil     float64     `json:"gpu_util"`
	GPUMemUsed  int64       `json:"gpu_mem_used"`
	ModelName   string      `json:"model_name"`
	CacheBlocks int64       `json:"cache_blocks"`
	HitRate     float64     `json:"hit_rate"`
	LastSeen    int64       `json:"last_seen"`
	Disks       []DiskUsage `json:"disks"`
}

// ClusterSummary is the aggregated cluster view.
type ClusterSummary struct {
	Nodes       []NodeSummary `json:"nodes"`
	CacheMode   CacheMode     `json:"cache_mode"`
	OnlineNodes int           `json:"online_nodes"`
	TotalNodes  int           `json:"total_nodes"`
	TotalBlocks int64         `json:"total_blocks"`
	TotalBytes  int64         `json:"total_bytes"`
	HitRate     float64       `json:"hit_rate"`
}

// NodeDetail is the full detail of a single node.
type NodeDetail struct {
	Info   *NodeInfo    `json:"info"`
	Status *MachineStatus `json:"status"`
	Recent []*CmdResult `json:"recent"`
}

// ============================================================================
// RPC Reply Types
// ============================================================================

type RegisterReply struct {
	NodeID      string    `json:"node_id"`
	ClusterSize int       `json:"cluster_size"`
	CacheMode   CacheMode `json:"cache_mode"`
	Accepted    bool      `json:"accepted"`
}

type HeartbeatReply struct {
	OK           bool       `json:"ok"`
	PendingCmds  []*Command `json:"pending_commands,omitempty"`
	RoleChange   string     `json:"role_change,omitempty"`
	ConfigUpdate string     `json:"config_update,omitempty"`
}

// OK is a generic success/error reply.
type OK struct {
	OK  bool   `json:"ok"`
	Err string `json:"err,omitempty"`
}

// Empty is a placeholder for methods that need no args/reply payload.
type Empty struct{}

// FormatHash formats a uint64 hash as a hex string.
func FormatHash(hash uint64) string {
	return fmt.Sprintf("%016x", hash)
}

// DispatchReq is the request payload for S端 AdminService.DispatchCommand.
type DispatchReq struct {
	Action  CommandAction        `json:"action"`
	Params  map[string]string    `json:"params"`
	Target  string               `json:"target"` // node_id or "*" for broadcast
	Timeout int                  `json:"timeout"`
}
