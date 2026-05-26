package cluster

import "context"

// ============================================================================
// rpcx Service Definitions
//
// Convention: each method has signature
//   Method(ctx context.Context, args *T, reply *R) error
// as required by rpcx.
//
// S端 implements these services; C端 calls them via rpcx client.
// For bidirectional push, C端 also implements AgentService and
// registers itself with the S端 rpcx server as a peer.
// ============================================================================

// ─── S端 Services (called by C端) ────────────────────────────────────────

// ClusterService handles node registration and periodic heartbeats.
// Every C端 agent MUST call Register once at startup, then Heartbeat periodically.
type ClusterService interface {
	// Register is called once by C端 at startup.
	// S端 records the node, assigns cluster config, returns a RegisterReply.
	Register(ctx context.Context, args *NodeInfo, reply *RegisterReply) error

	// Heartbeat is called periodically (every 5s) by C端.
	// S端 updates the node's status and returns any pending commands/updates
	// in the reply, piggybacking command dispatch to avoid extra RTT.
	Heartbeat(ctx context.Context, args *MachineStatus, reply *HeartbeatReply) error
}

// CommandService handles command pulling and result reporting.
// C端 fetches pending commands and reports execution results.
type CommandService interface {
	// FetchCommands returns all pending commands for a node.
	// Called by C端 after heartbeat indicates there are pending commands.
	FetchCommands(ctx context.Context, nodeID string, reply *[]*Command) error

	// ReportResult reports the result of a command execution.
	ReportResult(ctx context.Context, args *CmdResult, reply *OK) error
}

// QueryService provides cache lookup and cluster query APIs.
type QueryService interface {
	// CacheLookup looks up where a KV block is stored.
	// Returns the precise location (ip+disk_path for local_nvme, or shared_path).
	CacheLookup(ctx context.Context, hash uint64, reply *CacheLocation) error

	// ClusterStatus returns the aggregated cluster summary.
	ClusterStatus(ctx context.Context, _ *Empty, reply *ClusterSummary) error

	// NodeDetail returns full detail of a single node.
	NodeDetail(ctx context.Context, nodeID string, reply *NodeDetail) error
}

// AdminService provides management APIs.
// Called by the embedded REST API / CLI / Web UI.
type AdminService interface {
	// DispatchCommand dispatches a command to one or all C端 agents.
	// target="*" means broadcast to all nodes.
	DispatchCommand(ctx context.Context, args *DispatchReq, reply *OK) error

	// CommandHistory returns recent command results for display.
	CommandHistory(ctx context.Context, _ *Empty, reply *[]*CmdResult) error
}

// ─── C端 Services (called by S端 for push) ───────────────────────────────

// AgentService is implemented by C端 and called by S端 via rpcx bidirectional.
// This is the push path for urgent commands.
// C端 can also use the pull path (Heartbeat reply → FetchCommands) for normal commands.
type AgentService interface {
	// ExecCommand is called by S端 to push a command to C端.
	// The reply is the initial submission acknowledgement;
	// actual execution progress is reported via subsequent Heartbeat calls.
	ExecCommand(ctx context.Context, cmd *Command, reply *CmdResult) error

	// Ping is a liveness probe.
	Ping(ctx context.Context, _ *Empty, reply *OK) error
}
