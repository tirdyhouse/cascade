package agent

import (
	"context"
	"log"
	"sync"
	"time"

	"predict/engine/pkg/cluster"

	"github.com/smallnest/rpcx/client"
)

// RPCClient manages the rpcx connection to S端.
type RPCClient struct {
	serverAddr string

	mu      sync.Mutex
	conn    client.XClient // ClusterService: register + heartbeat
	cmdConn client.XClient // CommandService: command polling + result reports
	ctx     context.Context
	cancel  context.CancelFunc
}

// NewRPCClient creates an RPCClient.
func NewRPCClient(serverAddr string) *RPCClient {
	ctx, cancel := context.WithCancel(context.Background())
	return &RPCClient{
		serverAddr: serverAddr,
		ctx:        ctx,
		cancel:     cancel,
	}
}

// Connect establishes the rpcx connection to S端.
func (c *RPCClient) Connect() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	clusterDiscovery, err := client.NewPeer2PeerDiscovery("tcp@"+c.serverAddr, "")
	if err != nil {
		return err
	}
	commandDiscovery, err := client.NewPeer2PeerDiscovery("tcp@"+c.serverAddr, "")
	if err != nil {
		return err
	}

	c.conn = client.NewXClient("ClusterService", client.Failtry, client.RandomSelect, clusterDiscovery, client.DefaultOption)
	c.cmdConn = client.NewXClient("CommandService", client.Failtry, client.RandomSelect, commandDiscovery, client.DefaultOption)
	log.Printf("[rpcx] connected to S端 at %s", c.serverAddr)
	return nil
}

// Close disconnects the rpcx client (keeps context alive for reconnection).
func (c *RPCClient) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.conn != nil {
		c.conn.Close()
		c.conn = nil
	}
	if c.cmdConn != nil {
		c.cmdConn.Close()
		c.cmdConn = nil
	}
}

// Stop permanently shuts down the client.
func (c *RPCClient) Stop() {
	c.Close()
	c.cancel()
}

// Reconnect closes the old connection and creates a new one.
func (c *RPCClient) Reconnect() error {
	c.Close()
	time.Sleep(1 * time.Second)
	return c.Connect()
}

// Register calls S端 ClusterService.Register.
func (c *RPCClient) Register(info *cluster.NodeInfo) (*cluster.RegisterReply, error) {
	c.mu.Lock()
	conn := c.conn
	c.mu.Unlock()

	if conn == nil {
		return nil, errNotConnected
	}

	var reply cluster.RegisterReply
	err := conn.Call(c.ctx, "Register", info, &reply)
	if err != nil {
		return nil, err
	}
	return &reply, nil
}

// Heartbeat calls S端 ClusterService.Heartbeat.
func (c *RPCClient) Heartbeat(status *cluster.MachineStatus) (*cluster.HeartbeatReply, error) {
	c.mu.Lock()
	conn := c.conn
	c.mu.Unlock()

	if conn == nil {
		return nil, errNotConnected
	}

	var reply cluster.HeartbeatReply
	err := conn.Call(c.ctx, "Heartbeat", status, &reply)
	if err != nil {
		return nil, err
	}
	return &reply, nil
}

// ReportResult calls S端 CommandService.ReportResult.
func (c *RPCClient) ReportResult(result *cluster.CmdResult) error {
	c.mu.Lock()
	cmdConn := c.cmdConn
	c.mu.Unlock()
	if cmdConn == nil {
		return errNotConnected
	}

	var reply cluster.OK
	return cmdConn.Call(c.ctx, "ReportResult", result, &reply)
}

// FetchCommands polls the control plane's durable command queue separately
// from the slower telemetry heartbeat. This keeps an operator action
// responsive without increasing GPU / disk metric collection frequency.
func (c *RPCClient) FetchCommands(nodeID string) ([]*cluster.Command, error) {
	c.mu.Lock()
	cmdConn := c.cmdConn
	c.mu.Unlock()
	if cmdConn == nil {
		return nil, errNotConnected
	}

	var reply []*cluster.Command
	if err := cmdConn.Call(c.ctx, "FetchCommands", nodeID, &reply); err != nil {
		return nil, err
	}
	return reply, nil
}

var errNotConnected = &connError{"not connected to S端"}

type connError struct{ msg string }

func (e *connError) Error() string { return e.msg }
