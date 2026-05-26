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

	mu     sync.Mutex
	conn   client.XClient
	ctx    context.Context
	cancel context.CancelFunc
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

	d, err := client.NewPeer2PeerDiscovery("tcp@"+c.serverAddr, "")
	if err != nil {
		return err
	}

	c.conn = client.NewXClient("ClusterService", client.Failtry, client.RandomSelect, d, client.DefaultOption)
	log.Printf("[rpcx] connected to S端 at %s", c.serverAddr)
	return nil
}

// Close disconnects.
func (c *RPCClient) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.conn != nil {
		c.conn.Close()
	}
	c.cancel()
}

// Reconnect re-establishes the connection after failure.
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
	// Create a fresh connection for CommandService
	d, err := client.NewPeer2PeerDiscovery("tcp@"+c.serverAddr, "")
	if err != nil {
		return err
	}
	cmdConn := client.NewXClient("CommandService", client.Failtry, client.RandomSelect, d, client.DefaultOption)
	defer cmdConn.Close()

	var reply cluster.OK
	return cmdConn.Call(c.ctx, "ReportResult", result, &reply)
}

var errNotConnected = &connError{"not connected to S端"}

type connError struct{ msg string }

func (e *connError) Error() string { return e.msg }
