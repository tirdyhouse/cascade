package server

import (
	"fmt"
	"log"
	"strings"
	"time"

	"predict/engine/pkg/cluster"
)

// Dispatcher handles command creation, validation, and tracking.
type Dispatcher struct {
	registry *Registry
}

// NewDispatcher creates a new Dispatcher.
func NewDispatcher(registry *Registry) *Dispatcher {
	return &Dispatcher{registry: registry}
}

// Dispatch creates a command and enqueues it to the target node(s).
func (d *Dispatcher) Dispatch(req *cluster.DispatchReq) *cluster.OK {
	// Validate action
	validActions := map[cluster.CommandAction]bool{
		cluster.CmdStartVLLM:     true,
		cluster.CmdStopVLLM:      true,
		cluster.CmdRestartVLLM:   true,
		cluster.CmdLoadModel:     true,
		cluster.CmdUnloadModel:   true,
		cluster.CmdUpdateConfig:  true,
		cluster.CmdExecShell:     true,
	}
	if !validActions[req.Action] {
		return &cluster.OK{OK: false, Err: fmt.Sprintf("unknown action: %s", req.Action)}
	}

	// Normalize target
	target := req.Target
	if target == "" || target == "*" {
		target = "*"
	}

	timeout := req.Timeout
	if timeout <= 0 {
		timeout = 300 // default 5 minutes
	}

	cmd := &cluster.Command{
		CmdID:     d.genCmdID(target),
		Action:    req.Action,
		Params:    req.Params,
		Target:    target,
		CreatedAt: time.Now().UnixNano(),
		Timeout:   timeout,
	}

	d.registry.EnqueueCommand(cmd)

	log.Printf("[dispatch] cmd=%s action=%s target=%s", cmd.CmdID, cmd.Action, cmd.Target)
	return &cluster.OK{OK: true}
}

// genCmdID generates a unique command ID with target prefix.
func (d *Dispatcher) genCmdID(target string) string {
	seq := d.registry.NextSeq()
	prefix := strings.ReplaceAll(target, ".", "-")
	if len(prefix) > 16 {
		prefix = prefix[:16]
	}
	return fmt.Sprintf("cmd-%s-%d-%d", prefix, time.Now().UnixMilli(), seq)
}
