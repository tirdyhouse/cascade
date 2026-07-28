package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"predict/engine/pkg/cluster"
)

func TestDispatcherQueuesVisibleCommandEvent(t *testing.T) {
	registry := NewRegistry()
	if reply := registry.Register(&cluster.NodeInfo{
		NodeID:    "node-a",
		IP:        "10.0.0.1",
		CacheMode: cluster.CacheModeLocalNVMe,
	}); !reply.Accepted {
		t.Fatalf("Register() rejected node: %+v", reply)
	}
	registry.Heartbeat(&cluster.MachineStatus{
		NodeID: "node-a",
		AvailableModels: []cluster.LocalModel{{
			Name:   "model-a",
			Status: "ready",
		}},
	})

	result := NewDispatcher(registry).Dispatch(&cluster.DispatchReq{
		Action: cluster.CmdStartVLLM,
		Target: "node-a",
		Params: map[string]string{"model": "model-a"},
	})
	if !result.OK {
		t.Fatalf("Dispatch() failed: %+v", result)
	}
	if result.CmdID == "" {
		t.Fatal("Dispatch() did not return a command ID")
	}
	if result.TargetCount != 1 {
		t.Fatalf("Dispatch() target count = %d, want 1", result.TargetCount)
	}
	if duplicate := NewDispatcher(registry).Dispatch(&cluster.DispatchReq{
		Action: cluster.CmdStartVLLM,
		Target: "node-a",
		Params: map[string]string{"model": "model-a"},
	}); duplicate.OK || !strings.Contains(duplicate.Err, "in progress") {
		t.Fatalf("second lifecycle dispatch = %+v, want active-command rejection", duplicate)
	}

	pending := registry.FetchCommands("node-a")
	if len(pending) != 1 || pending[0].CmdID != result.CmdID {
		t.Fatalf("pending commands = %+v, want command %q", pending, result.CmdID)
	}

	history := registry.CommandHistory()
	if len(history) != 1 {
		t.Fatalf("command history length = %d, want 1", len(history))
	}
	event := history[0]
	if event.Status != "queued" || event.Action != cluster.CmdStartVLLM || event.Timestamp == 0 {
		t.Fatalf("queued event = %+v, want action, queued status, and timestamp", event)
	}
}

func TestDispatchCommandEndpointRejectsUnknownTarget(t *testing.T) {
	registry := NewRegistry()
	server := &Server{
		registry:   registry,
		dispatcher: NewDispatcher(registry),
	}

	req := httptest.NewRequest(http.MethodPost, "/api/v1/command", strings.NewReader(`{
		"action":"stop_vllm",
		"target":"missing-node"
	}`))
	req.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()
	server.apiDispatchCommand(recorder, req)

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusBadRequest)
	}
	var result cluster.OK
	if err := json.NewDecoder(recorder.Body).Decode(&result); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if result.OK || !strings.Contains(result.Err, "missing-node") {
		t.Fatalf("response = %+v, want rejected missing target", result)
	}
}

func TestDispatcherRejectsUnsupportedAndInvalidLifecycleActions(t *testing.T) {
	registry := NewRegistry()
	if reply := registry.Register(&cluster.NodeInfo{NodeID: "node-a", IP: "10.0.0.1", CacheMode: cluster.CacheModeLocalNVMe}); !reply.Accepted {
		t.Fatalf("Register() rejected node: %+v", reply)
	}
	dispatcher := NewDispatcher(registry)
	unsupported := dispatcher.Dispatch(&cluster.DispatchReq{Action: cluster.CmdExecShell, Target: "node-a", Params: map[string]string{"cmd": "id"}})
	if unsupported.OK || !strings.Contains(unsupported.Err, "not supported") {
		t.Fatalf("unsupported dispatch = %+v", unsupported)
	}
	stop := dispatcher.Dispatch(&cluster.DispatchReq{Action: cluster.CmdStopVLLM, Target: "node-a"})
	if stop.OK || !strings.Contains(stop.Err, "cannot stop") {
		t.Fatalf("stopped-node stop dispatch = %+v", stop)
	}
}

func TestUnhealthyManagedProcessCanBeRecoveredButNotStartedAgain(t *testing.T) {
	registry := NewRegistry()
	if reply := registry.Register(&cluster.NodeInfo{NodeID: "node-a", IP: "10.0.0.1", CacheMode: cluster.CacheModeLocalNVMe}); !reply.Accepted {
		t.Fatalf("Register() rejected node: %+v", reply)
	}
	registry.Heartbeat(&cluster.MachineStatus{
		NodeID:     "node-a",
		VLLMStatus: "error",
		ModelName:  "model-a",
	})

	if err := registry.ValidateOperation(cluster.CmdStartVLLM, "node-a"); err == nil || !strings.Contains(err.Error(), "restart or stop") {
		t.Fatalf("start unhealthy managed process error = %v, want recovery guidance", err)
	}
	if err := registry.ValidateOperation(cluster.CmdRestartVLLM, "node-a"); err != nil {
		t.Fatalf("restart unhealthy managed process: %v", err)
	}
	if err := registry.ValidateOperation(cluster.CmdStopVLLM, "node-a"); err != nil {
		t.Fatalf("stop unhealthy managed process: %v", err)
	}

	// A process that has already exited reports no managed model; starting it
	// again is the right recovery path.
	registry.Heartbeat(&cluster.MachineStatus{NodeID: "node-a", VLLMStatus: "error"})
	if err := registry.ValidateOperation(cluster.CmdStartVLLM, "node-a"); err != nil {
		t.Fatalf("start after process exit: %v", err)
	}
	if err := registry.ValidateOperation(cluster.CmdRestartVLLM, "node-a"); err == nil || !strings.Contains(err.Error(), "use start") {
		t.Fatalf("restart after process exit error = %v, want start guidance", err)
	}
	if err := registry.ValidateOperation(cluster.CmdStopVLLM, "node-a"); err == nil || !strings.Contains(err.Error(), "use start") {
		t.Fatalf("stop after process exit error = %v, want start guidance", err)
	}
}
