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
