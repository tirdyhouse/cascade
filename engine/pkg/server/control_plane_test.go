package server

import (
	"bytes"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"predict/engine/pkg/cluster"
)

func TestNodeChatUsesAdvertisedCacheMetadataURLAndV2Counters(t *testing.T) {
	var cacheCalls atomic.Int32
	cacheServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/stats" {
			http.NotFound(w, r)
			return
		}
		if cacheCalls.Add(1) == 1 {
			_, _ = w.Write([]byte(`{"BlocksRetrieved":1,"ChunksRetrieved":2,"BlocksStored":1,"ChunksStored":3,"MatchRequests":5,"MatchHits":4,"MatchedTokens":64}`))
			return
		}
		_, _ = w.Write([]byte(`{"BlocksRetrieved":2,"ChunksRetrieved":4,"BlocksStored":1,"ChunksStored":5,"MatchRequests":6,"MatchHits":5,"MatchedTokens":96}`))
	}))
	defer cacheServer.Close()

	vllmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte(`{"id":"chatcmpl-test","usage":{"prompt_tokens_details":{"cached_tokens":32}}}`))
	}))
	defer vllmServer.Close()
	endpoint, _ := url.Parse(vllmServer.URL)
	host, portText, err := net.SplitHostPort(endpoint.Host)
	if err != nil {
		t.Fatalf("split vLLM endpoint: %v", err)
	}
	port, _ := strconv.Atoi(portText)

	registry := NewRegistry()
	registry.Register(&cluster.NodeInfo{
		NodeID:           "node-a",
		IP:               host,
		VLLMPort:         port,
		CacheMode:        cluster.CacheModeSharedPool,
		SharedCacheID:    "pool-a",
		CacheMetadataURL: cacheServer.URL,
	})
	server := &Server{registry: registry, httpClient: &http.Client{Timeout: time.Second}}
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/api/v1/nodes/node-a/vllm/chat", bytes.NewBufferString(`{"model":"test","messages":[]}`))
	request.SetPathValue("id", "node-a")
	server.apiNodeVLLMChat(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("chat response = %d: %s", recorder.Code, recorder.Body.String())
	}
	if got := cacheCalls.Load(); got != 2 {
		t.Fatalf("cache stats calls = %d, want 2 via advertised endpoint", got)
	}
	var response map[string]interface{}
	if err := json.NewDecoder(recorder.Body).Decode(&response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	cache, ok := response["_cache"].(map[string]interface{})
	if !ok {
		t.Fatalf("response cache diagnostics = %#v", response["_cache"])
	}
	if got, want := cache["retrieved_objects"], float64(3); got != want {
		t.Fatalf("retrieved_objects = %v, want %v", got, want)
	}
	if got, want := cache["stored_objects"], float64(6); got != want {
		t.Fatalf("stored_objects = %v, want %v", got, want)
	}
	if got, want := cache["matched_tokens_total"], float64(96); got != want {
		t.Fatalf("matched_tokens_total = %v, want %v", got, want)
	}
}

func TestNodeLogsProxyReadsAdvertisedAgentDiagnostics(t *testing.T) {
	diagnostics := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/logs" || r.URL.Query().Get("lines") != "25" {
			http.NotFound(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(cluster.LogChunk{NodeID: "node-a", Filename: "vllm-model.log", Lines: "remote log", Offset: 11, EOF: true})
	}))
	defer diagnostics.Close()
	registry := NewRegistry()
	registry.Register(&cluster.NodeInfo{NodeID: "node-a", IP: "10.0.0.10", DiagnosticsURL: diagnostics.URL, CacheMode: cluster.CacheModeLocalNVMe})
	server := &Server{registry: registry, httpClient: &http.Client{Timeout: time.Second}}
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/api/v1/nodes/node-a/logs?lines=25", nil)
	request.SetPathValue("id", "node-a")
	server.apiNodeLogs(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("log response = %d: %s", recorder.Code, recorder.Body.String())
	}
	var chunk cluster.LogChunk
	if err := json.NewDecoder(recorder.Body).Decode(&chunk); err != nil {
		t.Fatalf("decode proxied log: %v", err)
	}
	if chunk.Lines != "remote log" || chunk.Filename != "vllm-model.log" {
		t.Fatalf("chunk = %+v", chunk)
	}
}

func TestPersistentRegistryRestoresUndeliveredCommandAndHistory(t *testing.T) {
	stateDir := t.TempDir()
	registry, err := NewPersistentRegistry(stateDir)
	if err != nil {
		t.Fatalf("NewPersistentRegistry() error = %v", err)
	}
	info := &cluster.NodeInfo{NodeID: "node-a", IP: "10.0.0.1", CacheMode: cluster.CacheModeLocalNVMe}
	if reply := registry.Register(info); !reply.Accepted {
		t.Fatalf("register = %+v", reply)
	}
	command := &cluster.Command{CmdID: "cmd-a", Action: cluster.CmdDownloadModel, Target: "node-a", CreatedAt: time.Now().UnixNano(), Timeout: 300}
	if _, err := registry.EnqueueCommand(command); err != nil {
		t.Fatalf("EnqueueCommand() error = %v", err)
	}
	if firstDelivery := registry.FetchCommands("node-a"); len(firstDelivery) != 1 || firstDelivery[0].CmdID != command.CmdID {
		t.Fatalf("first delivery = %+v, want %q", firstDelivery, command.CmdID)
	}

	restored, err := NewPersistentRegistry(stateDir)
	if err != nil {
		t.Fatalf("restore registry: %v", err)
	}
	if reply := restored.Register(info); !reply.Accepted {
		t.Fatalf("restored register = %+v", reply)
	}
	pending := restored.FetchCommands("node-a")
	if len(pending) != 1 || pending[0].CmdID != "cmd-a" {
		t.Fatalf("restored pending = %+v", pending)
	}
	if again := restored.FetchCommands("node-a"); len(again) != 0 {
		t.Fatalf("command delivered twice without a server restart: %+v", again)
	}
	history := restored.CommandHistory()
	if len(history) != 1 || history[0].Status != "queued" || history[0].CreatedAt != command.CreatedAt {
		t.Fatalf("restored history = %+v", history)
	}
}

func TestRegistryDeduplicatesReplayedResult(t *testing.T) {
	registry := NewRegistry()
	info := &cluster.NodeInfo{NodeID: "node-a", IP: "10.0.0.1", CacheMode: cluster.CacheModeLocalNVMe}
	if reply := registry.Register(info); !reply.Accepted {
		t.Fatalf("register = %+v", reply)
	}
	command := &cluster.Command{CmdID: "cmd-a", Action: cluster.CmdDownloadModel, Target: "node-a", CreatedAt: time.Now().UnixNano(), Timeout: 300}
	if _, err := registry.EnqueueCommand(command); err != nil {
		t.Fatalf("EnqueueCommand() error = %v", err)
	}
	result := &cluster.CmdResult{
		CmdID:     command.CmdID,
		NodeID:    "node-a",
		Action:    command.Action,
		CreatedAt: command.CreatedAt,
		Status:    "success",
		Progress:  100,
		Timestamp: time.Now().UnixNano(),
	}
	registry.RecordResult(result)
	registry.RecordResult(result)

	history := registry.CommandHistory()
	if len(history) != 2 {
		t.Fatalf("history = %+v, want queued plus one terminal result", history)
	}
	if pending := registry.FetchCommands("node-a"); len(pending) != 0 {
		t.Fatalf("pending = %+v, want terminal command removed", pending)
	}
}
