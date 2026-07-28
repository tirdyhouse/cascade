package server

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"predict/engine/pkg/agent"
	"predict/engine/pkg/cluster"
)

const fakeVLLMEnvironment = "CASCADE_TEST_FAKE_VLLM"

// The test binary doubles as a tiny OpenAI-compatible vLLM stand-in when the
// Agent starts it as a child process. This keeps the lifecycle and gateway
// test on the actual ProcessManager/RPC path without requiring a GPU in CI.
func init() {
	if os.Getenv(fakeVLLMEnvironment) == "1" {
		runFakeVLLM()
		os.Exit(0)
	}
}

func runFakeVLLM() {
	host, port := "127.0.0.1", "8000"
	for index := 1; index+1 < len(os.Args); index++ {
		switch os.Args[index] {
		case "--host":
			host = os.Args[index+1]
		case "--port":
			port = os.Args[index+1]
		}
	}

	listener, err := net.Listen("tcp", net.JoinHostPort(host, port))
	if err != nil {
		fmt.Fprintln(os.Stderr, "fake vLLM listen:", err)
		os.Exit(2)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"control-model","object":"model","owned_by":"fake-vllm"}]}`))
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-e2e","object":"chat.completion","model":"control-model","choices":[{"index":0,"message":{"role":"assistant","content":"ready"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5,"prompt_tokens_details":{"cached_tokens":4}}}`))
	})
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n"))
	})
	fmt.Fprintln(os.Stdout, "fake vLLM ready")
	if err := http.Serve(listener, mux); err != nil {
		fmt.Fprintln(os.Stderr, "fake vLLM serve:", err)
	}
}

// This is intentionally a real RPC path rather than a direct unit call. It
// protects the operator journey that matters: a catalog choice becomes one
// command, the Agent downloads and verifies it, and the control plane sees the
// terminal lifecycle plus a serving-ready model on the next heartbeat.
func TestControlPlaneDispatchesVerifiedModelToAgent(t *testing.T) {
	files := map[string][]byte{
		"config.json":    []byte(`{"model_type":"unit"}`),
		"weights.bin":    []byte("control-plane-verified-weights"),
		"tokenizer.json": []byte(`{"version":"1.0"}`),
	}
	manifest := cluster.ModelManifest{Name: "control-model"}
	for path, body := range files {
		hash := sha256.Sum256(body)
		manifest.Files = append(manifest.Files, cluster.ModelFile{
			Path:   path,
			Size:   int64(len(body)),
			SHA256: hex.EncodeToString(hash[:]),
		})
		manifest.TotalBytes += int64(len(body))
	}

	source := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/manifest" {
			_ = json.NewEncoder(w).Encode(manifest)
			return
		}
		const prefix = "/models/control-model/"
		if strings.HasPrefix(r.URL.Path, prefix) {
			if body, ok := files[strings.TrimPrefix(r.URL.Path, prefix)]; ok {
				_, _ = w.Write(body)
				return
			}
		}
		http.NotFound(w, r)
	}))
	defer source.Close()

	catalog := NewModelRegistry("", "")
	catalog.configured = []cluster.ModelInfo{{
		Name:        manifest.Name,
		DownloadURL: source.URL + "/models/control-model/",
		ManifestURL: source.URL + "/manifest",
	}}
	catalog.Scan()

	rpcPort := testFreeTCPPort(t)
	controlConfig := DefaultConfig()
	controlConfig.RPCPort = rpcPort
	controlConfig.StateDir = t.TempDir()
	control := New(controlConfig)
	control.models = catalog
	controlContext, cancelControl := context.WithCancel(context.Background())
	defer cancelControl()
	if err := control.startRPCX(controlContext); err != nil {
		t.Fatalf("start control RPC: %v", err)
	}
	defer control.Stop()

	agentConfig := agent.DefaultConfig()
	agentConfig.NodeID = "agent-e2e"
	agentConfig.ServerAddr = net.JoinHostPort("127.0.0.1", strconv.Itoa(rpcPort))
	agentConfig.WorkDir = t.TempDir()
	agentConfig.CachePath = ""
	agentConfig.DiagnosticsPort = 0
	agentConfig.VLLMPort = testFreeTCPPort(t)
	remoteAgent := agent.New(agentConfig)
	agentDone := make(chan error, 1)
	go func() { agentDone <- remoteAgent.Start() }()
	defer func() {
		remoteAgent.Stop()
		select {
		case err := <-agentDone:
			if err != nil {
				t.Errorf("agent stopped with error: %v", err)
			}
		case <-time.After(3 * time.Second):
			t.Error("agent did not stop")
		}
	}()

	waitForControlCondition(t, 5*time.Second, func() bool {
		detail := control.registry.NodeDetail(agentConfig.NodeID)
		return detail != nil && detail.Status != nil && detail.Status.Seq > 0
	}, "Agent initial heartbeat")

	request := &cluster.DispatchReq{
		Action:  cluster.CmdDownloadModel,
		Target:  agentConfig.NodeID,
		Params:  map[string]string{"model": manifest.Name},
		Timeout: 30,
	}
	if err := control.enrichDispatchRequest(request); err != nil {
		t.Fatalf("enrich download command: %v", err)
	}
	reply := control.dispatcher.Dispatch(request)
	if !reply.OK {
		t.Fatalf("dispatch download command: %+v", reply)
	}

	waitForControlCondition(t, 10*time.Second, func() bool {
		for _, event := range control.registry.CommandHistory() {
			if event.CmdID == reply.CmdID && event.Status == "success" {
				return true
			}
		}
		return false
	}, "download command success event")

	waitForControlCondition(t, 10*time.Second, func() bool {
		return control.registry.ValidateModelReady(agentConfig.NodeID, manifest.Name) == nil
	}, "verified local model in Agent heartbeat")
}

func TestControlPlaneServesThroughAgentGatewayLifecycle(t *testing.T) {
	t.Setenv(fakeVLLMEnvironment, "1")

	files := map[string][]byte{
		"config.json":    []byte(`{"model_type":"unit"}`),
		"weights.bin":    []byte("control-plane-gateway-weights"),
		"tokenizer.json": []byte(`{"version":"1.0"}`),
	}
	manifest := cluster.ModelManifest{Name: "control-model"}
	for path, body := range files {
		hash := sha256.Sum256(body)
		manifest.Files = append(manifest.Files, cluster.ModelFile{
			Path:   path,
			Size:   int64(len(body)),
			SHA256: hex.EncodeToString(hash[:]),
		})
		manifest.TotalBytes += int64(len(body))
	}

	source := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/manifest" {
			_ = json.NewEncoder(w).Encode(manifest)
			return
		}
		const prefix = "/models/control-model/"
		if strings.HasPrefix(r.URL.Path, prefix) {
			if body, ok := files[strings.TrimPrefix(r.URL.Path, prefix)]; ok {
				_, _ = w.Write(body)
				return
			}
		}
		http.NotFound(w, r)
	}))
	defer source.Close()

	cache := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/stats" {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte(`{"BlocksStored":1,"ChunksStored":3,"ChunksRetrieved":2,"DiskUsedBytes":2048,"MatchRequests":10,"MatchHits":8,"MatchedTokens":128}`))
	}))
	defer cache.Close()

	catalog := NewModelRegistry("", "")
	catalog.configured = []cluster.ModelInfo{{
		Name:        manifest.Name,
		DownloadURL: source.URL + "/models/control-model/",
		ManifestURL: source.URL + "/manifest",
	}}
	catalog.Scan()

	rpcPort := testFreeTCPPort(t)
	controlConfig := DefaultConfig()
	controlConfig.RPCPort = rpcPort
	controlConfig.StateDir = t.TempDir()
	control := New(controlConfig)
	control.models = catalog
	controlContext, cancelControl := context.WithCancel(context.Background())
	defer cancelControl()
	if err := control.startRPCX(controlContext); err != nil {
		t.Fatalf("start control RPC: %v", err)
	}
	defer control.Stop()

	apiMux := http.NewServeMux()
	control.registerAPI(apiMux)
	api := httptest.NewServer(apiMux)
	defer api.Close()

	agentConfig := agent.DefaultConfig()
	agentConfig.NodeID = "agent-gateway-e2e"
	agentConfig.ServerAddr = net.JoinHostPort("127.0.0.1", strconv.Itoa(rpcPort))
	agentConfig.WorkDir = t.TempDir()
	agentConfig.CachePath = cache.URL
	agentConfig.AdvertiseHost = "127.0.0.1"
	agentConfig.VLLMPath = os.Args[0]
	agentConfig.DiagnosticsHost = "127.0.0.1"
	agentConfig.DiagnosticsPort = testFreeTCPPort(t)
	agentConfig.VLLMHost = "127.0.0.1"
	agentConfig.VLLMPort = testFreeTCPPort(t)
	remoteAgent := agent.New(agentConfig)
	agentDone := make(chan error, 1)
	go func() { agentDone <- remoteAgent.Start() }()
	defer func() {
		remoteAgent.Stop()
		select {
		case err := <-agentDone:
			if err != nil {
				t.Errorf("agent stopped with error: %v", err)
			}
		case <-time.After(3 * time.Second):
			t.Error("agent did not stop")
		}
	}()

	waitForControlCondition(t, 5*time.Second, func() bool {
		detail := control.registry.NodeDetail(agentConfig.NodeID)
		return detail != nil && detail.Info != nil && detail.Info.IP == "127.0.0.1" && detail.Status != nil && detail.Status.Seq > 0
	}, "Agent registration with advertised gateway endpoint")

	downloadID := dispatchControlCommand(t, api.URL, cluster.DispatchReq{
		Action:  cluster.CmdDownloadModel,
		Target:  agentConfig.NodeID,
		Params:  map[string]string{"model": manifest.Name},
		Timeout: 30,
	})
	waitForCommandTerminalStatus(t, control, downloadID, "success")
	waitForControlCondition(t, 5*time.Second, func() bool {
		return control.registry.ValidateModelReady(agentConfig.NodeID, manifest.Name) == nil
	}, "verified local model")

	startID := dispatchControlCommand(t, api.URL, cluster.DispatchReq{
		Action: cluster.CmdStartVLLM,
		Target: agentConfig.NodeID,
		Params: map[string]string{
			"model":             manifest.Name,
			"gpu_util":          "0.10",
			"enable_disk_cache": "false",
		},
		Timeout: 30,
	})
	waitForCommandTerminalStatus(t, control, startID, "success")
	waitForControlCondition(t, 5*time.Second, func() bool {
		detail := control.registry.NodeDetail(agentConfig.NodeID)
		return detail != nil && detail.Status != nil && detail.Status.VLLMStatus == "running" && detail.Status.ModelName == manifest.Name && detail.Status.CacheMatchHits == 8
	}, "running Agent with cache telemetry")

	resp, err := http.Get(api.URL + "/health")
	if err != nil {
		t.Fatalf("gateway health: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		t.Fatalf("gateway health status = %d, want %d", resp.StatusCode, http.StatusOK)
	}
	resp.Body.Close()

	resp, err = http.Get(api.URL + "/v1/models")
	if err != nil {
		t.Fatalf("gateway models: %v", err)
	}
	var models struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&models); err != nil {
		resp.Body.Close()
		t.Fatalf("decode gateway models: %v", err)
	}
	resp.Body.Close()
	if len(models.Data) != 1 || models.Data[0].ID != manifest.Name {
		t.Fatalf("gateway models = %+v, want %q", models.Data, manifest.Name)
	}

	chatBody := strings.NewReader(`{"model":"control-model","messages":[{"role":"user","content":"hello"}]}`)
	resp, err = http.Post(api.URL+"/v1/chat/completions", "application/json", chatBody)
	if err != nil {
		t.Fatalf("gateway chat: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		t.Fatalf("gateway chat status = %d, want %d", resp.StatusCode, http.StatusOK)
	}
	if got := resp.Header.Get("X-Cascade-Node-ID"); got != agentConfig.NodeID {
		resp.Body.Close()
		t.Fatalf("gateway routed to %q, want %q", got, agentConfig.NodeID)
	}
	var chat struct {
		ID string `json:"id"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&chat); err != nil {
		resp.Body.Close()
		t.Fatalf("decode gateway chat: %v", err)
	}
	resp.Body.Close()
	if chat.ID != "chatcmpl-e2e" {
		t.Fatalf("gateway chat id = %q, want fake upstream response", chat.ID)
	}

	resp, err = http.Get(api.URL + "/api/v1/nodes/" + agentConfig.NodeID + "/cache/stats")
	if err != nil {
		t.Fatalf("node cache stats: %v", err)
	}
	var cacheStats map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&cacheStats); err != nil {
		resp.Body.Close()
		t.Fatalf("decode cache stats: %v", err)
	}
	resp.Body.Close()
	if got := cacheStats["cache_match_hits"]; got != float64(8) {
		t.Fatalf("cache_match_hits = %v, want 8", got)
	}

	resp, err = http.Get(api.URL + "/api/v1/nodes/" + agentConfig.NodeID + "/logs?lines=25")
	if err != nil {
		t.Fatalf("node logs: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		t.Fatalf("node log status = %d, want %d", resp.StatusCode, http.StatusOK)
	}
	var logChunk cluster.LogChunk
	if err := json.NewDecoder(resp.Body).Decode(&logChunk); err != nil {
		resp.Body.Close()
		t.Fatalf("decode node log: %v", err)
	}
	resp.Body.Close()
	if !strings.Contains(logChunk.Lines, "fake vLLM ready") {
		t.Fatalf("node log = %q, want fake vLLM readiness", logChunk.Lines)
	}

	stopID := dispatchControlCommand(t, api.URL, cluster.DispatchReq{
		Action:  cluster.CmdStopVLLM,
		Target:  agentConfig.NodeID,
		Params:  map[string]string{},
		Timeout: 30,
	})
	waitForCommandTerminalStatus(t, control, stopID, "success")
	waitForControlCondition(t, 5*time.Second, func() bool {
		detail := control.registry.NodeDetail(agentConfig.NodeID)
		return detail != nil && detail.Status != nil && detail.Status.VLLMStatus == "stopped"
	}, "stopped Agent")

	resp, err = http.Get(api.URL + "/health")
	if err != nil {
		t.Fatalf("gateway health after stop: %v", err)
	}
	if resp.StatusCode != http.StatusServiceUnavailable {
		resp.Body.Close()
		t.Fatalf("gateway health after stop = %d, want %d", resp.StatusCode, http.StatusServiceUnavailable)
	}
	resp.Body.Close()
}

func dispatchControlCommand(t *testing.T, apiURL string, request cluster.DispatchReq) string {
	t.Helper()
	body, err := json.Marshal(request)
	if err != nil {
		t.Fatalf("encode command request: %v", err)
	}
	response, err := http.Post(apiURL+"/api/v1/command", "application/json", strings.NewReader(string(body)))
	if err != nil {
		t.Fatalf("dispatch command: %v", err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		t.Fatalf("dispatch command status = %d", response.StatusCode)
	}
	var reply cluster.OK
	if err := json.NewDecoder(response.Body).Decode(&reply); err != nil {
		t.Fatalf("decode command reply: %v", err)
	}
	if !reply.OK || reply.CmdID == "" || reply.TargetCount != 1 {
		t.Fatalf("dispatch reply = %+v", reply)
	}
	return reply.CmdID
}

func waitForCommandTerminalStatus(t *testing.T, control *Server, cmdID, want string) {
	t.Helper()
	waitForControlCondition(t, 10*time.Second, func() bool {
		for _, event := range control.registry.CommandHistory() {
			if event.CmdID == cmdID && event.Status == want {
				return true
			}
		}
		return false
	}, "command "+cmdID+" status "+want)
}

func testFreeTCPPort(t *testing.T) int {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("reserve TCP port: %v", err)
	}
	defer listener.Close()
	return listener.Addr().(*net.TCPAddr).Port
}

func waitForControlCondition(t *testing.T, timeout time.Duration, condition func() bool, description string) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if condition() {
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", description)
}
