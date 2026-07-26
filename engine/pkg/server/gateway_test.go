package server

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"testing"
	"time"

	"predict/engine/pkg/cluster"
)

func registerGatewayNode(t *testing.T, registry *Registry, nodeID, upstreamURL, model string, queue int32, gpuUtil, cacheHitRate float64) {
	t.Helper()
	parsed, err := url.Parse(upstreamURL)
	if err != nil {
		t.Fatalf("parse upstream URL: %v", err)
	}
	host, portText, err := net.SplitHostPort(parsed.Host)
	if err != nil {
		t.Fatalf("split upstream host: %v", err)
	}
	port, err := strconv.Atoi(portText)
	if err != nil {
		t.Fatalf("parse upstream port: %v", err)
	}
	reply := registry.Register(&cluster.NodeInfo{
		NodeID:    nodeID,
		IP:        host,
		VLLMPort:  port,
		CacheMode: cluster.CacheModeLocalNVMe,
	})
	if !reply.Accepted {
		t.Fatalf("register %s: %+v", nodeID, reply)
	}
	registry.Heartbeat(&cluster.MachineStatus{
		NodeID:       nodeID,
		VLLMStatus:   "running",
		ModelName:    model,
		QueueLen:     queue,
		GPUUtil:      gpuUtil,
		CacheHitRate: cacheHitRate,
	})
}

func newGatewayHTTPServer(registry *Registry, maxInFlight int) *httptest.Server {
	cfg := DefaultConfig()
	cfg.GatewayMaxInFlightPerNode = maxInFlight
	server := &Server{
		registry:      registry,
		config:        cfg,
		gatewayClient: &http.Client{},
		scheduler:     NewGatewayScheduler(registry, maxInFlight),
	}
	mux := http.NewServeMux()
	server.registerAPI(mux)
	return httptest.NewServer(mux)
}

func TestGatewayRoutesToLowestLoadedMatchingNode(t *testing.T) {
	var gotAuthorization string
	upstreamA := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"node":"a"}`))
	}))
	defer upstreamA.Close()
	upstreamB := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuthorization = r.Header.Get("Authorization")
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("upstream path = %q", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"node":"b"}`))
	}))
	defer upstreamB.Close()

	registry := NewRegistry()
	registerGatewayNode(t, registry, "node-a", upstreamA.URL, "/models/Qwen2.5-7B", 4, 0.1, 99)
	registerGatewayNode(t, registry, "node-b", upstreamB.URL, "/models/Qwen2.5-7B", 0, 0.5, 0)
	gateway := newGatewayHTTPServer(registry, 4)
	defer gateway.Close()

	req, err := http.NewRequest(http.MethodPost, gateway.URL+"/v1/chat/completions", bytes.NewBufferString(`{"model":"Qwen2.5-7B","messages":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer test-token")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, body=%s", resp.StatusCode, body)
	}
	if got := resp.Header.Get("X-Cascade-Node-ID"); got != "node-b" {
		t.Fatalf("X-Cascade-Node-ID = %q, want node-b", got)
	}
	if gotAuthorization != "Bearer test-token" {
		t.Fatalf("authorization was not forwarded: %q", gotAuthorization)
	}
	if string(body) != `{"node":"b"}` {
		t.Fatalf("body = %s, want node-b response", body)
	}
}

func TestGatewayStreamsSSEWithoutBufferingOrRewriting(t *testing.T) {
	releaseSecondEvent := make(chan struct{})
	released := false
	defer func() {
		if !released {
			close(releaseSecondEvent)
		}
	}()
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		flusher := w.(http.Flusher)
		_, _ = w.Write([]byte("data: first\n\n"))
		flusher.Flush()
		<-releaseSecondEvent
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
		flusher.Flush()
	}))
	defer upstream.Close()

	registry := NewRegistry()
	registerGatewayNode(t, registry, "node-a", upstream.URL, "model-a", 0, 0, 0)
	gateway := newGatewayHTTPServer(registry, 2)
	defer gateway.Close()

	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Post(gateway.URL+"/v1/chat/completions", "application/json", bytes.NewBufferString(`{"model":"model-a","stream":true,"messages":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if got := resp.Header.Get("Content-Type"); got != "text/event-stream" {
		t.Fatalf("Content-Type = %q, want text/event-stream", got)
	}
	reader := bufio.NewReader(resp.Body)
	first, err := reader.ReadString('\n')
	if err != nil || first != "data: first\n" {
		t.Fatalf("first SSE line = %q, err=%v", first, err)
	}
	blank, err := reader.ReadString('\n')
	if err != nil || blank != "\n" {
		t.Fatalf("first SSE separator = %q, err=%v", blank, err)
	}
	close(releaseSecondEvent)
	released = true
	rest, _ := io.ReadAll(reader)
	if got := string(rest); got != "data: [DONE]\n\n" {
		t.Fatalf("remaining stream body = %q", got)
	}
}

func TestGatewayReturnsOpenAIErrorForNoReplica(t *testing.T) {
	gateway := newGatewayHTTPServer(NewRegistry(), 1)
	defer gateway.Close()

	resp, err := http.Post(gateway.URL+"/v1/chat/completions", "application/json", bytes.NewBufferString(`{"model":"missing","messages":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503", resp.StatusCode)
	}
	var body openAIErrorResponse
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode error response: %v", err)
	}
	if body.Error.Code != "model_not_available" {
		t.Fatalf("error body = %+v", body)
	}
}

func TestGatewayModelsListsRunningReplicasOnly(t *testing.T) {
	upstreamA := httptest.NewServer(http.NotFoundHandler())
	defer upstreamA.Close()
	upstreamB := httptest.NewServer(http.NotFoundHandler())
	defer upstreamB.Close()

	registry := NewRegistry()
	registerGatewayNode(t, registry, "node-a", upstreamA.URL, "/models/model-z", 0, 0, 0)
	registerGatewayNode(t, registry, "node-b", upstreamB.URL, "model-a", 0, 0, 0)
	gateway := newGatewayHTTPServer(registry, 1)
	defer gateway.Close()

	resp, err := http.Get(gateway.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var body openAIModelList
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	if len(body.Data) != 2 || body.Data[0].ID != "model-a" || body.Data[1].ID != "model-z" {
		t.Fatalf("models = %+v", body.Data)
	}
}

func TestGatewaySchedulerBalancesTiesAndEnforcesCapacity(t *testing.T) {
	upstreamA := httptest.NewServer(http.NotFoundHandler())
	defer upstreamA.Close()
	upstreamB := httptest.NewServer(http.NotFoundHandler())
	defer upstreamB.Close()
	registry := NewRegistry()
	registerGatewayNode(t, registry, "node-a", upstreamA.URL, "model", 0, 0, 0)
	registerGatewayNode(t, registry, "node-b", upstreamB.URL, "model", 0, 0, 0)
	scheduler := NewGatewayScheduler(registry, 1)

	first, err := scheduler.Acquire("model")
	if err != nil {
		t.Fatal(err)
	}
	second, err := scheduler.Acquire("model")
	if err != nil {
		t.Fatal(err)
	}
	if first.Target.NodeID == second.Target.NodeID {
		t.Fatalf("tie did not balance: first=%s second=%s", first.Target.NodeID, second.Target.NodeID)
	}
	if _, err := scheduler.Acquire("model"); !errors.Is(err, ErrGatewayCapacity) {
		t.Fatalf("third acquire error = %v, want ErrGatewayCapacity", err)
	}

	first.Release()
	third, err := scheduler.Acquire("model")
	if err != nil {
		t.Fatalf("acquire after release: %v", err)
	}
	third.Release()
	second.Release()
}

func TestGatewaySchedulerQuarantinesTransportFailures(t *testing.T) {
	upstreamA := httptest.NewServer(http.NotFoundHandler())
	defer upstreamA.Close()
	upstreamB := httptest.NewServer(http.NotFoundHandler())
	defer upstreamB.Close()
	registry := NewRegistry()
	registerGatewayNode(t, registry, "node-a", upstreamA.URL, "model", 0, 0, 0)
	registerGatewayNode(t, registry, "node-b", upstreamB.URL, "model", 0, 1, 0)
	scheduler := NewGatewayScheduler(registry, 2)

	first, err := scheduler.Acquire("model")
	if err != nil {
		t.Fatal(err)
	}
	if first.Target.NodeID != "node-a" {
		t.Fatalf("first target = %s, want node-a", first.Target.NodeID)
	}
	first.MarkUnavailable()
	first.Release()

	second, err := scheduler.Acquire("model")
	if err != nil {
		t.Fatal(err)
	}
	defer second.Release()
	if second.Target.NodeID != "node-b" {
		t.Fatalf("target after quarantine = %s, want node-b", second.Target.NodeID)
	}
}

func TestGatewayStatusEndpointReportsLiveState(t *testing.T) {
	upstream := httptest.NewServer(http.NotFoundHandler())
	defer upstream.Close()

	registry := NewRegistry()
	registerGatewayNode(t, registry, "node-a", upstream.URL, "model-a", 0, 0.1, 95)
	scheduler := NewGatewayScheduler(registry, 3)
	lease, err := scheduler.Acquire("model-a")
	if err != nil {
		t.Fatalf("Acquire() error = %v", err)
	}
	defer lease.Release()

	server := &Server{
		registry:  registry,
		config:    DefaultConfig(),
		scheduler: scheduler,
	}
	mux := http.NewServeMux()
	server.registerAPI(mux)
	api := httptest.NewServer(mux)
	defer api.Close()

	resp, err := http.Get(api.URL + "/api/v1/gateway/status")
	if err != nil {
		t.Fatalf("GET gateway status: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want %d", resp.StatusCode, http.StatusOK)
	}

	var status GatewayStatus
	if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
		t.Fatalf("decode gateway status: %v", err)
	}
	if status.EligibleNodes != 1 || len(status.Models) != 1 || status.Models[0] != "model-a" {
		t.Fatalf("gateway status = %+v, want one eligible model-a replica", status)
	}
	if status.MaxInFlightPerNode != 3 || status.InFlightByNode["node-a"] != 1 || status.UpdatedAt == 0 {
		t.Fatalf("gateway status = %+v, want live scheduler snapshot", status)
	}
}
