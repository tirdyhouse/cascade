package server

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

type gatewayRequest struct {
	Model string `json:"model"`
}

type openAIErrorResponse struct {
	Error openAIError `json:"error"`
}

type openAIError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Param   string `json:"param,omitempty"`
	Code    string `json:"code,omitempty"`
}

type openAIModelList struct {
	Object string        `json:"object"`
	Data   []openAIModel `json:"data"`
}

type openAIModel struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// GatewayStatus is the live routing state exposed to the operator console.
// It contains no request payloads or customer data.
type GatewayStatus struct {
	EligibleNodes      int              `json:"eligible_nodes"`
	Models             []string         `json:"models"`
	MaxInFlightPerNode int              `json:"max_inflight_per_node"`
	InFlightByNode     map[string]int   `json:"inflight_by_node"`
	QuarantinedUntil   map[string]int64 `json:"quarantined_until"`
	UpdatedAt          int64            `json:"updated_at"`
}

func (s *Server) gatewayStatus() GatewayStatus {
	status := GatewayStatus{
		InFlightByNode:   make(map[string]int),
		QuarantinedUntil: make(map[string]int64),
		UpdatedAt:        time.Now().UnixNano(),
	}
	if s == nil || s.registry == nil {
		return status
	}

	modelSet := make(map[string]struct{})
	for _, candidate := range s.registry.gatewayCandidates("", time.Now()) {
		status.EligibleNodes++
		if model := publicModelID(candidate.ModelName); model != "" {
			modelSet[model] = struct{}{}
		}
	}
	for model := range modelSet {
		status.Models = append(status.Models, model)
	}
	sort.Strings(status.Models)
	if s.scheduler != nil {
		status.MaxInFlightPerNode, status.InFlightByNode, status.QuarantinedUntil = s.scheduler.snapshot()
	}
	return status
}

// apiGatewayInference is the public cluster data plane for endpoints with an
// OpenAI-style model field. It preserves the upstream response, including SSE
// chunks, and only adds diagnostic response headers.
func (s *Server) apiGatewayInference(w http.ResponseWriter, r *http.Request) {
	maxBytes := int64(64 << 20)
	if s.config != nil && s.config.GatewayMaxRequestBytes > 0 {
		maxBytes = s.config.GatewayMaxRequestBytes
	}
	if r.ContentLength > maxBytes {
		writeOpenAIError(w, http.StatusRequestEntityTooLarge, "request body exceeds gateway limit", "invalid_request_error", "", "request_too_large")
		return
	}

	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxBytes))
	if err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			writeOpenAIError(w, http.StatusRequestEntityTooLarge, "request body exceeds gateway limit", "invalid_request_error", "", "request_too_large")
			return
		}
		writeOpenAIError(w, http.StatusBadRequest, "unable to read request body", "invalid_request_error", "", "invalid_request")
		return
	}

	var request gatewayRequest
	if err := json.Unmarshal(body, &request); err != nil || strings.TrimSpace(request.Model) == "" {
		writeOpenAIError(w, http.StatusBadRequest, "request must be a JSON object with a non-empty model", "invalid_request_error", "model", "invalid_model")
		return
	}

	if s.scheduler == nil {
		writeOpenAIError(w, http.StatusServiceUnavailable, "gateway scheduler is unavailable", "server_error", "", "cluster_unavailable")
		return
	}
	lease, err := s.scheduler.Acquire(request.Model)
	if err != nil {
		switch {
		case errors.Is(err, ErrGatewayCapacity):
			w.Header().Set("Retry-After", "1")
			writeOpenAIError(w, http.StatusTooManyRequests, "all replicas for this model are busy", "server_error", "model", "cluster_capacity")
		default:
			writeOpenAIError(w, http.StatusServiceUnavailable, "no healthy replica currently serves the requested model", "server_error", "model", "model_not_available")
		}
		return
	}
	defer lease.Release()

	target, err := vllmEndpointURL(lease.Target.IP, lease.Target.VLLMPort, r.URL.Path, r.URL.RawQuery)
	if err != nil {
		log.Printf("[gateway] invalid endpoint node=%s: %v", lease.Target.NodeID, err)
		writeOpenAIError(w, http.StatusBadGateway, "selected vLLM endpoint is invalid", "server_error", "", "upstream_unavailable")
		return
	}

	upstreamReq, err := http.NewRequestWithContext(r.Context(), r.Method, target, bytes.NewReader(body))
	if err != nil {
		writeOpenAIError(w, http.StatusInternalServerError, "unable to create upstream request", "server_error", "", "gateway_error")
		return
	}
	copyEndToEndHeaders(upstreamReq.Header, r.Header)
	if host, _, err := net.SplitHostPort(r.RemoteAddr); err == nil && host != "" {
		upstreamReq.Header.Add("X-Forwarded-For", host)
	}
	upstreamReq.Header.Set("X-Cascade-Node-ID", lease.Target.NodeID)

	client := s.gatewayClient
	if client == nil {
		client = http.DefaultClient
	}
	upstreamResp, err := client.Do(upstreamReq)
	if err != nil {
		if r.Context().Err() != nil {
			return
		}
		lease.MarkUnavailable()
		log.Printf("[gateway] upstream request failed node=%s target=%s: %v", lease.Target.NodeID, target, err)
		writeOpenAIError(w, http.StatusBadGateway, "selected vLLM replica is unavailable", "server_error", "", "upstream_unavailable")
		return
	}
	defer upstreamResp.Body.Close()

	copyEndToEndHeaders(w.Header(), upstreamResp.Header)
	w.Header().Set("X-Cascade-Node-ID", lease.Target.NodeID)
	w.Header().Set("X-Cascade-Model", publicModelID(lease.Target.Model))
	w.WriteHeader(upstreamResp.StatusCode)
	streamProxyBody(w, upstreamResp.Body)
}

// apiGatewayModels reports models with at least one fresh, running replica.
// The model IDs are derived from the running agent state rather than from a
// stale control-plane model catalog.
func (s *Server) apiGatewayModels(w http.ResponseWriter, _ *http.Request) {
	models := make(map[string]struct{})
	for _, candidate := range s.registry.gatewayCandidates("", time.Now()) {
		if id := publicModelID(candidate.ModelName); id != "" {
			models[id] = struct{}{}
		}
	}
	ids := make([]string, 0, len(models))
	for id := range models {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	response := openAIModelList{Object: "list", Data: make([]openAIModel, 0, len(ids))}
	for _, id := range ids {
		response.Data = append(response.Data, openAIModel{
			ID:      id,
			Object:  "model",
			Created: 0,
			OwnedBy: "cascade",
		})
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (s *Server) apiGatewayHealth(w http.ResponseWriter, _ *http.Request) {
	if len(s.registry.gatewayCandidates("", time.Now())) == 0 {
		writeOpenAIError(w, http.StatusServiceUnavailable, "no running vLLM replicas", "server_error", "", "cluster_unavailable")
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(`{"status":"ok"}`))
}

func writeOpenAIError(w http.ResponseWriter, status int, message, errorType, param, code string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(openAIErrorResponse{Error: openAIError{
		Message: message,
		Type:    errorType,
		Param:   param,
		Code:    code,
	}})
}

func vllmEndpointURL(host string, port int, path, rawQuery string) (string, error) {
	host = strings.TrimSpace(host)
	if host == "" {
		return "", errors.New("node IP is empty")
	}
	if port == 0 {
		port = defaultVLLMPort
	}
	if port < 1 || port > 65535 {
		return "", fmt.Errorf("invalid vLLM port %d", port)
	}
	if !strings.HasPrefix(path, "/") {
		return "", fmt.Errorf("invalid vLLM path %q", path)
	}
	return (&url.URL{
		Scheme:   "http",
		Host:     net.JoinHostPort(host, strconv.Itoa(port)),
		Path:     path,
		RawQuery: rawQuery,
	}).String(), nil
}

func copyEndToEndHeaders(dst, src http.Header) {
	connectionHeaders := make(map[string]struct{})
	for _, value := range src.Values("Connection") {
		for _, token := range strings.Split(value, ",") {
			connectionHeaders[strings.ToLower(strings.TrimSpace(token))] = struct{}{}
		}
	}
	for key, values := range src {
		lower := strings.ToLower(key)
		if _, skip := connectionHeaders[lower]; skip || isHopByHopHeader(lower) {
			continue
		}
		for _, value := range values {
			dst.Add(key, value)
		}
	}
}

func isHopByHopHeader(header string) bool {
	switch header {
	case "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade":
		return true
	default:
		return false
	}
}

func streamProxyBody(w http.ResponseWriter, body io.Reader) {
	flusher, _ := w.(http.Flusher)
	buffer := make([]byte, 32*1024)
	for {
		n, err := body.Read(buffer)
		if n > 0 {
			if _, writeErr := w.Write(buffer[:n]); writeErr != nil {
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
		if err == io.EOF {
			return
		}
		if err != nil {
			log.Printf("[gateway] upstream response stream failed: %v", err)
			return
		}
	}
}

func publicModelID(model string) string {
	model = strings.TrimSpace(strings.TrimSuffix(model, "/"))
	if model == "" {
		return ""
	}
	return filepath.Base(model)
}
