package server

import (
	"fmt"
	"log"
	"net/url"
	"strconv"
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
	if req == nil {
		return &cluster.OK{OK: false, Err: "dispatch request is required"}
	}
	if err := validateDispatchAction(req); err != nil {
		return &cluster.OK{OK: false, Err: err.Error()}
	}

	// Normalize target
	target := req.Target
	if target == "" || target == "*" {
		target = "*"
	}
	if err := d.registry.ValidateOperation(req.Action, target); err != nil {
		return &cluster.OK{OK: false, Err: err.Error()}
	}
	if req.Action == cluster.CmdStartVLLM || req.Action == cluster.CmdRestartVLLM {
		if err := d.registry.ValidateModelReady(target, req.Params["model"]); err != nil {
			return &cluster.OK{OK: false, Err: err.Error()}
		}
	}

	timeout := req.Timeout
	if timeout <= 0 {
		timeout = 300 // default 5 minutes
	}

	cmd := &cluster.Command{
		CmdID:     d.genCmdID(target),
		Action:    req.Action,
		Params:    cloneParams(req.Params),
		Target:    target,
		CreatedAt: time.Now().UnixNano(),
		Timeout:   timeout,
	}

	targetCount, err := d.registry.EnqueueCommand(cmd)
	if err != nil {
		return &cluster.OK{OK: false, Err: err.Error()}
	}

	log.Printf("[dispatch] cmd=%s action=%s target=%s targets=%d", cmd.CmdID, cmd.Action, cmd.Target, targetCount)
	return &cluster.OK{OK: true, CmdID: cmd.CmdID, TargetCount: targetCount}
}

func validateDispatchAction(req *cluster.DispatchReq) error {
	switch req.Action {
	case cluster.CmdStartVLLM, cluster.CmdRestartVLLM:
		return validateServeParams(req.Params)
	case cluster.CmdStopVLLM:
		if len(req.Params) > 0 {
			return fmt.Errorf("stop_vllm does not accept parameters")
		}
		return nil
	case cluster.CmdDownloadModel:
		return validateDownloadParams(req.Params)
	case cluster.CmdLoadModel, cluster.CmdUnloadModel, cluster.CmdUpdateConfig, cluster.CmdExecShell:
		return fmt.Errorf("action %q is not supported; use start_vllm, stop_vllm, restart_vllm, or download_model", req.Action)
	default:
		return fmt.Errorf("unknown action: %s", req.Action)
	}
}

func validateServeParams(params map[string]string) error {
	if params == nil {
		return fmt.Errorf("model parameter required")
	}
	for key := range params {
		switch key {
		case "model", "gpu_util", "enable_disk_cache", "enable_prefix_caching", "quantization", "raw_args":
		default:
			return fmt.Errorf("unsupported start parameter %q", key)
		}
	}
	if _, exists := params["raw_args"]; exists {
		return fmt.Errorf("raw_args is not supported; use validated structured start parameters")
	}
	if err := validateModelID(params["model"]); err != nil {
		return err
	}
	if raw := strings.TrimSpace(params["gpu_util"]); raw != "" {
		value, err := strconv.ParseFloat(raw, 64)
		if err != nil || value < 0.1 || value > 0.98 {
			return fmt.Errorf("gpu_util must be between 0.10 and 0.98")
		}
	}
	if params["enable_disk_cache"] == "true" && params["enable_prefix_caching"] == "true" {
		return fmt.Errorf("vLLM prefix caching cannot be enabled with Cascade disk cache")
	}
	for _, key := range []string{"enable_disk_cache", "enable_prefix_caching"} {
		if value := params[key]; value != "" && value != "true" && value != "false" {
			return fmt.Errorf("%s must be true or false", key)
		}
	}
	if quantization := strings.TrimSpace(params["quantization"]); quantization != "" && !safeIdentifier(quantization) {
		return fmt.Errorf("quantization contains unsupported characters")
	}
	return nil
}

func validateDownloadParams(params map[string]string) error {
	if params == nil {
		return fmt.Errorf("model parameter required")
	}
	for key := range params {
		switch key {
		case "model", "download_url", "manifest_url":
		default:
			return fmt.Errorf("unsupported download parameter %q", key)
		}
	}
	if err := validateModelID(params["model"]); err != nil {
		return err
	}
	if err := validateHTTPURL(params["download_url"], "download_url"); err != nil {
		return err
	}
	if raw := strings.TrimSpace(params["manifest_url"]); raw != "" {
		if err := validateHTTPURL(raw, "manifest_url"); err != nil {
			return err
		}
	}
	return nil
}

func validateModelID(raw string) error {
	name := strings.TrimSpace(raw)
	if name == "" {
		return fmt.Errorf("model parameter required")
	}
	if len(name) > 128 || name == "." || name == ".." || strings.ContainsAny(name, "/\\") {
		return fmt.Errorf("model must be a published model name, not a filesystem path")
	}
	if !safeIdentifier(name) {
		return fmt.Errorf("model contains unsupported characters")
	}
	return nil
}

func safeIdentifier(value string) bool {
	for _, character := range value {
		if (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') ||
			(character >= '0' && character <= '9') || character == '.' || character == '_' || character == '-' {
			continue
		}
		return false
	}
	return value != ""
}

func validateHTTPURL(raw, field string) error {
	parsed, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.User != nil {
		return fmt.Errorf("%s must be an http or https URL", field)
	}
	return nil
}

func cloneParams(params map[string]string) map[string]string {
	if len(params) == 0 {
		return nil
	}
	copy := make(map[string]string, len(params))
	for key, value := range params {
		copy[key] = value
	}
	return copy
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
