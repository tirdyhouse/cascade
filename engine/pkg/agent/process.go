package agent

import (
	"bytes"
	"fmt"
	"log"
	"os/exec"
	"sync"
)

// ProcessManager handles vLLM process lifecycle.
type ProcessManager struct {
	cfg *Config

	mu         sync.Mutex
	cmd        *exec.Cmd
	status     string // "stopped" | "running" | "loading" | "error"
	modelName  string
}

// NewProcessManager creates a ProcessManager.
func NewProcessManager(cfg *Config) *ProcessManager {
	return &ProcessManager{
		cfg:    cfg,
		status: "stopped",
	}
}

// Start launches a vLLM serve process.
func (pm *ProcessManager) Start(model, gpuUtil string) (string, error) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if pm.cmd != nil && pm.cmd.Process != nil {
		return "", fmt.Errorf("vLLM already running (model=%s)", pm.modelName)
	}

	args := []string{
		"serve", model,
		"--gpu-memory-utilization", gpuUtil,
		"--kv-connector", "disk-cache",
		"--disk-cache-path", pm.cfg.CachePath,
	}

	pm.cmd = exec.Command("vllm", args...)

	var stdout, stderr bytes.Buffer
	pm.cmd.Stdout = &stdout
	pm.cmd.Stderr = &stderr

	if err := pm.cmd.Start(); err != nil {
		pm.status = "error"
		return stderr.String(), fmt.Errorf("start vLLM: %w", err)
	}

	pm.status = "loading"
	pm.modelName = model
	log.Printf("[process] started vLLM (pid=%d) model=%s", pm.cmd.Process.Pid, model)

	// Wait in background
	go func() {
		err := pm.cmd.Wait()
		pm.mu.Lock()
		if err != nil {
			pm.status = "error"
			log.Printf("[process] vLLM exited: %v", err)
		} else {
			pm.status = "stopped"
			log.Println("[process] vLLM exited cleanly")
		}
		pm.modelName = ""
		pm.cmd = nil
		pm.mu.Unlock()
	}()

	return fmt.Sprintf("started pid=%d", pm.cmd.Process.Pid), nil
}

// Stop terminates the vLLM process.
func (pm *ProcessManager) Stop() (string, error) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if pm.cmd == nil || pm.cmd.Process == nil {
		return "", fmt.Errorf("vLLM not running")
	}

	pid := pm.cmd.Process.Pid
	if err := pm.cmd.Process.Kill(); err != nil {
		pm.status = "error"
		return "", fmt.Errorf("kill vLLM (pid=%d): %w", pid, err)
	}

	pm.status = "stopped"
	pm.modelName = ""
	log.Printf("[process] killed vLLM (pid=%d)", pid)
	return fmt.Sprintf("killed pid=%d", pid), nil
}

// LoadModel sends a model load request to a running vLLM instance.
// This is a simplified stub; real implementation would call vLLM API.
func (pm *ProcessManager) LoadModel(model string) (string, error) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if pm.cmd == nil || pm.cmd.Process == nil {
		return "", fmt.Errorf("vLLM not running")
	}

	pm.modelName = model
	pm.status = "loading"
	log.Printf("[process] loading model: %s", model)

	// In reality, this would call vLLM's /v1/models or similar
	return fmt.Sprintf("loading model %s...", model), nil
}

// UnloadModel unloads the current model.
func (pm *ProcessManager) UnloadModel() (string, error) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if pm.modelName == "" {
		return "", fmt.Errorf("no model loaded")
	}

	oldModel := pm.modelName
	pm.modelName = ""
	pm.status = "running"
	log.Printf("[process] unloaded model: %s", oldModel)
	return fmt.Sprintf("unloaded %s", oldModel), nil
}

// Status returns the current process status.
func (pm *ProcessManager) Status() string {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	return pm.status
}

// ModelName returns the currently loaded model name.
func (pm *ProcessManager) ModelName() string {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	return pm.modelName
}
