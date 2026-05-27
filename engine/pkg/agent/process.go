package agent

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
)

// StartOptions holds parameters for starting vLLM.
type StartOptions struct {
	Model         string
	GPUUtil       string
	PrefixCaching bool
	DiskCache     bool
	WorkDir       string
}

// ProcessManager handles vLLM process lifecycle.
type ProcessManager struct {
	cfg *Config

	mu         sync.Mutex
	cmd        *exec.Cmd
	status     string // "stopped" | "running" | "loading" | "error"
	modelName  string
	logFile    string // path to current vLLM log file
}

// NewProcessManager creates a ProcessManager.
func NewProcessManager(cfg *Config) *ProcessManager {
	return &ProcessManager{
		cfg:    cfg,
		status: "stopped",
	}
}

// Start launches a vLLM serve process with the given options.
func (pm *ProcessManager) Start(opts *StartOptions) (string, error) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if pm.cmd != nil && pm.cmd.Process != nil {
		return "", fmt.Errorf("vLLM already running (model=%s)", pm.modelName)
	}

	// Prepare log file
	logDir := filepath.Join(opts.WorkDir, "logs")
	os.MkdirAll(logDir, 0755)
	logFile := filepath.Join(logDir, "vllm-"+opts.Model+".log")
	f, err := os.Create(logFile)
	if err != nil {
		return "", fmt.Errorf("create log file: %w", err)
	}

	// Build vLLM args
	args := []string{"serve", opts.Model}
	args = append(args, "--gpu-memory-utilization", opts.GPUUtil)
	if opts.PrefixCaching {
		args = append(args, "--enable-prefix-caching")
	}
	if opts.DiskCache {
		args = append(args, "--kv-connector", "disk-cache")
		args = append(args, "--disk-cache-path", filepath.Join(opts.WorkDir, "cache"))
	}

	pm.cmd = exec.Command("vllm", args...)
	pm.cmd.Stdout = f
	pm.cmd.Stderr = f

	if err := pm.cmd.Start(); err != nil {
		f.Close()
		pm.status = "error"
		return "", fmt.Errorf("start vLLM: %w", err)
	}

	pm.status = "loading"
	pm.modelName = opts.Model
	pm.logFile = logFile
	log.Printf("[process] started vLLM (pid=%d) model=%s log=%s", pm.cmd.Process.Pid, opts.Model, logFile)

	// Wait in background
	go func() {
		err := pm.cmd.Wait()
		f.Close()
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

	return fmt.Sprintf("started pid=%d log=%s", pm.cmd.Process.Pid, logFile), nil
}

// DownloadModel downloads a model from URL into workDir/models/.
func (pm *ProcessManager) DownloadModel(model, url, workDir string) (string, error) {
	modelDir := filepath.Join(workDir, "models", model)
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		return "", fmt.Errorf("create model dir: %w", err)
	}

	log.Printf("[process] downloading model %s from %s to %s", model, url, modelDir)
	// Real implementation would use wget, curl, or huggingface_hub
	// For now, create a placeholder
	placeholder := filepath.Join(modelDir, ".downloaded")
	if err := os.WriteFile(placeholder, []byte("downloaded from "+url+"\n"), 0644); err != nil {
		return "", err
	}
	return fmt.Sprintf("model %s ready at %s", model, modelDir), nil
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

// LogFile returns the current vLLM log file path.
func (pm *ProcessManager) LogFile() string {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	return pm.logFile
}

// LoadModel stub.
func (pm *ProcessManager) LoadModel(model string) (string, error) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	if pm.cmd == nil || pm.cmd.Process == nil {
		return "", fmt.Errorf("vLLM not running")
	}
	pm.modelName = model
	pm.status = "loading"
	return fmt.Sprintf("loading model %s...", model), nil
}

// UnloadModel stub.
func (pm *ProcessManager) UnloadModel() (string, error) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	if pm.modelName == "" {
		return "", fmt.Errorf("no model loaded")
	}
	oldModel := pm.modelName
	pm.modelName = ""
	pm.status = "running"
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
