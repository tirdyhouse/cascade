package agent

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
)

// vllmBinary returns the path to the vllm binary.
// It checks common locations and falls back to "vllm" (system PATH).
func vllmBinary() string {
	for _, p := range []string{
		"/root/cascade/.venv-cascade/bin/vllm",
		"vllm",
	} {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return "vllm"
}

// StartOptions holds parameters for starting vLLM.
type StartOptions struct {
	Model         string
	GPUUtil       string
	PrefixCaching bool
	DiskCache     bool
	WorkDir       string
	Quantization  string // awq, gptq, etc.
	VLLMPath      string // path to vllm binary, auto-detected if empty


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
	if opts.Quantization != "" {
		args = append(args, "--quantization", opts.Quantization)
	}
	if opts.PrefixCaching {
		args = append(args, "--enable-prefix-caching")
	}
	if opts.DiskCache {
		diskCachePath := filepath.Join(opts.WorkDir, "cache")
		args = append(args, "--kv-connector", "DiskCacheConnector")
		args = append(args, "--kv-connector-extra-config", fmt.Sprintf(`{"disk_cache_path": %q}`, diskCachePath))
	}

	pm.cmd = exec.Command(vllmBinary(), args...)
	pm.cmd.Stdout = f
	pm.cmd.Stderr = f

	if err := pm.cmd.Start(); err != nil {
		f.Close()
		// Write the error to the log file so it shows in Live Log
		os.WriteFile(logFile, []byte(fmt.Sprintf("Failed to start vLLM: %v\n", err)), 0644)
		pm.status = "error"
		return fmt.Sprintf("Failed to start vLLM: %v", err), fmt.Errorf("start vLLM: %w", err)
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

// StartRaw starts vLLM with a raw command line string.
// The raw string is split and executed as: vllm serve <raw_args>

// StartRaw starts vLLM with a raw command line string.
// The raw string is split and executed as: vllm serve <raw_args>
// Example raw: "Qwen2.5-7B-Instruct --gpu-memory-utilization 0.9 --enable-prefix-caching --kv-connector disk-cache"
func (pm *ProcessManager) StartRaw(raw, workDir string) (string, error) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if pm.cmd != nil && pm.cmd.Process != nil {
		return "", fmt.Errorf("vLLM already running (model=%s)", pm.modelName)
	}

	// Split raw string into args
	args := []string{"serve"}
	args = append(args, splitArgs(raw)...)

	// Prepare log file
	logDir := filepath.Join(workDir, "logs")
	os.MkdirAll(logDir, 0755)
	modelName := ""
	if len(args) > 1 {
		modelName = args[1]
	}
	logFile := filepath.Join(logDir, "vllm-"+modelName+".log")
	f, err := os.Create(logFile)
	if err != nil {
		return "", fmt.Errorf("create log file: %w", err)
	}

	pm.cmd = exec.Command(vllmBinary(), args...)
	pm.cmd.Stdout = f
	pm.cmd.Stderr = f

	if err := pm.cmd.Start(); err != nil {
		f.Close()
		pm.status = "error"
		return "", fmt.Errorf("start vLLM: %w", err)
	}

	pm.status = "loading"
	pm.modelName = modelName
	pm.logFile = logFile
	log.Printf("[process] started vLLM (pid=%d) raw=%s", pm.cmd.Process.Pid, raw)

	go func() {
		err := pm.cmd.Wait()
		f.Close()
		pm.mu.Lock()
		if err != nil {
			pm.status = "error"
			log.Printf("[process] vLLM exited: %v", err)
		} else {
			pm.status = "stopped"
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

	log.Printf("[process] downloading model %s from %s", model, url)

	var cmd *exec.Cmd
	if strings.Contains(url, "huggingface.co") {
		cmd = exec.Command("huggingface-cli", "download",
			model, "--local-dir", modelDir, "--local-dir-use-symlinks", "False")
	} else if strings.HasPrefix(url, "http://") || strings.HasPrefix(url, "https://") {
		cmd = exec.Command("wget", "-c", "-P", modelDir, url)
	} else {
		marker := filepath.Join(modelDir, ".downloaded")
		os.WriteFile(marker, []byte("placeholder\n"), 0644)
		return fmt.Sprintf("placeholder model %s at %s", model, modelDir), nil
	}

	output, err := cmd.CombinedOutput()
	if err != nil {
		return string(output), fmt.Errorf("download failed: %w", err)
	}
	os.WriteFile(filepath.Join(modelDir, ".downloaded"), []byte("ok\n"), 0644)
	return fmt.Sprintf("downloaded %s (%.1f GB)", model, dirSizeGB(modelDir)), nil
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

// splitArgs splits a raw command string into args, respecting quoted strings.
func splitArgs(raw string) []string {
	var args []string
	var cur strings.Builder
	inQuote := false
	for i := 0; i < len(raw); i++ {
		c := raw[i]
		if c == '"' {
			inQuote = !inQuote
			continue
		}
		if c == ' ' && !inQuote {
			if cur.Len() > 0 {
				args = append(args, cur.String())
				cur.Reset()
			}
			continue
		}
		cur.WriteByte(c)
	}
	if cur.Len() > 0 {
		args = append(args, cur.String())
	}
	return args
}

