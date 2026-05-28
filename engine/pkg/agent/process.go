package agent

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"
)

// vllmBinary returns the path to the vllm binary.
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
// pythonBin returns the venv Python interpreter path.
func pythonBin() string {
	for _, p := range []string{
		"/root/cascade/.venv-cascade/bin/python3",
		"python3",
	} {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return "python3"
}

// StartOptions holds parameters for starting vLLM.
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
	logName := filepath.Base(opts.Model)
	logFile := filepath.Join(logDir, "vllm-"+logName+".log")
	f, err := os.Create(logFile)
	if err != nil {
		return "", fmt.Errorf("create log file: %w", err)
	}

	// Build vLLM args — model is just the name, prepend local models dir
	localModel := opts.Model
	if opts.WorkDir != "" && !strings.HasPrefix(opts.Model, "/") {
		localModel = filepath.Join(opts.WorkDir, "models", opts.Model)
	}
	args := []string{"serve", localModel}
	args = append(args, "--gpu-memory-utilization", opts.GPUUtil)
	if opts.Quantization != "" {
		args = append(args, "--quantization", opts.Quantization)
		if opts.Quantization == "awq" {
			args = append(args, "--dtype", "float16")
		}
	}
	// Disk cache: ensure cache dir exists and pass --kv-transfer-config
	if opts.DiskCache {
		diskCachePath := filepath.Join(opts.WorkDir, "cache")
		os.MkdirAll(diskCachePath, 0755)
		kvConfig := map[string]interface{}{
			"kv_connector": "DiskCacheConnector",
			"kv_role":      "kv_both",
			"kv_connector_extra_config": map[string]string{
				"disk_cache_path": diskCachePath,
			},
		}
		kvJSON, _ := json.Marshal(kvConfig)
		args = append(args, "--kv-transfer-config", string(kvJSON))
	}
	pm.cmd = exec.Command(vllmBinary(), args...)
	pm.cmd.Stdout = f
	pm.cmd.Stderr = f
	pm.cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

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
			if exitErr, ok := err.(*exec.ExitError); ok && exitErr.Exited() && !exitErr.Success() {
				// Killed intentionally via Stop() — keep current status
				if pm.status != "stopped" {
					pm.status = "error"
				}
			} else {
				pm.status = "error"
			}
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
	rawParts := splitArgs(raw)
	// Prepend local models dir if model is a relative name
	if len(rawParts) > 0 && !strings.HasPrefix(rawParts[0], "/") && workDir != "" {
		modelPath := filepath.Join(workDir, "models", rawParts[0])
		if _, err := os.Stat(modelPath); err == nil {
			rawParts[0] = modelPath
		}
	}
	args = append(args, rawParts...)
	// Prepare log file
	logDir := filepath.Join(workDir, "logs")
	os.MkdirAll(logDir, 0755)
	modelName := ""
	if len(args) > 1 {
		modelName = args[1]
	}
	logName := filepath.Base(modelName)
	logFile := filepath.Join(logDir, "vllm-"+logName+".log")
	f, err := os.Create(logFile)
	if err != nil {
		return "", fmt.Errorf("create log file: %w", err)
	}

	pm.cmd = exec.Command(vllmBinary(), args...)
	pm.cmd.Stdout = f
	pm.cmd.Stderr = f
	pm.cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

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
// Stop terminates the vLLM process and all its children.
func (pm *ProcessManager) Stop() (string, error) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if pm.cmd == nil || pm.cmd.Process == nil {
		return "", fmt.Errorf("vLLM not running")
	}

	pid := pm.cmd.Process.Pid
	// Kill the entire process group (vLLM spawns EngineCore subprocesses
	// that hold GPU memory; killing only the parent leaves orphans).
	pgid, err := syscall.Getpgid(pid)
	if err == nil {
		syscall.Kill(-pgid, syscall.SIGTERM)
		// Give it a moment to exit cleanly, then force-kill
		done := make(chan struct{})
		go func() {
			pm.cmd.Wait()
			close(done)
		}()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			syscall.Kill(-pgid, syscall.SIGKILL)
		}
	} else {
		// Fallback: kill just the process
		pm.cmd.Process.Kill()
	}

	pm.status = "stopped"
	pm.modelName = ""
	log.Printf("[process] killed vLLM process group (pid=%d)", pid)
	return fmt.Sprintf("killed pid=%d (process group)", pid), nil
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

