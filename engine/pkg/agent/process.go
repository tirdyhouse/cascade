package agent

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"predict/engine/pkg/cluster"
)

const (
	defaultVLLMHost = "0.0.0.0"
	defaultVLLMPort = 8000
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
type StartOptions struct {
	Model           string
	GPUUtil         string
	PrefixCaching   bool
	DiskCache       bool
	WorkDir         string
	Quantization    string // awq, gptq, etc.
	VLLMPath        string // path to vllm binary, auto-detected if empty
	CacheMode       cluster.CacheMode
	CacheEngineAddr string
	SharedCacheRoot string
	SharedCacheID   string
	VLLMHost        string
	VLLMPort        int
}

// ProcessManager handles vLLM process lifecycle.
type ProcessManager struct {
	cfg *Config

	mu        sync.Mutex
	cmd       *exec.Cmd
	status    string // "stopped" | "running" | "loading" | "error"
	modelName string
	logFile   string // path to current vLLM log file
}

const diskCacheConnectorModule = "adapter.vllm.connector_v21"

// NewProcessManager creates a ProcessManager.
func NewProcessManager(cfg *Config) *ProcessManager {
	return &ProcessManager{
		cfg:    cfg,
		status: "stopped",
	}
}

// buildVLLMArgs constructs the serving command without spawning a process.
// Keeping this separate makes the shared-cache contract testable before any
// GPU process is started.
func buildVLLMArgs(opts *StartOptions) ([]string, error) {
	if opts == nil {
		return nil, fmt.Errorf("start options are required")
	}
	if opts.DiskCache && opts.PrefixCaching {
		return nil, fmt.Errorf("vLLM prefix caching must be disabled when Cascade disk cache is enabled")
	}

	localModel := opts.Model
	if opts.WorkDir != "" && !strings.HasPrefix(opts.Model, "/") {
		localModel = filepath.Join(opts.WorkDir, "models", opts.Model)
	}
	args := []string{"serve", localModel, "--gpu-memory-utilization", opts.GPUUtil}
	vllmHost := strings.TrimSpace(opts.VLLMHost)
	if vllmHost == "" {
		vllmHost = defaultVLLMHost
	}
	vllmPort := opts.VLLMPort
	if vllmPort <= 0 {
		vllmPort = defaultVLLMPort
	}
	if vllmPort > 65535 {
		return nil, fmt.Errorf("invalid vLLM port %d", vllmPort)
	}
	args = append(args, "--host", vllmHost, "--port", strconv.Itoa(vllmPort))
	if opts.Quantization != "" {
		args = append(args, "--quantization", opts.Quantization)
		if opts.Quantization == "awq" {
			args = append(args, "--dtype", "float16")
		}
	}

	// Disk cache: local mode owns a node-private directory. Shared-pool mode
	// must use the already-mounted common root and one central metadata API.
	if !opts.DiskCache {
		return args, nil
	}

	diskCachePath := filepath.Join(opts.WorkDir, "cache")
	shared := opts.CacheMode == cluster.CacheModeSharedPool
	if shared {
		sharedRoot := strings.TrimSpace(opts.SharedCacheRoot)
		sharedID := strings.TrimSpace(opts.SharedCacheID)
		if sharedRoot == "" || sharedID == "" {
			return nil, fmt.Errorf("shared cache requires shared cache root and cache id")
		}
		if !filepath.IsAbs(sharedRoot) {
			return nil, fmt.Errorf("shared cache root must be absolute: %q", sharedRoot)
		}
		info, err := os.Stat(sharedRoot)
		if err != nil {
			return nil, fmt.Errorf("shared cache root is not accessible: %w", err)
		}
		if !info.IsDir() {
			return nil, fmt.Errorf("shared cache root is not a directory: %q", sharedRoot)
		}
		diskCachePath = sharedRoot
	} else if err := os.MkdirAll(diskCachePath, 0755); err != nil {
		return nil, fmt.Errorf("create local cache directory: %w", err)
	}

	engineAddr := strings.TrimSpace(opts.CacheEngineAddr)
	if engineAddr == "" {
		engineAddr = "http://127.0.0.1:9100"
	}
	extra := map[string]interface{}{
		"disk_cache_path":        diskCachePath,
		"disk_cache_engine_addr": engineAddr,
	}
	if shared {
		extra["disk_cache_shared"] = true
		extra["disk_cache_shared_id"] = strings.TrimSpace(opts.SharedCacheID)
	}
	kvConfig := map[string]interface{}{
		"kv_connector":              "DiskCacheConnector",
		"kv_role":                   "kv_both",
		"kv_connector_module_path":  diskCacheConnectorModule,
		"kv_connector_extra_config": extra,
	}
	kvJSON, err := json.Marshal(kvConfig)
	if err != nil {
		return nil, fmt.Errorf("encode KV transfer config: %w", err)
	}
	args = append(
		args,
		"--no-enable-prefix-caching",
		"--kv-transfer-config",
		string(kvJSON),
		"--enable-prompt-tokens-details",
	)
	return args, nil
}

// Start launches a vLLM serve process with the given options.
func (pm *ProcessManager) Start(opts *StartOptions) (string, error) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if pm.cmd != nil && pm.cmd.Process != nil {
		return "", fmt.Errorf("vLLM already running (model=%s)", pm.modelName)
	}

	args, err := buildVLLMArgs(opts)
	if err != nil {
		return "", err
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
	if !containsVLLMFlag(rawParts, "--host") {
		host := strings.TrimSpace(pm.cfg.VLLMHost)
		if host == "" {
			host = defaultVLLMHost
		}
		args = append(args, "--host", host)
	}
	if !containsVLLMFlag(rawParts, "--port") {
		port := pm.cfg.VLLMPort
		if port <= 0 {
			port = defaultVLLMPort
		}
		args = append(args, "--port", strconv.Itoa(port))
	}
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

func containsVLLMFlag(args []string, flag string) bool {
	for _, arg := range args {
		if arg == flag || strings.HasPrefix(arg, flag+"=") {
			return true
		}
	}
	return false
}
