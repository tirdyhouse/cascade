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
	done      chan struct{}
	status    string // "stopped" | "running" | "loading" | "error"
	modelName string
	logFile   string // path to current vLLM log file

	downloadMu sync.Mutex
	downloads  map[string]struct{}
}

const diskCacheConnectorModule = "adapter.vllm.connector_v21"

// NewProcessManager creates a ProcessManager.
func NewProcessManager(cfg *Config) *ProcessManager {
	return &ProcessManager{
		cfg:       cfg,
		status:    "stopped",
		downloads: make(map[string]struct{}),
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

	if opts == nil {
		return "", fmt.Errorf("start options are required")
	}
	if _, err := readyModelDir(opts.WorkDir, opts.Model); err != nil {
		return "", err
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

	vllmPath := strings.TrimSpace(opts.VLLMPath)
	if vllmPath == "" {
		vllmPath = vllmBinary()
	}
	cmd := exec.Command(vllmPath, args...)
	cmd.Stdout = f
	cmd.Stderr = f
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	if err := cmd.Start(); err != nil {
		f.Close()
		// Write the error to the log file so it shows in Live Log
		os.WriteFile(logFile, []byte(fmt.Sprintf("Failed to start vLLM: %v\n", err)), 0644)
		pm.status = "error"
		return fmt.Sprintf("Failed to start vLLM: %v", err), fmt.Errorf("start vLLM: %w", err)
	}

	pm.cmd = cmd
	pm.done = make(chan struct{})
	pm.status = "loading"
	pm.modelName = opts.Model
	pm.logFile = logFile
	log.Printf("[process] started vLLM (pid=%d) model=%s log=%s", cmd.Process.Pid, opts.Model, logFile)

	go pm.waitForExit(cmd, f)

	return fmt.Sprintf("started pid=%d log=%s", cmd.Process.Pid, logFile), nil
}

// waitForExit is the sole owner of cmd.Wait. Stop only signals the process
// group and waits on done; calling Wait from both paths races and can turn a
// clean operator stop into a false error state.
func (pm *ProcessManager) waitForExit(cmd *exec.Cmd, logFile *os.File) {
	err := cmd.Wait()
	_ = logFile.Close()

	pm.mu.Lock()
	defer pm.mu.Unlock()
	if pm.cmd != cmd {
		return
	}
	if err != nil {
		if pm.status != "stopped" {
			pm.status = "error"
		}
		log.Printf("[process] vLLM exited: %v", err)
	} else if pm.status != "stopped" {
		pm.status = "stopped"
		log.Println("[process] vLLM exited cleanly")
	}
	pm.modelName = ""
	pm.cmd = nil
	done := pm.done
	pm.done = nil
	if done != nil {
		close(done)
	}
}

// Stop terminates the vLLM process and all its children.
func (pm *ProcessManager) Stop() (string, error) {
	pm.mu.Lock()
	if pm.cmd == nil || pm.cmd.Process == nil {
		pm.mu.Unlock()
		return "", fmt.Errorf("vLLM not running")
	}

	pid := pm.cmd.Process.Pid
	cmd := pm.cmd
	done := pm.done
	// Mark the stop before signalling. The sole Wait owner preserves this
	// operator-requested state even when the process exits with SIGTERM.
	pm.status = "stopped"
	pm.modelName = ""
	pm.mu.Unlock()

	// Kill the entire process group (vLLM spawns EngineCore subprocesses
	// that hold GPU memory; killing only the parent leaves orphans).
	pgid, err := syscall.Getpgid(pid)
	if err == nil {
		_ = syscall.Kill(-pgid, syscall.SIGTERM)
		// Give it a moment to exit cleanly, then force-kill
		if done != nil {
			select {
			case <-done:
			case <-time.After(5 * time.Second):
				_ = syscall.Kill(-pgid, syscall.SIGKILL)
				select {
				case <-done:
				case <-time.After(time.Second):
					pm.markStopTimeout(cmd)
					return fmt.Sprintf("sent SIGTERM and SIGKILL to pid=%d", pid), fmt.Errorf("vLLM process did not exit after SIGKILL")
				}
			}
		}
	} else {
		// Fallback: kill just the process
		_ = cmd.Process.Kill()
		if done != nil {
			select {
			case <-done:
			case <-time.After(time.Second):
				pm.markStopTimeout(cmd)
				return fmt.Sprintf("sent kill to pid=%d", pid), fmt.Errorf("vLLM process did not exit")
			}
		}
	}

	log.Printf("[process] killed vLLM process group (pid=%d)", pid)
	return fmt.Sprintf("killed pid=%d (process group)", pid), nil
}

func (pm *ProcessManager) markStopTimeout(cmd *exec.Cmd) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	if pm.cmd == cmd {
		pm.status = "error"
	}
}

// LogFile returns the current vLLM log file path.
func (pm *ProcessManager) LogFile() string {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	return pm.logFile
}

// Status returns the current process status.
func (pm *ProcessManager) Status() string {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	return pm.status
}

// MarkRunning records a completed health check. Process creation alone is not
// considered a successful vLLM start because EngineCore can still fail during
// model initialization.
func (pm *ProcessManager) MarkRunning() {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	if pm.cmd != nil && pm.status == "loading" {
		pm.status = "running"
	}
}

// MarkError records an observed health-check failure for a process that has
// not yet exited. The exit watcher remains the sole owner of cleanup.
func (pm *ProcessManager) MarkError() {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	if pm.cmd != nil && pm.status != "stopped" {
		pm.status = "error"
	}
}

// ModelName returns the currently loaded model name.
func (pm *ProcessManager) ModelName() string {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	return pm.modelName
}
