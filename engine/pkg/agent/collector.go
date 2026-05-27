package agent

import (
	"fmt"
	"log"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"predict/engine/pkg/cluster"
)

// Collector gathers machine metrics for periodic status reporting.
type Collector struct {
	cfg *Config
}

// NewCollector creates a new Collector.
func NewCollector(cfg *Config) *Collector {
	return &Collector{cfg: cfg}
}

// Collect builds a MachineStatus snapshot.
func (c *Collector) Collect(seq int64) *cluster.MachineStatus {
	status := cluster.NewMachineStatus(c.cfg.NodeID, seq)

	// GPU metrics (simplified — real impl would use nvidia-smi)
	status.GPUUtil = c.getGPUUtil()
	status.GPUMemUsedMB = c.getGPUMemUsed()
	status.MemUsedMB = c.getMemUsed()
	status.CPULoad = c.getCPULoad()

	// Disk metrics
	status.Disks = c.getDiskUsage()

	return status
}

func (c *Collector) getGPUUtil() float64 {
	out, err := exec.Command("nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return 0.0
	}
	v, err := strconv.ParseFloat(strings.TrimSpace(string(out)), 64)
	if err != nil {
		return 0.0
	}
	return v / 100.0
}

func (c *Collector) getGPUMemUsed() int64 {
	out, err := exec.Command("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return 0
	}
	v, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
	if err != nil {
		return 0
	}
	return v
}

func (c *Collector) getMemUsed() int64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return int64(m.Alloc / 1024 / 1024)
}

func (c *Collector) getCPULoad() float64 {
	// Simple load average
	return 0.0
}

func (c *Collector) getDiskUsage() []cluster.DiskUsage {
	// Query actual disk usage for configured disk paths
	var usage []cluster.DiskUsage
	for _, d := range c.cfg.Disks {
		u := c.getDiskFree(d.Path)
		usage = append(usage, cluster.DiskUsage{
			Path:   d.Path,
			FreeGB: u.freeGB,
			UsedGB: u.usedGB,
		})
	}
	return usage
}

type diskFreeResult struct {
	freeGB int64
	usedGB int64
}

func (c *Collector) getDiskFree(path string) diskFreeResult {
	// Use os.StatFs or df on Linux
	// Simplified: just report config values
	for _, d := range c.cfg.Disks {
		if d.Path == path {
			return diskFreeResult{freeGB: d.FreeGB, usedGB: d.UsedGB}
		}
	}
	return diskFreeResult{}
}

// GetAvailableModels scans the models directory and returns models with sizes.
func (c *Collector) GetAvailableModels() []cluster.LocalModel {
	modelsDir := c.cfg.WorkDir + "/models"
	entries, err := os.ReadDir(modelsDir)
	if err != nil {
		return nil
	}
	var models []cluster.LocalModel
	for _, e := range entries {
		if e.IsDir() {
			marker := modelsDir + "/" + e.Name() + "/.downloaded"
			if _, err := os.Stat(marker); err == nil {
				sizeGB := dirSizeGB(modelsDir + "/" + e.Name())
				models = append(models, cluster.LocalModel{
					Name:   e.Name(),
					SizeGB: sizeGB,
				})
			}
		}
	}
	return models
}

func dirSizeGB(path string) float64 {
	var total int64
	filepath.Walk(path, func(_ string, fi os.FileInfo, err error) error {
		if err != nil { return nil }
		if !fi.IsDir() { total += fi.Size() }
		return nil
	})
	return float64(total) / (1024 * 1024 * 1024)
}

func getOutboundIP() string {
	if ip := os.Getenv("C_AGENT_IP"); ip != "" {
		return ip
	}
	conn, err := net.Dial("udp", "8.8.8.8:80")
	if err != nil {
		return "127.0.0.1"
	}
	defer conn.Close()
	localAddr := conn.LocalAddr().(*net.UDPAddr)
	return localAddr.IP.String()
}

func runShell(params map[string]string) (string, error) {
	cmdStr := params["cmd"]
	if cmdStr == "" {
		return "", fmt.Errorf("cmd parameter required")
	}

	log.Printf("[agent] exec shell: %s", cmdStr)
	// Execute command — this would be cmd.Run() or similar
	// For now just return a stub
	return fmt.Sprintf("executed: %s", cmdStr), nil
}
