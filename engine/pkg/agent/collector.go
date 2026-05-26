package agent

import (
	"fmt"
	"log"
	"os"
	"runtime"

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
	// Stub: real implementation would exec nvidia-smi or use NVML bindings.
	// For now return a default value.
	return 0.0
}

func (c *Collector) getGPUMemUsed() int64 {
	// Stub: real implementation would query nvidia-smi.
	return 0
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

// ── Utility functions ────────────────────────────────────────────────

func getOutboundIP() string {
	// Try environment first
	if ip := os.Getenv("C_AGENT_IP"); ip != "" {
		return ip
	}
	hostname, err := os.Hostname()
	if err != nil {
		return "127.0.0.1"
	}
	return hostname
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
