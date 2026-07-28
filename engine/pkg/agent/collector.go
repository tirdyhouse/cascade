package agent

import (
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"

	"predict/engine/pkg/cluster"
)

// Collector gathers machine metrics for periodic status reporting.
type Collector struct {
	cfg *Config

	mu          sync.Mutex
	previousCPU cpuTimes
	hasCPU      bool
	procRoot    string
}

// NewCollector creates a new Collector.
func NewCollector(cfg *Config) *Collector {
	return &Collector{cfg: cfg, procRoot: "/proc"}
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
	values := parseNvidiaValues(string(out))
	if len(values) == 0 {
		return 0.0
	}
	var total float64
	for _, value := range values {
		total += value
	}
	return total / float64(len(values)) / 100.0
}

func (c *Collector) getGPUMemUsed() int64 {
	out, err := exec.Command("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return 0
	}
	var total int64
	for _, value := range parseNvidiaValues(string(out)) {
		total += int64(value)
	}
	return total
}

func parseNvidiaValues(output string) []float64 {
	values := make([]float64, 0)
	for _, line := range strings.Split(output, "\n") {
		value := strings.TrimSpace(strings.SplitN(line, ",", 2)[0])
		if value == "" {
			continue
		}
		parsed, err := strconv.ParseFloat(value, 64)
		if err == nil {
			values = append(values, parsed)
		}
	}
	return values
}

func (c *Collector) getMemUsed() int64 {
	// Linux reports host-wide memory through MemAvailable. Do not use the Go
	// heap here: the dashboard is an operator surface and must describe the
	// machine, not the agent process.
	if data, err := os.ReadFile(filepath.Join(c.procRoot, "meminfo")); err == nil {
		if used, ok := parseMemUsedMB(string(data)); ok {
			return used
		}
	}

	// /proc is unavailable on non-Linux development machines. Keep a bounded
	// fallback so the agent remains usable there; production Linux hosts always
	// take the branch above.
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return int64(m.Alloc / 1024 / 1024)
}

func (c *Collector) getCPULoad() float64 {
	data, err := os.ReadFile(filepath.Join(c.procRoot, "stat"))
	if err != nil {
		return 0
	}
	current, ok := parseCPUTimes(string(data))
	if !ok {
		return 0
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.hasCPU {
		c.previousCPU = current
		c.hasCPU = true
		return 0
	}
	previous := c.previousCPU
	c.previousCPU = current
	totalDelta := current.total - previous.total
	idleDelta := current.idle - previous.idle
	if totalDelta <= 0 || idleDelta < 0 || idleDelta > totalDelta {
		return 0
	}
	load := float64(totalDelta-idleDelta) / float64(totalDelta)
	if load < 0 {
		return 0
	}
	if load > 1 {
		return 1
	}
	return load
}

func (c *Collector) getDiskUsage() []cluster.DiskUsage {
	usage := make([]cluster.DiskUsage, 0, len(c.cfg.Disks))
	for _, d := range c.cfg.Disks {
		u, err := diskSpace(d.Path)
		if err != nil {
			// A configured mount that is absent is deliberately not fabricated
			// from command-line capacity hints. Its absence is visible through
			// the missing runtime entry and the agent log.
			continue
		}
		usage = append(usage, cluster.DiskUsage{
			Path:   d.Path,
			FreeGB: u.freeGB,
			UsedGB: u.usedGB,
		})
	}
	return usage
}

type diskFreeResult struct {
	totalGB int64
	freeGB  int64
	usedGB  int64
}

func (c *Collector) getDiskFree(path string) diskFreeResult {
	space, err := diskSpace(path)
	if err != nil {
		return diskFreeResult{}
	}
	return space
}

// DiskInfo reports actual filesystem capacity at registration time. The
// --disks flag selects the mount points; its historic capacity suffix is not a
// source of truth.
func (c *Collector) DiskInfo() []cluster.DiskInfo {
	info := make([]cluster.DiskInfo, 0, len(c.cfg.Disks))
	for _, disk := range c.cfg.Disks {
		space, err := diskSpace(disk.Path)
		if err != nil {
			continue
		}
		info = append(info, cluster.DiskInfo{
			Path:    disk.Path,
			TotalGB: space.totalGB,
			FreeGB:  space.freeGB,
			UsedGB:  space.usedGB,
		})
	}
	return info
}

func diskSpace(path string) (diskFreeResult, error) {
	var stat syscall.Statfs_t
	if err := syscall.Statfs(path, &stat); err != nil {
		return diskFreeResult{}, err
	}

	blockSize := uint64(stat.Bsize)
	if blockSize == 0 {
		return diskFreeResult{}, syscall.EINVAL
	}
	totalBytes := uint64(stat.Blocks) * blockSize
	freeBytes := uint64(stat.Bavail) * blockSize
	if freeBytes > totalBytes {
		freeBytes = totalBytes
	}
	usedBytes := totalBytes - freeBytes
	const bytesPerGiB = uint64(1024 * 1024 * 1024)
	return diskFreeResult{
		totalGB: int64(totalBytes / bytesPerGiB),
		freeGB:  int64(freeBytes / bytesPerGiB),
		usedGB:  int64(usedBytes / bytesPerGiB),
	}, nil
}

func parseMemUsedMB(data string) (int64, bool) {
	var totalKB, availableKB int64
	var haveTotal, haveAvailable bool
	for _, line := range strings.Split(data, "\n") {
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		value, err := strconv.ParseInt(fields[1], 10, 64)
		if err != nil {
			continue
		}
		switch strings.TrimSuffix(fields[0], ":") {
		case "MemTotal":
			totalKB = value
			haveTotal = true
		case "MemAvailable":
			availableKB = value
			haveAvailable = true
		}
	}
	if !haveTotal || !haveAvailable || totalKB <= 0 || availableKB < 0 || availableKB > totalKB {
		return 0, false
	}
	return (totalKB - availableKB) / 1024, true
}

type cpuTimes struct {
	total uint64
	idle  uint64
}

func parseCPUTimes(data string) (cpuTimes, bool) {
	for _, line := range strings.Split(data, "\n") {
		fields := strings.Fields(line)
		if len(fields) < 5 || fields[0] != "cpu" {
			continue
		}
		var times cpuTimes
		for i, field := range fields[1:] {
			value, err := strconv.ParseUint(field, 10, 64)
			if err != nil {
				return cpuTimes{}, false
			}
			times.total += value
			// /proc/stat fields 4 (idle) and 5 (iowait) both describe time
			// unavailable to scheduled work and are treated as idle here.
			if i == 3 || i == 4 {
				times.idle += value
			}
		}
		return times, times.total > 0
	}
	return cpuTimes{}, false
}

// GetAvailableModels scans the agent-owned model directory. A model is ready
// only when the atomic, file-level readiness manifest exists; legacy markers
// remain visible but are never silently treated as serving-ready.
func (c *Collector) GetAvailableModels() []cluster.LocalModel {
	modelsDir := filepath.Join(c.cfg.WorkDir, "models")
	entries, err := os.ReadDir(modelsDir)
	if err != nil {
		return nil
	}
	var models []cluster.LocalModel
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		if strings.HasPrefix(e.Name(), ".partial-") {
			name := strings.TrimPrefix(e.Name(), ".partial-")
			if validateModelName(name) == nil {
				models = append(models, cluster.LocalModel{
					Name:   name,
					SizeGB: dirSizeGB(filepath.Join(modelsDir, e.Name())),
					Status: "partial",
				})
			}
			continue
		}
		if strings.HasPrefix(e.Name(), ".") {
			continue
		}
		modelDir := filepath.Join(modelsDir, e.Name())
		local := cluster.LocalModel{
			Name:   e.Name(),
			SizeGB: dirSizeGB(modelDir),
			Status: "invalid",
		}
		if state, err := readModelState(modelDir); err == nil && state.Name == e.Name() && len(state.Files) > 0 && state.TotalBytes > 0 {
			local.Status = "ready"
			local.SourceURL = state.SourceURL
		} else if _, legacyErr := os.Stat(filepath.Join(modelDir, legacyModelMarker)); legacyErr == nil {
			local.Status = "legacy_unverified"
		}
		models = append(models, local)
	}
	return models
}

func dirSizeGB(path string) float64 {
	var total int64
	filepath.Walk(path, func(_ string, fi os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if !fi.IsDir() {
			total += fi.Size()
		}
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
