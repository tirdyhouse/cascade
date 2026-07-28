package agent

import (
	"os"
	"path/filepath"
	"testing"
)

func TestParseNvidiaValuesSupportsMultipleGPUs(t *testing.T) {
	values := parseNvidiaValues("25\n75\n")
	if len(values) != 2 || values[0] != 25 || values[1] != 75 {
		t.Fatalf("parseNvidiaValues() = %#v, want [25 75]", values)
	}
}

func TestParseVLLMQueueLenSumsRunningAndWaiting(t *testing.T) {
	metrics := `
# HELP vllm:num_requests_running Number of running requests.
vllm:num_requests_running 2
vllm:num_requests_waiting{engine="0"} 3
unrelated_metric 99
`
	if got := parseVLLMQueueLen(metrics); got != 5 {
		t.Fatalf("parseVLLMQueueLen() = %d, want 5", got)
	}
}

func TestCollectorUsesHostProcMetrics(t *testing.T) {
	procRoot := t.TempDir()
	if err := os.WriteFile(filepath.Join(procRoot, "meminfo"), []byte("MemTotal:       8192000 kB\nMemAvailable:  2048000 kB\n"), 0644); err != nil {
		t.Fatalf("write meminfo: %v", err)
	}
	if err := os.WriteFile(filepath.Join(procRoot, "stat"), []byte("cpu  100 0 100 800 0 0 0 0 0 0\n"), 0644); err != nil {
		t.Fatalf("write first stat: %v", err)
	}
	collector := NewCollector(&Config{})
	collector.procRoot = procRoot
	if got, want := collector.getMemUsed(), int64(6000); got != want {
		t.Fatalf("getMemUsed() = %d, want %d", got, want)
	}
	if got := collector.getCPULoad(); got != 0 {
		t.Fatalf("first getCPULoad() = %v, want 0 baseline", got)
	}
	if err := os.WriteFile(filepath.Join(procRoot, "stat"), []byte("cpu  200 0 200 1000 0 0 0 0 0 0\n"), 0644); err != nil {
		t.Fatalf("write second stat: %v", err)
	}
	if got, want := collector.getCPULoad(), 0.5; got != want {
		t.Fatalf("getCPULoad() = %v, want %v", got, want)
	}
}

func TestParseMemUsedRejectsMissingAvailableValue(t *testing.T) {
	if _, ok := parseMemUsedMB("MemTotal: 1024 kB\n"); ok {
		t.Fatal("parseMemUsedMB accepted incomplete meminfo")
	}
}
