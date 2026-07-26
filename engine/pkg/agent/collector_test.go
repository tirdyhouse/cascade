package agent

import "testing"

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
