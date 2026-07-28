package agent

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"predict/engine/pkg/cluster"
)

func TestDiagnosticsLogsReadAgentWorkDirAndHonorOffsets(t *testing.T) {
	workDir := t.TempDir()
	logDir := filepath.Join(workDir, "logs")
	if err := os.MkdirAll(logDir, 0755); err != nil {
		t.Fatalf("create logs: %v", err)
	}
	if err := os.WriteFile(filepath.Join(logDir, "vllm-model.log"), []byte("first\nsecond\nthird\n"), 0644); err != nil {
		t.Fatalf("write log: %v", err)
	}

	first := httptest.NewRecorder()
	handleDiagnosticsLogs(first, httptest.NewRequest(http.MethodGet, "/v1/logs?lines=2", nil), "node-a", workDir)
	if first.Code != http.StatusOK {
		t.Fatalf("first log response = %d: %s", first.Code, first.Body.String())
	}
	var firstChunk cluster.LogChunk
	if err := json.NewDecoder(first.Body).Decode(&firstChunk); err != nil {
		t.Fatalf("decode first chunk: %v", err)
	}
	if firstChunk.Lines != "first\nsecond" || firstChunk.Offset <= 0 || firstChunk.EOF {
		t.Fatalf("first chunk = %+v", firstChunk)
	}

	second := httptest.NewRecorder()
	handleDiagnosticsLogs(second, httptest.NewRequest(http.MethodGet, "/v1/logs?offset="+itoa(firstChunk.Offset)+"&lines=2", nil), "node-a", workDir)
	if second.Code != http.StatusOK {
		t.Fatalf("second log response = %d: %s", second.Code, second.Body.String())
	}
	var secondChunk cluster.LogChunk
	if err := json.NewDecoder(second.Body).Decode(&secondChunk); err != nil {
		t.Fatalf("decode second chunk: %v", err)
	}
	if secondChunk.Lines != "third" || !secondChunk.EOF {
		t.Fatalf("second chunk = %+v", secondChunk)
	}
}

func TestDiagnosticsLogsRejectPathTraversal(t *testing.T) {
	recorder := httptest.NewRecorder()
	handleDiagnosticsLogs(recorder, httptest.NewRequest(http.MethodGet, "/v1/logs?filename=../secret.log", nil), "node-a", t.TempDir())
	if recorder.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusNotFound)
	}
}

func itoa(value int64) string {
	return fmt.Sprintf("%d", value)
}
