package agent

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"predict/engine/pkg/cluster"
)

func TestDownloadModelVerifiesCatalogManifestBeforePublishing(t *testing.T) {
	files := map[string][]byte{
		"config.json":    []byte(`{"architectures":["UnitTest"]}`),
		"weights.bin":    []byte("verified weights"),
		"tokenizer.json": []byte(`{"version":"1.0"}`),
	}
	manifest := cluster.ModelManifest{Name: "unit-model"}
	for path, body := range files {
		hash := sha256.Sum256(body)
		manifest.Files = append(manifest.Files, cluster.ModelFile{
			Path:   path,
			Size:   int64(len(body)),
			SHA256: hex.EncodeToString(hash[:]),
		})
		manifest.TotalBytes += int64(len(body))
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/manifest" {
			_ = json.NewEncoder(w).Encode(manifest)
			return
		}
		const prefix = "/models/unit-model/"
		if strings.HasPrefix(r.URL.Path, prefix) {
			if body, ok := files[r.URL.Path[len(prefix):]]; ok {
				_, _ = w.Write(body)
				return
			}
		}
		http.NotFound(w, r)
	}))
	defer server.Close()

	workDir := t.TempDir()
	var updates []int32
	output, err := NewProcessManager(&Config{}).DownloadModel(
		context.Background(),
		"unit-model",
		server.URL+"/models/unit-model/",
		server.URL+"/manifest",
		workDir,
		func(progress int32, _ string) { updates = append(updates, progress) },
	)
	if err != nil {
		t.Fatalf("DownloadModel() error = %v, output=%s", err, output)
	}
	if len(updates) == 0 || updates[len(updates)-1] != 100 {
		t.Fatalf("progress = %#v, want terminal 100", updates)
	}
	modelDir, err := readyModelDir(workDir, "unit-model")
	if err != nil {
		t.Fatalf("readyModelDir() error = %v", err)
	}
	if got, want := filepath.Base(modelDir), "unit-model"; got != want {
		t.Fatalf("model directory = %q, want basename %q", modelDir, want)
	}

	models := NewCollector(&Config{WorkDir: workDir}).GetAvailableModels()
	if len(models) != 1 || models[0].Name != "unit-model" || models[0].Status != "ready" {
		t.Fatalf("available models = %#v, want verified ready unit-model", models)
	}
	if err := os.WriteFile(filepath.Join(workDir, "models", "unit-model", "weights.bin"), []byte("corrupted"), 0644); err != nil {
		t.Fatalf("corrupt verified model: %v", err)
	}
	if _, err := readyModelDir(workDir, "unit-model"); err == nil {
		t.Fatal("readyModelDir accepted a model whose files no longer match the manifest")
	}
}

func TestDownloadModelDoesNotPublishChecksumMismatch(t *testing.T) {
	body := []byte("actual bytes")
	manifest := cluster.ModelManifest{
		Name:       "broken-model",
		TotalBytes: int64(len(body)),
		Files: []cluster.ModelFile{{
			Path:   "weights.bin",
			Size:   int64(len(body)),
			SHA256: "0000000000000000000000000000000000000000000000000000000000000000",
		}},
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/manifest":
			_ = json.NewEncoder(w).Encode(manifest)
		case "/models/broken-model/weights.bin":
			_, _ = w.Write(body)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	workDir := t.TempDir()
	_, err := NewProcessManager(&Config{}).DownloadModel(context.Background(), "broken-model", server.URL+"/models/broken-model/", server.URL+"/manifest", workDir, nil)
	if err == nil {
		t.Fatal("DownloadModel() succeeded with a checksum mismatch")
	}
	if _, err := readyModelDir(workDir, "broken-model"); err == nil {
		t.Fatal("checksum-mismatched model was published as ready")
	}
}

func TestCollectorReportsPartialModelDirectory(t *testing.T) {
	workDir := t.TempDir()
	partial := filepath.Join(workDir, "models", ".partial-unit-model")
	if err := os.MkdirAll(partial, 0755); err != nil {
		t.Fatalf("create partial model: %v", err)
	}
	if err := os.WriteFile(filepath.Join(partial, "weights.bin"), []byte("partial"), 0644); err != nil {
		t.Fatalf("write partial model: %v", err)
	}

	models := NewCollector(&Config{WorkDir: workDir}).GetAvailableModels()
	if len(models) != 1 || models[0].Name != "unit-model" || models[0].Status != "partial" {
		t.Fatalf("available models = %#v, want partial unit-model", models)
	}
}
