package server

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"predict/engine/pkg/agent"
	"predict/engine/pkg/cluster"
)

func TestModelRegistryPublishesManifestForScannedModel(t *testing.T) {
	modelsDir := t.TempDir()
	modelDir := filepath.Join(modelsDir, "unit-model")
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		t.Fatalf("create model dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "config.json"), []byte(`{"quantization_config":{"quant_method":"awq"}}`), 0644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "weights.bin"), []byte("weights"), 0644); err != nil {
		t.Fatalf("write weights: %v", err)
	}

	registry := NewModelRegistry(modelsDir, "http://control.example:18090/models/")
	registry.Scan()
	model := registry.Find("unit-model")
	if model == nil {
		t.Fatal("scanned model not found")
	}
	if got, want := model.DownloadURL, "http://control.example:18090/models/unit-model/"; got != want {
		t.Fatalf("DownloadURL = %q, want %q", got, want)
	}
	if got, want := model.ManifestURL, "http://control.example:18090/api/v1/models/unit-model/manifest"; got != want {
		t.Fatalf("ManifestURL = %q, want %q", got, want)
	}
	if model.Quantization != "awq" {
		t.Fatalf("Quantization = %q, want awq", model.Quantization)
	}
	if !model.DistributionReady {
		t.Fatalf("DistributionReady = false, want true for published local model")
	}
	manifest, err := registry.Manifest("unit-model")
	if err != nil {
		t.Fatalf("Manifest() error = %v", err)
	}
	if len(manifest.Files) != 2 || manifest.TotalBytes <= 0 {
		t.Fatalf("manifest = %+v", manifest)
	}
}

func TestModelRegistryMarksOnlyVerifiableSourcesReady(t *testing.T) {
	registry := NewModelRegistry("", "")
	registry.configured = []cluster.ModelInfo{
		{Name: "generic", DownloadURL: "https://models.example/generic/"},
		{Name: "manifested", DownloadURL: "https://models.example/manifested/", ManifestURL: "https://models.example/manifested.json"},
		{Name: "huggingface", DownloadURL: "https://huggingface.co/org/model"},
	}
	registry.Scan()

	if model := registry.Find("generic"); model == nil || model.DistributionReady {
		t.Fatalf("generic catalog model = %+v, want not distribution-ready", model)
	}
	if model := registry.Find("manifested"); model == nil || !model.DistributionReady {
		t.Fatalf("manifest catalog model = %+v, want distribution-ready", model)
	}
	if model := registry.Find("huggingface"); model == nil || !model.DistributionReady {
		t.Fatalf("Hugging Face catalog model = %+v, want distribution-ready", model)
	}
}

func TestEnrichDownloadUsesOnlyVerifiedCatalogEndpoints(t *testing.T) {
	registry := NewModelRegistry("", "")
	registry.configured = []cluster.ModelInfo{
		{
			Name:        "verified",
			DownloadURL: "https://models.example/verified/",
			ManifestURL: "https://models.example/verified-manifest.json",
		},
		{Name: "generic", DownloadURL: "https://models.example/generic/"},
	}
	registry.Scan()
	server := &Server{models: registry}

	req := &cluster.DispatchReq{Action: cluster.CmdDownloadModel, Params: map[string]string{"model": "verified"}}
	if err := server.enrichDispatchRequest(req); err != nil {
		t.Fatalf("enrich verified model: %v", err)
	}
	if got, want := req.Params["download_url"], "https://models.example/verified/"; got != want {
		t.Fatalf("download URL = %q, want %q", got, want)
	}
	if got, want := req.Params["manifest_url"], "https://models.example/verified-manifest.json"; got != want {
		t.Fatalf("manifest URL = %q, want %q", got, want)
	}

	override := &cluster.DispatchReq{Action: cluster.CmdDownloadModel, Params: map[string]string{
		"model":        "verified",
		"download_url": "https://attacker.example/model/",
	}}
	if err := server.enrichDispatchRequest(override); err == nil {
		t.Fatal("enrich accepted an arbitrary download source")
	}

	generic := &cluster.DispatchReq{Action: cluster.CmdDownloadModel, Params: map[string]string{"model": "generic"}}
	if err := server.enrichDispatchRequest(generic); err == nil {
		t.Fatal("enrich accepted a generic URL without a manifest")
	}
}

func TestAdminRPCAlsoAppliesModelCatalogPolicy(t *testing.T) {
	registry := NewModelRegistry("", "")
	registry.configured = []cluster.ModelInfo{{Name: "generic", DownloadURL: "https://models.example/generic/"}}
	registry.Scan()
	server := &Server{models: registry}
	service := &adminSvc{disp: NewDispatcher(NewRegistry()), reg: NewRegistry(), enrich: server.enrichDispatchRequest}
	reply := cluster.OK{}
	if err := service.DispatchCommand(context.Background(), &cluster.DispatchReq{
		Action: cluster.CmdDownloadModel,
		Params: map[string]string{"model": "generic"},
		Target: "node-a",
	}, &reply); err != nil {
		t.Fatalf("AdminService.DispatchCommand() error = %v", err)
	}
	if reply.OK || reply.Err == "" {
		t.Fatalf("AdminService reply = %+v, want catalog policy rejection", reply)
	}
}

func TestCatalogRoutesDistributeAndVerifyAModel(t *testing.T) {
	modelsDir := t.TempDir()
	modelDir := filepath.Join(modelsDir, "unit-model")
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		t.Fatalf("create model directory: %v", err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "config.json"), []byte(`{"model_type":"unit"}`), 0644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "weights.bin"), []byte("verified weights"), 0644); err != nil {
		t.Fatalf("write weights: %v", err)
	}

	var control *Server
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/v1/models/{name}/manifest", func(w http.ResponseWriter, r *http.Request) {
		control.apiModelManifest(w, r)
	})
	mux.Handle("/models/", http.StripPrefix("/models/", http.FileServer(http.Dir(modelsDir))))
	published := httptest.NewServer(mux)
	defer published.Close()

	registry := NewModelRegistry(modelsDir, published.URL+"/models/")
	registry.Scan()
	control = &Server{models: registry}
	model := registry.Find("unit-model")
	if model == nil || !model.DistributionReady {
		t.Fatalf("published model = %+v, want distribution-ready", model)
	}

	workDir := t.TempDir()
	output, err := agent.NewProcessManager(&agent.Config{}).DownloadModel(
		context.Background(), model.Name, model.DownloadURL, model.ManifestURL, workDir, nil,
	)
	if err != nil {
		t.Fatalf("catalog download failed: %v, output=%s", err, output)
	}
	for _, filename := range []string{"config.json", "weights.bin", ".cascade-model.json"} {
		if _, err := os.Stat(filepath.Join(workDir, "models", model.Name, filename)); err != nil {
			t.Fatalf("published Agent file %s: %v", filename, err)
		}
	}
}
