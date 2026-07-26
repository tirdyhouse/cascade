package agent

import (
	"encoding/json"
	"testing"

	"predict/engine/pkg/cluster"
)

func argValue(args []string, name string) string {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == name {
			return args[i+1]
		}
	}
	return ""
}

func hasArg(args []string, wanted string) bool {
	for _, arg := range args {
		if arg == wanted {
			return true
		}
	}
	return false
}

func TestBuildVLLMArgsSharedPoolUsesCentralMetadata(t *testing.T) {
	root := t.TempDir()
	args, err := buildVLLMArgs(&StartOptions{
		Model:           "model",
		GPUUtil:         "0.9",
		DiskCache:       true,
		WorkDir:         t.TempDir(),
		CacheMode:       cluster.CacheModeSharedPool,
		CacheEngineAddr: "http://metadata.service:9100",
		SharedCacheRoot: root,
		SharedCacheID:   "nvfile-prod",
	})
	if err != nil {
		t.Fatalf("buildVLLMArgs() error = %v", err)
	}
	if !hasArg(args, "--no-enable-prefix-caching") {
		t.Fatalf("args = %q, want --no-enable-prefix-caching", args)
	}

	var config map[string]interface{}
	if err := json.Unmarshal([]byte(argValue(args, "--kv-transfer-config")), &config); err != nil {
		t.Fatalf("decode KV transfer config: %v", err)
	}
	if config["kv_connector_module_path"] != diskCacheConnectorModule {
		t.Fatalf("module = %q, want %q", config["kv_connector_module_path"], diskCacheConnectorModule)
	}
	extra, ok := config["kv_connector_extra_config"].(map[string]interface{})
	if !ok {
		t.Fatalf("extra config = %#v, want object", config["kv_connector_extra_config"])
	}
	if extra["disk_cache_path"] != root ||
		extra["disk_cache_engine_addr"] != "http://metadata.service:9100" ||
		extra["disk_cache_shared"] != true ||
		extra["disk_cache_shared_id"] != "nvfile-prod" {
		t.Fatalf("shared extra config = %#v", extra)
	}
}

func TestBuildVLLMArgsRejectsBuiltInPrefixCacheWithCascade(t *testing.T) {
	_, err := buildVLLMArgs(&StartOptions{
		DiskCache:     true,
		PrefixCaching: true,
		WorkDir:       t.TempDir(),
	})
	if err == nil {
		t.Fatal("buildVLLMArgs accepted conflicting prefix cache settings")
	}
}

func TestBuildVLLMArgsExposesConfiguredEndpoint(t *testing.T) {
	args, err := buildVLLMArgs(&StartOptions{
		Model:    "model",
		GPUUtil:  "0.9",
		WorkDir:  t.TempDir(),
		VLLMHost: "0.0.0.0",
		VLLMPort: 8012,
	})
	if err != nil {
		t.Fatalf("buildVLLMArgs() error = %v", err)
	}
	if got := argValue(args, "--host"); got != "0.0.0.0" {
		t.Fatalf("--host = %q, want 0.0.0.0", got)
	}
	if got := argValue(args, "--port"); got != "8012" {
		t.Fatalf("--port = %q, want 8012", got)
	}
}
