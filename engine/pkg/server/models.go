package server

import (
	"encoding/json"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"predict/engine/pkg/cluster"
)

// ModelRegistry manages models on S端.
// Models are auto-discovered from a directory + optionally loaded from a JSON file.
// Directory format: each subdirectory = one model.
// JSON can supplement with metadata (download_url etc).
type ModelRegistry struct {
	mu        sync.RWMutex
	modelsDir string      // scanned for model directories
	baseURL   string      // base URL for download links (e.g. http://S_IP:18080/models/)
	extra     []cluster.ModelInfo // from --models-file (supplemental)
}

// NewModelRegistry creates a ModelRegistry.
func NewModelRegistry(modelsDir, baseURL string) *ModelRegistry {
	return &ModelRegistry{
		modelsDir: modelsDir,
		baseURL:   baseURL,
	}
}

// LoadFromFile loads supplemental model metadata from JSON.
func (r *ModelRegistry) LoadFromFile(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var models []cluster.ModelInfo
	if err := json.Unmarshal(data, &models); err != nil {
		return err
	}
	r.mu.Lock()
	r.extra = models
	r.mu.Unlock()
	log.Printf("[models] loaded %d models from %s", len(models), path)
	return nil
}

// Scan refreshes the model list from disk.
func (r *ModelRegistry) Scan() {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Build map: name → model
	modelMap := make(map[string]*cluster.ModelInfo)

	// 1. Scan directory
	if r.modelsDir != "" {
		entries, err := os.ReadDir(r.modelsDir)
		if err == nil {
			for _, e := range entries {
				if !e.IsDir() {
					continue
				}
				if strings.HasPrefix(e.Name(), ".") {
					continue
				}
				name := e.Name()
				sizeGB := dirSizeGB(filepath.Join(r.modelsDir, name))

				downloadURL := ""
				quant := detectQuantization(r.modelsDir, name)

				if r.baseURL != "" {
					downloadURL = r.baseURL + name + "/"
				}
				modelMap[name] = &cluster.ModelInfo{
					Name:             name,
					DownloadURL:      downloadURL,
					DefaultGPUMem:    "0.9",
					SupportsPrefix:   true,
					SupportsDiskCache: true,
					SizeGB:           sizeGB,
					Quantization:     quant,
				}
			}
		}
	}

	// 2. Merge JSON extras (override directory-scanned values)
	for _, m := range r.extra {
		if existing, ok := modelMap[m.Name]; ok {
			if m.DownloadURL != "" {
				existing.DownloadURL = m.DownloadURL
			}
			if m.DefaultGPUMem != "" {
				existing.DefaultGPUMem = m.DefaultGPUMem
			}
		} else {
			modelMap[m.Name] = &cluster.ModelInfo{
				Name:              m.Name,
				DownloadURL:       m.DownloadURL,
				DefaultGPUMem:     m.DefaultGPUMem,
				SupportsPrefix:    m.SupportsPrefix,
				SupportsDiskCache: m.SupportsDiskCache,
				Quantization:      m.Quantization,
			}
		}
	}

	// Convert to slice
	result := make([]cluster.ModelInfo, 0, len(modelMap))
	for _, m := range modelMap {
		result = append(result, *m)
	}

	r.extra = result // reuse extra field as the full list
	log.Printf("[models] scan complete: %d models from %s", len(result), r.modelsDir)
}

// List returns all models.
func (r *ModelRegistry) List() []cluster.ModelInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()
	result := make([]cluster.ModelInfo, len(r.extra))
	copy(result, r.extra)
	return result
}

// Find returns a model by name.
func (r *ModelRegistry) Find(name string) *cluster.ModelInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, m := range r.extra {
		if m.Name == name {
			return &m
		}
	}
	return nil
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
func detectQuantization(modelsDir, name string) string {
	cfgPath := filepath.Join(modelsDir, name, "config.json")
	data, err := os.ReadFile(cfgPath)
	if err != nil {
		return ""
	}
	var cfg struct {
		QuantizationConfig *struct {
			QuantMethod string `json:"quant_method"`
		} `json:"quantization_config"`
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return ""
	}
	if cfg.QuantizationConfig != nil && cfg.QuantizationConfig.QuantMethod != "" {
		return cfg.QuantizationConfig.QuantMethod
	}
	return ""
}

