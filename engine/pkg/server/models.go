package server

import (
	"encoding/json"
	"os"
	"sync"

	"predict/engine/pkg/cluster"
)

// ModelRegistry manages the list of models available on S端.
// Models can be loaded from a JSON file or registered programmatically.
type ModelRegistry struct {
	mu     sync.RWMutex
	models []cluster.ModelInfo
}

// NewModelRegistry creates a ModelRegistry.
func NewModelRegistry() *ModelRegistry {
	return &ModelRegistry{}
}

// LoadFromFile loads models from a JSON file.
// Format: [{"name":"Qwen-72B","download_url":"http://...","default_gpu_mem":"0.9","supports_prefix":true,"supports_disk_cache":true}]
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
	r.models = models
	r.mu.Unlock()
	return nil
}

// Register adds a model.
func (r *ModelRegistry) Register(m cluster.ModelInfo) {
	r.mu.Lock()
	r.models = append(r.models, m)
	r.mu.Unlock()
}

// List returns all registered models.
func (r *ModelRegistry) List() []cluster.ModelInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()
	result := make([]cluster.ModelInfo, len(r.models))
	copy(result, r.models)
	return result
}

// Find returns a model by name.
func (r *ModelRegistry) Find(name string) *cluster.ModelInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, m := range r.models {
		if m.Name == name {
			return &m
		}
	}
	return nil
}
