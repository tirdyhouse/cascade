package server

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"predict/engine/pkg/cluster"
)

// ModelRegistry manages models published by the control server. The scanned
// catalog and the configured catalog are retained separately: a periodic scan
// must not erase an operator-supplied source URL or integrity metadata.
type ModelRegistry struct {
	mu         sync.RWMutex
	modelsDir  string
	baseURL    string
	configured []cluster.ModelInfo
	models     []cluster.ModelInfo
}

func NewModelRegistry(modelsDir, baseURL string) *ModelRegistry {
	return &ModelRegistry{modelsDir: modelsDir, baseURL: strings.TrimRight(baseURL, "/")}
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
	r.configured = cloneModelInfos(models)
	r.mu.Unlock()
	log.Printf("[models] loaded %d configured entries from %s", len(models), path)
	return nil
}

// Scan refreshes the model catalog from disk and overlays configured metadata.
func (r *ModelRegistry) Scan() {
	r.mu.Lock()
	defer r.mu.Unlock()

	modelMap := make(map[string]cluster.ModelInfo)
	if r.modelsDir != "" {
		entries, err := os.ReadDir(r.modelsDir)
		if err != nil {
			log.Printf("[models] scan %s: %v", r.modelsDir, err)
		} else {
			for _, entry := range entries {
				if !entry.IsDir() || strings.HasPrefix(entry.Name(), ".") || !safeIdentifier(entry.Name()) {
					continue
				}
				name := entry.Name()
				modelMap[name] = cluster.ModelInfo{
					Name:              name,
					Path:              filepath.Join(r.modelsDir, name),
					DownloadURL:       r.modelDownloadURL(name),
					ManifestURL:       r.modelManifestURL(name),
					SupportsPrefix:    true,
					SupportsDiskCache: true,
					SizeGB:            dirSizeGB(filepath.Join(r.modelsDir, name)),
					Quantization:      detectQuantization(r.modelsDir, name),
				}
			}
		}
	}

	for _, configured := range r.configured {
		if !safeIdentifier(configured.Name) {
			log.Printf("[models] ignoring configured model with invalid name %q", configured.Name)
			continue
		}
		if scanned, exists := modelMap[configured.Name]; exists {
			modelMap[configured.Name] = mergeModelInfo(scanned, configured)
		} else {
			modelMap[configured.Name] = configured
		}
	}

	result := make([]cluster.ModelInfo, 0, len(modelMap))
	for name, model := range modelMap {
		// Distribution readiness is a derived fact, not operator-supplied
		// decoration. A catalog row can exist without being safe to hand to an
		// Agent, but it must never be presented as downloadable in that state.
		model.DistributionReady = r.distributionReadyLocked(name, model)
		result = append(result, model)
	}
	sort.Slice(result, func(left, right int) bool { return result[left].Name < result[right].Name })
	r.models = result
	log.Printf("[models] scan complete: %d models from %s", len(result), r.modelsDir)
}

func mergeModelInfo(scanned, configured cluster.ModelInfo) cluster.ModelInfo {
	if configured.DownloadURL != "" {
		scanned.DownloadURL = configured.DownloadURL
	}
	if configured.ManifestURL != "" {
		scanned.ManifestURL = configured.ManifestURL
	}
	if configured.DefaultGPUMem != "" {
		scanned.DefaultGPUMem = configured.DefaultGPUMem
	}
	if configured.Quantization != "" {
		scanned.Quantization = configured.Quantization
	}
	if configured.Path != "" {
		scanned.Path = configured.Path
	}
	if configured.SizeGB > 0 {
		scanned.SizeGB = configured.SizeGB
	}
	// A configured false is meaningful only for an entry without a scanned
	// directory; scanned local models always support the connector contract.
	if configured.SupportsPrefix {
		scanned.SupportsPrefix = true
	}
	if configured.SupportsDiskCache {
		scanned.SupportsDiskCache = true
	}
	return scanned
}

func (r *ModelRegistry) modelDownloadURL(name string) string {
	if r.baseURL == "" {
		return ""
	}
	endpoint, err := url.JoinPath(r.baseURL+"/", name)
	if err != nil {
		return ""
	}
	return endpoint + "/"
}

func (r *ModelRegistry) modelManifestURL(name string) string {
	if r.baseURL == "" {
		return ""
	}
	endpoint, err := url.Parse(r.baseURL)
	if err != nil || endpoint.Scheme == "" || endpoint.Host == "" {
		return ""
	}
	path := strings.TrimRight(endpoint.Path, "/")
	index := strings.LastIndex(path, "/models")
	if index < 0 || path[index:] != "/models" {
		return ""
	}
	endpoint.Path = path[:index] + "/api/v1/models/" + url.PathEscape(name) + "/manifest"
	endpoint.RawQuery = ""
	return endpoint.String()
}

// distributionReadyLocked reports whether an Agent can obtain this model via
// a catalog-controlled, integrity-verifiable source. Callers hold r.mu.
//
// There are exactly two supported source forms:
//   - a published local directory with the control server's matching manifest;
//   - an operator-configured HTTP(S) source paired with a file manifest, or a
//     Hugging Face repository snapshot handled by huggingface-cli.
//
// A generic URL alone deliberately remains visible in the catalog but is not
// dispatchable. Treating it as ready would make an index page, a single shard,
// or an unverified mirror look like a usable model.
func (r *ModelRegistry) distributionReadyLocked(name string, model cluster.ModelInfo) bool {
	if !safeIdentifier(name) || validateHTTPURL(model.DownloadURL, "download_url") != nil {
		return false
	}

	manifestURL := strings.TrimSpace(model.ManifestURL)
	if manifestURL == "" {
		return isHuggingFaceRepository(model.DownloadURL)
	}
	if validateHTTPURL(manifestURL, "manifest_url") != nil {
		return false
	}

	// The default scanned model is fully owned by this control server. Keep the
	// check exact: if an operator changes either endpoint, it becomes an
	// external distribution and therefore needs its own manifest contract.
	if model.DownloadURL == r.modelDownloadURL(name) && model.ManifestURL == r.modelManifestURL(name) {
		return r.isLocallyPublishedLocked(name, model)
	}
	return true
}

func (r *ModelRegistry) isLocallyPublishedLocked(name string, model cluster.ModelInfo) bool {
	if r.modelsDir == "" {
		return false
	}
	root, err := filepath.Abs(r.modelsDir)
	if err != nil {
		return false
	}
	modelDir := filepath.Join(root, name)
	if !pathWithin(root, modelDir) || strings.TrimSpace(model.Path) == "" {
		return false
	}
	configuredPath, err := filepath.Abs(model.Path)
	if err != nil || filepath.Clean(configuredPath) != filepath.Clean(modelDir) {
		return false
	}
	info, err := os.Stat(modelDir)
	return err == nil && info.IsDir()
}

func isHuggingFaceRepository(raw string) bool {
	endpoint, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || !strings.EqualFold(endpoint.Scheme, "https") || endpoint.User != nil {
		return false
	}
	host := strings.ToLower(endpoint.Hostname())
	if host != "huggingface.co" && host != "www.huggingface.co" {
		return false
	}
	parts := strings.Split(strings.Trim(endpoint.Path, "/"), "/")
	return len(parts) >= 2 && safeIdentifier(parts[0]) && safeIdentifier(parts[1])
}

func (r *ModelRegistry) List() []cluster.ModelInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return cloneModelInfos(r.models)
}

func (r *ModelRegistry) Find(name string) *cluster.ModelInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, model := range r.models {
		if model.Name == name {
			copy := model
			return &copy
		}
	}
	return nil
}

// Manifest builds a file-level integrity manifest only for a model physically
// published by this control server. Configured external sources intentionally
// do not get a fabricated manifest.
func (r *ModelRegistry) Manifest(name string) (cluster.ModelManifest, error) {
	if !safeIdentifier(name) {
		return cluster.ModelManifest{}, fmt.Errorf("invalid model name %q", name)
	}
	r.mu.RLock()
	root := r.modelsDir
	model := r.findLocked(name)
	r.mu.RUnlock()
	if root == "" || model == nil {
		return cluster.ModelManifest{}, os.ErrNotExist
	}
	root, err := filepath.Abs(root)
	if err != nil {
		return cluster.ModelManifest{}, err
	}
	modelDir := filepath.Join(root, name)
	if !pathWithin(root, modelDir) {
		return cluster.ModelManifest{}, errors.New("model directory escapes published models root")
	}
	modelPath, err := filepath.Abs(model.Path)
	if err != nil || model.Path == "" || filepath.Clean(modelPath) != filepath.Clean(modelDir) {
		return cluster.ModelManifest{}, errors.New("model is not a locally published directory")
	}
	return buildModelManifest(name, modelDir)
}

func (r *ModelRegistry) findLocked(name string) *cluster.ModelInfo {
	for _, model := range r.models {
		if model.Name == name {
			copy := model
			return &copy
		}
	}
	return nil
}

func buildModelManifest(name, modelDir string) (cluster.ModelManifest, error) {
	manifest := cluster.ModelManifest{Name: name, CreatedAt: time.Now().UnixNano()}
	err := filepath.WalkDir(modelDir, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			return nil
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return fmt.Errorf("published model contains unsupported symlink %s", path)
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("published model contains non-regular file %s", path)
		}
		relative, err := filepath.Rel(modelDir, path)
		if err != nil {
			return fmt.Errorf("resolve model file path %s: %w", path, err)
		}
		if !pathWithin(modelDir, path) {
			return fmt.Errorf("model file path escapes published model: %s", path)
		}
		file, err := os.Open(path)
		if err != nil {
			return err
		}
		hash := sha256.New()
		_, copyErr := io.Copy(hash, file)
		closeErr := file.Close()
		if copyErr != nil {
			return copyErr
		}
		if closeErr != nil {
			return closeErr
		}
		manifest.Files = append(manifest.Files, cluster.ModelFile{
			Path:   filepath.ToSlash(relative),
			Size:   info.Size(),
			SHA256: hex.EncodeToString(hash.Sum(nil)),
		})
		manifest.TotalBytes += info.Size()
		return nil
	})
	if err != nil {
		return cluster.ModelManifest{}, fmt.Errorf("build model manifest: %w", err)
	}
	if len(manifest.Files) == 0 || manifest.TotalBytes <= 0 {
		return cluster.ModelManifest{}, errors.New("published model has no non-empty files")
	}
	return manifest, nil
}

func cloneModelInfos(models []cluster.ModelInfo) []cluster.ModelInfo {
	result := make([]cluster.ModelInfo, len(models))
	copy(result, models)
	return result
}

func pathWithin(root, candidate string) bool {
	relative, err := filepath.Rel(root, candidate)
	if err != nil {
		return false
	}
	return relative != ".." && !strings.HasPrefix(relative, ".."+string(filepath.Separator)) && !filepath.IsAbs(relative)
}

func dirSizeGB(path string) float64 {
	var total int64
	_ = filepath.Walk(path, func(_ string, info os.FileInfo, err error) error {
		if err == nil && info != nil && !info.IsDir() {
			total += info.Size()
		}
		return nil
	})
	return float64(total) / (1024 * 1024 * 1024)
}

func detectQuantization(modelsDir, name string) string {
	configPath := filepath.Join(modelsDir, name, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return ""
	}
	var config struct {
		QuantizationConfig *struct {
			QuantMethod string `json:"quant_method"`
		} `json:"quantization_config"`
	}
	if err := json.Unmarshal(data, &config); err != nil {
		return ""
	}
	if config.QuantizationConfig != nil {
		return config.QuantizationConfig.QuantMethod
	}
	return ""
}
