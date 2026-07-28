package agent

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"predict/engine/pkg/cluster"
)

const (
	modelReadyMarker   = ".cascade-model.json"
	legacyModelMarker  = ".downloaded"
	maxManifestBytes   = 32 << 20
	modelDownloadLimit = 10 * time.Minute
)

type localModelState struct {
	Name       string              `json:"name"`
	SourceURL  string              `json:"source_url"`
	Files      []cluster.ModelFile `json:"files"`
	TotalBytes int64               `json:"total_bytes"`
	Completed  int64               `json:"completed_at"`
}

type downloadProgress func(progress int32, message string)

// DownloadModel makes a model available atomically. HTTP catalog downloads
// require a signed-by-content manifest (per-file SHA-256 and size); an explicit
// Hugging Face repository remains supported through its verified snapshot
// client. A generic URL is intentionally rejected because downloading an index
// page or one shard must never be reported as a usable model.
func (pm *ProcessManager) DownloadModel(ctx context.Context, model, downloadURL, manifestURL, workDir string, progress downloadProgress) (string, error) {
	if err := validateModelName(model); err != nil {
		return "", err
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if deadline, ok := ctx.Deadline(); !ok || time.Until(deadline) <= 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, modelDownloadLimit)
		defer cancel()
	}
	if err := pm.beginDownload(model); err != nil {
		return "", err
	}
	defer pm.endDownload(model)

	root, targetDir, stagingDir, err := modelPaths(workDir, model)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(root, 0755); err != nil {
		return "", fmt.Errorf("create models directory: %w", err)
	}

	if strings.TrimSpace(manifestURL) != "" {
		manifest, err := fetchModelManifest(ctx, manifestURL)
		if err != nil {
			return "", err
		}
		if manifest.Name != model {
			return "", fmt.Errorf("manifest model %q does not match requested model %q", manifest.Name, model)
		}
		if verified, err := manifestMatchesDirectory(targetDir, manifest); err != nil {
			return "", err
		} else if verified {
			if _, err := readModelState(targetDir); err != nil {
				if err := writeModelState(targetDir, localModelState{
					Name:       model,
					SourceURL:  downloadURL,
					Files:      manifest.Files,
					TotalBytes: manifest.TotalBytes,
					Completed:  time.Now().UnixNano(),
				}); err != nil {
					return "", err
				}
			}
			notify(progress, 100, "model already verified")
			return fmt.Sprintf("model %s is already verified at %s", model, targetDir), nil
		}
		notify(progress, 2, "validated model manifest")
		if err := ensureStagingDirectory(stagingDir); err != nil {
			return "", err
		}
		if err := downloadManifestFiles(ctx, stagingDir, downloadURL, manifest, progress); err != nil {
			return "", err
		}
		if err := writeModelState(stagingDir, localModelState{
			Name:       model,
			SourceURL:  downloadURL,
			Files:      manifest.Files,
			TotalBytes: manifest.TotalBytes,
			Completed:  time.Now().UnixNano(),
		}); err != nil {
			return "", err
		}
		if err := publishModel(stagingDir, targetDir, manifest); err != nil {
			return "", err
		}
		notify(progress, 100, "model verified and ready")
		return fmt.Sprintf("downloaded and verified %s (%.1f GiB)", model, dirSizeGB(targetDir)), nil
	}

	repo, ok := huggingFaceRepo(downloadURL)
	if !ok {
		return "", errors.New("model download requires a catalog manifest or a Hugging Face repository URL")
	}
	if ready, err := readyModelDir(workDir, model); err == nil && ready == targetDir {
		notify(progress, 100, "model is already ready")
		return fmt.Sprintf("model %s is already ready at %s", model, targetDir), nil
	}
	if err := ensureStagingDirectory(stagingDir); err != nil {
		return "", err
	}
	notify(progress, 5, "starting Hugging Face snapshot download")
	output, err := exec.CommandContext(ctx, "huggingface-cli", "download", repo, "--local-dir", stagingDir, "--local-dir-use-symlinks", "False").CombinedOutput()
	if err != nil {
		return string(output), fmt.Errorf("Hugging Face download failed: %w", err)
	}
	notify(progress, 85, "verifying downloaded snapshot")
	manifest, err := manifestFromDirectory(model, stagingDir)
	if err != nil {
		return string(output), err
	}
	if err := writeModelState(stagingDir, localModelState{
		Name:       model,
		SourceURL:  downloadURL,
		Files:      manifest.Files,
		TotalBytes: manifest.TotalBytes,
		Completed:  time.Now().UnixNano(),
	}); err != nil {
		return string(output), err
	}
	if err := publishModel(stagingDir, targetDir, manifest); err != nil {
		return string(output), err
	}
	notify(progress, 100, "model verified and ready")
	return fmt.Sprintf("downloaded and verified %s (%.1f GiB)", model, dirSizeGB(targetDir)), nil
}

func (pm *ProcessManager) beginDownload(model string) error {
	pm.downloadMu.Lock()
	defer pm.downloadMu.Unlock()
	if _, exists := pm.downloads[model]; exists {
		return fmt.Errorf("model %q is already downloading", model)
	}
	pm.downloads[model] = struct{}{}
	return nil
}

func (pm *ProcessManager) endDownload(model string) {
	pm.downloadMu.Lock()
	defer pm.downloadMu.Unlock()
	delete(pm.downloads, model)
}

func notify(progress downloadProgress, percent int32, message string) {
	if progress == nil {
		return
	}
	if percent < 0 {
		percent = 0
	}
	if percent > 100 {
		percent = 100
	}
	progress(percent, message)
}

func validateModelName(name string) error {
	name = strings.TrimSpace(name)
	if name == "" {
		return errors.New("model name is required")
	}
	if filepath.Base(name) != name || name == "." || name == ".." {
		return fmt.Errorf("model name must be a single directory name: %q", name)
	}
	for _, character := range name {
		if (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') ||
			(character >= '0' && character <= '9') || character == '.' || character == '_' || character == '-' {
			continue
		}
		return fmt.Errorf("model name contains an unsupported character: %q", name)
	}
	return nil
}

func modelPaths(workDir, model string) (root, target, staging string, err error) {
	if err = validateModelName(model); err != nil {
		return "", "", "", err
	}
	if strings.TrimSpace(workDir) == "" {
		return "", "", "", errors.New("work directory is required")
	}
	root, err = filepath.Abs(filepath.Join(workDir, "models"))
	if err != nil {
		return "", "", "", fmt.Errorf("resolve models directory: %w", err)
	}
	target = filepath.Join(root, model)
	staging = filepath.Join(root, ".partial-"+model)
	if !pathWithin(root, target) || !pathWithin(root, staging) {
		return "", "", "", errors.New("model path escapes agent models directory")
	}
	return root, target, staging, nil
}

func pathWithin(root, candidate string) bool {
	rel, err := filepath.Rel(root, candidate)
	if err != nil {
		return false
	}
	return rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) && !filepath.IsAbs(rel)
}

func readyModelDir(workDir, model string) (string, error) {
	_, target, _, err := modelPaths(workDir, model)
	if err != nil {
		return "", err
	}
	info, err := os.Stat(target)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", fmt.Errorf("model %q is not downloaded on this Agent", model)
		}
		return "", fmt.Errorf("inspect model %q: %w", model, err)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("model %q is not a directory", model)
	}
	state, err := readModelState(target)
	if err != nil {
		if _, legacyErr := os.Stat(filepath.Join(target, legacyModelMarker)); legacyErr == nil {
			return "", fmt.Errorf("model %q uses a legacy unverified marker; redistribute it through the control plane", model)
		}
		return "", fmt.Errorf("model %q is not verified and ready: %w", model, err)
	}
	if state.Name != model || len(state.Files) == 0 || state.TotalBytes <= 0 {
		return "", fmt.Errorf("model %q has an invalid readiness manifest", model)
	}
	manifest := cluster.ModelManifest{
		Name:       state.Name,
		Files:      state.Files,
		TotalBytes: state.TotalBytes,
	}
	if err := validateManifest(manifest); err != nil {
		return "", fmt.Errorf("model %q has an invalid readiness manifest: %w", model, err)
	}
	matches, err := manifestMatchesDirectory(target, manifest)
	if err != nil {
		return "", fmt.Errorf("verify model %q before serving: %w", model, err)
	}
	if !matches {
		return "", fmt.Errorf("model %q no longer matches its verified manifest; redistribute it before serving", model)
	}
	return target, nil
}

func readModelState(modelDir string) (localModelState, error) {
	data, err := os.ReadFile(filepath.Join(modelDir, modelReadyMarker))
	if err != nil {
		return localModelState{}, err
	}
	var state localModelState
	if err := json.Unmarshal(data, &state); err != nil {
		return localModelState{}, fmt.Errorf("decode readiness manifest: %w", err)
	}
	return state, nil
}

func writeModelState(modelDir string, state localModelState) error {
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return fmt.Errorf("encode readiness manifest: %w", err)
	}
	temporary := filepath.Join(modelDir, modelReadyMarker+".tmp")
	if err := os.WriteFile(temporary, data, 0644); err != nil {
		return fmt.Errorf("write readiness manifest: %w", err)
	}
	if err := os.Rename(temporary, filepath.Join(modelDir, modelReadyMarker)); err != nil {
		return fmt.Errorf("publish readiness manifest: %w", err)
	}
	// Retain the historic marker for external tooling, but only after the
	// stronger file-level manifest has been written successfully.
	if err := os.WriteFile(filepath.Join(modelDir, legacyModelMarker), []byte("verified\n"), 0644); err != nil {
		return fmt.Errorf("write compatibility model marker: %w", err)
	}
	return nil
}

func ensureStagingDirectory(path string) error {
	if info, err := os.Lstat(path); err == nil {
		if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
			return fmt.Errorf("model staging path is not a directory: %s", path)
		}
		return nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("inspect model staging path: %w", err)
	}
	if err := os.MkdirAll(path, 0755); err != nil {
		return fmt.Errorf("create model staging directory: %w", err)
	}
	return nil
}

func fetchModelManifest(ctx context.Context, manifestURL string) (cluster.ModelManifest, error) {
	parsed, err := url.Parse(manifestURL)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
		return cluster.ModelManifest{}, fmt.Errorf("invalid manifest URL %q", manifestURL)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, parsed.String(), nil)
	if err != nil {
		return cluster.ModelManifest{}, err
	}
	response, err := (&http.Client{Timeout: 30 * time.Second}).Do(req)
	if err != nil {
		return cluster.ModelManifest{}, fmt.Errorf("fetch model manifest: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return cluster.ModelManifest{}, fmt.Errorf("fetch model manifest: unexpected HTTP %d", response.StatusCode)
	}
	data, err := io.ReadAll(io.LimitReader(response.Body, maxManifestBytes+1))
	if err != nil {
		return cluster.ModelManifest{}, fmt.Errorf("read model manifest: %w", err)
	}
	if len(data) > maxManifestBytes {
		return cluster.ModelManifest{}, errors.New("model manifest exceeds maximum size")
	}
	var manifest cluster.ModelManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return cluster.ModelManifest{}, fmt.Errorf("decode model manifest: %w", err)
	}
	if err := validateManifest(manifest); err != nil {
		return cluster.ModelManifest{}, err
	}
	return manifest, nil
}

func validateManifest(manifest cluster.ModelManifest) error {
	if err := validateModelName(manifest.Name); err != nil {
		return fmt.Errorf("invalid manifest: %w", err)
	}
	if len(manifest.Files) == 0 {
		return errors.New("invalid manifest: no model files")
	}
	var total int64
	seen := make(map[string]struct{}, len(manifest.Files))
	for _, file := range manifest.Files {
		if err := validateManifestFile(file); err != nil {
			return err
		}
		if _, exists := seen[file.Path]; exists {
			return fmt.Errorf("invalid manifest: duplicate file %q", file.Path)
		}
		seen[file.Path] = struct{}{}
		total += file.Size
		if total < 0 {
			return errors.New("invalid manifest: total size overflow")
		}
	}
	if total != manifest.TotalBytes || total <= 0 {
		return fmt.Errorf("invalid manifest total size: got %d, want %d", manifest.TotalBytes, total)
	}
	return nil
}

func validateManifestFile(file cluster.ModelFile) error {
	if file.Path == "" || filepath.IsAbs(file.Path) || filepath.Clean(file.Path) != file.Path ||
		strings.HasPrefix(file.Path, ".."+string(filepath.Separator)) || strings.Contains(file.Path, "\\") {
		return fmt.Errorf("invalid manifest file path %q", file.Path)
	}
	if file.Size < 0 {
		return fmt.Errorf("invalid manifest size for %q", file.Path)
	}
	if len(file.SHA256) != 64 {
		return fmt.Errorf("invalid SHA-256 for %q", file.Path)
	}
	if _, err := hex.DecodeString(file.SHA256); err != nil {
		return fmt.Errorf("invalid SHA-256 for %q", file.Path)
	}
	return nil
}

func manifestMatchesDirectory(dir string, manifest cluster.ModelManifest) (bool, error) {
	info, err := os.Stat(dir)
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	if !info.IsDir() {
		return false, nil
	}
	for _, file := range manifest.Files {
		matches, err := fileMatches(filepath.Join(dir, filepath.FromSlash(file.Path)), file)
		if err != nil {
			return false, err
		}
		if !matches {
			return false, nil
		}
	}
	return true, nil
}

func downloadManifestFiles(ctx context.Context, stagingDir, downloadURL string, manifest cluster.ModelManifest, progress downloadProgress) error {
	base, err := url.Parse(downloadURL)
	if err != nil || (base.Scheme != "http" && base.Scheme != "https") || base.Host == "" {
		return fmt.Errorf("invalid model download URL %q", downloadURL)
	}
	completed := int64(0)
	verified := make(map[string]bool, len(manifest.Files))
	for _, file := range manifest.Files {
		if matches, err := fileMatches(filepath.Join(stagingDir, filepath.FromSlash(file.Path)), file); err != nil {
			return err
		} else if matches {
			verified[file.Path] = true
			completed += file.Size
		}
	}
	notify(progress, int32(2+completed*88/manifest.TotalBytes), "resuming verified model files")

	for _, file := range manifest.Files {
		target := filepath.Join(stagingDir, filepath.FromSlash(file.Path))
		if !verified[file.Path] {
			fileURL, err := url.JoinPath(base.String(), strings.Split(file.Path, "/")...)
			if err != nil {
				return fmt.Errorf("build download URL for %q: %w", file.Path, err)
			}
			if err := downloadOneFile(ctx, fileURL, target, file); err != nil {
				return err
			}
			completed += file.Size
		}
		if completed > manifest.TotalBytes {
			completed = manifest.TotalBytes
		}
		notify(progress, int32(2+completed*88/manifest.TotalBytes), "verified "+file.Path)
	}
	return nil
}

func downloadOneFile(ctx context.Context, sourceURL, target string, expected cluster.ModelFile) error {
	if !pathWithin(filepath.Dir(target), target) {
		return errors.New("model file path escapes staging directory")
	}
	if err := os.MkdirAll(filepath.Dir(target), 0755); err != nil {
		return fmt.Errorf("create model file directory: %w", err)
	}
	var offset int64
	if info, err := os.Lstat(target); err == nil {
		if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() {
			return fmt.Errorf("model file target is not a regular file: %s", target)
		}
		offset = info.Size()
		if offset > expected.Size {
			if err := os.Remove(target); err != nil {
				return fmt.Errorf("reset oversized partial file: %w", err)
			}
			offset = 0
		}
		if offset == expected.Size {
			// The caller already verified this file and only reaches this path
			// when its checksum differs. A range request at EOF would produce a
			// 416 response, so restart the corrupted partial file cleanly.
			if err := os.Remove(target); err != nil {
				return fmt.Errorf("reset corrupted partial file: %w", err)
			}
			offset = 0
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, sourceURL, nil)
	if err != nil {
		return err
	}
	if offset > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
	}
	response, err := (&http.Client{Timeout: 5 * time.Minute}).Do(req)
	if err != nil {
		return fmt.Errorf("download %s: %w", sourceURL, err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK && response.StatusCode != http.StatusPartialContent {
		return fmt.Errorf("download %s: unexpected HTTP %d", sourceURL, response.StatusCode)
	}
	appendMode := offset > 0 && response.StatusCode == http.StatusPartialContent
	if !appendMode {
		offset = 0
	}
	flags := os.O_CREATE | os.O_WRONLY
	if appendMode {
		flags |= os.O_APPEND
	} else {
		flags |= os.O_TRUNC
	}
	file, err := os.OpenFile(target, flags, 0644)
	if err != nil {
		return err
	}
	remaining := expected.Size - offset
	written, copyErr := io.Copy(file, io.LimitReader(response.Body, remaining+1))
	closeErr := file.Close()
	if copyErr != nil {
		return fmt.Errorf("write %s: %w", expected.Path, copyErr)
	}
	if closeErr != nil {
		return fmt.Errorf("close %s: %w", expected.Path, closeErr)
	}
	if written != remaining {
		return fmt.Errorf("download %s: got %d bytes, want %d", expected.Path, offset+written, expected.Size)
	}
	matches, err := fileMatches(target, expected)
	if err != nil {
		return err
	}
	if !matches {
		return fmt.Errorf("checksum mismatch for %s", expected.Path)
	}
	return nil
}

func fileMatches(path string, expected cluster.ModelFile) (bool, error) {
	info, err := os.Lstat(path)
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.Mode().IsRegular() || info.Size() != expected.Size {
		return false, nil
	}
	file, err := os.Open(path)
	if err != nil {
		return false, err
	}
	defer file.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return false, err
	}
	return strings.EqualFold(hex.EncodeToString(hash.Sum(nil)), expected.SHA256), nil
}

func publishModel(stagingDir, targetDir string, manifest cluster.ModelManifest) error {
	if matches, err := manifestMatchesDirectory(targetDir, manifest); err != nil {
		return err
	} else if matches {
		return nil
	}
	if _, err := os.Lstat(targetDir); err == nil {
		return fmt.Errorf("model destination %s already exists but is not a verified match; refusing to replace it", targetDir)
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	if err := os.Rename(stagingDir, targetDir); err != nil {
		return fmt.Errorf("publish verified model: %w", err)
	}
	return nil
}

func huggingFaceRepo(rawURL string) (string, bool) {
	parsed, err := url.Parse(rawURL)
	if err != nil || !strings.EqualFold(parsed.Scheme, "https") {
		return "", false
	}
	host := strings.ToLower(parsed.Hostname())
	if host != "huggingface.co" && host != "www.huggingface.co" {
		return "", false
	}
	parts := strings.Split(strings.Trim(parsed.Path, "/"), "/")
	if len(parts) < 2 || parts[0] == "" || parts[1] == "" {
		return "", false
	}
	return parts[0] + "/" + parts[1], true
}

func manifestFromDirectory(name, dir string) (cluster.ModelManifest, error) {
	manifest := cluster.ModelManifest{Name: name, CreatedAt: time.Now().UnixNano()}
	err := filepath.WalkDir(dir, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			return nil
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return fmt.Errorf("model download contains unsupported symlink %s", path)
		}
		relative, err := filepath.Rel(dir, path)
		if err != nil {
			return err
		}
		if relative == modelReadyMarker || relative == legacyModelMarker || strings.HasPrefix(relative, ".") {
			return nil
		}
		info, err := entry.Info()
		if err != nil {
			return err
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
	if err := validateManifest(manifest); err != nil {
		return cluster.ModelManifest{}, err
	}
	return manifest, nil
}
