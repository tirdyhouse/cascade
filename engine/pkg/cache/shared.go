package cache

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const (
	// SharedCacheMarkerFile lives at the root of a shared cache filesystem.
	// Every producer and consumer verifies it before using the cache.
	SharedCacheMarkerFile = ".cascade-shared-cache.json"

	// SharedCacheFormatVersion changes when the shared object layout or
	// publication contract becomes incompatible.
	SharedCacheFormatVersion = 1
)

// SharedCacheMarker binds a mounted cache root to one operator-provided
// logical cache ID. It lets a node reject a locally-created fallback
// directory or the wrong shared filesystem before it reads cache metadata.
type SharedCacheMarker struct {
	FormatVersion int    `json:"format_version"`
	SharedCacheID string `json:"shared_cache_id"`
}

// EnsureSharedCacheMarker creates the shared-root marker with create-if-absent
// semantics or validates the marker already present. Only the central metadata
// service should call this creation path.
func EnsureSharedCacheMarker(cacheRoot, sharedCacheID string) (SharedCacheMarker, error) {
	sharedCacheID = strings.TrimSpace(sharedCacheID)
	if sharedCacheID == "" {
		return SharedCacheMarker{}, fmt.Errorf("%w: shared cache id is required", ErrInvalidArgument)
	}
	rootAbs, err := filepath.Abs(cacheRoot)
	if err != nil {
		return SharedCacheMarker{}, fmt.Errorf("%w: resolve cache root: %v", ErrInvalidArgument, err)
	}
	info, err := os.Stat(rootAbs)
	if err != nil {
		return SharedCacheMarker{}, fmt.Errorf("stat shared cache root %q: %w", rootAbs, err)
	}
	if !info.IsDir() {
		return SharedCacheMarker{}, fmt.Errorf("%w: shared cache root %q is not a directory", ErrInvalidArgument, rootAbs)
	}

	markerPath := filepath.Join(rootAbs, SharedCacheMarkerFile)
	if marker, err := ReadSharedCacheMarker(rootAbs); err == nil {
		if marker.SharedCacheID != sharedCacheID {
			return SharedCacheMarker{}, fmt.Errorf(
				"%w: shared cache marker id %q does not match configured id %q",
				ErrInvalidArgument,
				marker.SharedCacheID,
				sharedCacheID,
			)
		}
		return marker, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return SharedCacheMarker{}, err
	}

	marker := SharedCacheMarker{
		FormatVersion: SharedCacheFormatVersion,
		SharedCacheID: sharedCacheID,
	}
	payload, err := json.Marshal(marker)
	if err != nil {
		return SharedCacheMarker{}, fmt.Errorf("encode shared cache marker: %w", err)
	}
	tmp, err := os.CreateTemp(rootAbs, ".cascade-shared-cache-marker-*")
	if err != nil {
		return SharedCacheMarker{}, fmt.Errorf("create shared cache marker temp file: %w", err)
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if _, err := tmp.Write(append(payload, '\n')); err != nil {
		tmp.Close()
		return SharedCacheMarker{}, fmt.Errorf("write shared cache marker: %w", err)
	}
	if err := tmp.Chmod(0644); err != nil {
		tmp.Close()
		return SharedCacheMarker{}, fmt.Errorf("chmod shared cache marker: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return SharedCacheMarker{}, fmt.Errorf("sync shared cache marker: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return SharedCacheMarker{}, fmt.Errorf("close shared cache marker: %w", err)
	}

	if err := os.Link(tmpPath, markerPath); err == nil {
		return marker, nil
	} else if !errors.Is(err, os.ErrExist) {
		return SharedCacheMarker{}, fmt.Errorf("publish shared cache marker: %w", err)
	}

	// Another metadata-service instance won the create race. Its marker must
	// still describe precisely the same cache pool.
	existing, err := ReadSharedCacheMarker(rootAbs)
	if err != nil {
		return SharedCacheMarker{}, err
	}
	if existing.SharedCacheID != sharedCacheID {
		return SharedCacheMarker{}, fmt.Errorf(
			"%w: shared cache marker id %q does not match configured id %q",
			ErrInvalidArgument,
			existing.SharedCacheID,
			sharedCacheID,
		)
	}
	return existing, nil
}

// ReadSharedCacheMarker reads and validates the marker on a mounted shared
// cache root. Consumers use this before connecting to the metadata service.
func ReadSharedCacheMarker(cacheRoot string) (SharedCacheMarker, error) {
	rootAbs, err := filepath.Abs(cacheRoot)
	if err != nil {
		return SharedCacheMarker{}, fmt.Errorf("%w: resolve cache root: %v", ErrInvalidArgument, err)
	}
	markerPath := filepath.Join(rootAbs, SharedCacheMarkerFile)
	info, err := os.Lstat(markerPath)
	if err != nil {
		return SharedCacheMarker{}, err
	}
	if !info.Mode().IsRegular() {
		return SharedCacheMarker{}, fmt.Errorf("%w: shared cache marker is not a regular file", ErrInvalidArgument)
	}
	payload, err := os.ReadFile(markerPath)
	if err != nil {
		return SharedCacheMarker{}, fmt.Errorf("read shared cache marker: %w", err)
	}
	var marker SharedCacheMarker
	if err := json.Unmarshal(payload, &marker); err != nil {
		return SharedCacheMarker{}, fmt.Errorf("%w: decode shared cache marker: %v", ErrInvalidArgument, err)
	}
	if marker.FormatVersion != SharedCacheFormatVersion {
		return SharedCacheMarker{}, fmt.Errorf(
			"%w: unsupported shared cache marker format %d",
			ErrInvalidArgument,
			marker.FormatVersion,
		)
	}
	if strings.TrimSpace(marker.SharedCacheID) == "" {
		return SharedCacheMarker{}, fmt.Errorf("%w: shared cache marker has empty id", ErrInvalidArgument)
	}
	return marker, nil
}

// ValidatePublishedChunkObjects verifies that every v2 metadata entry points
// at a complete, visible object below cacheRoot. A shared metadata service
// calls this before publishing metadata so consumers never discover a file
// that its storage mount cannot see yet.
//
// This validates publication visibility and containment, not the chunk-object
// binary format. Consumers continue to validate the immutable object header
// before injecting KV into a model.
func ValidatePublishedChunkObjects(cacheRoot string, objects []ChunkObject) error {
	if err := validateChunkObjects(objects); err != nil {
		return err
	}

	rootAbs, err := filepath.Abs(cacheRoot)
	if err != nil {
		return fmt.Errorf("%w: resolve cache root: %v", ErrInvalidArgument, err)
	}
	rootInfo, err := os.Stat(rootAbs)
	if err != nil {
		return fmt.Errorf("stat cache root %q: %w", rootAbs, err)
	}
	if !rootInfo.IsDir() {
		return fmt.Errorf("%w: cache root %q is not a directory", ErrInvalidArgument, rootAbs)
	}
	resolvedRoot, err := filepath.EvalSymlinks(rootAbs)
	if err != nil {
		return fmt.Errorf("resolve cache root %q: %w", rootAbs, err)
	}

	for i, obj := range objects {
		fullPath, err := rootedCachePath(rootAbs, obj.FilePath)
		if err != nil {
			return fmt.Errorf("%w: objects[%d]: %v", ErrInvalidArgument, i, err)
		}
		info, err := os.Stat(fullPath)
		if err != nil {
			return fmt.Errorf("published object %q is not visible: %w", obj.FilePath, err)
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("%w: objects[%d]: %q is not a regular file", ErrInvalidArgument, i, obj.FilePath)
		}
		if info.Size() != obj.Size {
			return fmt.Errorf(
				"published object %q size is %d, metadata declares %d",
				obj.FilePath,
				info.Size(),
				obj.Size,
			)
		}
		resolvedPath, err := filepath.EvalSymlinks(fullPath)
		if err != nil {
			return fmt.Errorf("resolve published object %q: %w", obj.FilePath, err)
		}
		rel, err := filepath.Rel(resolvedRoot, resolvedPath)
		if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			return fmt.Errorf("%w: objects[%d]: file_path escapes cache root", ErrInvalidArgument, i)
		}
	}
	return nil
}
