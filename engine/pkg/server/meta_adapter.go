package server

import (
	"predict/engine/pkg/cluster"
	"predict/engine/pkg/metadata"
)

// MetaAdapter adapts the existing metadata.Store to the CacheMetaBackend interface.
// This allows the S端 to reuse the existing disk-cache engine as the KV cache
// metadata center for the entire cluster.
type MetaAdapter struct {
	store *metadata.Store
}

// NewMetaAdapter creates a MetaAdapter backed by the existing metadata store.
func NewMetaAdapter(path string) (*MetaAdapter, error) {
	store, err := metadata.Open(path)
	if err != nil {
		return nil, err
	}
	return &MetaAdapter{store: store}, nil
}

// LookupHash checks if a hash exists in the metadata store.
// For local_nvme mode, this returns the node that has this block.
// For shared_pool mode, this returns the shared path.
func (a *MetaAdapter) LookupHash(hash uint64) (*cluster.CacheLocation, error) {
	meta, err := a.store.Get(hash)
	if err != nil {
		return nil, err
	}
	if meta == nil {
		return nil, nil
	}

	return &cluster.CacheLocation{
		Hash:     meta.Hash,
		Size:     meta.Size,
		FilePath: meta.FilePath,
	}, nil
}

// RecordHash stores the mapping from hash to a cache location.
func (a *MetaAdapter) RecordHash(loc *cluster.CacheLocation) error {
	return a.store.Put(&metadata.BlockMeta{
		Hash:     loc.Hash,
		FilePath: loc.FilePath,
		Size:     loc.Size,
	})
}

// Close shuts down the underlying store.
func (a *MetaAdapter) Close() error {
	return a.store.Close()
}
