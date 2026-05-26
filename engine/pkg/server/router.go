package server

import (
	"log"

	"predict/engine/pkg/cluster"
)

// Router handles KV cache routing decisions.
// In local_nvme mode, it consults the registry for node IP + DiskTracker for best disk.
// In shared_pool mode, it returns the shared storage path.
type Router struct {
	registry    *Registry
	diskTracker *DiskTracker
	cacheMeta   CacheMetaBackend // interface to the existing disk-cache engine
}

// CacheMetaBackend abstracts the existing disk-cache engine's metadata queries.
type CacheMetaBackend interface {
	// LookupHash returns the location for a given hash.
	// Returns nil if not found.
	LookupHash(hash uint64) (*cluster.CacheLocation, error)

	// RecordHash stores the location for a hash.
	RecordHash(loc *cluster.CacheLocation) error
}

// NewRouter creates a Router.
func NewRouter(reg *Registry, dt *DiskTracker, meta CacheMetaBackend) *Router {
	return &Router{
		registry:    reg,
		diskTracker: dt,
		cacheMeta:   meta,
	}
}

// Lookup finds where a KV block is cached.
func (r *Router) Lookup(hash uint64) (*cluster.CacheLocation, error) {
	// Try metadata backend first
	loc, err := r.cacheMeta.LookupHash(hash)
	if err == nil && loc != nil {
		return loc, nil
	}

	// Not found
	return nil, nil
}

// RecommendTarget recommends where to store a new KV block.
// For local_nvme: picks the node+disk with most free space.
// For shared_pool: returns a shared path.
func (r *Router) RecommendTarget(cacheMode cluster.CacheMode, blockHash uint64, blockSize int64) *cluster.CacheLocation {
	switch cacheMode {
	case cluster.CacheModeLocalNVMe:
		return r.recommendLocalNVMe(blockHash, blockSize)
	case cluster.CacheModeSharedPool:
		return r.recommendSharedPool(blockHash, blockSize)
	default:
		log.Printf("[router] unknown cache mode: %s, falling back to local_nvme", cacheMode)
		return r.recommendLocalNVMe(blockHash, blockSize)
	}
}

func (r *Router) recommendLocalNVMe(blockHash uint64, blockSize int64) *cluster.CacheLocation {
	summary := r.registry.Summary()
	if summary == nil {
		return nil
	}

	var bestNode *cluster.NodeSummary
	var bestFree int64 = -1

	for _, node := range summary.Nodes {
		if node.Status != cluster.NodeOnline {
			continue
		}
		// Check this node's disk free space
		for _, d := range node.Disks {
			if d.FreeGB > bestFree {
				bestFree = d.FreeGB
				bestNode = &node
			}
		}
	}

	if bestNode == nil {
		return nil
	}

	// Get the best disk path for this node
	diskPath, _ := r.diskTracker.BestDisk(bestNode.NodeID)

	return &cluster.CacheLocation{
		Hash:     blockHash,
		Size:     blockSize,
		NodeID:   bestNode.NodeID,
		IP:       bestNode.IP,
		DiskPath: diskPath,
		FilePath: formatBlockPath(blockHash),
	}
}

func (r *Router) recommendSharedPool(blockHash uint64, blockSize int64) *cluster.CacheLocation {
	return &cluster.CacheLocation{
		Hash:       blockHash,
		Size:       blockSize,
		SharedPath: formatSharedPath(blockHash),
	}
}

func formatBlockPath(hash uint64) string {
	// Partition into sub-directories to avoid too many files per dir
	h := hashToString(hash)
	if len(h) >= 4 {
		return h[:2] + "/" + h[2:4] + "/" + h[4:]
	}
	return h
}

func formatSharedPath(hash uint64) string {
	return "/mnt/shared/kv-cache/" + formatBlockPath(hash)
}

func hashToString(hash uint64) string {
	return cluster.FormatHash(hash)
}
