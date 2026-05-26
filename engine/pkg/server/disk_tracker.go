package server

import "predict/engine/pkg/cluster"

// DiskSnapshot holds per-disk usage for a node.
type DiskSnapshot struct {
	NodeID string
	Disks  []cluster.DiskUsage
}

// DiskTracker tracks disk usage across all nodes.
// In local_nvme mode, the S端 uses this to recommend which disk to store new blocks on.
type DiskTracker struct {
	// diskFree: (node_id, disk_path) → free_gb
	diskFree map[string]map[string]int64
}

// NewDiskTracker creates a DiskTracker.
func NewDiskTracker() *DiskTracker {
	return &DiskTracker{
		diskFree: make(map[string]map[string]int64),
	}
}

// Update refreshes disk usage for a node from heartbeat data.
func (dt *DiskTracker) Update(nodeID string, disks []cluster.DiskUsage) {
	m, ok := dt.diskFree[nodeID]
	if !ok {
		m = make(map[string]int64)
		dt.diskFree[nodeID] = m
	}
	for _, d := range disks {
		m[d.Path] = d.FreeGB
	}
}

// BestDisk returns the disk path with the most free space for a node.
// Returns empty string if no disks tracked.
func (dt *DiskTracker) BestDisk(nodeID string) (path string, freeGB int64) {
	disks, ok := dt.diskFree[nodeID]
	if !ok {
		return "", 0
	}
	bestPath := ""
	bestFree := int64(-1)
	for p, f := range disks {
		if f > bestFree {
			bestFree = f
			bestPath = p
		}
	}
	return bestPath, bestFree
}

// Remove clears tracking data for a node (on unregister).
func (dt *DiskTracker) Remove(nodeID string) {
	delete(dt.diskFree, nodeID)
}
