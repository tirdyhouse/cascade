package server

import (
	"testing"

	"predict/engine/pkg/cluster"
)

func sharedNode(nodeID, cacheID string) *cluster.NodeInfo {
	return &cluster.NodeInfo{
		NodeID:           nodeID,
		IP:               "10.0.0.1",
		CacheMode:        cluster.CacheModeSharedPool,
		SharedCacheID:    cacheID,
		CacheMetadataURL: "http://metadata.service:9100",
	}
}

func TestRegistryRejectsMismatchedSharedCacheID(t *testing.T) {
	registry := NewRegistry()
	first := registry.Register(sharedNode("node-a", "nvfile-a"))
	if !first.Accepted {
		t.Fatalf("first shared node rejected: %+v", first)
	}

	second := registry.Register(sharedNode("node-b", "nvfile-b"))
	if second.Accepted {
		t.Fatalf("mismatched shared node accepted: %+v", second)
	}
	if second.Reason == "" {
		t.Fatal("mismatched shared node rejection did not include a reason")
	}
	if summary := registry.Summary(); summary.TotalNodes != 1 {
		t.Fatalf("TotalNodes = %d, want 1", summary.TotalNodes)
	}
}

func TestRegistrySummaryDeduplicatesSharedCacheStats(t *testing.T) {
	registry := NewRegistry()
	for _, nodeID := range []string{"node-a", "node-b"} {
		if reply := registry.Register(sharedNode(nodeID, "nvfile-a")); !reply.Accepted {
			t.Fatalf("Register(%s) rejected: %+v", nodeID, reply)
		}
		registry.Heartbeat(&cluster.MachineStatus{
			NodeID:         nodeID,
			CacheBlocks:    9,
			CacheBytes:     1024,
			CacheHitRate:   75,
			CacheRetrieved: 30,
		})
	}

	summary := registry.Summary()
	if summary.TotalBlocks != 9 {
		t.Fatalf("TotalBlocks = %d, want deduplicated 9", summary.TotalBlocks)
	}
	if summary.TotalBytes != 1024 {
		t.Fatalf("TotalBytes = %d, want deduplicated 1024", summary.TotalBytes)
	}
	if summary.HitRate != 75 {
		t.Fatalf("HitRate = %v, want deduplicated 75", summary.HitRate)
	}
	if len(summary.Nodes) != 2 {
		t.Fatalf("node summaries = %d, want 2", len(summary.Nodes))
	}
	for _, node := range summary.Nodes {
		if node.CacheMode != cluster.CacheModeSharedPool || node.SharedCacheID != "nvfile-a" {
			t.Fatalf("node summary = %+v, want shared cache identity", node)
		}
	}
}
