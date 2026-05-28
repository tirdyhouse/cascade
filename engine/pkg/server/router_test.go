package server

import (
	"testing"

	"predict/engine/pkg/cluster"
)

type memoryCacheMeta struct {
	locations map[uint64]*cluster.CacheLocation
	err       error
}

func (m *memoryCacheMeta) LookupHash(hash uint64) (*cluster.CacheLocation, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.locations[hash], nil
}

func (m *memoryCacheMeta) RecordHash(loc *cluster.CacheLocation) error {
	if m.locations == nil {
		m.locations = make(map[uint64]*cluster.CacheLocation)
	}
	m.locations[loc.Hash] = loc
	return nil
}

func TestRouterLookupUsesMetadataBackend(t *testing.T) {
	want := &cluster.CacheLocation{
		Hash:     0x1234,
		NodeID:   "node-a",
		IP:       "10.0.0.1",
		DiskPath: "/mnt/nvme0",
		FilePath: "12/34/block",
	}
	router := NewRouter(NewRegistry(), NewDiskTracker(), &memoryCacheMeta{
		locations: map[uint64]*cluster.CacheLocation{0x1234: want},
	})

	got, err := router.Lookup(0x1234)
	if err != nil {
		t.Fatalf("Lookup() error = %v", err)
	}
	if got != want {
		t.Fatalf("Lookup() = %+v, want backend location %+v", got, want)
	}

	missing, err := router.Lookup(0xbeef)
	if err != nil {
		t.Fatalf("Lookup(missing) error = %v", err)
	}
	if missing != nil {
		t.Fatalf("Lookup(missing) = %+v, want nil", missing)
	}
}

func TestRecommendTargetLocalNVMEPicksOnlineNodeWithMostFreeDisk(t *testing.T) {
	reg := NewRegistry()
	reg.Register(&cluster.NodeInfo{
		NodeID:    "node-small",
		IP:        "10.0.0.1",
		CacheMode: cluster.CacheModeLocalNVMe,
		Disks: []cluster.DiskInfo{{
			Path:   "/mnt/small",
			FreeGB: 50,
		}},
	})
	reg.Register(&cluster.NodeInfo{
		NodeID:    "node-large",
		IP:        "10.0.0.2",
		CacheMode: cluster.CacheModeLocalNVMe,
		Disks: []cluster.DiskInfo{{
			Path:   "/mnt/large-a",
			FreeGB: 100,
		}, {
			Path:   "/mnt/large-b",
			FreeGB: 200,
		}},
	})

	dt := NewDiskTracker()
	dt.Update("node-small", []cluster.DiskUsage{{Path: "/mnt/small", FreeGB: 50}})
	dt.Update("node-large", []cluster.DiskUsage{
		{Path: "/mnt/large-a", FreeGB: 100},
		{Path: "/mnt/large-b", FreeGB: 200},
	})
	router := NewRouter(reg, dt, &memoryCacheMeta{})

	got := router.RecommendTarget(cluster.CacheModeLocalNVMe, 0x1234abcd, 4096)
	if got == nil {
		t.Fatal("RecommendTarget() = nil, want location")
	}
	if got.NodeID != "node-large" || got.IP != "10.0.0.2" || got.DiskPath != "/mnt/large-b" {
		t.Fatalf("RecommendTarget() = %+v, want node-large /mnt/large-b", got)
	}
	if got.FilePath != "00/00/00001234abcd" {
		t.Fatalf("FilePath = %q, want partitioned hash path", got.FilePath)
	}
	if got.Size != 4096 {
		t.Fatalf("Size = %d, want 4096", got.Size)
	}
}

func TestRecommendTargetSharedPool(t *testing.T) {
	router := NewRouter(NewRegistry(), NewDiskTracker(), &memoryCacheMeta{})

	got := router.RecommendTarget(cluster.CacheModeSharedPool, 0xfeed, 512)
	if got == nil {
		t.Fatal("RecommendTarget(shared_pool) = nil, want location")
	}
	if got.SharedPath != "/mnt/shared/kv-cache/00/00/00000000feed" {
		t.Fatalf("SharedPath = %q, want formatted shared path", got.SharedPath)
	}
	if got.Hash != 0xfeed || got.Size != 512 {
		t.Fatalf("location = %+v, want hash and size preserved", got)
	}
}
