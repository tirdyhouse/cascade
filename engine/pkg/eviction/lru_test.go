package eviction

import (
	"testing"
)

func TestLRURecordUpdatesSize(t *testing.T) {
	p := NewLRU(1000)

	// First record
	p.Record("key1", 100)
	if p.TotalBytes() != 100 {
		t.Fatalf("TotalBytes after first Record = %d, want 100", p.TotalBytes())
	}

	// Record same key with larger size — should update delta
	p.Record("key1", 200)
	if p.TotalBytes() != 200 {
		t.Fatalf("TotalBytes after size increase = %d, want 200", p.TotalBytes())
	}
	if p.Len() != 1 {
		t.Fatalf("Len after update = %d, want 1", p.Len())
	}

	// Record same key with smaller size — should update delta (negative)
	p.Record("key1", 50)
	if p.TotalBytes() != 50 {
		t.Fatalf("TotalBytes after size decrease = %d, want 50", p.TotalBytes())
	}

	// Another key should not affect key1's size
	p.Record("key2", 30)
	if p.TotalBytes() != 80 {
		t.Fatalf("TotalBytes after adding key2 = %d, want 80", p.TotalBytes())
	}

	// Evict key1 and verify totals
	evicted := p.Evict(50)
	if len(evicted) != 1 || evicted[0] != "key1" {
		t.Fatalf("Evict() = %v, want [key1]", evicted)
	}
	if p.TotalBytes() != 30 {
		t.Fatalf("TotalBytes after eviction = %d, want 30", p.TotalBytes())
	}
}

func TestLRURecordMovesToFront(t *testing.T) {
	p := NewLRU(1000)

	p.Record("a", 10)
	p.Record("b", 20)
	p.Record("c", 30)

	// Access 'a' again — should move to front
	p.Record("a", 10)

	// Evict should now evict 'b' (least recently used)
	evicted := p.Evict(20)
	if len(evicted) != 1 || evicted[0] != "b" {
		t.Fatalf("Evict() = %v, want [b] (least recently used)", evicted)
	}
}
