// Package eviction provides cache eviction policies.
package eviction

import (
	"container/list"
	"sync"
)

// Policy defines the eviction strategy interface.
type Policy interface {
	// Record marks a key as accessed.
	Record(key string, size int64)

	// Evict returns keys to evict to free at least targetBytes.
	Evict(targetBytes int64) []string

	// Remove deletes a key from the policy tracker.
	Remove(key string)

	// Len returns the number of tracked entries.
	Len() int
}

// entry holds key and size for the LRU list.
type entry struct {
	key  string
	size int64
}

// LRU implements a least-recently-used eviction policy.
type LRU struct {
	mu       sync.Mutex
	ll       *list.List
	items    map[string]*list.Element
	total    int64
	maxBytes int64 // 0 = unlimited
}

// NewLRU creates an LRU eviction policy.
// maxBytes: maximum tracked bytes before eviction triggers (0 = unlimited).
func NewLRU(maxBytes int64) *LRU {
	return &LRU{
		ll:       list.New(),
		items:    make(map[string]*list.Element),
		maxBytes: maxBytes,
	}
}

// Record marks a key as recently used. Adds it if new.
func (p *LRU) Record(key string, size int64) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if elem, ok := p.items[key]; ok {
		// Move to front (most recently used)
		p.ll.MoveToFront(elem)
		return
	}

	// Add new entry
	e := &entry{key: key, size: size}
	elem := p.ll.PushFront(e)
	p.items[key] = elem
	p.total += size
}

// Evict removes the least recently used entries until targetBytes are freed.
func (p *LRU) Evict(targetBytes int64) []string {
	p.mu.Lock()
	defer p.mu.Unlock()

	var evicted []string
	var freed int64

	for freed < targetBytes && p.ll.Len() > 0 {
		// Back of the list = least recently used
		elem := p.ll.Back()
		if elem == nil {
			break
		}
		e := elem.Value.(*entry)
		p.ll.Remove(elem)
		delete(p.items, e.key)
		p.total -= e.size
		freed += e.size
		evicted = append(evicted, e.key)
	}

	return evicted
}

// Remove deletes a key from the policy.
func (p *LRU) Remove(key string) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if elem, ok := p.items[key]; ok {
		e := elem.Value.(*entry)
		p.ll.Remove(elem)
		delete(p.items, key)
		p.total -= e.size
	}
}

// Len returns the number of tracked entries.
func (p *LRU) Len() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.ll.Len()
}

// TotalBytes returns the total tracked bytes.
func (p *LRU) TotalBytes() int64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.total
}
