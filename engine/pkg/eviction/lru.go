package eviction

import (
	"container/list"
	"sync"
)

// Policy defines the eviction strategy interface.
type Policy interface {
	Record(key string, size int64)
	Evict(targetBytes int64) []string
	Remove(key string)
	Len() int
	TotalBytes() int64
}

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
	maxBytes int64
}

func NewLRU(maxBytes int64) *LRU {
	return &LRU{
		ll:       list.New(),
		items:    make(map[string]*list.Element),
		maxBytes: maxBytes,
	}
}

func (p *LRU) Record(key string, size int64) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if elem, ok := p.items[key]; ok {
		p.ll.MoveToFront(elem)
		return
	}
	e := &entry{key: key, size: size}
	elem := p.ll.PushFront(e)
	p.items[key] = elem
	p.total += size
}

func (p *LRU) Evict(targetBytes int64) []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	var evicted []string
	var freed int64
	for freed < targetBytes && p.ll.Len() > 0 {
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

func (p *LRU) Len() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.ll.Len()
}

func (p *LRU) TotalBytes() int64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.total
}
