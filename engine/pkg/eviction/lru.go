package eviction

import (
	"container/list"
	"sync"
)

// Policy defines the eviction strategy interface.
type Policy interface {
	Record(key string, size int64)
	Candidates(targetBytes int64) []string
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
		e := elem.Value.(*entry)
		delta := size - e.size
		e.size = size
		p.total += delta
		return
	}
	e := &entry{key: key, size: size}
	elem := p.ll.PushFront(e)
	p.items[key] = elem
	p.total += size
}

func (p *LRU) Candidates(targetBytes int64) []string {
	p.mu.Lock()
	defer p.mu.Unlock()

	var candidates []string
	var covered int64
	for elem := p.ll.Back(); elem != nil && covered < targetBytes; elem = elem.Prev() {
		e := elem.Value.(*entry)
		candidates = append(candidates, e.key)
		covered += e.size
	}
	return candidates
}

func (p *LRU) Evict(targetBytes int64) []string {
	keys := p.Candidates(targetBytes)
	for _, key := range keys {
		p.Remove(key)
	}
	return keys
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
