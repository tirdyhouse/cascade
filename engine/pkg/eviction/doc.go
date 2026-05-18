// Package eviction provides cache eviction policies.
package eviction

// Policy defines the eviction strategy interface.
type Policy interface {
	Record(key string, size int64)
	Evict(targetBytes int64) []string
	Remove(key string)
	Len() int
	TotalBytes() int64
}
