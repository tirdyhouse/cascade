// Package storage provides storage backend interfaces and implementations.
package storage

import (
	"io"
)

// Backend defines the storage operations for KV cache data.
type Backend interface {
	// Write saves data to storage. Returns the file path and bytes written.
	Write(key string, data []byte) (path string, n int64, err error)

	// Read loads data from storage into the provided buffer.
	Read(path string) ([]byte, error)

	// Delete removes data from storage.
	Delete(path string) error

	// Size returns the usable capacity in bytes (0 = unlimited).
	Capacity() int64

	// Used returns the bytes currently used.
	Used() int64

	// Close shuts down the backend.
	Close() error
}

// Compactor is an optional interface for backends that support compaction.
type Compactor interface {
	// Compact reclaims disk space.
	Compact() error
}

// NopCloser wraps a Backend with a no-op Close.
type NopCloser struct {
	Backend
}

func (n *NopCloser) Close() error { return nil }

// LimitWriter returns an io.WriteCloser that limits total written bytes.
type LimitWriter struct {
	io.WriteCloser
	Total int64
	Max   int64
}

func (w *LimitWriter) Write(p []byte) (int, error) {
	if w.Total+int64(len(p)) > w.Max {
		return 0, ErrDiskFull
	}
	n, err := w.WriteCloser.Write(p)
	w.Total += int64(n)
	return n, err
}

var (
	ErrDiskFull    = io.ErrShortWrite
	ErrNotFound    = io.EOF
	ErrBackendFail = io.ErrUnexpectedEOF
)
