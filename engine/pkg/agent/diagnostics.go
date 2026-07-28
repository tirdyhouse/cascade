package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"predict/engine/pkg/cluster"
)

const (
	defaultDiagnosticsHost = "0.0.0.0"
	defaultDiagnosticsPort = 9002
	maxDiagnosticsLines    = 2000
	maxDiagnosticsBytes    = 2 << 20
)

// diagnosticsServer exposes a deliberately small, read-only endpoint owned by
// the Agent. The control server uses it to retrieve the log generated on the
// actual node rather than reading a path on its own host.
type diagnosticsServer struct {
	listener net.Listener
	server   *http.Server
}

func startDiagnosticsServer(cfg *Config) (*diagnosticsServer, error) {
	if cfg == nil || cfg.DiagnosticsPort <= 0 {
		return nil, nil
	}
	host := strings.TrimSpace(cfg.DiagnosticsHost)
	if host == "" {
		host = defaultDiagnosticsHost
	}
	if cfg.WorkDir == "" {
		return nil, errors.New("work directory is required for diagnostics")
	}

	listener, err := net.Listen("tcp", net.JoinHostPort(host, strconv.Itoa(cfg.DiagnosticsPort)))
	if err != nil {
		return nil, fmt.Errorf("listen diagnostics on %s:%d: %w", host, cfg.DiagnosticsPort, err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})
	mux.HandleFunc("GET /v1/logs", func(w http.ResponseWriter, r *http.Request) {
		handleDiagnosticsLogs(w, r, cfg.NodeID, cfg.WorkDir)
	})

	d := &diagnosticsServer{
		listener: listener,
		server: &http.Server{
			Handler:           mux,
			ReadHeaderTimeout: 5 * time.Second,
			IdleTimeout:       30 * time.Second,
		},
	}
	go func() {
		_ = d.server.Serve(listener)
	}()
	return d, nil
}

func (d *diagnosticsServer) Stop() {
	if d == nil || d.server == nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	_ = d.server.Shutdown(ctx)
}

func (d *diagnosticsServer) port() int {
	if d == nil || d.listener == nil {
		return 0
	}
	if address, ok := d.listener.Addr().(*net.TCPAddr); ok {
		return address.Port
	}
	return 0
}

func handleDiagnosticsLogs(w http.ResponseWriter, r *http.Request, nodeID, workDir string) {
	offset, err := parseNonNegativeInt64(r.URL.Query().Get("offset"))
	if err != nil {
		http.Error(w, "invalid offset", http.StatusBadRequest)
		return
	}
	maxLines := 100
	if raw := r.URL.Query().Get("lines"); raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil || parsed < 1 {
			http.Error(w, "invalid lines", http.StatusBadRequest)
			return
		}
		maxLines = parsed
	}
	if maxLines > maxDiagnosticsLines {
		maxLines = maxDiagnosticsLines
	}

	chunk, err := readLogChunk(nodeID, workDir, r.URL.Query().Get("filename"), offset, maxLines)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			http.Error(w, "no vLLM log is available on this agent", http.StatusNotFound)
			return
		}
		http.Error(w, "read agent log: "+err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(chunk)
}

func parseNonNegativeInt64(raw string) (int64, error) {
	if raw == "" {
		return 0, nil
	}
	value, err := strconv.ParseInt(raw, 10, 64)
	if err != nil || value < 0 {
		return 0, errors.New("must be a non-negative integer")
	}
	return value, nil
}

func readLogChunk(nodeID, workDir, filename string, offset int64, maxLines int) (cluster.LogChunk, error) {
	logFile, err := findAgentLog(filepath.Join(workDir, "logs"), filename)
	if err != nil {
		return cluster.LogChunk{}, err
	}
	info, err := os.Stat(logFile)
	if err != nil {
		return cluster.LogChunk{}, err
	}
	if offset >= info.Size() {
		return cluster.LogChunk{
			NodeID:   nodeID,
			Filename: filepath.Base(logFile),
			Offset:   info.Size(),
			EOF:      true,
		}, nil
	}

	file, err := os.Open(logFile)
	if err != nil {
		return cluster.LogChunk{}, err
	}
	defer file.Close()
	if _, err := file.Seek(offset, io.SeekStart); err != nil {
		return cluster.LogChunk{}, err
	}

	remaining := info.Size() - offset
	if remaining > maxDiagnosticsBytes {
		remaining = maxDiagnosticsBytes
	}
	data, err := io.ReadAll(io.LimitReader(file, remaining))
	if err != nil {
		return cluster.LogChunk{}, err
	}
	consumed := len(data)
	if maxLines > 0 {
		lineEnds := 0
		for index, b := range data {
			if b != '\n' {
				continue
			}
			lineEnds++
			if lineEnds >= maxLines {
				consumed = index + 1
				break
			}
		}
	}
	data = data[:consumed]
	newOffset := offset + int64(consumed)
	return cluster.LogChunk{
		NodeID:   nodeID,
		Filename: filepath.Base(logFile),
		Lines:    string(bytes.TrimSuffix(data, []byte("\n"))),
		Offset:   newOffset,
		EOF:      newOffset >= info.Size(),
	}, nil
}

func findAgentLog(logDir, filename string) (string, error) {
	if filename != "" {
		if filepath.Base(filename) != filename || !isVLLMLogName(filename) {
			return "", os.ErrNotExist
		}
		return filepath.Join(logDir, filename), nil
	}

	entries, err := os.ReadDir(logDir)
	if err != nil {
		return "", err
	}
	var latest string
	var latestTime time.Time
	for _, entry := range entries {
		if entry.IsDir() || !isVLLMLogName(entry.Name()) {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		if latest == "" || info.ModTime().After(latestTime) {
			latest = filepath.Join(logDir, entry.Name())
			latestTime = info.ModTime()
		}
	}
	if latest == "" {
		return "", os.ErrNotExist
	}
	return latest, nil
}

func isVLLMLogName(name string) bool {
	return strings.HasPrefix(name, "vllm-") && strings.HasSuffix(name, ".log")
}
