.PHONY: build-engine build-so build-server build-agent clean test test-smoke test-storage bench-storage test-vllm-cache deps all run-server run-agent

# Build Go engine binary
build-engine:
	cd engine && go build -o ../bin/disk-cache ./cmd/disk-cache

# Build Go shared library for Python connector
build-so:
	cd engine && go build -o ../bin/libdiskcache.so -buildmode=c-shared ./cmd/disk-cache

# Run all tests
test:
	cd engine && go test ./pkg/...

# Run local disk-cache HTTP smoke test
test-smoke:
	bash scripts/smoke_disk_cache.sh

# Run storage backend smoke/diagnostics (override STORAGE_BACKEND/DEVICE/SHAPE/DTYPE)
test-storage:
	PYTHONPATH=. python3 scripts/validate_storage_backend.py

# Benchmark storage backends (requires adapter deps; defaults to GPU/GDS-oriented settings)
bench-storage:
	PYTHONPATH=. python3 scripts/benchmark_storage_backend.py

# Run real vLLM + disk-cache validation (requires MODEL_PATH, GPU, and vLLM)
test-vllm-cache:
	bash scripts/validate_vllm_disk_cache.sh

# Build S端 cluster-server
build-server:
	cd engine && go build -o ../bin/cluster-server ./cmd/cluster-server

# Build C端 agent
build-agent:
	cd engine && go build -o ../bin/c-agent ./cmd/c-agent

# Build all CS components
build-cs: build-server build-agent

# Run S端 locally (development)
run-server:
	cd engine && go run ./cmd/cluster-server --rpcx-port 9000 --http-port 8080

# Run C端 locally (development)
run-agent:
	cd engine && go run ./cmd/c-agent --server 127.0.0.1:9000 --node-id dev-node --cache-mode local_nvme \
		--disks /tmp/nvme0:100

# Clean build artifacts
clean:
	rm -rf bin/

# Install dependencies
deps:
	cd engine && go mod tidy

# Default target
all: build-engine
