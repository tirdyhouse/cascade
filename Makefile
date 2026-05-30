.PHONY: all build-engine build-so build-server build-agent build-cs clean deps test test-go test-adapter test-smoke test-storage bench-storage test-vllm-cache ci ci-gpu ci-vllm run-server run-agent

PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest
GO_TEST_FLAGS ?=

STORAGE_BACKEND ?= posix
STORAGE_DEVICE ?= cpu
STORAGE_SHAPE ?= 2,4,8
STORAGE_DTYPE ?= float32

STORAGE_BENCH_BACKENDS ?= posix,gds
STORAGE_BENCH_DEVICE ?= cuda:0
STORAGE_BENCH_SHAPE ?= 4096,4096
STORAGE_BENCH_DTYPE ?= float16
STORAGE_BENCH_ITERATIONS ?= 3
STORAGE_BENCH_WARMUP ?= 1
STORAGE_BENCH_DIR ?=
STORAGE_BENCH_MARKDOWN ?=

# Default target
all: build-engine

# Build Go engine binary
build-engine:
	mkdir -p bin
	cd engine && go build -o ../bin/disk-cache ./cmd/disk-cache

# Build Go shared library for Python connector
build-so:
	mkdir -p bin
	cd engine && go build -o ../bin/libdiskcache.so -buildmode=c-shared ./cmd/disk-cache

# Build S端 cluster-server
build-server:
	mkdir -p bin
	cd engine && go build -o ../bin/cluster-server ./cmd/cluster-server

# Build C端 agent
build-agent:
	mkdir -p bin
	cd engine && go build -o ../bin/c-agent ./cmd/c-agent

# Build all CS components
build-cs: build-server build-agent

# Fast default test target: Go engine, commands, and packages.
test: test-go

# Run all Go tests under engine/.
test-go:
	cd engine && go test $(GO_TEST_FLAGS) ./...

# Run Python adapter/helper tests (requires adapter test dependencies).
test-adapter:
	PYTHONPATH=.$${PYTHONPATH:+:$$PYTHONPATH} $(PYTEST) adapter/tests

# CI-friendly validation: no long-running vLLM service and no GPU benchmark.
ci: test-go test-adapter test-storage

# GPU host validation: CI bundle plus storage backend benchmark.
ci-gpu: ci bench-storage

# Full validation: CI bundle plus real vLLM + disk-cache smoke.
ci-vllm: ci test-vllm-cache

# Run local disk-cache HTTP smoke test
test-smoke:
	bash scripts/smoke_disk_cache.sh

# Run storage backend smoke/diagnostics (override STORAGE_BACKEND/DEVICE/SHAPE/DTYPE)
test-storage:
	PYTHONPATH=.$${PYTHONPATH:+:$$PYTHONPATH} \
	STORAGE_BACKEND=$(STORAGE_BACKEND) \
	STORAGE_DEVICE=$(STORAGE_DEVICE) \
	STORAGE_SHAPE=$(STORAGE_SHAPE) \
	STORAGE_DTYPE=$(STORAGE_DTYPE) \
	$(PYTHON) scripts/validate_storage_backend.py

# Benchmark storage backends (requires adapter deps; defaults to GPU/GDS-oriented settings)
bench-storage:
	PYTHONPATH=.$${PYTHONPATH:+:$$PYTHONPATH} \
	STORAGE_BENCH_BACKENDS=$(STORAGE_BENCH_BACKENDS) \
	STORAGE_BENCH_DEVICE=$(STORAGE_BENCH_DEVICE) \
	STORAGE_BENCH_SHAPE=$(STORAGE_BENCH_SHAPE) \
	STORAGE_BENCH_DTYPE=$(STORAGE_BENCH_DTYPE) \
	STORAGE_BENCH_ITERATIONS=$(STORAGE_BENCH_ITERATIONS) \
	STORAGE_BENCH_WARMUP=$(STORAGE_BENCH_WARMUP) \
	STORAGE_BENCH_DIR=$(STORAGE_BENCH_DIR) \
	STORAGE_BENCH_MARKDOWN=$(STORAGE_BENCH_MARKDOWN) \
	$(PYTHON) scripts/benchmark_storage_backend.py

# Run real vLLM + disk-cache validation (requires MODEL_PATH, GPU, and vLLM)
test-vllm-cache:
	PYTHON_BIN=$(PYTHON) bash scripts/validate_vllm_disk_cache.sh

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

# Install Go dependencies
deps:
	cd engine && go mod tidy
