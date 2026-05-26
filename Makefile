.PHONY: build-engine build-so build-server build-agent clean test deps all run-server run-agent

# Build Go engine binary
build-engine:
	cd engine && go build -o ../bin/disk-cache ./cmd/disk-cache

# Build Go shared library for Python connector
build-so:
	cd engine && go build -o ../bin/libdiskcache.so -buildmode=c-shared ./cmd/disk-cache

# Run all tests
test:
	cd engine && go test ./pkg/...

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
