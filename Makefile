.PHONY: build-engine build-so clean test deps all

# Build Go engine binary
build-engine:
	cd engine && go build -o ../bin/disk-cache ./cmd/disk-cache

# Build Go shared library for Python connector
build-so:
	cd engine && go build -o ../bin/libdiskcache.so -buildmode=c-shared ./cmd/disk-cache

# Run all tests
test:
	cd engine && go test ./pkg/...

# Clean build artifacts
clean:
	rm -rf bin/

# Install dependencies
deps:
	cd engine && go mod tidy

# Default target
all: build-engine
