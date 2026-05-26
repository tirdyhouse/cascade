<p align="center">
  🇬🇧 <a href="README.md">English</a> | 🇨🇳 <a href="README.zh-CN.md">简体中文</a>
</p>

# Cascade

> **Extend LLM inference context windows beyond GPU memory limits with a high-performance disk-backed KV cache layer.**

[![Go Version](https://img.shields.io/badge/Go-1.23+-00ADD8?logo=go)](https://golang.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](#license)
[![vLLM](https://img.shields.io/badge/vLLM-Compatible-8A2BE2)](https://github.com/vllm-project/vllm)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](#contributing)

---

## Vision

LLM inference is fundamentally memory-bound. GPU HBM (~80 GB per H100) constrains how many tokens a model can process, forcing operators to choose between context length and batch size.

**Cascade** decouples KV cache from GPU memory by adding a high-performance, distributed disk cache layer underneath existing inference engines. The result: longer context windows, higher throughput, and dramatically lower cost per token — without modifying the model or buying more GPUs.

### Why disk?

| | GPU HBM | Local NVMe | Remote NVMe (RDMA) |
|---|---|---|---|
| Capacity | 80 GB | 2–30 TB | ∞ (cluster) |
| Latency | ~1 µs | ~10 µs | ~100 µs |
| Bandwidth | 2000 GB/s | 7 GB/s | 100–500 GB/s |
| Cost/GB | ~$100 | ~$0.10 | ~$0.05 |

The key insight: **latency and bandwidth of NVMe are viable for KV cache**, and the cost advantage is overwhelming. By keeping hot data on GPU and seamlessly tiering cold data to disk, we enable practical 1M+ token contexts without rebuilding infrastructure.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Inference Engine (vLLM)                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │            DiskCache KVConnector (Python)            │    │
│  │  • Scheduler hooks: cache-hit detection, eviction    │    │
│  │  • Worker hooks:   save/load KV tensors to disk     │    │
│  └───────────────────────┬─────────────────────────────┘    │
│                          │ HTTP / local filesystem           │
├──────────────────────────┼──────────────────────────────────┤
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Go DiskCache Engine                      │    │
│  │                                                       │    │
│  │  ┌──────────────┐  ┌──────────────┐                  │    │
│  │  │  Metadata     │  │  Eviction    │                  │    │
│  │  │  (Pebble/LSM) │  │  LRU         │                  │    │
│  │  └──────────────┘  └──────────────┘                  │    │
│  │                                                       │    │
│  │  ┌────────────────────────────────────────────────┐  │    │
│  │  │  Storage Backends                               │  │    │
│  │  │  POSIX │ io_uring │ GPUDirect Storage (future)  │  │    │
│  │  └────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Cluster Manager (future)                 │    │
│  │  • etcd-based node discovery                         │    │
│  │  • Dynamic role assignment (prefill/decode/storage)  │    │
│  │  • RDMA data transfer                                │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Project Structure

```
cascade/
├── adapter/           # Inference engine adapters
│   └── vllm/          # vLLM KVConnector implementation
├── engine/            # Go core engine
│   ├── cmd/           # Entry points (disk-cache daemon)
│   └── pkg/           # Core libraries
│       ├── cache/     # Cache engine (Pebble metadata + LRU)
│       ├── eviction/  # Eviction policies
│       ├── metadata/  # Block metadata store
│       └── storage/   # Storage backends
├── csrc/              # C storage primitives (GDS, RDMA, io_uring)
├── deploy/            # Deployment (Helm charts)
├── docs/              # Documentation
├── scripts/           # Benchmarking & utility scripts
└── test/              # Integration & benchmark tests
```

---

## Current Status

> **Phase 1 — Local Disk Cache MVP** ✅

| Component | Status | Description |
|---|---|---|
| Go engine core | ✅ **Done** | Pebble-backed metadata store, LRU eviction, HTTP API |
| vLLM connector | ✅ **Done** | Full KVConnectorBase_V1 implementation (~185 LOC) |
| Disk I/O benchmarks | ✅ **Done** | Sequential/random read-write, latency profiling |
| Benchmark suite | ✅ **Done** | Compare native vLLM vs LMCache vs DiskCache |
| GDS / RDMA | 📋 **Planned** | C storage primitives scaffolded |
| SGLang adapter | 📋 **Planned** | Directory structure ready |
| Cluster manager | 📋 **Planned** | Architecture designed in DESIGN.md |
| Helm deployment | 📋 **Planned** | Chart scaffolded |

---

## Quick Start

### 1. Build the Go Engine

```bash
make build-engine
# Output: bin/disk-cache
```

### 2. Start the DiskCache Daemon

```bash
./bin/disk-cache \
    --cache-path /mnt/nvme/kv-cache \
    --metadata-path /tmp/disk-cache-meta \
    --max-size 100GB \
    --listen :9100
```

### 3. Start vLLM with DiskCache Connector

```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --tensor-parallel-size 8 \
    --max-model-len 100000 \
    --kv-transfer-config '{
        "kv_connector": "disk-cache",
        "kv_connector_extra_config": {
            "disk_cache_path": "/mnt/nvme/kv-cache",
            "disk_cache_engine_addr": "http://localhost:9100"
        }
    }'
```

### 4. Verify

```bash
# Engine stats
curl http://localhost:9100/stats

# Run benchmark
python3 test/scripts/run_bench.py --mode diskcache
```

---

## Roadmap

### Phase 1: Local Disk Cache MVP ✅ *(current)*
- [x] Go engine: Pebble metadata + LRU eviction + HTTP API
- [x] vLLM KVConnector: save/load KV tensors to disk
- [x] Benchmark suite: compare native / LMCache / DiskCache
- [x] Disk I/O profiling tools

### Phase 2: GPUDirect Storage Acceleration 📋 *(next)*
- [ ] C GDS backend (NVIDIA cuFile)
- [ ] Go CGo bindings for GPU↔NVMe zero-copy
- [ ] Automatic fallback: GDS → POSIX
- [ ] Throughput target: 3–5× vs CPU bounce buffer

### Phase 3: Distributed Cluster 📋
- [ ] Cluster manager: etcd-based node registration & discovery
- [ ] Dynamic role assignment (prefill / decode / storage)
- [ ] RDMA data transfer between nodes
- [ ] SGLang adapter
- [ ] Fault tolerance & data migration

### Phase 4: Production Hardening 📋
- [ ] Helm chart (Kubernetes deployment)
- [ ] Prometheus / Grafana metrics
- [ ] Admin dashboard
- [ ] Multi-tenancy
- [ ] Extensive documentation & examples

---

## Benchmarking

```bash
# 1. Profile your NVMe drive first
python3 scripts/disk-bench.py /mnt/nvme

# 2. Run the cache engine benchmark
python3 scripts/disk-bench-cache.py http://localhost:9100

# 3. Compare inference engines
python3 test/scripts/run_bench.py --mode native
python3 test/scripts/run_bench.py --mode diskcache
python3 test/scripts/gen_report.py
```

Results are stored in `test/results/`. See `docs/benchmark-plan.md` for detailed methodology.

---

## Comparison

### Design Philosophy

| | LMCache | Mooncake | **Cascade** |
|---|---|---|---|---|
| **Role** | Tiered cache engine | Distributed KV transport engine | **Cluster disk cache** |
| **Disk role** | Warm data tier (CPU→Disk) | Eviction overflow target | **🎯 Primary storage layer** |
| **Data path** | GPU → CPU → Disk | GPU memory ↔ RDMA → peer GPU | **GPU → NVMe (GDS) → cluster (RDMA)** |
| **Storage node** | ❌ Must have GPU | ❌ Must have GPU | **✅ Pure disk node planned** |

### Current Implementation (Phase 1 MVP)

| Feature | LMCache | Mooncake | **Cascade** |
|---|---|---|---|
| **Disk cache** | ✅ LocalDiskBackend | ✅ FileStorage (SSD offload) | **✅ Core design** |
| **KV data I/O** | Python `open/write` | C++ io_uring / POSIX | **Python safetensors*** |
| **Metadata store** | In-memory Python dict | etcd + Master Service | **Pebble (LSM tree)** |
| **Metadata persistent** | ❌ Lost on restart | ✅ etcd | **✅ Pebble** |
| **Eviction policy** | ✅ LRU / LFU / FIFO / MRU | ✅ LRU / FIFO | **✅ LRU** |
| **Prefix matching** | ✅ TokenDatabase | ❌ Opaque key only | **✅ SHA-256 incremental + sentinel** |
| **vLLM integration** | ✅ Deep integration | ✅ mooncake-integration | **✅ KVConnectorBase_V1** |
| **Codebase (core engine)** | ~79K lines Python | ~220K lines C++ | **~400 lines Go** |

> *Phase 1 uses Python safetensors via CPU bounce buffer as a temporary path
> (hardware limitation). The Go engine manages metadata + eviction over HTTP.
> Future phases will move all storage I/O to the Go/C layer.

### Planned Architecture (Design Target)

| Feature | LMCache | Mooncake | **Cascade** |
|---|---|---|---|
| **I/O stack** | Python native | C++ native | **Python connector → Go engine → C backend** |
| **GPU↔NVMe** | ✅ GdsBackend (partial) | ❌ Not supported | **📋 GPUDirect Storage (cufile)** |
| **Cross-node transfer** | ❌ No RDMA | ✅ RDMA (core competency) | **📋 RDMA (ibverbs)** |
| **Async disk I/O** | ❌ Not supported | ✅ io_uring | **📋 io_uring** |
| **Cluster manager** | ❌ P2P only (ZMQ) | ✅ Master + etcd + HA | **📋 ClusterManager + etcd** |
| **Pure storage node** | ❌ No such concept | ❌ GPU required | **📋 GPU-free storage node** |
| **SGLang adapter** | ❌ Not available | ✅ Supported | **📋 Scaffolded** |

### Architecture Evolution

```
Phase 1 (current)   Python safetensors writes/reads disk directly
                    Go engine manages metadata (Pebble) + eviction (LRU) via HTTP
                    
Phase 2 (next)      Go engine takes over all storage I/O
                    C backends: GPUDirect Storage (GPU↔NVMe zero-copy)
                                io_uring (async disk I/O)
                    
Phase 3 (planned)   Cluster mode: etcd node discovery + ClusterManager
                    RDMA cross-node transfer + pure disk storage nodes
```

---

## Documentation

- [Design Document](./DESIGN.md) — Detailed architecture and rationale
- [Benchmark Plan](./docs/benchmark-plan.md) — Testing methodology
- [vLLM Baseline Setup](./docs/baseline-vllm-deepseek-v4.md) — Reference deployment

---

## Contributing

Contributions are welcome! This project is in active early development, so there are many opportunities to make an impact:

- **Engineers**: Help implement GDS, RDMA, cluster manager, or storage backends
- **ML practitioners**: Run benchmarks, report results, suggest optimizations
- **Infrastructure folks**: Improve deployment, monitoring, and observability

Please open an issue or pull request. For major changes, start with a discussion.

---

## License

Cascade is **dual-licensed**:

- **Apache 2.0** — Free for open-source projects, individual developers, and non-commercial use.
- **Commercial License** — Required for embedding in hardware appliances, proprietary products, or commercial solutions.

See [COMMERCIAL_LICENSE.md](./COMMERCIAL_LICENSE.md) for details.

[Apache License 2.0](./LICENSE)

---

<p align="center">
  <b>LLM inference shouldn't be memory-bound.</b><br>
  Cascade — extending context, one NVMe at a time.
</p>
