<p align="center">
  🇬🇧 <a href="README.md">English</a> | 🇨🇳 <a href="README.zh-CN.md">简体中文</a>
</p>

<h1 align="center">Cascade</h1>

<p align="center">
  <strong>Disk-backed KV cache for long-context LLM inference.</strong><br>
  Extend vLLM beyond GPU memory with a Go metadata engine, POSIX/GDS storage backends, and a roadmap toward pooled NVMe clusters.
</p>

<p align="center">
  <a href="https://golang.org"><img alt="Go" src="https://img.shields.io/badge/Go-1.24+-00ADD8?logo=go"></a>
  <a href="https://github.com/vllm-project/vllm"><img alt="vLLM" src="https://img.shields.io/badge/vLLM-0.21%20validated-8A2BE2"></a>
  <img alt="A100" src="https://img.shields.io/badge/A100-validated-76B900?logo=nvidia">
  <img alt="GDS" src="https://img.shields.io/badge/GDS-backend-0B7285">
  <a href="#license"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0%20%2B%20Commercial-blue"></a>
</p>

---

## Why Cascade

LLM serving is increasingly constrained by KV cache memory. GPU HBM is fast but expensive and finite; long prompts, agentic workloads, and repeated retrieval-heavy conversations quickly turn context length into an infrastructure problem.

**Cascade** treats local NVMe as a first-class KV cache tier. It keeps the inference engine API surface small, moves metadata and eviction into a compact Go service, and lets the Python vLLM connector save/load KV tensors through pluggable storage backends:

- **POSIX backend** for portable CPU-bounce-buffer storage.
- **GPUDirect Storage (GDS) backend** for GPU↔NVMe transfer on supported hosts.
- **Cluster roadmap** for RDMA-accessible pooled SSD and GPU-aware scheduling.

The goal is simple: **make long-context inference cheaper and more elastic without changing the model.**

### Why disk?

| | GPU HBM | Local NVMe | Remote NVMe (RDMA) |
|---|---:|---:|---:|
| Capacity | ~80 GB / GPU | 2–30 TB / node | Cluster-scale |
| Latency | ~1 µs | ~10–100 µs | ~100 µs+ |
| Bandwidth | ~2 TB/s | ~7 GB/s | 100–500 GB/s |
| Cost/GB | High | Low | Lower at scale |

NVMe is not a replacement for hot GPU KV. It is a high-capacity tier for cold or reusable KV blocks, enabling operators to trade a small amount of storage latency for a large reduction in HBM pressure.

---

## What works today

| Area | Status | Notes |
|---|---|---|
| vLLM integration | ✅ Working | `KVConnectorBase_V1` connector for vLLM 0.21-style V1 execution. |
| DiskCache engine | ✅ Working | Go service with HTTP API, Pebble metadata, LRU eviction, and persistent block/chunk indexes. |
| Storage backends | ✅ Working | POSIX backend plus GDS/NvFile backend with automatic fallback. |
| Cache-hit validation | ✅ Working | Isolated script starts disk-cache + vLLM and checks retrieved block counters and cached token stats. |
| A100 validation | ✅ Working | Real A100 run with Qwen2.5-7B-Instruct and vLLM 0.21.0. |
| Cluster scheduling | 🚧 Roadmap | GPU-aware dispatch, RDMA pooled NVMe, and multi-node coordination are planned. |

---

## Validation snapshot

Observed on the project A100 validation host. These numbers are environment samples, not universal performance guarantees.

### Real vLLM + disk-cache smoke

| Item | Value |
|---|---|
| GPU | NVIDIA A100-PCIE-40GB |
| vLLM | 0.21.0 |
| Model | Qwen2.5-7B-Instruct |
| Prompt size | 6,629 prompt tokens (`REQUEST_REPETITIONS=200`, `max_model_len=8192`) |
| First request | `1.704s`, retrieved blocks `0` |
| Second request | `0.199s`, retrieved blocks `28`, cached tokens `6624` |

See [release notes](./docs/release-notes.md) for the full validation command and environment notes.

### POSIX vs GDS storage backend benchmark

| Backend | Selected implementation | Save median | Load median |
|---|---|---:|---:|
| POSIX | `PosixBackend` | `0.066639s` / `480.20 MiB/s` | `0.009055s` / `3534.14 MiB/s` |
| GDS | `NvFileBackend` | `0.020430s` / `1566.36 MiB/s` | `0.005160s` / `6202.14 MiB/s` |

See [A100 storage benchmark](./docs/storage-benchmark-a100.md) for the reproducible marker-aware report.

---

## Architecture

```text
┌──────────────────────────────────────────────────────────────────────┐
│                            vLLM / LLM Engine                         │
│                                                                      │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │ DiskCache KVConnector (Python)                               │   │
│   │ - scheduler-side prefix/cache-hit lookup                     │   │
│   │ - worker-side KV tensor save/load                            │   │
│   │ - POSIX/GDS backend selection                                │   │
│   └───────────────────────────────┬──────────────────────────────┘   │
└───────────────────────────────────┼──────────────────────────────────┘
                                    │ HTTP metadata + local file I/O
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         Go DiskCache Engine                           │
│                                                                      │
│   ┌────────────────────┐   ┌────────────────────┐   ┌────────────┐  │
│   │ Pebble metadata    │   │ LRU eviction        │   │ HTTP stats │  │
│   │ blocks/sentinels   │   │ capacity control    │   │ diagnostics│  │
│   └────────────────────┘   └────────────────────┘   └────────────┘  │
│                                                                      │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │ Storage layer                                                │   │
│   │ POSIX today | GDS today | io_uring/RDMA planned              │   │
│   └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        Cluster layer (roadmap)                        │
│ GPU-aware request routing | pooled SSD | RDMA | node registry         │
└──────────────────────────────────────────────────────────────────────┘
```

### Project layout

```text
cascade/
├── adapter/           # Inference engine adapters
│   └── vllm/          # vLLM KVConnector implementation
├── engine/            # Go disk-cache engine
│   ├── cmd/           # disk-cache, c-agent, cluster-server entry points
│   └── pkg/           # cache, metadata, server, storage, cluster packages
├── docs/              # Design docs, benchmark plans, release notes
├── scripts/           # Validation and benchmark scripts
├── test/              # Integration and benchmark helpers
└── images/            # README/community assets
```

---

## Quick start

### 1. Build the engine

```bash
make build-engine
# Output: bin/disk-cache
```

### 2. Run the disk-cache service

```bash
./bin/disk-cache \
  -cache-path /mnt/nvme/kv-cache \
  -metadata-path /tmp/disk-cache-meta \
  -max-size 100GB \
  -listen :9100
```

### 3. Prefer the reproducible vLLM smoke

On a GPU host with vLLM and a local model path:

```bash
MODEL_PATH=/tmp/models/Qwen2.5-7B-Instruct \
VLLM_EXTRA_ARGS='--tensor-parallel-size 1 --max-model-len 8192' \
REQUEST_REPETITIONS=200 \
MAX_TOKENS=8 \
make test-vllm-cache
```

The script builds the engine, starts isolated disk-cache and vLLM services, sends two repeated prompts, and fails if the second request does not retrieve cached KV chunks.

### 4. Manual vLLM integration

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
vllm serve /path/to/model \
  --no-enable-prefix-caching \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --kv-transfer-config '{
    "kv_connector": "DiskCacheConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "adapter.vllm.connector_v21",
    "kv_connector_extra_config": {
      "disk_cache_path": "/mnt/nvme/kv-cache",
      "disk_cache_engine_addr": "http://localhost:9100",
      "target_device": "auto",
      "storage_backend": "auto",
      "disk_cache_chunk_size_mb": 128
    }
  }'
```

Use `target_device=auto` for multi-GPU safety: the connector saves/loads to the actual KV tensor device unless an explicit device override is required.

---

## Validation and CI targets

```bash
# Go packages and engine tests
make test-go

# Python adapter/helper tests
make test-adapter

# CI-friendly bundle: Go + adapter tests + POSIX storage smoke
make ci

# GPU host validation: CI bundle + POSIX/GDS benchmark
make ci-gpu

# Real vLLM + disk-cache smoke
make test-vllm-cache
```

Useful overrides:

```bash
# Storage smoke on a specific backend/device
STORAGE_BACKEND=gds STORAGE_DEVICE=cuda:0 make test-storage

# Storage backend benchmark
STORAGE_BENCH_BACKENDS=posix,gds \
STORAGE_BENCH_DEVICE=cuda:0 \
STORAGE_BENCH_MARKDOWN=docs/storage-benchmark-a100.md \
make bench-storage
```

---

## Roadmap

### Done

- [x] Go DiskCache engine: Pebble metadata, LRU eviction, HTTP API, stats.
- [x] vLLM `KVConnectorBase_V1` integration for save/load of KV tensors.
- [x] Persistent prefix/chunk metadata with sentinel entries and hit diagnostics.
- [x] POSIX storage backend using safetensors-compatible CPU bounce path.
- [x] GDS/NvFile backend with automatic POSIX fallback.
- [x] A100 validation workflow and marker-aware storage benchmark report.

### Next

- [ ] Better operator-facing docs and deployment examples.
- [ ] GPU-aware scheduling: VRAM/task/load monitoring and smarter dispatch.
- [ ] Pooled SSD design: RDMA-accessible shared NVMe across nodes.
- [ ] Async disk I/O experiments (`io_uring`) and larger benchmark matrix.
- [ ] SGLang adapter and additional inference-engine integration points.

### Later

- [ ] Cluster manager with node registry/discovery.
- [ ] Multi-GPU gang scheduling for tensor-parallel models.
- [ ] Fault tolerance, data migration, and production observability.
- [ ] Helm chart and managed deployment patterns.

---

## Benchmarking

```bash
# 1. Profile a local NVMe drive
python3 scripts/disk-bench.py /mnt/nvme

# 2. Exercise the cache engine over HTTP
python3 scripts/disk-bench-cache.py http://localhost:9100

# 3. Compare POSIX/GDS storage backends on a GPU host
make bench-storage

# 4. Refresh the checked-in A100 benchmark report
STORAGE_BENCH_MARKDOWN=docs/storage-benchmark-a100.md make bench-storage
```

See [Benchmark Plan](./docs/benchmark-plan.md) for inference benchmarking methodology and [A100 Storage Backend Benchmark](./docs/storage-benchmark-a100.md) for the latest checked-in storage report.

---

## Comparison

| Dimension | LMCache | Mooncake | **Cascade** |
|---|---|---|---|
| Primary role | Tiered KV cache | Distributed KV transfer | **Disk-backed KV cache + future pooled SSD** |
| Disk role | Warm tier | Overflow/offload target | **Primary high-capacity KV tier** |
| Metadata | Python/in-memory paths | Master/etcd service | **Pebble-backed Go engine** |
| vLLM integration | Deep integration | Integration plugin | **KVConnectorBase_V1** |
| GPU↔NVMe path | GDS backend exists | Not the main design center | **POSIX + GDS backend abstraction** |
| Cluster direction | Cache tiering | RDMA transfer | **GPU-aware scheduling + pooled NVMe roadmap** |

Cascade is intentionally small and infrastructure-oriented: keep the Python connector thin, keep metadata durable, make storage backends explicit, and evolve from one-node NVMe to pooled SSD clusters.

---

## Documentation

- [Design Document](./DESIGN.md) — architecture and rationale.
- [Benchmark Plan](./docs/benchmark-plan.md) — benchmarking methodology.
- [A100 Storage Backend Benchmark](./docs/storage-benchmark-a100.md) — reproducible POSIX/GDS sample report.
- [Release Notes](./docs/release-notes.md) — latest validation notes.
- [vLLM Baseline Setup](./docs/baseline-vllm-deepseek-v4.md) — reference deployment notes.

---

## Contributing

Cascade is early and practical. Good contributions include:

- Running the validation scripts on different GPUs, SSDs, filesystems, and vLLM versions.
- Improving storage backends, cache-hit diagnostics, or benchmark coverage.
- Hardening deployment, observability, and operational docs.
- Prototyping cluster scheduling, RDMA transfer, or SGLang integration.

Please open an issue or pull request. For major design changes, start with a short proposal.

---

## License

Cascade is **dual-licensed**:

- **Apache 2.0** — free for open-source projects, individual developers, and non-commercial use.
- **Commercial License** — required for embedding in hardware appliances, proprietary products, or commercial solutions.

See [COMMERCIAL_LICENSE.md](./COMMERCIAL_LICENSE.md) for details.

[Apache License 2.0](./LICENSE)

---

<p align="center">
  <strong>LLM inference should not be memory-bound.</strong><br>
  Cascade — stretch context with NVMe, one cache block at a time.
</p>

---

## Community

<p align="center">
  <img src="./images/qr-community.jpg" width="280" alt="WeChat Cascade community QR code" /><br>
  <em>Scan to join the Cascade community and discuss KV cache, long-context inference, and LLM systems engineering.</em>
</p>
