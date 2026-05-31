<p align="center">
  🇬🇧 <a href="README.md">English</a> | 🇨🇳 <a href="README.zh-CN.md">简体中文</a>
</p>

<h1 align="center">Cascade</h1>

<p align="center">
  <strong>面向长上下文 LLM 推理的磁盘 KV Cache。</strong><br>
  用 Go 元数据引擎、POSIX/GDS 存储后端和池化 NVMe 集群路线图，让 vLLM 突破显存边界。
</p>

<p align="center">
  <a href="https://golang.org"><img alt="Go" src="https://img.shields.io/badge/Go-1.24+-00ADD8?logo=go"></a>
  <a href="https://github.com/vllm-project/vllm"><img alt="vLLM" src="https://img.shields.io/badge/vLLM-0.21%20validated-8A2BE2"></a>
  <img alt="A100" src="https://img.shields.io/badge/A100-validated-76B900?logo=nvidia">
  <img alt="GDS" src="https://img.shields.io/badge/GDS-backend-0B7285">
  <a href="#许可证"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0%20%2B%20Commercial-blue"></a>
</p>

---

## 为什么做 Cascade

LLM 服务越来越受 KV cache 显存占用限制。GPU HBM 极快，但昂贵且容量有限；长 prompt、Agent 工作流、重复检索型对话都会把上下文长度变成基础设施问题。

**Cascade** 把本地 NVMe 视为一层真正可用的 KV cache。它保持推理引擎接入面很小，把元数据和淘汰逻辑放进紧凑的 Go 服务，并让 Python vLLM connector 通过可插拔存储后端保存/加载 KV tensor：

- **POSIX 后端**：可移植的 CPU bounce-buffer 存储路径。
- **GPUDirect Storage (GDS) 后端**：在支持的主机上实现 GPU↔NVMe 传输。
- **集群路线图**：面向 RDMA 池化 SSD 和 GPU 感知调度。

目标很直接：**不改模型，也能让长上下文推理更便宜、更弹性。**

### 为什么用磁盘？

| | GPU HBM | 本地 NVMe | 远端 NVMe (RDMA) |
|---|---:|---:|---:|
| 容量 | ~80 GB / GPU | 2–30 TB / 节点 | 集群级 |
| 延迟 | ~1 µs | ~10–100 µs | ~100 µs+ |
| 带宽 | ~2 TB/s | ~7 GB/s | 100–500 GB/s |
| 成本/GB | 高 | 低 | 规模化后更低 |

NVMe 不是替代 GPU 上的热 KV，而是作为冷 KV / 可复用 KV block 的高容量层。用少量存储访问延迟，换取显存压力的大幅下降。

---

## 现在已经能跑什么

| 方向 | 状态 | 说明 |
|---|---|---|
| vLLM 集成 | ✅ 可用 | 面向 vLLM 0.21 风格 V1 执行路径的 `KVConnectorBase_V1` connector。 |
| DiskCache 引擎 | ✅ 可用 | Go 服务，包含 HTTP API、Pebble 元数据、LRU 淘汰、持久 block/chunk 索引。 |
| 存储后端 | ✅ 可用 | POSIX 后端 + GDS/NvFile 后端，支持自动降级。 |
| 缓存命中验证 | ✅ 可用 | 脚本自动启动 disk-cache + vLLM，检查 retrieved block 计数和 cached token 统计。 |
| A100 真机验证 | ✅ 可用 | 已在 A100 + Qwen2.5-7B-Instruct + vLLM 0.21.0 上验证。 |
| 集群调度 | 🚧 路线图 | GPU 感知分发、RDMA 池化 NVMe、多节点协调仍在规划中。 |

---

## 验证快照

以下结果来自项目 A100 验证主机，仅代表该环境样本，不是通用性能承诺。

### 真实 vLLM + disk-cache smoke

| 项目 | 数值 |
|---|---|
| GPU | NVIDIA A100-PCIE-40GB |
| vLLM | 0.21.0 |
| 模型 | Qwen2.5-7B-Instruct |
| Prompt 规模 | 6,629 prompt tokens（`REQUEST_REPETITIONS=200`, `max_model_len=8192`） |
| 第一次请求 | `1.704s`，retrieved blocks `0` |
| 第二次请求 | `0.199s`，retrieved blocks `28`，cached tokens `6624` |

完整验证命令和环境说明见 [release notes](./docs/release-notes.md)。

### POSIX vs GDS 存储后端基准

| 后端 | 实际实现 | Save median | Load median |
|---|---|---:|---:|
| POSIX | `PosixBackend` | `0.066639s` / `480.20 MiB/s` | `0.009055s` / `3534.14 MiB/s` |
| GDS | `NvFileBackend` | `0.020430s` / `1566.36 MiB/s` | `0.005160s` / `6202.14 MiB/s` |

可复现、marker-aware 的报告见 [A100 storage benchmark](./docs/storage-benchmark-a100.md)。

---

## 架构

```text
┌──────────────────────────────────────────────────────────────────────┐
│                            vLLM / LLM Engine                         │
│                                                                      │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │ DiskCache KVConnector (Python)                               │   │
│   │ - 调度侧 prefix/cache-hit 查询                                │   │
│   │ - worker 侧 KV tensor 保存/加载                               │   │
│   │ - POSIX/GDS 后端选择                                          │   │
│   └───────────────────────────────┬──────────────────────────────┘   │
└───────────────────────────────────┼──────────────────────────────────┘
                                    │ HTTP 元数据 + 本地文件 I/O
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         Go DiskCache Engine                           │
│                                                                      │
│   ┌────────────────────┐   ┌────────────────────┐   ┌────────────┐  │
│   │ Pebble 元数据      │   │ LRU 淘汰            │   │ HTTP stats │  │
│   │ blocks/sentinels   │   │ 容量控制            │   │ 诊断信息   │  │
│   └────────────────────┘   └────────────────────┘   └────────────┘  │
│                                                                      │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │ Storage layer                                                │   │
│   │ POSIX 已可用 | GDS 已可用 | io_uring/RDMA 规划中             │   │
│   └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        Cluster layer（路线图）                         │
│ GPU 感知路由 | 池化 SSD | RDMA | 节点注册                              │
└──────────────────────────────────────────────────────────────────────┘
```

### 项目结构

```text
cascade/
├── adapter/           # 推理框架适配器
│   └── vllm/          # vLLM KVConnector 实现
├── engine/            # Go disk-cache 引擎
│   ├── cmd/           # disk-cache、c-agent、cluster-server 入口
│   └── pkg/           # cache、metadata、server、storage、cluster 等包
├── docs/              # 设计文档、benchmark 方案、release notes
├── scripts/           # 验证与 benchmark 脚本
├── test/              # 集成测试与 benchmark helper
└── images/            # README / 社区资源
```

---

## 快速开始

### 1. 编译引擎

```bash
make build-engine
# 输出: bin/disk-cache
```

### 2. 启动 disk-cache 服务

```bash
./bin/disk-cache \
  -cache-path /mnt/nvme/kv-cache \
  -metadata-path /tmp/disk-cache-meta \
  -max-size 100GB \
  -listen :9100
```

### 3. 优先使用可复现的 vLLM smoke

在已安装 vLLM 且有本地模型路径的 GPU 主机上：

```bash
MODEL_PATH=/tmp/models/Qwen2.5-7B-Instruct \
VLLM_EXTRA_ARGS='--tensor-parallel-size 1 --max-model-len 8192' \
REQUEST_REPETITIONS=200 \
MAX_TOKENS=8 \
make test-vllm-cache
```

该脚本会编译引擎，启动隔离的 disk-cache 和 vLLM 服务，发送两次重复 prompt；如果第二次请求没有取回缓存 KV chunk，则直接失败。

### 4. 手动接入 vLLM

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

多 GPU 场景建议使用 `target_device=auto`：connector 会根据实际 KV tensor 所在设备保存/加载；只有明确需要固定设备时才显式覆盖。

---

## 验证与 CI targets

```bash
# Go 包与引擎测试
make test-go

# Python adapter/helper 测试
make test-adapter

# CI 友好组合：Go + adapter tests + POSIX storage smoke
make ci

# GPU 主机验证：CI 组合 + POSIX/GDS benchmark
make ci-gpu

# 真实 vLLM + disk-cache smoke
make test-vllm-cache
```

常用覆盖参数：

```bash
# 指定后端/设备做 storage smoke
STORAGE_BACKEND=gds STORAGE_DEVICE=cuda:0 make test-storage

# 存储后端 benchmark
STORAGE_BENCH_BACKENDS=posix,gds \
STORAGE_BENCH_DEVICE=cuda:0 \
STORAGE_BENCH_MARKDOWN=docs/storage-benchmark-a100.md \
make bench-storage
```

---

## 路线图

### 已完成

- [x] Go DiskCache 引擎：Pebble 元数据、LRU 淘汰、HTTP API、stats。
- [x] vLLM `KVConnectorBase_V1` 集成，支持 KV tensor 保存/加载。
- [x] 持久 prefix/chunk 元数据、sentinel entries 与命中诊断。
- [x] POSIX 存储后端：基于 safetensors-compatible CPU bounce path。
- [x] GDS/NvFile 后端：支持自动降级到 POSIX。
- [x] A100 验证工作流与 marker-aware 存储 benchmark 报告。

### 下一步

- [ ] 更完整的 operator 文档和部署示例。
- [ ] GPU 感知调度：显存/任务/负载监控与智能分发。
- [ ] 池化 SSD 设计：跨节点 RDMA 共享 NVMe。
- [ ] 异步磁盘 I/O 实验（`io_uring`）和更大的 benchmark matrix。
- [ ] SGLang 适配器与更多推理引擎接入点。

### 后续

- [ ] 带节点注册/发现能力的集群管理器。
- [ ] 面向 tensor-parallel 模型的多 GPU gang scheduling。
- [ ] 故障恢复、数据迁移、生产级可观测性。
- [ ] Helm chart 与托管部署模式。

---

## 基准测试

```bash
# 1. 测试本地 NVMe 磁盘
python3 scripts/disk-bench.py /mnt/nvme

# 2. 通过 HTTP 压测 cache engine
python3 scripts/disk-bench-cache.py http://localhost:9100

# 3. 在 GPU 主机对比 POSIX/GDS 存储后端
make bench-storage

# 4. 刷新仓库内 A100 benchmark 报告
STORAGE_BENCH_MARKDOWN=docs/storage-benchmark-a100.md make bench-storage
```

推理 benchmark 方法见 [Benchmark Plan](./docs/benchmark-plan.md)，最新存储报告见 [A100 Storage Backend Benchmark](./docs/storage-benchmark-a100.md)。

---

## 方案对比

| 维度 | LMCache | Mooncake | **Cascade** |
|---|---|---|---|
| 主要定位 | 分层 KV cache | 分布式 KV 传输 | **磁盘 KV cache + 未来池化 SSD** |
| 磁盘角色 | 温数据层 | 溢出/卸载目标 | **高容量 KV 主存储层** |
| 元数据 | Python / 内存路径 | Master/etcd 服务 | **Pebble-backed Go 引擎** |
| vLLM 集成 | 深度集成 | Integration plugin | **KVConnectorBase_V1** |
| GPU↔NVMe 路径 | 有 GDS backend | 不是核心设计重点 | **POSIX + GDS 后端抽象** |
| 集群方向 | Cache tiering | RDMA transfer | **GPU 感知调度 + 池化 NVMe 路线图** |

Cascade 刻意保持小而基础设施友好：Python connector 尽量薄，元数据持久化，存储后端显式可切换，再从单机 NVMe 演进到池化 SSD 集群。

---

## 文档

- [设计文档](./DESIGN.md) — 架构与设计原因。
- [Benchmark Plan](./docs/benchmark-plan.md) — benchmark 方法论。
- [A100 Storage Backend Benchmark](./docs/storage-benchmark-a100.md) — 可复现 POSIX/GDS 样本报告。
- [Release Notes](./docs/release-notes.md) — 最新验证记录。
- [vLLM Baseline Setup](./docs/baseline-vllm-deepseek-v4.md) — 参考部署笔记。

---

## 贡献

Cascade 仍处于早期，但已经能进行真实验证。适合贡献的方向包括：

- 在不同 GPU、SSD、文件系统和 vLLM 版本上运行验证脚本并反馈结果。
- 改进存储后端、缓存命中诊断或 benchmark 覆盖。
- 强化部署、可观测性和运维文档。
- 原型验证集群调度、RDMA 传输或 SGLang 集成。

欢迎提交 Issue 或 Pull Request。重大设计变更建议先写一个简短 proposal。

---

## 许可证

Cascade 采用**双授权**模式：

- **Apache 2.0** — 开源项目、个人开发者、非商业使用免费。
- **商业授权** — 嵌入硬件设备、商业产品或专有解决方案需购买。

详见 [COMMERCIAL_LICENSE.md](./COMMERCIAL_LICENSE.md)。

[Apache License 2.0](./LICENSE)

---

<p align="center">
  <strong>LLM 推理不应该被显存限制。</strong><br>
  Cascade — 用 NVMe 拉伸上下文，一次缓存一个 block。
</p>

---

## 爱好者交流群

<p align="center">
  <img src="./images/qr-community.jpg" width="280" alt="微信爱好者交流群" /><br>
  <em>扫码加入 Cascade 爱好者交流群，讨论 KV cache、长上下文推理与大模型工程实践。</em>
</p>
