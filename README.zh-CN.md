<p align="center">
  🇬🇧 <a href="README.md">English</a> | 🇨🇳 <a href="README.zh-CN.md">简体中文</a>
</p>

# Cascade

> **高性能磁盘 KV 缓存层，让 LLM 推理突破显存限制，支持更长上下文。**

[![Go Version](https://img.shields.io/badge/Go-1.23+-00ADD8?logo=go)](https://golang.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](#license)
[![vLLM](https://img.shields.io/badge/vLLM-Compatible-8A2BE2)](https://github.com/vllm-project/vllm)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](#contributing)

---

## 愿景

LLM 推理本质上是显存受限的。GPU HBM（每张 H100 约 80 GB）限制了模型能处理的 token 数量，迫使运维人员在上下文长度和批处理大小之间做取舍。

**Cascade** 通过在推理引擎下方增加一层高性能、分布式的磁盘缓存，将 KV cache 与 GPU 显存解耦。结果是：更长的上下文窗口、更高的吞吐量、更低的每 token 成本——无需修改模型，也无需购买更多 GPU。

### 为什么用磁盘？

| | GPU HBM | 本地 NVMe | 远端 NVMe (RDMA) |
|---|---|---|---|
| 容量 | 80 GB | 2–30 TB | ∞（集群） |
| 延迟 | ~1 µs | ~10 µs | ~100 µs |
| 带宽 | 2000 GB/s | 7 GB/s | 100–500 GB/s |
| 成本/GB | ~$100 | ~$0.10 | ~$0.05 |

核心洞察：**NVMe 的延迟和带宽对 KV cache 来说是可行的**，而且成本优势巨大。将热数据放在 GPU 上，冷数据无缝分层到磁盘，即可实现 1M+ token 上下文而无需重建基础设施。

---

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                   推理引擎 (vLLM)                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │            DiskCache KVConnector (Python)            │    │
│  │  • 调度器钩子: 缓存命中检测、淘汰决策               │    │
│  │  • Worker 钩子:  KV tensor 的磁盘读写               │    │
│  └───────────────────────┬─────────────────────────────┘    │
│                          │ HTTP / 本地文件系统                │
├──────────────────────────┼──────────────────────────────────┤
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Go 引擎 (DiskCache Engine)                │    │
│  │                                                       │    │
│  │  ┌──────────────┐  ┌──────────────┐                  │    │
│  │  │  元数据管理    │  │  淘汰策略    │                  │    │
│  │  │  (Pebble/LSM) │  │  LRU         │                  │    │
│  │  └──────────────┘  └──────────────┘                  │    │
│  │                                                       │    │
│  │  ┌────────────────────────────────────────────────┐  │    │
│  │  │  存储后端                                       │  │    │
│  │  │  POSIX │ GDS │ io_uring/RDMA（计划）               │  │    │
│  │  └────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              集群管理器 (未来)                        │    │
│  │  • GPU 资源监控（显存、任务数、负载）                 │    │
│  │  • GPU 感知请求分发                                  │    │
│  │  • 池化 SSD: 跨节点 RDMA 共享 NVMe                   │    │
│  │  • 动态角色分配 (prefill/decode/storage)             │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 项目结构

```
cascade/
├── adapter/           # 推理框架适配器
│   └── vllm/          # vLLM KVConnector 实现
├── engine/            # Go 引擎核心
│   ├── cmd/           # 入口 (disk-cache 守护进程)
│   └── pkg/           # 核心库
│       ├── cache/     # 缓存引擎 (Pebble 元数据 + LRU)
│       ├── eviction/  # 淘汰策略
│       ├── metadata/  # 块元数据存储
│       └── storage/   # 存储后端
├── csrc/              # C 存储原语 (GDS, RDMA, io_uring)
├── deploy/            # 部署 (Helm Charts)
├── docs/              # 文档
├── scripts/           # 基准测试和工具脚本
└── test/              # 集成测试和基准测试
```

---

## 当前进度

> **阶段 1 — 本地磁盘缓存 MVP** ✅

| 组件 | 状态 | 说明 |
|---|---|---|
| Go 引擎核心 | ✅ **完成** | Pebble 元数据存储、LRU 淘汰、HTTP API |
| vLLM 连接器 | ✅ **完成** | 完整 KVConnectorBase_V1 实现 (~185 行) |
| 磁盘 I/O 基准测试 | ✅ **完成** | 顺序/随机读写、延迟分析 |
| 基准测试套件 | ✅ **完成** | 对比原生 vLLM vs LMCache vs Cascade |
| GDS 存储后端 | ✅ **完成** | POSIX/GDS 后端抽象，支持自动降级 |
| GPU 感知调度 | 📋 **计划中** | 每卡显存/任务监控 + 智能分发 |
| 池化 SSD 集群 | 📋 **计划中** | 跨节点 RDMA 共享存储池 |
| SGLang 适配器 | 📋 **计划中** | 目录结构已就绪 |
| Helm 部署 | 📋 **计划中** | Chart 脚手架已就绪 |

---

## 验证与 CI targets

```bash
# 快速本地 Go 覆盖
make test-go

# Python adapter/helper 覆盖（需要 adapter 依赖）
make test-adapter

# CI 友好组合：Go + adapter 测试 + POSIX 存储 smoke
make ci

# GPU 存储验证：CI 组合 + POSIX/GDS 基准测试
make ci-gpu

# 真实 vLLM + disk-cache 验证（需要 MODEL_PATH、GPU 和 vLLM）
make test-vllm-cache
```

- `make test-storage` 封装 `scripts/validate_storage_backend.py`；可覆盖 `STORAGE_BACKEND`、`STORAGE_DEVICE`、`STORAGE_SHAPE` 或 `STORAGE_DTYPE` 执行 POSIX/GDS/cuda smoke。
- `make bench-storage` 封装 `scripts/benchmark_storage_backend.py`；可覆盖 `STORAGE_BENCH_*` 变量，或设置 `STORAGE_BENCH_MARKDOWN=docs/storage-benchmark-a100.md` 刷新带 marker 的报告。
- `make test-vllm-cache` 封装 `scripts/validate_vllm_disk_cache.sh`，启动隔离的 disk-cache + vLLM 服务并检查命中统计。
- `engine/pkg/cache/Stats` 同时保留旧字段和更细粒度的元数据/事件计数，便于诊断。

## 快速开始

### 1. 编译 Go 引擎

```bash
make build-engine
# 输出: bin/disk-cache
```

### 2. 启动 DiskCache 守护进程

```bash
./bin/disk-cache \
    --cache-path /mnt/nvme/kv-cache \
    --metadata-path /tmp/disk-cache-meta \
    --max-size 100GB \
    --listen :9100
```

### 3. 启动 vLLM 并启用 DiskCache 连接器

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --no-enable-prefix-caching \
    --tensor-parallel-size 8 \
    --max-model-len 100000 \
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

如需可复现的真实 vLLM smoke，优先使用 `make test-vllm-cache`；它会构建
engine、启动隔离服务并检查 retrieval 统计。

### 4. 验证

```bash
# 引擎状态
curl http://localhost:9100/stats

# 本地 CI 友好检查
make ci

# 可选：在 GPU 主机刷新 A100 POSIX/GDS 基准报告
STORAGE_BENCH_MARKDOWN=docs/storage-benchmark-a100.md make bench-storage
```

---

## 路线图
### 阶段 1: 本地磁盘缓存 MVP ✅ *(当前)*
- [x] Go 引擎: Pebble 元数据 + LRU 淘汰 + HTTP API
- [x] vLLM KVConnector: KV tensor 的磁盘读写
- [x] 基准测试套件: 对比原生 / LMCache / Cascade
- [x] 磁盘 I/O 分析工具

### 阶段 2: GPUDirect Storage 加速 🚀 *(已完成)*
- [x] Python StorageBackend 抽象层 (GDS + POSIX 自动降级)
- [x] NvFileBackend: GPU↔NVMe 零拷贝 (cuFile/nvfile/hipfile API)
- [x] PosixBackend: 自动降级 (cudaMemcpy + safetensors)
- [x] 自动选择: GDS → POSIX 回退 (可通过 `storage_backend` 配置)
- [x] Level 1+2 测试: mock / GPU 降级路径 (17 个测试，无 GDS 硬件也能跑)

### 阶段 3: GPU 感知集群调度 📋
- [ ] **GPU 资源监控**: 每卡显存占用、运行任务数、利用率
- [ ] **GPU 感知请求分发**: 根据显存容量和 GPU 负载路由请求，替代 nginx 轮询
- [ ] **池化 SSD 存储**: 全节点 RDMA 共享 NVMe 池
- [ ] **显存准入控制**: 无 GPU 有足够显存时，淘汰冷 KV block 到池化 SSD 腾出空间
- [ ] **动态角色分配**: 节点根据实时负载自动切换 prefill/decode/storage 角色
- [ ] **多卡协同调度**: 为张量并行模型同时预留 N 块 GPU
- [ ] **基于 etcd 的节点注册与发现**
- [ ] **SGLang 适配器**
- [ ] **故障转移与数据迁移**

### 阶段 4: 产品化 📋
- [ ] Helm Chart (Kubernetes 部署)
- [ ] Prometheus / Grafana 监控
- [ ] 管理面板
- [ ] 多租户
- [ ] 完整文档和示例

---

## 基准测试

```bash
# 1. 先测试 NVMe 磁盘性能
python3 scripts/disk-bench.py /mnt/nvme

# 2. 运行缓存引擎基准测试
python3 scripts/disk-bench-cache.py http://localhost:9100

# 3. 在 GPU 主机对比存储后端吞吐
make bench-storage

# 4. 刷新仓库内 A100 POSIX/GDS 报告
STORAGE_BENCH_MARKDOWN=docs/storage-benchmark-a100.md make bench-storage
```

推理 benchmark 方法见 `docs/benchmark-plan.md`，最新 A100 存储后端报告见 `docs/storage-benchmark-a100.md`。

---

## 方案对比

### 设计理念

| | LMCache | Mooncake | **Cascade** |
|---|---|---|---|
| **定位** | 分层缓存引擎 | 分布式 KV 传输引擎 | **集群磁盘缓存** |
| **磁盘角色** | 温数据层（CPU→Disk） | 内存溢出卸载目标 | **🎯 主存储层** |
| **数据路径** | GPU → CPU → Disk | GPU 内存 ↔ RDMA → 远端 GPU | **GPU → NVMe (GDS) → 集群 (RDMA)** |
| **存储节点** | ❌ 必须有 GPU | ❌ 必须有 GPU | **✅ 支持纯磁盘节点** |

### 当前实现（Phase 1 MVP）

| 特性 | LMCache | Mooncake | **Cascade** |
|---|---|---|---|
| **磁盘缓存** | ✅ LocalDiskBackend | ✅ FileStorage (SSD offload) | **✅ 核心设计** |
| **KV 数据 I/O** | Python `open/write` | C++ io_uring / POSIX | **Python safetensors*** |
| **元数据存储** | Python 内存 dict | etcd + Master 服务 | **Pebble (LSM 树)** |
| **元数据持久化** | ❌ 重启丢失 | ✅ etcd | **✅ Pebble** |
| **淘汰策略** | ✅ LRU / LFU / FIFO / MRU | ✅ LRU / FIFO | **✅ LRU** |
| **前缀匹配** | ✅ TokenDatabase | ❌ 不透明 key | **✅ SHA-256 增量哈希 + 哨兵** |
| **vLLM 集成** | ✅ 深度集成 | ✅ mooncake-integration | **✅ KVConnectorBase_V1** |
| **核心引擎代码量** | ~79K 行 Python | ~220K 行 C++ | **~400 行 Go** |

> *Phase 1: Python safetensors 经 CPU bounce buffer 直接读写磁盘。
> Phase 2: 新增 Python StorageBackend 抽象层，支持 GPU↔NVMe 零拷贝 (GDS) 和自动降级。

### 规划架构（设计目标）

| 特性 | LMCache | Mooncake | **Cascade** |
|---|---|---|---|
| **I/O 栈** | Python 原生 | C++ 原生 | **Python connector → Go 引擎 → C 后端** |
| **GPU↔NVMe** | ✅ GdsBackend (部分) | ❌ 不支持 | **✅ GPUDirect Storage (cuFile/nvfile)** |
| **跨节点传输** | ❌ 不支持 RDMA | ✅ RDMA (核心能力) | **📋 RDMA (ibverbs)** |
| **异步磁盘 I/O** | ❌ 不支持 | ✅ io_uring | **📋 io_uring** |
| **GPU 资源调度** | ❌ 无 | ❌ 仅手动角色分配 | **📋 GPU 感知: 显存/任务监控 + 智能分发** |
| **集群管理** | ❌ 仅 P2P (ZMQ) | ✅ Master + etcd + HA | **📋 ClusterManager + etcd** |
| **池化 SSD 存储** | ❌ 无 | ❌ 仅本地 offload | **📋 RDMA 共享 NVMe 池** |
| **纯存储节点** | ❌ 无此概念 | ❌ 必须有 GPU | **📋 支持无 GPU 节点** |
| **SGLang 适配** | ❌ 不支持 | ✅ 已支持 | **📋 脚手架就绪** |

### 架构演进

```
Phase 1 (已完成) Python safetensors 经 CPU bounce buffer 直接读写磁盘
                 Go 引擎管理元数据 (Pebble) + 淘汰策略 (LRU) — 通过 HTTP

Phase 2 (已完成) Python StorageBackend 抽象层
                   ├── NvFileBackend: GPU↔NVMe 零拷贝 (cuFile/nvfile)
                   └── PosixBackend: CPU bounce buffer 降级 (safetensors)
                 自动选择: GDS → POSIX，无需改代码

Phase 3 (规划中)   GPU 感知集群: 每卡显存/任务监控
                 池化 SSD: 全节点 RDMA 共享 NVMe
                 io_uring 异步磁盘 I/O
                 SGLang 适配器
```

---

## 文档

- [设计文档](./DESIGN.md) — 详细架构和设计原理
- [基准测试方案](./docs/benchmark-plan.md) — 测试方法论
- [vLLM 基础部署](./docs/baseline-vllm-deepseek-v4.md) — 参考部署方案

---

## 贡献

欢迎贡献！本项目处于早期活跃开发阶段，有大量机会可以产生影响：

- **工程师**: 帮助实现 GDS、RDMA、集群管理器或存储后端
- **ML 从业者**: 运行基准测试、报告结果、提出优化建议
- **基础设施专家**: 改进部署、监控和可观测性

请提交 Issue 或 Pull Request。重大变更请先发起讨论。

---

## 许可证

Cascade 采用**双授权**模式：

- **Apache 2.0** — 开源项目、个人开发者、非商业使用免费
- **商业授权** — 嵌入硬件设备、商业产品或专有解决方案需购买

详见 [COMMERCIAL_LICENSE.md](./COMMERCIAL_LICENSE.md)

[Apache License 2.0](./LICENSE)

---

<p align="center">
  <b>LLM 推理不应该被显存限制。</b><br>
  Cascade — 池化 NVMe，延伸上下文。
</p>
