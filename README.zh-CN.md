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
| 带宽 | 2000 GB/s | 7 GB/s | 25 GB/s |
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
│  │  │  POSIX │ io_uring │ GPUDirect Storage (未来)    │  │    │
│  │  └────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              集群管理器 (未来)                        │    │
│  │  • 基于 etcd 的节点发现                              │    │
│  │  • 动态角色分配 (prefill/decode/storage)             │    │
│  │  • RDMA 数据传输                                    │    │
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
| GDS / RDMA | 📋 **计划中** | C 存储原语已建目录 |
| SGLang 适配器 | 📋 **计划中** | 目录结构已就绪 |
| 集群管理器 | 📋 **计划中** | 架构已在 DESIGN.md 中设计 |
| Helm 部署 | 📋 **计划中** | Chart 脚手架已就绪 |

---

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

### 4. 验证

```bash
# 引擎状态
curl http://localhost:9100/stats

# 运行基准测试
python3 test/scripts/run_bench.py --mode diskcache
```

---

## 路线图

### 阶段 1: 本地磁盘缓存 MVP ✅ *(当前)*
- [x] Go 引擎: Pebble 元数据 + LRU 淘汰 + HTTP API
- [x] vLLM KVConnector: KV tensor 的磁盘读写
- [x] 基准测试套件: 对比原生 / LMCache / Cascade
- [x] 磁盘 I/O 分析工具

### 阶段 2: GPUDirect Storage 加速 📋 *(下一步)*
- [ ] C GDS 后端 (NVIDIA cuFile)
- [ ] Go CGo 绑定，实现 GPU↔NVMe 零拷贝
- [ ] 自动回退: GDS → POSIX
- [ ] 吞吐目标: 比 CPU bounce buffer 快 3–5 倍

### 阶段 3: 分布式集群 📋
- [ ] 集群管理器: 基于 etcd 的节点注册与发现
- [ ] 动态角色分配 (prefill / decode / storage)
- [ ] 节点间 RDMA 数据传输
- [ ] SGLang 适配器
- [ ] 故障转移与数据迁移

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

# 3. 对比推理引擎
python3 test/scripts/run_bench.py --mode native
python3 test/scripts/run_bench.py --mode diskcache
python3 test/scripts/gen_report.py
```

结果保存在 `test/results/` 目录。详细方法见 `docs/benchmark-plan.md`。

---

## 方案对比

| 特性 | 原生 vLLM | LMCache | Cascade |
|---|---|---|---|
| 磁盘缓存 | ❌ 无 | ✅ 有 | ✅ 有 |
| 淘汰策略 | ❌ 仅 APC | LRU | **LRU + 分层** |
| 元数据引擎 | 内存 | SQLite | **Pebble (LSM 树)** |
| GDS 支持 | ❌ 无 | 部分 | **计划中** |
| 集群感知 | ❌ 无 | ❌ 无 | **计划中** |
| 可观测性 | 极少 | 基本 | **Stats API** |
| 纯存储节点 | ❌ 无 | ❌ 无 | **计划中** |

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
  Cascade — 每次一块 NVMe，扩展无限上下文。
</p>
