# Cascade Benchmark 测试报告

> 测试日期：2026-07-22
> 测试环境：Tesla T4 (16GB)；历史记录称 nvfile/GDS，后续检查显示该路径实际位于 `/dev/sda3` XFS 且 cuFile 为 compatibility mode

---

## 测试环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA Tesla T4 (16GB) |
| 存储 | `/mnt/nvfile` 路径；后续实测为 `/dev/sda3` XFS |
| 模型 | Qwen2.5-7B-Instruct-AWQ |
| vLLM | 0.25.1 |
| LMCache | 0.4.3 |
| Go Engine | Cascade disk-cache (Pebble on nvfile) |

### 存储路径

| 组件 | 路径 | 说明 |
|------|------|------|
| Cascade KV 缓存 | `/mnt/nvfile/test-1/cascade-gds/` | 历史 GDS 配置；direct-GDS 前置条件未留证 |
| Cascade 元数据 | `/mnt/nvfile/test-1/cascade-meta/` | Pebble 数据库 |
| LMCache GDS 缓存 | `/mnt/nvfile/cache/` | 历史 GDS 配置；direct-GDS 前置条件未留证 |

---

## 测试参数

| 参数 | 值 |
|------|-----|
| 文档数量 | 10 |
| 文档长度 | ~2760 tokens (4096 字符目标) |
| 生成长度 | 10 tokens |
| 并发数 | 1 (串行) |
| 重复模式 | tile (相同文档重复1次) |

---

## 测试结果

> **口径校正（2026-07-23）**：以下数字保留为历史结果。当时只确认配置和 `NvFileBackend` 路径，没有保存 `nvidia_fs`、`gdscheck` 与实际挂载证据；当前同机检查为 cuFile compatibility mode。因此下文“GDS”应理解为历史 cuFile/NvFile 路径，不是已证明的 direct GDS。

### GDS 版本对比（严格测试，全新状态）

| 指标 | Cascade GDS | LMCache GDS |
|------|:----------:|:----------:|
| **Warmup TTFT** | 8.041s | 8.257s |
| **Query TTFT** | 0.389s | **0.092s** |
| **Query 总耗时** | 6.598s | 3.618s |
| **加速比** | 20.7x | 89.8x |
| **命中率** | 90% | - |

**分析**：
- Warmup（冷启动）两者相当，约 8 秒
- Query（缓存命中）LMCache 快 4.2 倍
- LMCache 优势：直接从 GDS 加载到 GPU，路径更短
- Cascade 瓶颈：Go 引擎 HTTP 调用开销

### POSIX 版本测试

| 指标 | Cascade POSIX |
|------|:----------:|
| **Warmup TTFT** | 8.105s |
| **Query TTFT** | 0.390s |
| **Query 总耗时** | 6.621s |
| **加速比** | 20.8x |
| **命中率** | 96% |

### LMCache CPU RAM 版本（同参数验证）

| 指标 | LMCache CPU RAM |
|------|:----------:|
| **Warmup TTFT** | 8.178s |
| **Query TTFT** | **0.093s** |

注：使用相同参数（10文档×4K tokens）验证，LMCache CPU RAM 与 GDS 性能一致（93ms vs 92ms）。

---

## 完整对比表

| 方案 | Warmup TTFT | Query TTFT | 加速比 | 命中率 | 说明 |
|------|:----------:|:----------:|:------:|:------:|------|
| **Cascade GDS** | 8.041s | 0.389s | 20.7x | 90% | GPU 直接读写 nvfile |
| **Cascade POSIX** | 8.105s | 0.390s | 20.8x | 96% | CPU 绕路读写 nvfile |
| **LMCache GDS** | 8.257s | **0.092s** | 89.8x | - | GPU 直接读写 nvfile |
| **LMCache CPU RAM** | 8.178s | **0.093s** | 88.0x | - | CPU 内存缓存 |

注：LMCache CPU RAM 使用相同参数（10文档×4K tokens）验证，与 GDS 性能一致。

---

## 关键发现

### 1. Block-level Hashing 修复

**问题**：早期 `prefix_key` 只 hash 前 16 个 token，导致 chat template 使所有请求共享同一 key，chunks 互相覆盖。

**修复**：改用 block-level cumulative hashing，每个 block 使用 `hash(tokens[0:block_end])` 作为存储 key。

**效果**：命中率从 47% 提升到 90-96%。

### 2. 必须禁用 vLLM 内置 Prefix Caching

使用 `--no-enable-prefix-caching` 启动 vLLM，否则 vLLM 的内置缓存会绕过 Cascade 的 match 接口，导致 `External prefix cache hit rate: 0%`。

### 3. GDS vs POSIX

- GDS 和 POSIX 的 Query TTFT 基本相同（0.389s vs 0.390s）
- 瓶颈不在 I/O，而在 Go 引擎 HTTP 调用和文件路径解析
- GDS 优势主要体现在大文件顺序读写场景

### 4. Cascade vs LMCache 差距分析

LMCache Query TTFT (~0.093s) 比 Cascade (~0.389s) 快 **4.2 倍**，原因：

| 因素 | LMCache | Cascade |
|------|---------|---------|
| 数据路径 | GPU 直接加载 | Go 引擎 → 文件系统 → Python → GPU |
| 元数据查询 | 内存 hash 表 | Pebble + HTTP API |
| 语言开销 | Python + C++ (进程内) | Go + HTTP + Python (跨进程) |

**关键洞察**：LMCache 无论 CPU RAM 还是 GDS 都是 ~93ms，说明瓶颈不在 I/O，而在 Cascade 的 Go 引擎 HTTP 调用开销。

---

## 优化建议

1. **减少 Go 引擎 HTTP 开销**：考虑 gRPC 或 Unix Socket 替代 HTTP
2. **Pebble 查询优化**：预热缓存、批量查询
3. **并行加载**：多层 KV 并行加载而非串行
4. **内存元数据缓存**：在 connector 端缓存热门 prefix 的匹配结果

---

## 测试脚本

### 启动 Cascade GDS

```bash
# 启动 Go 引擎（Pebble 在 nvfile 上）
/opt/cascade/disk-cache \
  -listen :9100 \
  -cache-path /mnt/nvfile/test-1/cascade-storage \
  -metadata-path /mnt/nvfile/test-1/cascade-meta \
  -max-size 50GB

# 启动 vLLM + Cascade
vllm serve /mnt/data/Qwen2.5-7B-Instruct-AWQ \
  --served-model-name qwen25-7b \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --kv-transfer-config '{
    "kv_connector": "DiskCacheConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "adapter.vllm.connector",
    "kv_connector_extra_config": {
      "disk_cache_path": "/mnt/nvfile/test-1/cascade-gds",
      "disk_cache_engine_addr": "http://localhost:9100",
      "target_device": "auto",
      "disk_cache_chunk_size_tokens": 256,
      "storage_backend": "gds"
    }
  }'
```

### 启动 LMCache GDS

```bash
export LMCACHE_CONFIG_FILE=/mnt/data/vllmtest/configs/gds-config.yaml

vllm serve /mnt/data/Qwen2.5-7B-Instruct-AWQ \
  --served-model-name qwen25-7b \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```

### 运行 Benchmark

本报告中的 `128.258ms` Cascade 结果属于 **T4 单请求 TTFT 诊断口径**，参数如下：

```bash
python3 LMCache/benchmarks/long_doc_qa/long_doc_qa.py \
  --model qwen25-7b \
  --host 127.0.0.1 --port 8000 \
  --num-documents 10 \
  --document-length 4096 \
  --output-len 10 \
  --repeat-count 1 \
  --repeat-mode tile \
  --max-inflight-requests 1 \
  --sleep-time-after-warmup 1 \
  --json-output
```

它同时缩小了文档数、文档长度、输出长度和并发，不能代表官方长文档压力口径。正式 GDS/POSIX/LMCache A/B 应使用 `lmcache bench engine` 的共享、模型归一化 `bench_config.json`；`46 × 10000 × output 100 × inflight 4` 仅保留为历史固定压力配置，对当前模型约对应 26.5GB KV，而不是 10GB。

---

## GDS 合并读取验证（2026-07-23）

> 当前测试机缺少 `nvidia_fs`，`gdscheck.py -p` 显示 `properties.use_compat_mode : true`，且 `/mnt/nvfile` 实际位于 `/dev/sda3` XFS。因此以下结果是 **NvFile/cuFile compatibility-mode**，不是 direct GDS。

### Storage 层微基准

28 个连续 layer slices、14,680,064 bytes payload，20 样本：

| 实现 | cuFile reads/chunk | Mean | Median | Min–Max |
|------|-------------------:|-----:|-------:|--------:|
| 旧版逐层读取 | 28 | 3.281ms | 3.227ms | 3.210–4.103ms |
| 新版合并读取 | 1 | 1.726ms | 1.718ms | 1.690–1.876ms |

median speedup 为 `1.878×`，新旧 28 层 tensor 逐字节一致。

### 4K 单请求 TTFT 诊断

同一 `10 documents × 4096 tokens × output 10 × tile × inflight 1` 口径：

| 后端 | 样本 | Mean TTFT |
|------|-----:|----------:|
| Cascade POSIX 合并读取 | 20 | 128.258ms |
| Cascade GDS compatibility-mode run 1 | 10 | 104.295ms |
| Cascade GDS compatibility-mode run 2 | 10 | 99.318ms |
| **GDS compatibility-mode aggregate** | **20** | **101.807ms** |

GDS compatibility-mode 在该诊断口径下比 POSIX 快约 `26.451ms / 20.6%`。所有 query 成功，内置 prefix cache 关闭，run 2 engine 增量为 `MatchHits +25`、`MatchedTokens +87650`、`ChunksRetrieved +365`、`ChunksStored +0`。

### LMCache 官方 `bench engine` 方法

正式配置使用 `10GB / 10000 tokens / 1 query per document / tile / inflight 4`。`tokens_per_gb_kvcache` 必须按当前模型测量；Qwen3-8B 官方教程中的 `46020` 不能直接用于 Qwen2.5-7B-AWQ。当前 256-token 完整 chunk 实测为 14,745,600 bytes，对应约 `17,361 tokens/GB`，因此派生 17 个文档。

正式 GDS compatibility-mode run：

- 17/17 请求成功，0 失败；工作集实际占用 9,865,150,464 bytes。
- Mean TTFT `1367.082ms`，p50 `958.639ms`，p90 `2523.867ms`，p99 `2524.856ms`。
- Engine 增量：`MatchRequests +34`、`MatchHits +17`、`MatchedTokens +169728`、`ChunksStored +697`、`ChunksRetrieved +663`。
- 该 run 证明 GDS 路线可在官方压力方法下正确工作；由于本轮未重跑同配置 POSIX/LMCache，不能用它单独得出正式后端性能排名。

结果保存在远端 `/mnt/data/vllmtest/cascade-gds-coalesced/results/`。

---

## 附录：Go Engine 统计数据

### Cascade GDS 测试结束时

```json
{
  "BlocksStored": 700,
  "BlocksRetrieved": 672,
  "MatchRequests": 25,
  "MatchHits": 24,
  "MatchedTokens": 41344,
  "SentinelEntries": 2876,
  "ChunksRetrieved": 672
}
```

**命中率**: 24/25 = 96%
