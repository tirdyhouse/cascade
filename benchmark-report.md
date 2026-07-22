# Cascade Benchmark 测试报告

> 测试日期：2026-07-22
> 测试环境：Tesla T4 (16GB) + nvfile (自研高性能存储集群)

---

## 测试环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA Tesla T4 (16GB) |
| 存储 | nvfile 自研高性能存储集群 (113TB) |
| 模型 | Qwen2.5-7B-Instruct-AWQ |
| vLLM | 0.25.1 |
| LMCache | 0.4.3 |
| Go Engine | Cascade disk-cache (Pebble on nvfile) |

### 存储路径

| 组件 | 路径 | 说明 |
|------|------|------|
| Cascade KV 缓存 | `/mnt/nvfile/test-1/cascade-gds/` | GDS 直接读写 |
| Cascade 元数据 | `/mnt/nvfile/test-1/cascade-meta/` | Pebble 数据库 |
| LMCache GDS 缓存 | `/mnt/nvfile/cache/` | GDS 直接读写 |

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

### LMCache CPU RAM 版本（参考）

| 指标 | LMCache CPU RAM |
|------|:----------:|
| **Warmup TTFT** | 59.867s |
| **Query TTFT** | 0.208s |
| **命中率** | ~50% |

注：LMCache CPU RAM 测试使用不同参数（46 文档 × 10K tokens），数据仅供参考。

---

## 完整对比表

| 方案 | Warmup TTFT | Query TTFT | 加速比 | 命中率 | 说明 |
|------|:----------:|:----------:|:------:|:------:|------|
| **Cascade GDS** | 8.041s | 0.389s | 20.7x | 90% | GPU 直接读写 nvfile |
| **Cascade POSIX** | 8.105s | 0.390s | 20.8x | 96% | CPU 绕路读写 nvfile |
| **LMCache GDS** | 8.257s | **0.092s** | 89.8x | - | GPU 直接读写 nvfile |
| **LMCache CPU RAM** | 59.867s | 0.208s | - | ~50% | CPU 内存缓存 |

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

LMCache Query TTFT (0.092s) 比 Cascade (0.389s) 快 4.2 倍，原因：

| 因素 | LMCache | Cascade |
|------|---------|---------|
| 数据路径 | GDS → GPU 直接加载 | GDS → Go 引擎 → GPU |
| 元数据查询 | 内存 hash 表 | Pebble + HTTP API |
| 语言开销 | Python + C++ | Go + HTTP + Python |

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
      "disk_cache_chunk_size_mb": 64,
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
