# 基准测试方案

> 对比 vLLM 原生 vs LMCache vs 我们的 DiskCache 在 DeepSeek V4 Flash 上的性能。

---

## 一、测试环境

### 硬件

| 组件 | 规格 |
|------|------|
| GPU | 2× A800 80GB |
| 磁盘 | 500GB+ NVMe SSD |
| CPU | ≥32 核 |
| 内存 | ≥256 GB |

### 软件

| 组件 | 版本 |
|------|------|
| vLLM | 最新（支持 DeepSeek V4） |
| LMCache | latest（集成在 vLLM 中） |
| 我们的 DiskCache | 当前代码 |
| CUDA | ≥12.4 |

---

## 二、测试场景

### 场景 1：短上下文（2k tokens）

测试 prefill + decode 基础性能，磁盘缓存尚未发挥作用。

```
prompt: 2048 tokens
gen:    128 tokens
并发:   1 / 4 / 8 请求
```

### 场景 2：中等上下文（32k tokens）

磁盘缓存开始起作用——前缀缓存命中。

```
prompt: 32768 tokens
gen:    128 tokens
测试 A: 单请求，首次（缓存未命中）
测试 B: 相同前缀，第二次（缓存命中）
```

### 场景 3：长上下文（100k tokens）

磁盘缓存的核心场景——显存不够，磁盘来凑。

```
prompt: 100000 tokens
gen:    128 tokens
测试:   首次 vs 后续（前缀匹配）
```

### 场景 4：长时间会话

模拟 Agent 场景，多轮对话积累 KV cache。

```
第一轮: 5000 tokens → 生成 500 tokens
第二轮: +2000 tokens → 生成 500 tokens
...
直到磁盘写满，触发 LRU 淘汰
```

---

## 三、测试配置

### 1. vLLM 原生（基准）

```bash
# 2×A80，32k 上下文
vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --tensor-parallel-size 2 \
    --dtype auto \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.95 \
    --enable-prefix-caching \
    --port 8000
```

### 2. vLLM + LMCache

```bash
# 安装 LMCache
pip install lmcache

# 启动，启用本地磁盘缓存
vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --tensor-parallel-size 2 \
    --dtype auto \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --kv-transfer-config \
      '{"kv_connector":"lmcache","kv_connector_extra_config":{
        "local_disk":"/mnt/nvme/lmcache-cache",
        "max_local_disk_size":100
      }}'
```

### 3. vLLM + 我们的 DiskCache

```bash
# 启动 Go 引擎（先开一个终端）
./bin/disk-cache \
    --cache-path /mnt/nvme/disk-cache \
    --max-size 100GB \
    --listen :9100

# 启动 vLLM
vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --tensor-parallel-size 2 \
    --dtype auto \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.95 \
    --kv-transfer-config \
      '{"kv_connector":"disk-cache","kv_connector_extra_config":{
        "disk_cache_path":"/mnt/nvme/disk-cache",
        "disk_cache_engine_addr":"http://localhost:9100"
      }}'
```

---

## 四、关键指标

| 指标 | 单位 | 说明 |
|------|------|------|
| **TTFT** | ms | 首 token 延迟 |
| **TPOT** | ms/token | 每 token 生成时间 |
| **Prefill Throughput** | tokens/s | 预填充吞吐 |
| **Gen Throughput** | tokens/s | 生成吞吐 |
| **Cache Hit Rate** | % | 磁盘缓存命中率（相比重新计算） |
| **GPU Memory Saved** | GB | 释放的显存量 |
| **Max Context** | tokens | 不 OOM 的最大上下文 |

---

## 五、预期差异

| 场景 | 原生 vLLM | LMCache | 我们的 DiskCache |
|------|----------|---------|-----------------|
| 短上下文 | ✅ 正常 | ✅ 正常 | ✅ 正常 |
| 中等上下文（首次） | ✅ 正常 | ✅ 正常 | ✅ 正常（写盘略慢） |
| 中等上下文（命中） | ✅ APC 命中 | ✅ 磁盘命中 | ✅ 磁盘命中 |
| 长上下文（显存不够） | ❌ OOM | ✅ 换到磁盘 | ✅ 换到磁盘 |
| LRU 淘汰 | ❌ 无 | ✅ 有 | ✅ 有 |
| 可观测性 | ❌ 少 | 基本 | ✅ stats API |

---

## 六、测试脚本

```bash
# 安装依赖
pip install aiohttp openai

# 运行对比测试
python3 scripts/bench_compare.py \
    --vllm-url http://localhost:8000/v1 \
    --prompt-lens 2048,32768,65536,100000 \
    --gen-tokens 128 \
    --concurrency 1,4,8 \
    --output results.csv
```
