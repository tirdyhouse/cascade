# 基准方案：单机 vLLM 运行 DeepSeek V4 Flash

> 本文档描述如何使用原生 vLLM 在单机上部署 DeepSeek V4 Flash 模型，
> 作为后续磁盘缓存方案的性能基准比较对象。

---

## 一、模型概述

### DeepSeek V4 Flash

| 属性 | 值 |
|------|-----|
| 参数总量 | 284B |
| 每 token 激活参数 | ~13B（MoE） |
| 上下文长度 | 1M tokens |
| 模型架构 | MLA（Multi-head Latent Attention）+ DeepSeekMoE |
| 推荐精度 | FP8（原生支持） |
| HuggingFace | `deepseek-ai/DeepSeek-V4-Flash` |
| License | DeepSeek License |

### 为什么选这个模型

- vLLM **原生支持** DeepSeek V4（代码中已有 `DeepseekV4ForCausalLM` 完整实现）
- MLA 压缩 KV cache 技术本身就是 DeepSeek 的招牌
- 1M 上下文窗口是磁盘缓存的最佳应用场景
- 284B 参数 / 13B 激活，FP8 下可以在 8×H100 上运行

---

## 二、硬件要求

### 最低配置

| 组件 | 规格 | 说明 |
|------|------|------|
| GPU | **8× NVIDIA H100 80GB** | FP8 推理必需，单卡 80GB |
| CPU | AMD EPYC / Intel Xeon, ≥32 核 | 模型加载、prefill 调度 |
| 内存 | ≥256 GB | 模型权重加载 + CPU 内存池 |
| 磁盘 | **NVMe SSD ≥2TB** | 模型权重存储 + 测试数据 |
| 网络 | InfiniBand / RoCE（可选） | 多卡通信（NVLink 已够） |
| CUDA | ≥12.4 | FP8 支持 |
| PyTorch | ≥2.5 | vLLM 必需 |

### 为什么是 8×H100

DeepSeek V4 Flash 在 FP8 下的显存估算：

```
模型权重（FP8）:  ~284B × 1 byte = 284 GB
KV cache (100k): ~每个 token 约 1.5 MB × 100k = 150 GB
其他开销:         ~30 GB
总计:             ~464 GB

8×H100 (80GB):   640 GB 总量 ✅（有余量）
4×H100 (80GB):   320 GB ❌（不够）
```

> 注：vLLM 的 Prefix Caching 和 MLA 压缩可以显著降低 KV cache 需求，
> 但作为基准方案，我们按最保守估算。

### 替代GPU（如果 H100 不可用）

| GPU | 数量 | 可行性 | 说明 |
|-----|------|--------|------|
| H100 80GB | 8 | ✅ 推荐 | 官方推荐配置 |
| A100 80GB | 8 | ✅ 可行 | 无 FP8，需 FP16，显存需求翻倍，上下文受限 |
| H200 141GB | 4 | ✅ 可行 | 大显存，4 卡够 |
| A100 40GB | 8 | ❌ 不够 | 40GB 太小，284B 模型装不下 |
| L40S 48GB | 8 | ❌ 不够 | 同上 |

---

## 三、vLLM 安装

### 方式一：pip 安装（推荐）

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装 vLLM（需要 CUDA 12.4+）
pip install --upgrade pip
pip install vllm
```

### 方式二：Docker（生产推荐）

```bash
# 拉取官方镜像
docker pull vllm/vllm-openai:latest

# 启动容器
docker run --gpus all \
    --shm-size 32g \
    -v /mnt/nvme/models:/models \
    -v /mnt/nvme/kv-cache:/kv-cache \
    -p 8000:8000 \
    vllm/vllm-openai:latest
```

### 方式三：源码安装（开发用）

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

### 验证安装

```bash
# 检查 vLLM 版本
python -c "from vllm import __version__; print(__version__)"

# 检查 CUDA 可见
python -c "import torch; print(torch.cuda.device_count(), 'GPUs')"
```

---

## 四、模型下载

### 从 HuggingFace 下载

```bash
# 安装 huggingface-cli（如果未安装）
pip install huggingface_hub

# 登录（如果需要）
huggingface-cli login

# 下载 DeepSeek V4 Flash
# 注意：模型约 284GB，确保有足够磁盘空间
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
    --local-dir /mnt/nvme/models/DeepSeek-V4-Flash \
    --local-dir-use-symlinks False
```

---

## 五、启动推理服务

### 基本启动命令

```bash
# 8×H80，FP8，100k 上下文
vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --tensor-parallel-size 8 \
    --dtype auto \
    --max-model-len 100000 \
    --gpu-memory-utilization 0.95 \
    --port 8000
```

### 参数说明

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `--tensor-parallel-size` | 8 | 8 卡 TP，模型分片到所有 GPU |
| `--dtype` | auto | 自动检测 FP8（模型原生支持） |
| `--max-model-len` | 100000 | 最大上下文长度（按需调整） |
| `--gpu-memory-utilization` | 0.95 | 显存利用率，预留 KV cache 空间 |
| `--port` | 8000 | API 端口 |
| `--max-num-seqs` | 256 | 最大并发序列数 |
| `--enable-prefix-caching` | （加此参数） | 启用前缀缓存，复用公共前缀 |

### 带前缀缓存的启动（推荐）

```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --tensor-parallel-size 8 \
    --dtype auto \
    --max-model-len 100000 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --port 8000
```

### 验证服务

```bash
# 检查模型是否加载成功
curl http://localhost:8000/v1/models

# 发送测试请求
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek-ai/DeepSeek-V4-Flash",
        "messages": [{"role": "user", "content": "Hello, who are you?"}],
        "max_tokens": 100,
        "temperature": 0.6
    }'
```

---

## 六、基准测试方法

### 测试工具

```bash
# 安装 vLLM 自带的 benchmark
pip install aiohttp  # benchmark 依赖

# 运行 benchmark（vLLM 自带的）
python -m vllm.benchmarks.benchmark_serving \
    --backend vllm \
    --model deepseek-ai/DeepSeek-V4-Flash \
    --dataset-name sharegpt \
    --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 100 \
    --request-rate 10 \
    --port 8000
```

### 关键指标

| 指标 | 单位 | 说明 |
|------|------|------|
| **TTFT** | ms | 首 token 延迟（Time to First Token） |
| **TPOT** | ms | 每 token 输出时间（Time per Output Token） |
| **Throughput** | req/s | 服务吞吐量 |
| **Prefill t/s** | tokens/s | 预填充速度 |
| **Gen t/s** | tokens/s | 生成速度 |
| **KV Cache 使用量** | GB | 实际 KV cache 占用的显存 |

### 不同上下文长度的基准

```bash
# 短上下文（2k）
vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --tensor-parallel-size 8 \
    --max-model-len 2048

# 中等上下文（32k）
vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --tensor-parallel-size 8 \
    --max-model-len 32768

# 长上下文（100k）
vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --tensor-parallel-size 8 \
    --max-model-len 100000
```

---

## 七、预期性能（参考）

### 基于 vLLM 官方数据（类似规模模型）

| 上下文 | 配置 | Prefill t/s | Gen t/s | TTFT |
|--------|------|------------|---------|------|
| 2k | 8×H100, FP8 | ~500,000 | ~5,000 | ~50ms |
| 32k | 8×H100, FP8 | ~300,000 | ~3,000 | ~200ms |
| 100k | 8×H100, FP8 | ~100,000 | ~1,500 | ~1s |
| 1M | 8×H100, FP8 | ~20,000 | ~500 | ~10s |

> ⚠️ 以上为估算值，实际性能取决于具体硬件、vLLM 版本和模型版本。
> DeepSeek V4 Flash 是较新模型，vLLM 社区持续优化中。

---

## 八、常见问题

### Q: 显存不够怎么办？

1. 减小 `--max-model-len`（降低 KV cache 需求）
2. 降低 `--gpu-memory-utilization`（但会影响吞吐）
3. 增加 GPU 数量（`--tensor-parallel-size`）
4. 启用 `--enable-prefix-caching`（复用公共前缀）

### Q: 加载模型太慢？

- 使用 `--load-format safetensors`（默认）
- 确保存储是 NVMe SSD（不是 HDD）
- 首次加载需要下载并转换模型权重

### Q: 如何调试？

```bash
# 启用详细日志
vllm serve ... --verbose

# 查看 CUDA 内存
nvidia-smi -l 1

# vLLM 内置指标
curl http://localhost:8000/metrics
```

---

## 九、后续计划

```
Phase 1（当前）: 原生 vLLM 基准
  └─ 记录 baseline 性能数据

Phase 2: 接入 LMCache（本地磁盘缓存）
  └─ 对比: 磁盘缓存 vs 纯 GPU 的性能差异

Phase 3: 实现我们的 DiskCacheConnector
  └─ 对比: 我们 vs LMCache vs 原生 vLLM
```

---

> 文档版本: v1.0
> 更新日期: 2025-07-19
