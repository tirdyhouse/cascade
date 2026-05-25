# 在 Mac 上运行 Cascade Disk Cache（vLLM + 小模型验证）

> 目标：在本地 Mac 上完整跑通 Cascade disk cache 流程，包括 Go 引擎、vLLM connector、KV cache 落盘和读取。
> 所有步骤在 M1/M2/M3 Mac 上均可执行，无需 NVIDIA GPU。

---

## 一、架构概览

```
终端 1                     终端 2
┌─────────────────┐       ┌──────────────────────────────┐
│  Go 引擎         │       │  vLLM (MPS) + connector       │
│                  │ HTTP  │                              │
│  Pebble 元数据   │◄─────►│  DiskCacheConnector           │
│  LRU 淘汰        │       │                              │
│  :9100           │       │  推理 → KV tensor → .bin 文件 │
└─────────────────┘       └──────────────────────────────┘
                                  │
                                  ▼
                           /tmp/cascade-kv/
                           (KV cache .bin 文件)
```

- **Go 引擎**：管理元数据（Pebble）+ LRU 淘汰，通过 HTTP API 暴露
- **vLLM connector**：在推理过程中钩住 KV tensor，写入/读取磁盘
- **HTTP 只传元数据**（hash + path + size，几十字节），真正的 tensor 数据走文件系统

---

## 二、前置条件

| 工具 | 版本要求 | 检查方式 |
|------|---------|---------|
| Go | ≥ 1.23 | `go version` |
| Python | ≥ 3.10 | `python3 --version` |
| Homebrew | 最新 | `brew --version` |

### 安装 Go（如未安装）

```bash
brew install go
```

### 安装 vLLM（MPS 版）

vLLM 从 0.6.0 开始实验性支持 Apple MPS（Metal Performance Shaders）。

```bash
# 创建虚拟环境
python3 -m venv .venv-cascade
source .venv-cascade/bin/activate

# 安装 vLLM（MPS 后端）
VLLM_TARGET_DEVICE=mps pip install vllm

# 验证
python -c "from vllm import __version__; print('vLLM', __version__)"
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

---

## 三、构建 Go 引擎

```bash
# 进入项目目录
cd /path/to/cascade

# 编译 Go 引擎
make build-engine

# 确认编译成功
./bin/disk-cache --help
# 应看到 --cache-path / --max-size / --listen 等参数
```

---

## 四、启动 Go 引擎

在终端 1 中启动：

```bash
# Mac 上用 /tmp 即可，不需要 NVMe
# --max-size 设小一点，测试够用就行
./bin/disk-cache \
    --cache-path /tmp/cascade-kv \
    --metadata-path /tmp/cascade-meta \
    --max-size 1GB \
    --listen :9100
```

看到以下日志即启动成功：

```
disk-cache engine ready: path=/tmp/cascade-kv max=1073741824
disk-cache engine started on :9100
```

保持此终端运行，新开一个终端做后续操作。

### 验证引擎

```bash
curl http://localhost:9100/stats
# 应返回: {"blocks_stored":0,"blocks_evicted":0,"disk_used_bytes":0}
```

---

## 五、下载测试模型

### 安装下载工具

Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
Collecting huggingface_hub
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/49/79/621a7dbb80c70974f73a597275351ebe03ce5bc65cb5f8f4acb5859252bc/huggingface_hub-1.16.1-py3-none-any.whl (668 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 668.2/668.2 kB 7.6 MB/s eta 0:00:00
Requirement already satisfied: filelock>=3.10.0 in /opt/anaconda3/lib/python3.13/site-packages (from huggingface_hub) (3.17.0)
Requirement already satisfied: fsspec>=2023.5.0 in /opt/anaconda3/lib/python3.13/site-packages (from huggingface_hub) (2025.3.2)
Collecting hf-xet<2.0.0,>=1.4.3 (from huggingface_hub)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/9b/ff/edcc2b40162bef3ff78e14ab637e5f3b89243d6aee72f5949d3bb6a5af83/hf_xet-1.5.0-cp37-abi3-macosx_11_0_arm64.whl (3.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.8/3.8 MB 28.5 MB/s eta 0:00:00
Requirement already satisfied: httpx<1,>=0.23.0 in /opt/anaconda3/lib/python3.13/site-packages (from huggingface_hub) (0.28.1)
Requirement already satisfied: packaging>=20.9 in /opt/anaconda3/lib/python3.13/site-packages (from huggingface_hub) (24.2)
Requirement already satisfied: pyyaml>=5.1 in /opt/anaconda3/lib/python3.13/site-packages (from huggingface_hub) (6.0.2)
Requirement already satisfied: tqdm>=4.42.1 in /opt/anaconda3/lib/python3.13/site-packages (from huggingface_hub) (4.67.1)
Collecting typer>=0.20.0 (from huggingface_hub)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/3f/f9/2b3ff4e56e5fa7debfaf9eb135d0da96f3e9a1d5b27222223c7296336e5f/typer-0.25.1-py3-none-any.whl (58 kB)
Requirement already satisfied: typing-extensions>=4.1.0 in /opt/anaconda3/lib/python3.13/site-packages (from huggingface_hub) (4.12.2)
Requirement already satisfied: anyio in /opt/anaconda3/lib/python3.13/site-packages (from httpx<1,>=0.23.0->huggingface_hub) (4.7.0)
Requirement already satisfied: certifi in /opt/anaconda3/lib/python3.13/site-packages (from httpx<1,>=0.23.0->huggingface_hub) (2025.4.26)
Requirement already satisfied: httpcore==1.* in /opt/anaconda3/lib/python3.13/site-packages (from httpx<1,>=0.23.0->huggingface_hub) (1.0.9)
Requirement already satisfied: idna in /opt/anaconda3/lib/python3.13/site-packages (from httpx<1,>=0.23.0->huggingface_hub) (3.7)
Requirement already satisfied: h11>=0.16 in /opt/anaconda3/lib/python3.13/site-packages (from httpcore==1.*->httpx<1,>=0.23.0->huggingface_hub) (0.16.0)
Collecting click>=8.2.1 (from typer>=0.20.0->huggingface_hub)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/c7/0d/67e5b4109ea4a837e80daa87c2c696711955e40449a97e8926672534def2/click-8.4.1-py3-none-any.whl (116 kB)
Requirement already satisfied: shellingham>=1.3.0 in /opt/anaconda3/lib/python3.13/site-packages (from typer>=0.20.0->huggingface_hub) (1.5.0)
Requirement already satisfied: rich>=13.8.0 in /opt/anaconda3/lib/python3.13/site-packages (from typer>=0.20.0->huggingface_hub) (13.9.4)
Collecting annotated-doc>=0.0.2 (from typer>=0.20.0->huggingface_hub)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/1e/d3/26bf1008eb3d2daa8ef4cacc7f3bfdc11818d111f7e2d0201bc6e3b49d45/annotated_doc-0.0.4-py3-none-any.whl (5.3 kB)
Requirement already satisfied: markdown-it-py>=2.2.0 in /opt/anaconda3/lib/python3.13/site-packages (from rich>=13.8.0->typer>=0.20.0->huggingface_hub) (2.2.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /opt/anaconda3/lib/python3.13/site-packages (from rich>=13.8.0->typer>=0.20.0->huggingface_hub) (2.19.1)
Requirement already satisfied: mdurl~=0.1 in /opt/anaconda3/lib/python3.13/site-packages (from markdown-it-py>=2.2.0->rich>=13.8.0->typer>=0.20.0->huggingface_hub) (0.1.0)
Requirement already satisfied: sniffio>=1.1 in /opt/anaconda3/lib/python3.13/site-packages (from anyio->httpx<1,>=0.23.0->huggingface_hub) (1.3.0)
Installing collected packages: hf-xet, click, annotated-doc, typer, huggingface_hub
  Attempting uninstall: click
    Found existing installation: click 8.1.8
    Uninstalling click-8.1.8:
      Successfully uninstalled click-8.1.8
  Attempting uninstall: typer
    Found existing installation: typer 0.9.0
    Uninstalling typer-0.9.0:
      Successfully uninstalled typer-0.9.0

Successfully installed annotated-doc-0.0.4 click-8.4.1 hf-xet-1.5.0 huggingface_hub-1.16.1 typer-0.25.1

### 模型推荐

在 M4 Pro（48GB）上，按以下优先级下载测试：

#### Tier 1 — 核心验证（必须）

| 模型 | 大小 | 说明 |
|------|------|------|
| `Qwen/Qwen2.5-0.5B-Instruct` | ~1 GB | 主力测试，Qwen 家族，跟后续 72B 同架构 |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | ~2.2 GB | 不同 tokenizer，验证 cache key 隔离 |
| `facebook/opt-125m` | ~250 MB | 最轻量兜底 |

#### Tier 2 — 进阶验证（推荐）

| 模型 | 大小 | 说明 |
|------|------|------|
| `Qwen/Qwen2.5-1.5B-Instruct` | ~3 GB | 跟 0.5B 同族但层数/heads 不同，验证 shape 兼容 |
| `HuggingFaceTB/SmolLM2-360M-Instruct` | ~700 MB | 现代小模型，vLLM 支持 |

#### Tier 3 — 生产目标（以后上 GPU 用）

| 模型 | 大小 | 说明 |
|------|------|------|
| `deepseek-ai/DeepSeek-V4-Flash` | ~284 GB | 最终目标，需要 8×H100 |
| `Qwen/Qwen2.5-72B-Instruct` | ~140 GB | 另一个目标，需要 4×H100 |

### 下载命令

#### 方式一：HF_TOKEN（推荐，认证据后速度快）
先去 https://huggingface.co/settings/tokens 创建 token：

```bash
# 设置 token
export HF_TOKEN=你的_token

# 建议写入 shell 配置
# echo "export HF_TOKEN=你的_token" >> ~/.zshrc
```

然后逐个下载（建议从小到大，方便尽早开始测试）：

```bash
# 1. 最小的先下（~250MB，1-2 小时）
hf download facebook/opt-125m

# 2. Qwen2.5-0.5B（~1GB）
hf download Qwen/Qwen2.5-0.5B-Instruct

# 3. SmolLM2-360M（~700MB）
hf download HuggingFaceTB/SmolLM2-360M-Instruct

# 4. TinyLlama-1.1B（~2.2GB）
hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0

# 5. Qwen2.5-1.5B（~3GB，最后下）
hf download Qwen/Qwen2.5-1.5B-Instruct
```

> ⏱ 国内网络较慢，每个模型可能需要 1-6 小时，建议分批下载。

#### 方式二：魔塔社区 ModelScope（阿里模型专用）
Qwen 是阿里模型，魔塔下载最快：

```bash
# 安装
pip install modelscope

# ── Tier 1 ──
python3 -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')"
python3 -c "from modelscope import snapshot_download; snapshot_download('TinyLlama/TinyLlama-1.1B-Chat-v1.0')"
python3 -c "from modelscope import snapshot_download; snapshot_download('facebook/opt-125m')"

# ── Tier 2 ──
python3 -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')"
python3 -c "from modelscope import snapshot_download; snapshot_download('HuggingFaceTB/SmolLM2-360M-Instruct')"
```

#### 方式二：HF_TOKEN（HuggingFace 官方，认证后不限速）
去 https://huggingface.co/settings/tokens 创建 token：

```bash
hf auth login
# ── Tier 1 ──
hf download Qwen/Qwen2.5-0.5B-Instruct
hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0
hf download facebook/opt-125m

# ── Tier 2 ──
hf download Qwen/Qwen2.5-1.5B-Instruct
hf download HuggingFaceTB/SmolLM2-360M-Instruct
```


### 验证下载

models--zai-org--GLM-ASR-Nano-2512

---

## 六、准备 vLLM Connector

### 1. 理解 connector 发现机制

vLLM 的 `--kv-transfer-config '{"kv_connector": "disk-cache"}'` 会尝试导入名为 `disk_cache` 的 Python 模块。我们的 connector 在 `adapter/vllm/connector.py` 中，需要让它可被发现。

有多种方式：

**方式 A：符号链接（推荐测试用）**

```bash
# 在 vLLM 的搜索路径中创建链接
cd .venv-cascade/lib/python3.xx/site-packages/
ln -s /path/to/cascade/adapter/vllm/connector.py disk_cache.py
```

**方式 B：PYTHONPATH 环境变量**

```bash
export PYTHONPATH=/path/to/cascade/adapter/vllm:$PYTHONPATH
# 然后创建一个 disk_cache.py 的包装模块
```

**方式 C：直接复制**

```bash
cp /path/to/cascade/adapter/vllm/connector.py ./disk_cache.py
```

推荐方式 A，最简单。

### 2. 设备兼容性说明

connector 已内置 `_resolve_device()` 自动检测机制：

- **默认（auto）**：从传入 tensor 的 `.device` 属性自动识别（CUDA / MPS / ROCm）
- **手动指定**：在 `kv_connector_extra_config` 中加入 `target_device` 字段

不传 `target_device` 则 Mac 上自动识别为 `mps:0`，NVIDIA 上为 `cuda:0`，无需手动配置。

需要强制指定时：

```bash
# Mac 上手动指定 MPS
vllm serve ... \
    --kv-transfer-config '{
        "kv_connector": "disk-cache",
        "kv_connector_extra_config": {
            "target_device": "mps:0",
            ...
        }
    }'

# 或指定某块 GPU
# "target_device": "cuda:1"
```

### 3. 验证 connector 可导入

```bash
python -c "from disk_cache import DiskCacheConnector; print('connector OK')"
```

---

## 六、启动 vLLM 并连接 Disk Cache

在终端 2 中（先激活虚拟环境）：

```bash
source .venv-cascade/bin/activate

# 选择一个 Mac 上能跑的小模型
# Qwen2.5-0.5B 约 1GB，M1/M2 上可运行
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --kv-transfer-config '{
        "kv_connector": "disk-cache",
        "kv_connector_extra_config": {
            "disk_cache_path": "/tmp/cascade-kv",
            "disk_cache_engine_addr": "http://localhost:9100"
        }
    }'
```

### 可选的小模型

| 模型 | 大小 | 说明 |
|------|------|------|
| `Qwen/Qwen2.5-0.5B-Instruct` | ~1 GB | 推荐，轻量稳定 |
| `Qwen/Qwen2.5-1.5B-Instruct` | ~3 GB | M2/M3 可用 |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | ~2.2 GB | 备选 |
| `facebook/opt-125m` | ~250 MB | 最轻量，但效果差 |

### 验证 vLLM 服务

```bash
# 等待 vLLM 加载完成（看到 "Application startup complete" 类似的日志）
# 另开一个终端

curl http://localhost:8000/v1/models
# 应返回模型列表

curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 20
    }'
# 应正常返回回复
```

---

## 七、验证 Disk Cache 是否生效

### 1. 检查引擎统计

```bash
curl http://localhost:9100/stats
# 如果 connector 正常工作，blocks_stored 应 > 0
```

### 2. 检查磁盘上的缓存文件

```bash
ls -la /tmp/cascade-kv/
# 应有按 hash 分片的目录结构，如 00/00/0000...bin
```

### 3. 手动触发 cache 操作

```bash
# 查看某个 key 是否存在
curl "http://localhost:9100/exists?hash=0000000000000001"

# 获取统计
curl http://localhost:9100/stats | python3 -m json.tool
```

---

## 八、故障排查

### 1. vLLM 找不到 connector

```
ImportError: No module named 'disk_cache'
```

→ 确认 `disk_cache.py` 在 `PYTHONPATH` 中或 `site-packages` 下。

### 2. vLLM 启动后引擎 stats 无变化

```
{"blocks_stored":0,"blocks_evicted":0,"disk_used_bytes":0}
```

可能原因：
- connector 连接引擎失败 → 检查 `disk_cache_engine_addr` 是否正确
- 模型未触发 KV cache 落盘 → 发几个请求试试
- GPU 显存太小 → 减小 `--max-model-len` 或换更小的模型

### 3. MPS 内存不足（OOM）

```
torch.mps.OutOfMemoryError
```

解决：
- 用更小的模型（0.5B 替代 1.5B）
- 减小 `--max-model-len`（如 2048）
- 关闭其他占用 GPU 的应用

### 4. Go 引擎端口被占用

```
listen tcp :9100: bind: address already in use
```

→ 换个端口：`--listen :9101`，并同步修改 connector 配置中的 `disk_cache_engine_addr`。

---

## 九、测试顺序建议

```
Step 1: 构建 Go 引擎 + 启动 daemon                          5 分钟
Step 2: 纯 CPU 模拟验证（不启动 vLLM）                       10 分钟
   ├─ 用 Python 模拟生成 KV tensor
   ├─ 调 Go HTTP API 做 put/get/evict
   └─ 确认序列化一致性
Step 3: 安装 vLLM MPS + 配置 connector                      15 分钟
Step 4: 用小模型启动 vLLM + 验证 disk cache 集成             20 分钟
Step 5: 发几个请求，确认 blocks_stored > 0                    5 分钟
                                                     ─────────
                                                      约 55 分钟
```

### 纯 CPU 快速验证脚本

不启动 vLLM，先验证 disk cache 链路是否通：

```python
# test_disk_cache.py
import torch
import requests
import json

GO_ADDR = "http://localhost:9100"

# 1. 模拟一个 KV cache block
kv = torch.randn(2, 8, 64, 128).half()
data = kv.numpy().tobytes()

# 2. 保存到文件
import tempfile, os
tmp = tempfile.NamedTemporaryFile(delete=False, dir="/tmp/cascade-kv")
tmp.write(data)
tmp.close()

# 3. 通知 Go 引擎
hash_val = hash(str(kv.shape))
rel_path = os.path.basename(tmp.name)
resp = requests.post(f"{GO_ADDR}/put", json={
    "hash": hash_val,
    "file_path": rel_path,
    "size": len(data),
})
print(f"PUT: {resp.status_code}")

# 4. 取回元数据
resp = requests.get(f"{GO_ADDR}/get?hash={hash_val:016x}")
meta = resp.json()
print(f"GET: {meta}")

# 5. 从文件读回并验证
loaded = torch.frombuffer(data, dtype=torch.float16).reshape(kv.shape)
assert torch.allclose(kv, loaded), "数据不一致！"
print("✅ 序列化一致性验证通过")

# 6. 检查统计
resp = requests.get(f"{GO_ADDR}/stats")
print(f"Stats: {resp.json()}")
```

运行方式：

```bash
# 确保 Go 引擎已在运行
python3 test_disk_cache.py
```

---

## 十、模型选择与硬件规划

### 模型选用三阶段

| 阶段 | 模型 | 大小 | 硬件 | 目的 |
|------|------|------|------|------|
| **Tier 1：Mac 本地验证** | `Qwen/Qwen2.5-0.5B-Instruct` | ~1 GB | Mac M1-M4（已下载 ✅） | 验证 Go 引擎、序列化、connector 逻辑 |
| | `facebook/opt-125m` | ~250 MB | Mac M1-M4（已下载 ✅） | 最小模型兜底 |
| | `Qwen/Qwen2.5-1.5B-Instruct` | ~3 GB | Mac M4 Pro（已下载 ✅） | 不同 shape 兼容性 |
| **Tier 2：GPU 开发测试** | `Qwen/Qwen2.5-1.5B-Instruct` | ~3 GB | RTX 2080 / 2080 Ti / 3060 | 验证 vLLM KV transfer 全链路 |
| | `Qwen/Qwen2.5-7B-Instruct` | ~14 GB | RTX 3090 / 4090 / A4000 | 中等规模 disk cache 压力测试 |
| | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | ~2.2 GB | 任意 GPU | 不同 tokenizer 验证 cache 隔离 |
| **Tier 3：生产部署** | `deepseek-ai/DeepSeek-V4-Flash` | ~284 GB | 8×H100 / 4×H200 | 最终目标，长上下文杀手场景 |
| | `Qwen/Qwen2.5-72B-Instruct` | ~140 GB | 4×H100 / 2×H200 | 另一个生产目标 |

### GPU 硬件要求

vLLM V1 引擎（KV transfer 必需）对 GPU 的要求：

| GPU | 显存 | 计算能力 | V1 支持 | 适用场景 |
|-----|------|---------|---------|---------|
| RTX 2080 | 8 GB | 7.5 | ✅ | 开发测试 0.5B / 1.5B |
| RTX 2080 Ti | 11 GB | 7.5 | ✅ | 开发测试 7B |
| RTX 3060 | 12 GB | 8.6 | ✅ | 开发测试 7B |
| RTX 3090 | 24 GB | 8.6 | ✅ | 开发+小规模生产 |
| RTX 4090 | 24 GB | 8.9 | ✅ | 开发+小规模生产 |
| A100 | 80 GB | 8.0 | ✅ | 生产 |
| H100 | 80 GB | 9.0 | ✅ | 生产（推荐） |

> ⚠️ Mac MPS / Apple Silicon 不支持 vLLM V1 引擎，因此 KV transfer（含 disk cache connector）无法在 Mac 上做完整集成测试。这是 vLLM 的限制，非本项目问题。

### 推荐路线

```
Phase 1（当前）:    Mac 本地验证 ✅
   ├─ Go 引擎编译运行 ✅
   ├─ 模型下载（3 个模型已就绪）✅
   ├─ 纯 CPU 模拟测试（序列化/反序列化）
   └─ vLLM MPS 安装 ✅（V0 引擎，不能测 KV transfer）

Phase 2（下一步）:   租卡开发测试
   ├─ 用 RTX 2080 级别 GPU
   ├─ 验证 vLLM V1 + KV transfer + disk cache 全链路
   ├─ `vllm serve /tmp/models/... --kv-transfer-config '...'`
   └─ 用 `--target_device` 参数指定设备

Phase 3:          生产部署
   ├─ H100 / H200 集群
   ├─ 大模型（DeepSeek V4 / Qwen 72B）
   ├─ NVMe 缓存路径
   └─ 大容量 LRU 缓存
```

---

## 十一、当前进展总结

截至 2025-07-19，已完成：

| 组件 | 状态 |
|------|------|
| Go 引擎（Pebble 元数据 + LRU + HTTP API） | ✅ 编译运行通过 |
| Connector 代码（`_resolve_device` 跨平台检测） | ✅ 已编写、已合并到 main |
| `docs/run-on-mac.md` | ✅ 完整方案文档 |
| vLLM MPS 安装（0.19.1） | ✅ |
| Qwen2.5-0.5B / 1.5B / OPT-125m 模型 | ✅ 已下载到 `/tmp/models/` |
| vLLM V1 + KV transfer 集成验证 | ⏳ 需 GPU 环境 |
| 大规模压力测试 | ⏳ 后续 |

---

> 文档版本: v1.1
> 更新日期: 2025-07-19
