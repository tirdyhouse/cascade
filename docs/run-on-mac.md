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

## 五、准备 vLLM Connector

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

## 十、后续上 GPU

在 Mac 上验证通过后，上 GPU 服务器只需要改：

1. **安装 vLLM（CUDA 版）** — `pip install vllm`（不需要 `VLLM_TARGET_DEVICE=mps`）
2. **connector.py 的 `.to(device)` 已改好**，无需再改代码
3. **模型换大的** — 如 DeepSeek V4、Qwen2.5-72B
4. **Go 引擎的 `--cache-path` 指向 NVMe 盘** — 如 `/mnt/nvme/kv-cache`
5. **`--max-size` 设大** — 如 `1TB`

Mac 上验证通过的核心逻辑（Go 引擎 + Pebble 元数据 + LRU 淘汰 + tensor 序列化）在 GPU 上完全一致，不需要重新验证。

---

> 文档版本: v1.0
> 更新日期: 2025-07-19
