# GDS (GPUDirect Storage) 配置

## 概述

Cascade 支持通过 NVIDIA GPUDirect Storage (GDS) 实现 **GPU↔NVMe 直读直写**，跳过 CPU bounce buffer，大幅降低 KV cache 落盘延迟。

## 前置条件

### 硬件要求

- NVIDIA GPU (H100/A100/V100 等)
- NVMe SSD（本地直连，非网络挂载）
- Linux 系统

### 软件要求

```bash
# 1. CUDA driver + CUDA Toolkit（已安装）
# 2. 安装 cuda-python（提供 GDS 绑定）
pip install cuda-python numpy

# 3. 确认 GDS 驱动可用
python -c "
from cuda.bindings import cufile
cufile.driver_open()
print('GDS driver OK')
cufile.driver_close()
"
```

> 注意: `nvidia-cufile` pip 包仅提供 `libcufile.so`，但 **不** 提供 Python 绑定。
> 我们需要 `cuda-python` (`pip install cuda-python`)，它包含了 `cuda.bindings.cufile`。

## 配置

在 `kv_connector_extra_config` 中加入 `storage_backend` 字段：

```json
{
  "kv_connector": "DiskCacheConnector",
  "kv_role": "kv_both",
  "kv_connector_module_path": "disk_cache",
  "kv_connector_extra_config": {
    "disk_cache_path": "/tmp/cascade-kv",
    "disk_cache_engine_addr": "http://localhost:9100",
    "target_device": "cuda:0",
    "disk_cache_chunk_size_mb": 128,
    "storage_backend": "gds"
  }
}
```

`storage_backend` 可选值：

| 值 | 行为 |
|-----|------|
| `"auto"`（默认） | 自动检测：有 cuda.bindings.cufile → NvFileBackend，否则 PosixBackend |
| `"gds"` | 强制使用 GDS，不可用时降级到 PosixBackend（日志 warning） |
| `"posix"` | 强制使用 CPU bounce buffer + safetensors |

## 验证 GDS 生效

### 1. 启动时日志

vLLM 启动日志中会出现：

```
[connector.py] DiskCacheConnector using NvFileBackend(binding=cuda.bindings.cufile)
[connector.py] DiskCacheConnector ready: ... backend=NvFileBackend
```

如果看到 `backend=NvFileBackend` 说明 GDS 已启用。
如果看到 `backend=PosixBackend` 说明走的是 CPU bounce buffer 降级路径。

### 2. 手动测试

```bash
python -c "
from adapter.storage import create_storage_backend
b = create_storage_backend(prefer='gds')
print(repr(b))  # 应为 NvFileBackend(binding=cuda.bindings.cufile)
"
```

### 3. 全链路测试

```bash
# 启动 Go engine
disk-cache --cache-path /tmp/cascade-kv --metadata-path /tmp/cascade-meta --listen :9100 &

# 启动 vLLM（配置 storage_backend: gds）
vllm serve /path/to/model \
  --kv-transfer-config '{"kv_connector": "DiskCacheConnector", "kv_role": "kv_both", "kv_connector_module_path": "disk_cache", "kv_connector_extra_config": {"disk_cache_path": "/tmp/cascade-kv", "disk_cache_engine_addr": "http://localhost:9100", "target_device": "cuda:0", "disk_cache_chunk_size_mb": 128, "storage_backend": "gds"}}'

# 发推理请求
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/path/to/model","messages":[{"role":"user","content":"Hello!"}],"max_tokens":10}'
```

## GDS 后端实现说明

Cascade 的 `NvFileBackend` 支持三种 GDS 绑定方式（自动探测）：

| 优先级 | 绑定 | 安装方式 | 来源 |
|--------|------|---------|------|
| 1 | `cuda.bindings.cufile` | `pip install cuda-python` | NVIDIA CUDA Python |
| 2 | `cufile` (Python 模块) | NVIDIA GDS SDK | NVIDIA 官方 |
| 3 | `nvfile` / `hipfile` | 供应商提供 | 厂商 / AMD ROCm |

## 文件格式

GDS 后端使用 **raw 格式**（4KB JSON header + 原始 tensor 数据），区别于 PosixBackend 的 safetensors 格式。两种格式可以共存，后端会根据文件内容自动识别。

## 故障排除

### GDS 驱动未加载

```
RuntimeError: No GDS library found.
  Install:  pip install cuda-python
```

解决: `pip install cuda-python numpy`

### cuFileHandleRegister 失败

```
OSError: cuFileHandleRegister failed
```

原因: 缺少 `nvidia-fs` 内核模块。

解决: 安装 NVIDIA GDS 驱动:

```bash
# Ubuntu/Debian
apt install nvidia-gds

# 确认内核模块已加载
lsmod | grep nvidia_fs
ls /dev/nvidia-fs*
```

### GDS 降级到 POSIX

如果日志中看到 `backend=PosixBackend` 但期望 GDS，检查：

1. `cuda-python` 是否安装: `pip list | grep cuda-python`
2. `cuda.bindings.cufile` 是否可导入:
   ```bash
   python -c "from cuda.bindings import cufile; print('OK')"
   ```
3. `/root/cascade` 是否在 Python 的 `sys.path` 中
