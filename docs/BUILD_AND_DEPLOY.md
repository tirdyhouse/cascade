# 构建与部署指南

## 目录结构

```
engine/
├── cmd/
│   ├── cluster-server/       # S端 入口
│   └── c-agent/              # C端 入口
├── pkg/
│   ├── cluster/              # C/S 共享类型 + rpcx 接口定义
│   │   ├── types.go          # 所有数据结构
│   │   └── rpc_service.go    # rpcx 服务接口
│   ├── server/               # S端 实现
│   │   ├── server.go         # 主服务 + REST API + Web UI
│   │   ├── registry.go       # 节点注册中心
│   │   ├── dispatcher.go     # 指令调度
│   │   ├── models.go         # 模型注册表
│   │   ├── router.go         # KV cache 路由
│   │   ├── disk_tracker.go   # 多磁盘追踪
│   │   ├── meta_adapter.go   # disk-cache 引擎桥接
│   │   └── web/static/       # Web UI 静态文件
│   └── agent/                # C端 实现
│       ├── agent.go          # 主循环
│       ├── process.go        # vLLM 进程管理
│       ├── collector.go      # 状态采集（GPU/磁盘）
│       ├── cache_proxy.go    # 本地 cache 代理
│       ├── rpc_client.go     # rpcx 客户端
│       └── launch_vllm.py    # vLLM 启动包装脚本
├── adapter/
│   ├── vllm/                 # vLLM connector 适配器
│   └── storage/              # 存储后端适配器（GDS/POSIX）
│       ├── nvfile_backend.py # GDS 后端（cuFile）
│       ├── posix_backend.py  # POSIX 后端（CPU bounce buffer）
│       └── backend.py        # 存储后端工厂
└── docs/
    ├── gds-config.md         # GDS 配置文档
    ├── cache-design.md       # Cache 设计文档
    └── BUILD_AND_DEPLOY.md   # 本文件
```

---

## 1. 构建

### 前置条件

- Go 1.23+
- 交叉编译到 linux/amd64（部署到 GPU 服务器）

### 编译命令

```bash
# S端
cd engine && GOOS=linux GOARCH=amd64 go build -o cluster-server-linux ./cmd/cluster-server/

# C端
cd engine && GOOS=linux GOARCH=amd64 go build -o c-agent-linux ./cmd/c-agent/

# Disk Cache Go Engine（独立二进制）
cd engine && GOOS=linux GOARCH=amd64 go build -o disk-cache-linux ./cmd/disk-cache/
```

---

## 2. S端 部署（Server）

### 启动命令

```bash
./cluster-server \
  --rpcx-port 9000 \           # rpcx 服务端口，C端 通过此端口连接
  --http-port 18080 \          # HTTP API + Web UI 端口
  --models-dir /tmp/models \   # 模型文件目录（S端 扫描此目录发现模型）
  --models-file /root/cascade/models.json  # 模型列表 JSON 文件
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--rpcx-port` | `9000` | rpcx 服务端口，C端 Agent 通过此端口注册和心跳 |
| `--http-port` | `18080` | HTTP API + 嵌入式 Web UI |
| `--models-dir` | - | S端 扫描此目录自动发现模型（读取 config.json 检测量化类型） |
| `--models-file` | - | 模型列表 JSON 文件路径，格式见下方 |

### models.json 格式

```json
[
  {
    "name": "Qwen2.5-7B-Instruct-AWQ",
    "path": "/tmp/models/Qwen2.5-7B-Instruct-AWQ",
    "size_gb": 15,
    "quantization": "awq",
    "download_url": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ"
  }
]
```

### 启动日志示例

```
2026/05/28 11:34:35 === S端 Cluster Server ===
2026/05/28 11:34:35 rpcx :9000  ← C端 agents connect here
2026/05/28 11:34:35 HTTP :18080  ← Web UI: http://localhost:18080
2026/05/28 11:34:35 [server] rpcx listening on :9000
2026/05/28 11:34:35 [server] HTTP + Web UI on :18080
```

---

## 3. C端 部署（Agent）

### 启动命令

```bash
./c-agent \
  --server 127.0.0.1:9000 \          # S端 rpcx 地址
  --node-id a100-test \               # 节点 ID（唯一标识）
  --cache-mode local_nvme \           # 缓存模式
  --gpu-type A100-PCIE-40GB \         # GPU 型号
  --gpu-mem 40960 \                   # GPU 显存 MB
  --gpu-count 1 \                     # GPU 数量
  --disks /root/cache:100 \           # 磁盘列表：路径:总容量GB
  --work-dir /root/cascade/agent \    # 工作目录
  --rpcx-port 9001                    # 本地 rpcx 端口（可选，用于 C端 服务注册）
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--server` | - | S端 rpcx 地址（必填） |
| `--node-id` | hostname | 节点唯一标识 |
| `--cache-mode` | `local_nvme` | 缓存模式：`local_nvme` / `shared_pool` |
| `--gpu-type` | - | GPU 型号，用于 S端 展示 |
| `--gpu-mem` | - | 单卡显存（MB） |
| `--gpu-count` | `1` | GPU 数量 |
| `--disks` | - | 磁盘列表，格式 `路径:总容量GB`，逗号分隔 |
| `--work-dir` | - | 工作目录，包含子目录：`models/` `logs/` `cache/` |
| `--rpcx-port` | `9001` | 本地 rpcx 端口（C端 注册的服务端口） |

### 工作目录结构

```
<work-dir>/
├── models/          # 本地模型文件（通过分发下载）
│   └── Qwen2.5-7B-Instruct-AWQ/
├── logs/            # vLLM 日志文件
│   └── vllm-Qwen2.5-7B-Instruct-AWQ.log
└── cache/           # Disk Cache 存储目录
    └── .../         # KV cache 块文件（.safetensors）
```

---

## 3.5 DiskCache 引擎部署（前置依赖）

DiskCache 引擎是一个**全局唯一的独立守护进程**，每台机器运行一个实例，监听 `:9100`。
它必须**在 C端 Agent 和 vLLM 之前启动**，否则 DiskCacheConnector 无法连接。

### 启动命令

```bash
# 编译（也可在本地交叉编译后上传）
cd engine && GOOS=linux GOARCH=amd64 go build -o ../bin/disk-cache ./cmd/disk-cache/

# 启动
./bin/disk-cache \
  --cache-path /tmp/cascade-kv \       # KV 缓存数据文件目录
  --metadata-path /tmp/cascade-meta \   # Pebble 元数据存储目录
  --max-size 200GB \                    # 缓存上限
  --listen :9100                        # HTTP API 监听地址
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cache-path` | `/tmp/disk-cache` | KV 缓存块文件存储路径（建议放在大容量 NVMe 盘） |
| `--metadata-path` | `/tmp/disk-cache-meta` | Pebble 元数据数据库路径 |
| `--max-size` | `100GB` | 最大缓存容量（支持 `MB`, `GB`, `TB` 单位） |
| `--listen` | `:9100` | HTTP API 监听地址，C端 通过 `--cache-path http://<ip>:9100` 连接 |

### 验证

```bash
# 检查是否正常运行
curl http://localhost:9100/stats
# 输出示例：{"BlocksStored":0,"BlocksRetrieved":0,"BlocksEvicted":0,"DiskUsedBytes":0}
```

### 架构说明

```
┌──────────────┐     HTTP API (:9100)     ┌──────────────────┐
│  C端 Agent   │◄────────────────────────►│  DiskCache Engine │
│ (cache_proxy)│                          │  (独立 Go 进程)   │
└──────────────┘                          └──────────────────┘
┌─────────────────────┐                   ┌──────────────────┐
│  vLLM Process       │─── /put /match ──►│  Pebble 元数据   │
│  DiskCacheConnector │◄── /get /chunk ───│  + KV 块文件     │
└─────────────────────┘                   └──────────────────┘
```

C端 Agent 的心跳通过 `cache_proxy.go` 定时拉取引擎统计（`BlocksStored`、`BlocksRetrieved` 等），上报到 S端 展示在 Web UI 上。

---

## 4. Disk Cache 磁盘缓存

### 工作原理

Disk Cache 通过 vLLM 的 `KVTransferConfig` 机制集成。当 C端 启动 vLLM 时：

1. `agent.go` 收到 `start_vllm` 指令
2. `process.go` 在 vLLM 命令行中添加 `--kv-transfer-config` JSON 参数
3. vLLM 启动时加载 `DiskCacheConnector`，初始化 GDS/POSIX 存储后端
4. 推理过程中，KV cache 块通过 `DiskCacheConnector` 写入 NVMe 磁盘

### 命令行参数（由 C端 自动添加）

```
--kv-transfer-config '{"kv_connector":"DiskCacheConnector","kv_role":"kv_both","kv_connector_extra_config":{"disk_cache_path":"<work-dir>/cache"}}'
```

### Web UI 操作

1. 进入 Node Detail 页面
2. 选择 **Auto-generate** 模式
3. 勾选 **💾 Disk Cache**
4. 选模型，点启动

### 存储后端

| 后端 | 检测条件 | 说明 |
|------|----------|------|
| `NvFileBackend` (GDS) | `cuda.bindings.cufile` 可用 | GPU↔NVMe 直读直写，最高性能 |
| `PosixBackend` | `cuda.bindings.cufile` 不可用 | CPU bounce buffer + safetensors |

详见 [gds-config.md](./gds-config.md)。

---

## 5. vLLM 启动方式

C端 支持两种 vLLM 启动模式：

### Auto-generate（结构化参数）

由 Web UI 或 API 传入结构化参数，`process.go` 构建 vLLM 命令行：

| 参数 | 对应 vLLM 参数 | 说明 |
|------|----------------|------|
| `model` | `serve <model-path>` | 模型名（自动拼接 work-dir/models/） |
| `gpu_util` | `--gpu-memory-utilization` | GPU 显存利用率 |
| `quantization` | `--quantization` | 量化类型（awq/gptq 等） |
| `enable_prefix_caching` | `--enable-prefix-caching` | 前缀缓存 |
| `enable_disk_cache` | `--kv-transfer-config` | 磁盘 KV 缓存 |

### Raw Args（自定义命令行）

直接传入原始命令行参数，透传给 vLLM：

```
Qwen2.5-7B-Instruct-AWQ --gpu-memory-utilization 0.90 --quantization awq --dtype float16 --kv-transfer-config '{...}'
```

> ⚠️ Raw Args 模式下，Disk Cache 选项不生效。如需启用 Disk Cache，需手动在 raw args 中加入 `--kv-transfer-config` JSON。

---

## 6. 已知问题

### 6.1 GDS cuFile `TypeError: an integer is required`

**现象**：vLLM 启动后处理第一个推理请求时 crash，日志报：
```
File "nvfile_backend.py", line 81, in write
    return self._C.write(handle, gpu_ptr, size, file_offset, dev_offset)
TypeError: an integer is required
```

**根因**：`_CuFileHandle.write()` 的低级 API 路径传了 `ctypes.c_void_p(gpu_addr)` 给 `cuda.bindings.cufile.write()`，但后者声明的 `buf_ptr_base` 参数类型为 `intptr_t`（Python int），不接受 `c_void_p`。

**修复**：`adapter/storage/nvfile_backend.py` 第 249/261 行，低级 API 路径用 `gpu_addr`（raw int）替代 `ptr`（c_void_p）。

**为何不装高级 API**：高级 `cufile` Python 包（`import cufile`）可从 NVIDIA pip index 安装，但该索引（pypi.nvidia.org）在中国服务器上无法解析 DNS。

### 6.2 Stop 后 GPU 显存不释放

**现象**：Web UI 点击 Stop 后，nvidia-smi 显示显存仍被占用。

**根因**：vLLM V1 引擎会 fork EngineCore 子进程持有 GPU 显存。`Process.Kill()` 只杀主进程，EngineCore 变成孤儿进程继续占用显存。

**修复**：`process.go` 设置 `Setpgid: true` 创建进程组，Stop 时向整个进程组发 SIGTERM → 5s 超时 → SIGKILL。

### 6.3 C端 停止后 EngineCore 僵尸进程

**现象**：`ps` 看到 `[VLLM::EngineCor] <defunct>`。

**说明**：僵尸进程是子进程已退出但父进程未 wait 的标志。C端 重启后 init 进程会自动回收，不影响系统运行。

### 6.4 `launch_vllm.py` 环境变量不生效

---

## 7. vLLM 0.21 已知 Bug 与 Fix

以下是在 vLLM 0.21.0（当前最新 release）中发现的 Bug 及修复方式。vLLM 官方已在 main 分支修复（Issue [#36755](https://github.com/vllm-project/vllm/issues/36755)、[#36533](https://github.com/vllm-project/vllm/issues/36533)），将在下一个 release 中包含。

### 7.1 Prometheus Counter 负值崩溃

**现象**：多轮对话或高并发时，vLLM API 返回 `400 Bad Request`，错误信息 `"Counters can only be incremented by non-negative amounts."`，随后 vLLM 进程退出。

**根因**：调度器 `_preempt_request()` 抢占请求时，只重置了 `num_computed_tokens`，但未重置 `num_cached_tokens`。恢复后 token 统计不一致，导致 Prometheus counter 收到负值。

**修复**（手动 backport，等待 vLLM 0.22+）：

```bash
# 1. scheduler.py: _preempt_request() 复位 num_cached_tokens
# 修改前：
request.num_computed_tokens = 0
# 修改后：
request.num_computed_tokens = 0
request.num_cached_tokens = 0
# 文件位置：vllm/v1/core/sched/scheduler.py

# 2. stats.py: PromptTokenStats.update_from_output() 加 max(0) 防护
# 文件位置：vllm/v1/metrics/stats.py
# self.computed += max(0, prefill_stats.num_computed_tokens)
# self.cached_tokens += max(0, prefill_stats.num_cached_tokens)  
# self.local_cache_hit += max(0, prefill_stats.num_local_cached_tokens)
# self.external_kv_transfer += max(0, prefill_stats.num_external_cached_tokens)

# 3. loggers.py: counter_connector_prefix_cache_hits 加 max(0) 防护
# 文件位置：vllm/v1/metrics/loggers.py 第 1095 行
# .inc(max(0, scheduler_stats.connector_prefix_cache_stats.hits))
```

### 7.2 `--enable-prompt-tokens-details` 必需

**说明**：`usage.prompt_tokens_details.cached_tokens` 是 OpenAI 兼容的缓存命中字段，vLLM 默认关闭。C端 `process.go` 在 DiskCache 模式下会自动添加 `--enable-prompt-tokens-details` 参数。如果手动启动 vLLM，需要加上此参数才能在 API 响应中看到缓存命中信息。

**现象**：设置了 `VLLM_KV_CONNECTOR` 环境变量但 DiskCacheConnector 未加载。

**根因**：vLLM 0.21 **不读取** `VLLM_KV_CONNECTOR` / `VLLM_KV_CONNECTOR_EXTRA_CONFIG` 环境变量。正确方式是通过 `--kv-transfer-config` CLI 参数传递 JSON 配置。

---

## 7. 常见运维操作

### 查看节点状态

```bash
curl http://localhost:18080/api/v1/nodes
```

### 下发启动命令

```bash
curl -X POST http://localhost:18080/api/v1/command \
  -H 'Content-Type: application/json' \
  -d '{
    "action": "start_vllm",
    "target": "a100-test",
    "params": {
      "model": "Qwen2.5-7B-Instruct-AWQ",
      "gpu_util": "0.90",
      "quantization": "awq",
      "enable_disk_cache": "true"
    },
    "timeout": 300
  }'
```

### 下发停止命令

```bash
curl -X POST http://localhost:18080/api/v1/command \
  -H 'Content-Type: application/json' \
  -d '{
    "action": "stop_vllm",
    "target": "a100-test",
    "params": {},
    "timeout": 30
  }'
```

### 查看 vLLM 日志

```bash
curl "http://localhost:18080/api/v1/nodes/a100-test/logs?offset=0&lines=50"
```

### 查看 GPU 状态

```bash
# 在 A100 上直接查看
nvidia-smi

# 通过 Web UI
curl http://localhost:18080/api/v1/nodes | python3 -m json.tool
```

---

## 8. 完整部署流程

```bash
# 1. 本地交叉编译
cd engine
GOOS=linux GOARCH=amd64 go build -o /tmp/cluster-server-linux ./cmd/cluster-server/
GOOS=linux GOARCH=amd64 go build -o /tmp/c-agent-linux ./cmd/c-agent/

# 2. 上传到服务器
scp /tmp/cluster-server-linux root@A100:/root/cascade/bin/cluster-server
scp /tmp/c-agent-linux root@A100:/root/cascade/bin/c-agent

# 3. 启动 S端
ssh root@A100 "cd /root/cascade && nohup ./bin/cluster-server \
  --rpcx-port 9000 --http-port 18080 \
  --models-dir /tmp/models --models-file /root/cascade/models.json \
  > /tmp/cluster-server.log 2>&1 &"

# 4. 启动 C端
ssh root@A100 "cd /root/cascade && nohup ./bin/c-agent \
  --server 127.0.0.1:9000 --node-id a100-test \
  --cache-mode local_nvme \
  --gpu-type A100-PCIE-40GB --gpu-mem 40960 --gpu-count 1 \
  --disks /root/cache:100 \
  --work-dir /root/cascade/agent \
  > /tmp/c-agent.log 2>&1 &"

# 5. 验证
curl http://A100:18080/api/v1/nodes
```
