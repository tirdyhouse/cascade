# Cascade 测试指南

> 本文档记录如何在远程 GPU 机器上测试 Cascade 磁盘缓存系统。

---

## ⚠️ 关键配置要求

### 必须禁用 vLLM 内置 prefix caching

```bash
vllm serve ... --no-enable-prefix-caching
```

**原因**：vLLM 的 `--enable-prefix-caching` 会在 GPU 内存中缓存 KV blocks，完全绕过 Cascade 的 match 接口。启用后会出现：
- `External prefix cache hit rate: 0.0%`
- Go engine 的 `MatchRequests` 不增长
- Cascade 缓存完全失效

官方验证脚本 `scripts/validate_vllm_disk_cache.sh` 默认设置 `DISABLE_PREFIX_CACHING=1`。

---

## 测试环境

### 远程机器 (192.168.56.29)

| 项目 | 详情 |
|------|------|
| IP | 192.168.56.29 |
| 用户/密码 | root/123456 |
| 系统 | Rocky Linux 8.9 |
| GPU | NVIDIA Tesla T4 (16GB) |
| 内存 | 125GB |
| 存储 | 本轮检查 `/mnt/nvfile` 落在 `/dev/sda3` 的 XFS；不是已证明的 GPFS/nvfile direct-GDS 挂载 |
| Go | 1.26.5 (`/usr/local/go/bin`) |
| Python | 3.12.13 (venv at `/mnt/data/vllmtest/.venv`) |
| vLLM | 0.25.1 |
| LMCache | 0.4.3 |

### 目录结构

```
/opt/cascade/              # Cascade Go engine
├── disk-cache             # Go 二进制
├── data/                  # 元数据和存储
└── cascade_engine.log     # 引擎日志

/mnt/data/vllmtest/        # 测试目录
├── .venv/                 # Python 虚拟环境
├── adapter/               # Cascade Python adapter (rsync from local)
├── scripts/               # 测试脚本
├── results/               # 测试结果
├── logs/                  # vLLM 日志
└── LMCache/               # LMCache 仓库 (用于 benchmark 脚本)

/mnt/nvfile/test-1/        # nvfile 高性能存储
├── cascade-bench/         # Cascade benchmark 数据
│   ├── posix/             # POSIX 后端测试数据
│   └── gds/               # GDS 后端测试数据
└── cascade-debug/         # 调试测试数据
```

---

## 部署流程

### 1. 本地交叉编译

```bash
cd /Users/steven/code/predict
GOOS=linux GOARCH=amd64 go build -o disk-cache-linux ./engine/cmd/disk-cache/
```

### 2. 上传到远程

```bash
# 上传 Go engine
sshpass -p '123456' scp disk-cache-linux root@192.168.56.29:/opt/cascade/disk-cache

# 上传 Python adapter
rsync -az --delete -e 'sshpass -p '\''123456'\'' ssh -o StrictHostKeyChecking=no' \
  --include='*.go' --include='*.py' --include='*/' --exclude='*' \
  adapter/ root@192.168.56.29:/mnt/data/vllmtest/adapter/
```

### 3. 一键部署脚本

```bash
./deploy-remote.sh  # 本地 rsync + 远程编译 + 重启
```

---

## 测试流程

### 启动 Go Engine

```bash
sshpass -p '123456' ssh root@192.168.56.29 '
rm -rf /opt/cascade/data
mkdir -p /opt/cascade/data
nohup /opt/cascade/disk-cache \
  -listen :9100 \
  -cache-path /opt/cascade/data/storage \
  -metadata-path /opt/cascade/data/meta \
  -max-size 50GB > /opt/cascade/cascade_engine.log 2>&1 &
'
```

### 启动 vLLM + Cascade (POSIX)

```bash
sshpass -p '123456' ssh root@192.168.56.29 '
source /mnt/data/vllmtest/.venv/bin/activate
export PYTHONPATH=/mnt/data/vllmtest:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

CACHE_DIR=/mnt/nvfile/test-1/cascade-bench/posix
mkdir -p $CACHE_DIR

KV_CONFIG=$(python3 -c "import json; print(json.dumps({
    \"kv_connector\": \"DiskCacheConnector\",
    \"kv_role\": \"kv_both\",
    \"kv_connector_module_path\": \"adapter.vllm.connector\",
    \"kv_connector_extra_config\": {
        \"disk_cache_path\": \"$CACHE_DIR\",
        \"disk_cache_engine_addr\": \"http://localhost:9100\",
        \"target_device\": \"auto\",
        \"disk_cache_chunk_size_tokens\": 256,
        \"storage_backend\": \"posix\"
    }
}, separators=(\",\", \":\")))")

vllm serve /mnt/data/Qwen2.5-7B-Instruct-AWQ \
  --served-model-name qwen25-7b \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --kv-transfer-config "$KV_CONFIG"
'
```

### 启动 vLLM + Cascade (GDS)

将 `storage_backend` 改为 `"gds"`，`CACHE_DIR` 改为 `/mnt/nvfile/test-1/cascade-bench/gds`。

正式 GDS 对比必须同时启用严格选择，禁止 `gds` 静默降级为 POSIX：

```json
"storage_backend": "gds",
"storage_backend_strict": true
```

启动后必须从日志确认 `NvFileBackend` 被实际选中；仅配置字段为 `gds` 不足以证明路径生效：

```bash
grep -E "DiskCacheConnector.*(NvFileBackend|PosixBackend)" \
  /mnt/data/vllmtest/logs/vllm_gds.log | tail -n 5
```

还必须验证内核和文件系统条件。`cuda.bindings.cufile` 可导入只表示 cuFile binding 可用；若 `gdscheck -p` 显示 `use_compat_mode: true`、没有 `nvidia_fs`，或测试目录不是支持的直连 NVMe / GPFS 文件系统，则结果只能标为 **cuFile compatibility-mode**，不能标为真实 direct GDS：

```bash
lsmod | grep nvidia_fs
/usr/local/cuda/gds/tools/gdscheck.py -p
findmnt -T "$CACHE_DIR" -o TARGET,SOURCE,FSTYPE,OPTIONS
```


### 启动 vLLM + LMCache (对比基准)

```bash
sshpass -p '123456' ssh root@192.168.56.29 '
source /mnt/data/vllmtest/.venv/bin/activate
export LMCACHE_CONFIG_FILE=/mnt/data/vllmtest/configs/lmcache-local.yaml

vllm serve /mnt/data/Qwen2.5-7B-Instruct-AWQ \
  --served-model-name qwen25-7b \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --kv-transfer-config "{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}"
'
```

---

## 验证测试

### 单请求验证

```bash
# 第一次请求（冷启动）
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen25-7b","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":10,"temperature":0}'

# 检查 Go engine 状态
curl -s http://localhost:9100/stats

# 第二次请求（应命中缓存）
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen25-7b","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":10,"temperature":0}'

# 检查缓存命中
curl -s http://localhost:9100/stats
# 期望: MatchHits > 0, BlocksRetrieved > 0
```

### 检查 vLLM 日志

```bash
# 查看外部缓存命中率
grep "External prefix cache hit rate" /mnt/data/vllmtest/logs/vllm_*.log | tail -5

# 查看 connector 日志
grep "Disk cache HIT\|Chunk.*missing" /mnt/data/vllmtest/logs/vllm_*.log | tail -10
```

---

## Benchmark 测试

### 正式长文档 A/B：LMCache 官方方法

当前 LMCache 0.4.3 的推荐入口是 `lmcache bench engine`。官方教程通过一次导出的共享配置，对不同 vLLM 后端重放同一 workload；项目采用 10GB KV 工作集、10000-token 文档、每文档 1 个 query、`tile` 顺序和 4 并发。请求数由模型相关的 `tokens_per_gb_kvcache` 派生，不能硬编码为官方 Qwen3-8B 示例中的 46：当前 Qwen2.5-7B-AWQ 实测约为 `17,361 tokens/GB`，因此派生约 17 个请求：

```json
{
  "model": "qwen25-7b",
  "workload": "long-doc-qa",
  "kv_cache_volume": 10.0,
  "tokens_per_gb_kvcache": 17361,
  "ldqa_document_length": 10000,
  "ldqa_query_per_document": 1,
  "ldqa_shuffle_policy": "tile",
  "ldqa_num_inflight_requests": 4
}
```

保存为 `bench_config.json` 后，对 Cascade POSIX、Cascade GDS 和 LMCache 分别启动同模型服务，再执行完全相同的命令：

```bash
lmcache bench engine \
  --engine-url http://127.0.0.1:8000 \
  --config bench_config.json \
  --output-dir /mnt/data/vllmtest/results/<backend> \
  --json
```

如果没有已知的 `tokens_per_gb_kvcache`，先按 LMCache 官方教程通过 `--lmcache-url` 交互导出配置，或按实际完整 chunk bytes/token 计算，再把同一配置重放到所有后端。不要复制其他模型的教程示例值。正式比较必须保持模型、chat template、vLLM 参数、内置 prefix cache 设置、workload 配置和 TTFT 统计方法一致。

### 历史连续性：官方旧版 `long_doc_qa.py`

仓库已有的 `46 × 10000` 命令使用 LMCache 官方旧 benchmark 脚本，但 `46` 是项目历史固定参数，不是脚本默认值或模型归一化的通用官方值。对当前 Qwen2.5-7B-AWQ，它约对应 `46 × 10000 / 17361 = 26.5GB` KV 压力，仅保留用于复现历史结果；正式 10GB A/B 应使用上方 `bench engine` 配置派生的 17 个请求：

```bash
source /mnt/data/vllmtest/.venv/bin/activate
export PYTHONPATH=/mnt/data/vllmtest:$PYTHONPATH

python /mnt/data/vllmtest/LMCache/benchmarks/long_doc_qa/long_doc_qa.py \
  --model qwen25-7b \
  --host 127.0.0.1 --port 8000 \
  --num-documents 46 \
  --document-length 10000 \
  --output-len 100 \
  --repeat-count 1 \
  --repeat-mode tile \
  --max-inflight-requests 4 \
  --sleep-time-after-warmup 1 \
  --json-output \
  2>&1 | tee /mnt/data/vllmtest/results/<backend>_long_doc_10k.log
```

### T4 单请求 TTFT 诊断口径

下面这组不是正式容量/并发 benchmark，只用于在 15GB T4 上隔离单请求热命中路径并避免并发 prefill OOM：

```bash
python /mnt/data/vllmtest/LMCache/benchmarks/long_doc_qa/long_doc_qa.py \
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

相对正式旧脚本口径，它同时改变了四项：文档数 `46→10`、文档长度 `10000→4096`、输出长度 `100→10`、并发 `4→1`。因此这组数据只能横向比较同一诊断配置下的 POSIX/GDS/LMCache，不能替代正式压力测试。

### 结果门禁

每轮结果至少同时保存：

- benchmark JSON/CSV、成功请求数、mean/median/min/max TTFT；
- vLLM 启动参数和实际 connector backend 日志；
- Cascade `/stats` 前后增量，确认 `MatchHits`、`MatchedTokens`、`ChunksRetrieved`；
- `External prefix cache hit rate`，并确认所有 external-only 方案均使用 `--no-enable-prefix-caching`；
- GDS 的 `gdscheck -p`、`lsmod | grep nvidia_fs` 和 `findmnt -T` 输出。

> 远端遗留的 `/mnt/data/vllmtest/scripts/run_cascade_bench.sh` 使用了 `--enable-prefix-caching`，不能作为 external cache 公平结果；修正或弃用后才能运行。

## 已知问题

### Chunk 0 missing 警告

**现象**：vLLM 日志中出现大量 `Chunk 0 missing for ...` 警告。

**原因**：vLLM 批量处理请求时，第一个请求的 `save_kv_layer` 还在写文件，第二个请求的 `start_load_kv` 就开始尝试加载。Go engine 已通过 `chunk_put` 注册了 chunk，但文件可能还没写完。

**影响**：不影响功能。后续请求能正确命中缓存（`External prefix cache hit rate` 正常增长）。

**根本原因**：`prefix_key` 只 hash 前 16 个 token，多个请求共享同一 prefix_key 时会产生竞态。

### vLLM 版本兼容性

- vLLM 0.21.0：验证通过
- vLLM 0.25.1：需要 `--no-enable-prefix-caching`，connector 接口兼容

---

## 性能基准

### 已验证结果 (A100 + vLLM 0.21)

| 指标 | 值 |
|------|-----|
| 冷启动 TTFT | 1.704s |
| 缓存命中 TTFT | 0.199s |
| Retrieved blocks | 28 |
| Cached tokens | 6624 |

### LMCache 对比基准 (T4 + vLLM 0.25)

| 指标 | 值 |
|------|-----|
| Warmup TTFT | 59.867s |
| Query TTFT | 0.208s |
| Hit rate | ~49.7% |

---

## 故障排查

### Go engine 不响应

```bash
# 检查进程
ps aux | grep disk-cache

# 检查日志
tail -20 /opt/cascade/cascade_engine.log

# 重启
pkill -f disk-cache
nohup /opt/cascade/disk-cache -listen :9100 ... &
```

### vLLM 启动失败

```bash
# 检查端口占用
ss -tlnp | grep 8000

# 检查日志
tail -50 /mnt/data/vllmtest/logs/vllm_*.log

# 常见问题
# 1. 端口被占用 → pkill -9 -f vllm
# 2. GPU 内存不足 → 降低 --gpu-memory-utilization
# 3. 模型路径错误 → 检查 /mnt/data/Qwen2.5-7B-Instruct-AWQ
```

### External prefix cache hit rate = 0%

检查是否使用了 `--no-enable-prefix-caching`。如果使用了 `--enable-prefix-caching`，vLLM 会绕过 Cascade。

### SSH 连接超时

远程机器 GPU 负载高时 SSH 可能超时。解决方案：
1. 等待 GPU 负载降低
2. 使用 `nohup` 启动长时间任务
3. 使用 `remote_connect` 工具建立持久连接
