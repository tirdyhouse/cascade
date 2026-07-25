# Cascade / LMCache 六组性能稳定复现指南

本文用于在 GPU 测试机上稳定复现以下六组单请求 TTFT：

1. Cascade 4GiB pinned host tier；
2. Cascade POSIX 文件；
3. Cascade cuFile compatibility；
4. LMCache LocalCPU；
5. LMCache LocalDisk；
6. LMCache GdsBackend compatibility。

当前参考成绩与分析见仓库根目录的 `benchmark-report.md`。本文只描述可重复执行的
部署、运行和验收步骤，不在仓库中保存测试机密码。

## 1. 固定测试口径

参考环境：

| 项目 | 固定值 |
|---|---|
| GPU | NVIDIA Tesla T4 15,360MiB |
| Driver / CUDA | 595.71.05 / CUDA 13.2 |
| 模型 | `/mnt/data/Qwen2.5-7B-Instruct-AWQ` |
| Python venv | `/mnt/data/vllmtest/.venv` |
| vLLM | 0.25.1 |
| LMCache | 当前测试机的 0.4.3 系列版本 |
| 存储 | `/mnt/nvfile` |
| 文档 | 10 份互异 prompt，每份 API 实测 4,088 tokens |
| 输出 | 10 tokens |
| 并发 | 1，串行 |
| chunk | 256 tokens |
| vLLM prefix cache | 关闭 |
| CUDA graph | PIECEWISE |

公平性规则：

- 两端均按完整 4,088 tokens 做逻辑 lookup；
- vLLM 为首 token logits 统一只物理注入 4,087 tokens；
- 输出 hash 不一致单独统计，但 TTFT 样本不得删除；
- Cascade 的 compiled transfer 必须在日志中显示为 `True`；
- LMCache 必须出现真实 `Retrieved` 日志，不能只看逻辑 hit；
- 当前 `gdscheck` 为 compatibility mode，因此不得把结果标成 direct GDS。

## 2. 提交前本地验证

在仓库根目录执行：

```bash
python3 -m pytest adapter/tests
go test ./engine/pkg/cache ./engine/cmd/disk-cache
python3 -m black --check \
  adapter/vllm/compiled_transfer.py \
  adapter/vllm/connector_common.py \
  adapter/vllm/go_client.py \
  adapter/tests/test_connector_common.py \
  adapter/tests/test_vllm_helpers.py \
  scripts/benchmark_cache_ttft.py \
  scripts/summarize_cache_retest.py
bash -n scripts/run_remote_gds_retest.sh
git diff --check
```

当前版本预期为 `174 passed, 7 skipped`，两个 Go package 测试通过。

## 3. 部署到测试机

先通过 SSH agent 或临时 `SSHPASS` 配置认证，不要把密码写入脚本或 Git。部署必须从
已经提交的 `HEAD` 创建快照，避免把工作区里的临时改动混入测试版本。以下命令在开发机
仓库根目录执行：

```bash
export GPU_HOST=root@192.168.56.29
export REMOTE_BASE=/mnt/data/vllmtest
export RELEASE_ID=cascade-repro-$(date +%Y%m%d-%H%M%S)
export REMOTE_RELEASE=$REMOTE_BASE/$RELEASE_ID
export REMOTE_TOOLS=$REMOTE_BASE/gds-retest-tools
export REMOTE_GO=${REMOTE_GO:-/usr/local/go/bin/go}
export COMMIT_SHA=$(git rev-parse HEAD)

ssh "$GPU_HOST" "mkdir -p '$REMOTE_RELEASE' '$REMOTE_TOOLS/lmcache_compat/unregistered_cufile'"

git archive --format=tar "$COMMIT_SHA" | \
  ssh "$GPU_HOST" "tar -xf - -C '$REMOTE_RELEASE'"

ssh "$GPU_HOST" "
  printf '%s\n' '$COMMIT_SHA' > '$REMOTE_RELEASE/GIT_COMMIT'
  cp '$REMOTE_RELEASE/scripts/benchmark_cache_ttft.py' \
     '$REMOTE_RELEASE/scripts/summarize_cache_retest.py' \
     '$REMOTE_RELEASE/scripts/run_remote_gds_retest.sh' \
     '$REMOTE_TOOLS/'
  cp '$REMOTE_RELEASE'/test/configs/*.yaml '$REMOTE_TOOLS/'
  cp '$REMOTE_RELEASE/test/lmcache_compat/unregistered_cufile/sitecustomize.py' \
     '$REMOTE_TOOLS/lmcache_compat/unregistered_cufile/'
  chmod 0755 '$REMOTE_TOOLS/run_remote_gds_retest.sh'
"
```

在测试机编译 Go engine：

```bash
ssh "$GPU_HOST" "cd '$REMOTE_RELEASE' && \
  test -x '$REMOTE_GO' && \
  mkdir -p bin && \
  '$REMOTE_GO' build -o bin/disk-cache ./engine/cmd/disk-cache && \
  ln -sfnT '$REMOTE_RELEASE' '$REMOTE_BASE/cascade-current'"
```

`cascade-current` 只在代码快照与 Go engine 均部署成功后切换；runner 默认使用该软链。
正式归档仍以 `environment.txt` 中记录的 release 路径、commit SHA 和 engine SHA-256 为准。

Runner 会优先选择测试机已有的 GCC Toolset 13。基础系统的 GCC 8 不能编译当前
PyTorch CUDA extension；如果机器没有 `/opt/rh/gcc-toolset-13`，必须先提供 GCC 9+
并通过 `CC`、`CXX` 显式指定。

## 4. 运行前环境检查

登录测试机后执行：

```bash
source /mnt/data/vllmtest/.venv/bin/activate

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
/usr/local/go/bin/go version
vllm --version
python -c 'import lmcache, vllm; print(lmcache.__file__); print(vllm.__version__)'
findmnt -T /mnt/nvfile -o TARGET,SOURCE,FSTYPE,OPTIONS
lsmod | grep '^nvidia_fs'
cat /proc/driver/nvidia-fs/version
python /usr/local/cuda-13.2/gds/tools/gdscheck.py -p
df -h /mnt/nvfile
ss -ltn | grep -E ':(18205|19205)$' || true
```

验收条件：

- 模型、venv、`bin/disk-cache` 均存在；
- 18205 和 19205 没有其他服务监听；
- `/mnt/nvfile` 至少保留约 30GiB 可用空间；
- `nvidia_fs` 已加载；
- 如果 `gdscheck` 显示 `properties.use_compat_mode : true`，结果必须继续标为
  compatibility，而不是 direct GDS。

## 5. 一键运行六组

在测试机 shell 中设置本次 release：

```bash
export CASCADE_ROOT=/mnt/data/vllmtest/<RELEASE_ID>
export TOOLS_DIR=/mnt/data/vllmtest/gds-retest-tools
export RUN_ID=cache-sixway-$(date +%Y%m%d-%H%M%S)

env \
  CASCADE_ROOT="$CASCADE_ROOT" \
  TOOLS_DIR="$TOOLS_DIR" \
  RUN_ID="$RUN_ID" \
  MODE_FILTER=all \
  NUM_DOCUMENTS=10 \
  DOCUMENT_TOKENS=4096 \
  MAX_TOKENS=10 \
  FORCE_PIECEWISE_CUDAGRAPH=1 \
  CASCADE_COMPILED_TRANSFER=1 \
  CASCADE_HOST_CACHE_GIB=4 \
  CASCADE_POSIX_HOST_CACHE_GIB=0 \
  CASCADE_POSIX_PINNED_STAGING_MIB=256 \
  CASCADE_POSIX_LOAD_WORKERS=4 \
  CASCADE_GDS_LOAD_WORKERS=6 \
  CASCADE_GDS_STAGING_BUFFER_MIB=192 \
  bash "$TOOLS_DIR/run_remote_gds_retest.sh"
```

Runner 会依次执行：

```text
cascade-host
cascade-posix
cascade-gds
lmcache-local-cpu
lmcache-local-disk
lmcache-gds
```

每组使用独立 cache/metadata 目录。LMCache GDS 会先写入持久缓存、结束 producer
进程、启动新 vLLM，再执行 query，避免本进程状态伪装成持久命中。整轮通常需要
20–30 分钟。

结果目录：

```text
/mnt/data/vllmtest/gds-retest-results/<RUN_ID>/
```

KV 数据目录：

```text
/mnt/nvfile/codex-gds-retest/<RUN_ID>/
```

## 6. 单组与 profile 复测

`MODE_FILTER` 支持逗号分隔，或只指定一组：

```bash
MODE_FILTER=cascade-host bash "$TOOLS_DIR/run_remote_gds_retest.sh"
MODE_FILTER=cascade-posix bash "$TOOLS_DIR/run_remote_gds_retest.sh"
MODE_FILTER=cascade-gds bash "$TOOLS_DIR/run_remote_gds_retest.sh"
MODE_FILTER=lmcache-local-cpu bash "$TOOLS_DIR/run_remote_gds_retest.sh"
MODE_FILTER=lmcache-local-disk bash "$TOOLS_DIR/run_remote_gds_retest.sh"
MODE_FILTER=lmcache-gds bash "$TOOLS_DIR/run_remote_gds_retest.sh"
```

阶段 profile 使用 3 份文档，避免 profile 同步污染正式成绩：

```bash
RUN_ID=cascade-gds-profile-$(date +%Y%m%d-%H%M%S) \
MODE_FILTER=cascade-gds \
NUM_DOCUMENTS=3 \
CASCADE_LOAD_PROFILE=1 \
bash "$TOOLS_DIR/run_remote_gds_retest.sh"
```

正式成绩必须保持 `CASCADE_LOAD_PROFILE=0`。

## 7. 结果验收

Runner 在 `MODE_FILTER=all` 时自动生成：

```text
summary.json
summary.md
summary.log
```

先查看总表：

```bash
RUN_DIR=/mnt/data/vllmtest/gds-retest-results/<RUN_ID>
cat "$RUN_DIR/summary.md"
```

再执行强制检查：

```bash
grep -a 'compiled_transfer=True' "$RUN_DIR"/cascade-*/vllm.log
grep -a 'POSIX pinned staging allocator warmed' \
  "$RUN_DIR/cascade-posix/vllm.log"
grep -a 'GDS read-ahead:.*workers=6.*registered=True' \
  "$RUN_DIR/cascade-gds/vllm.log"
grep -a 'Retrieved 4088 out of 4088 required tokens' \
  "$RUN_DIR/lmcache-gds/restart-vllm.log"

for result in \
  "$RUN_DIR/cascade-host/query.json" \
  "$RUN_DIR/cascade-posix/query.json" \
  "$RUN_DIR/cascade-gds/query.json" \
  "$RUN_DIR/lmcache-local-cpu/query.json" \
  "$RUN_DIR/lmcache-local-disk/query.json" \
  "$RUN_DIR/lmcache-gds/restart-query.json"; do
  jq '{file: input_filename, summary: .summary}' "$result"
done
```

每个 Cascade 组还应满足：

```bash
jq . "$RUN_DIR/cascade-host/stats-after-query.json"
jq . "$RUN_DIR/cascade-posix/stats-after-query.json"
jq . "$RUN_DIR/cascade-gds/stats-after-query.json"
```

10 文档参考值为：

- `MatchHits` query 增量 10/10；
- `MatchedTokens` query 增量 40,880；
- `ChunksRetrieved` query 增量 160；
- `matching_outputs=10`；
- query transport success 10/10。

LMCache CPU/GDS 可能出现 9/10 输出 hash 的近似数值分叉，但其 TTFT 样本必须保留。

## 8. 同机性能告警线

这些是回归告警线，不是跨硬件 SLA：

| 方案 | Mean 告警线 | P50 告警线 |
|---|---:|---:|
| Cascade host | >75ms | >70ms |
| LMCache LocalCPU | >75ms | >70ms |
| Cascade cuFile compatibility | >90ms | >85ms |
| LMCache GDS compatibility | >110ms | >100ms |
| Cascade POSIX | >110ms | >100ms |
| LMCache LocalDisk | >180ms | >170ms |

超过告警线时，先检查首样本、CUDA extension、页缓存、GDS worker、输出 hash 和真实
`Retrieved`，不要直接用剔除慢样本的方式修正数据。

## 9. 常见无效复现

以下结果不能进入正式对比：

- `FORCE_PIECEWISE_CUDAGRAPH=0` 时 LMCache 约 46–50ms，但没有真实 `Retrieved`；
- LMCache 使用 192MiB registered pool，却只恢复约 3,328/4,088 tokens；
- LMCache layerwise 仅保存/恢复约 2,040/4,088 tokens；
- Cascade 配置了 compiled transfer，但日志实际为 `compiled_transfer=False`；
- 人为少做一个逻辑 token lookup；
- 输出 hash 不一致后把该请求从 TTFT 中删除；
- `gdscheck` 仍为 compatibility mode，却把结果写成 direct GDS。

## 10. 产物留存

每轮至少保留：

- `environment.txt`；
- `gdscheck.txt`；
- 六组 warmup/query JSON；
- vLLM、restart-vLLM、Go engine 日志；
- Cascade query 前后 stats；
- `summary.json` 和 `summary.md`；
- 对应 Git commit SHA 与 `RUN_ID`。

不要覆盖旧结果目录。需要释放空间时，应先归档报告，并只处理已经确认的具体
`RUN_ID` 目录。
