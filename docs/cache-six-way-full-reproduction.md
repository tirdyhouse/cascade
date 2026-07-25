# Cascade / LMCache 六组测试：完整环境安装与复现

本文用于从一台新 GPU 测试机开始，完成系统环境、CUDA/GDS、nvfile、Python、
Cascade 编译、服务启动和六组 benchmark。已经装好环境、只想重新跑成绩时，使用精简版
文档：`docs/cache-six-way-reproduction.md`。

本文不保存 SSH 密码、存储认证文件或 Hugging Face token。

## 1. 已验证的基准环境

以下版本已经在 `192.168.56.29` 上完成单并发和 4 并发六组测试：

| 项目 | 版本或路径 |
|---|---|
| OS | Rocky Linux 8.9，kernel `4.18.0-513.5.1.el8_9.x86_64` |
| GPU | NVIDIA Tesla T4，15,360MiB |
| NVIDIA driver | `595.71.05` |
| CUDA toolkit | `13.2`，`/usr/local/cuda-13.2` |
| GDS / cuFile | GDS `1.17.1.22`，`libcufile 13.2` |
| nvidia-fs | `2.28.4`，内核模块显示 `2.28` |
| nvfile client | `7.4.6.6`，挂载点 `/mnt/nvfile` |
| Go | `1.26.5`，`/usr/local/go/bin/go`；项目最低 `1.24` |
| C/C++ | GCC Toolset 13，`/opt/rh/gcc-toolset-13` |
| Python | `3.12.13`，venv `/mnt/data/vllmtest/.venv` |
| PyTorch | `2.11.0+cu130` |
| vLLM | `0.25.1` |
| LMCache | `0.4.3` |
| cuda-python | `13.3.1` |
| 模型 | `/mnt/data/Qwen2.5-7B-Instruct-AWQ` |
| 复现代码 | runtime commit `d06674f`；含报告 commit `93126d1` |

当前机器虽然已经安装 `nvidia-fs`，但 `gdscheck.py -p` 仍显示：

```text
properties.use_compat_mode : true
```

因此两组 GDS 成绩必须称为 **cuFile/GDS compatibility**。只有存储供应商使
`use_compat_mode` 变为 `false` 后，才能标记为 direct GDS。

## 2. 目录和端口约定

在测试机上统一使用：

```bash
export BENCH_ROOT=/mnt/data/vllmtest
export VENV=/mnt/data/vllmtest/.venv
export MODEL_PATH=/mnt/data/Qwen2.5-7B-Instruct-AWQ
export NVFILE_ROOT=/mnt/nvfile
export TOOLS_DIR=/mnt/data/vllmtest/gds-retest-tools
```

正式 runner 使用以下端口：

| 服务 | 地址 |
|---|---|
| vLLM OpenAI API | `127.0.0.1:18205` |
| Cascade Go engine | `127.0.0.1:19205` |

运行前必须确认端口没有被其他服务占用：

```bash
ss -ltn | grep -E ':(18205|19205)$' || true
```

## 3. 安装基础系统依赖

以下步骤只在首次装机时执行。GPU driver、DKMS 和 kernel 包会影响整机，不能在正在
提供服务的节点上直接升级。

```bash
dnf install -y epel-release dnf-plugins-core
dnf install -y \
  git rsync curl jq make tar gzip which pciutils \
  dkms kernel-devel-$(uname -r) kernel-headers-$(uname -r)

dnf install -y \
  gcc-toolset-13-gcc \
  gcc-toolset-13-gcc-c++ \
  gcc-toolset-13-libstdc++-devel
```

验证编译器：

```bash
/opt/rh/gcc-toolset-13/root/usr/bin/gcc --version
/opt/rh/gcc-toolset-13/root/usr/bin/g++ --version
```

Go 需要 1.24 或更高版本。以下示例将 Go 1.26.5 安装到独立版本目录，再创建固定软链；
也可以从组织内部镜像取得同一压缩包：

```bash
export GO_VERSION=1.26.5
export GO_ARCHIVE=/tmp/go${GO_VERSION}.linux-amd64.tar.gz
export GO_INSTALL=/opt/go-${GO_VERSION}

curl -fL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" \
  -o "$GO_ARCHIVE"
test ! -e "$GO_INSTALL"
mkdir -p "$GO_INSTALL"
tar -xzf "$GO_ARCHIVE" -C "$GO_INSTALL" --strip-components=1
ln -sfnT "$GO_INSTALL" /usr/local/go
```

生产环境应从 Go 下载页取得对应 SHA-256 并先校验 `GO_ARCHIVE`。然后验证：

```bash
/usr/local/go/bin/go version
```

参考机器输出为：

```text
go version go1.26.5 linux/amd64
```

## 4. 安装 NVIDIA driver、CUDA、cuFile 和 nvidia-fs

先配置 NVIDIA RHEL 8 官方仓库或导入离线 local-repo RPM。参考机器最终安装的关键
RPM 为：

```text
nvidia-driver-595.71.05
kmod-nvidia-open-dkms-595.71.05
cuda-toolkit-13-2
nvidia-gds-13-2
libcufile-13-2
libcufile-devel-13-2
nvidia-fs-2.28.4
nvidia-fs-dkms-2.28.4
```

仓库配置完成后安装：

```bash
dnf install -y \
  nvidia-driver nvidia-driver-cuda kmod-nvidia-open-dkms \
  cuda-toolkit-13-2 nvidia-gds-13-2 \
  libcufile-13-2 libcufile-devel-13-2

dnf install -y nvidia-fs-dkms-2.28.4-1 nvidia-fs-2.28.4-1
```

如果普通仓库没有 `nvidia-fs 2.28.4`，使用存储/GDS 环境提供的对应 RPM；安装前仍
必须存在与 `uname -r` 完全一致的 `kernel-devel`。安装完成后重启机器：

```bash
reboot
```

重连后加载并设置开机加载：

```bash
modprobe nvidia_fs
printf '%s\n' nvidia_fs > /etc/modules-load.d/nvidia-fs.conf

nvidia-smi
/usr/local/cuda-13.2/bin/nvcc --version
lsmod | grep '^nvidia_fs'
cat /proc/driver/nvidia-fs/version
```

官方安装参考：

- NVIDIA Driver Installation Guide：
  `https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/`
- CUDA Installation Guide for Linux：
  `https://docs.nvidia.com/cuda/cuda-installation-guide-linux/`
- GPUDirect Storage Troubleshooting Guide：
  `https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/`

## 5. 安装并挂载 nvfile 客户端

nvfile 是供应商组件，不在 Cascade 仓库内。供应商需要提供：

- nvfile 7.4.6.6 客户端 RPM；
- `/etc/nvfile/nvfile-client.conf`；
- `/etc/nvfile/connauthfile`；
- 管理节点地址、DNS/hosts、RDMA 网卡和防火墙要求。

客户端节点至少安装：

```bash
dnf install -y \
  ./nvfile-common-7.4.6.6-el8.noarch.rpm \
  ./nvfile-client-7.4.6.6-el8.noarch.rpm \
  ./nvfile-helperd-7.4.6.6-el8.x86_64.rpm \
  ./nvfile-utils-7.4.6.6-el8.x86_64.rpm
```

放置供应商配置。认证文件必须限制权限，不得复制进 Git：

```bash
install -d -m 0755 /etc/nvfile /mnt/nvfile
install -m 0644 /path/from/vendor/nvfile-client.conf \
  /etc/nvfile/nvfile-client.conf
install -m 0600 /path/from/vendor/connauthfile \
  /etc/nvfile/connauthfile

systemctl enable --now nvfile-helperd nvfile-client
mount -t nvfile nvfile_nodev /mnt/nvfile \
  -o cfgFile=/etc/nvfile/nvfile-client.conf
```

需要开机自动挂载时，在 `/etc/fstab` 中加入等价的 `_netdev` 配置；具体参数以供应商
交付为准。

验证挂载和空间：

```bash
findmnt -T /mnt/nvfile -o TARGET,SOURCE,FSTYPE,OPTIONS
df -h /mnt/nvfile
systemctl --no-pager --full status nvfile-helperd nvfile-client
```

预期 `FSTYPE` 为 `nvfile`，参考机器约有 40TiB 空间。benchmark 至少预留 30GiB。

## 6. 安装 Python 3.12、vLLM 和 LMCache

参考机器使用 uv 管理 Python：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

/root/.local/bin/uv python install 3.12.13
/root/.local/bin/uv venv --python 3.12.13 /mnt/data/vllmtest/.venv

/root/.local/bin/uv pip install \
  --python /mnt/data/vllmtest/.venv/bin/python \
  'vllm==0.25.1' \
  'lmcache==0.4.3' \
  'cuda-python==13.3.1' \
  'openai==2.46.0' \
  'ninja==1.13.0' \
  'pytest==8.4.2' \
  'requests-mock==1.12.1' \
  'black==25.11.0'
```

如果机器不能访问公网，应从同一内部 Python 镜像安装以上固定版本。vLLM 安装说明：
`https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html`；LMCache 安装说明：
`https://docs.lmcache.ai/getting_started/installation.html`。

验证环境：

```bash
source /mnt/data/vllmtest/.venv/bin/activate

python --version
vllm --version
python - <<'PY'
import lmcache
import torch
import vllm
from cuda.bindings import cufile

print('torch:', torch.__version__, 'torch CUDA:', torch.version.cuda)
print('vllm:', vllm.__version__)
print('lmcache:', lmcache.__file__)
print('cufile binding:', cufile.__file__)
PY
```

## 7. 准备模型

模型必须固定为同一个本地快照。能够访问 Hugging Face 时可执行：

```bash
source /mnt/data/vllmtest/.venv/bin/activate
hf download Qwen/Qwen2.5-7B-Instruct-AWQ \
  --local-dir /mnt/data/Qwen2.5-7B-Instruct-AWQ
```

需要认证时通过 shell 环境或 `hf auth login` 提供 token，不要把 token 写入文档或
脚本。验证：

```bash
test -f /mnt/data/Qwen2.5-7B-Instruct-AWQ/config.json
du -sh /mnt/data/Qwen2.5-7B-Instruct-AWQ
```

参考快照约为 5.2GiB。

## 8. 部署固定 Cascade 提交

下面命令在持有 Cascade Git 仓库的开发机执行。它使用 `git archive`，只部署已提交
快照，不会把开发机未提交的文件带到测试机。

```bash
cd /path/to/cascade

export GPU_HOST=root@192.168.56.29
export REPRO_COMMIT=93126d1
export REMOTE_BASE=/mnt/data/vllmtest
export REMOTE_RELEASE=$REMOTE_BASE/cascade-repro-$REPRO_COMMIT

ssh "$GPU_HOST" "mkdir -p '$REMOTE_RELEASE'"
git archive --format=tar "$REPRO_COMMIT" | \
  ssh "$GPU_HOST" "tar -xf - -C '$REMOTE_RELEASE'"
ssh "$GPU_HOST" \
  "printf '%s\n' '$REPRO_COMMIT' > '$REMOTE_RELEASE/GIT_COMMIT'"
```

如果这些提交还没有推到远端仓库，必须使用上面的 archive 方式；不要在测试机上假设
`git checkout 93126d1` 一定可用。

## 9. 编译、测试并发布 Cascade

以下命令在测试机执行：

```bash
export CASCADE_RELEASE=/mnt/data/vllmtest/cascade-repro-93126d1
cd "$CASCADE_RELEASE"

mkdir -p bin
/usr/local/go/bin/go build -o bin/disk-cache ./engine/cmd/disk-cache

/usr/local/go/bin/go test ./engine/pkg/cache ./engine/cmd/disk-cache
source /mnt/data/vllmtest/.venv/bin/activate
python -m pytest adapter/tests
python -m black --check \
  adapter/vllm/compiled_transfer.py \
  adapter/vllm/connector_common.py \
  scripts/benchmark_cache_ttft.py \
  scripts/summarize_cache_retest.py
bash -n scripts/run_remote_gds_retest.sh
```

将共享 runner、配置和 LMCache compatibility shim 安装到固定工具目录：

```bash
export TOOLS_DIR=/mnt/data/vllmtest/gds-retest-tools
mkdir -p "$TOOLS_DIR/lmcache_compat/unregistered_cufile"

cp scripts/benchmark_cache_ttft.py \
   scripts/summarize_cache_retest.py \
   scripts/run_remote_gds_retest.sh \
   "$TOOLS_DIR/"
cp test/configs/*.yaml "$TOOLS_DIR/"
cp test/lmcache_compat/unregistered_cufile/sitecustomize.py \
   "$TOOLS_DIR/lmcache_compat/unregistered_cufile/"
chmod 0755 "$TOOLS_DIR/run_remote_gds_retest.sh"

ln -sfnT "$CASCADE_RELEASE" /mnt/data/vllmtest/cascade-current
readlink -f /mnt/data/vllmtest/cascade-current
sha256sum /mnt/data/vllmtest/cascade-current/bin/disk-cache
```

## 10. 检查 GDS 和 nvfile 状态

```bash
source /mnt/data/vllmtest/.venv/bin/activate

export LD_LIBRARY_PATH=/usr/local/cuda-13.2/gds/lib64:/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-}

lsmod | grep '^nvidia_fs'
cat /proc/driver/nvidia-fs/version
findmnt -T /mnt/nvfile -o TARGET,SOURCE,FSTYPE,OPTIONS
python /usr/local/cuda-13.2/gds/tools/gdscheck.py -p
```

判定规则：

- `nvidia_fs` 未加载：先修复驱动/DKMS，不能继续 GDS 测试；
- `use_compat_mode : true`：可以公平比较两套 compatibility 路径，但不能称 direct GDS；
- `use_compat_mode : false`：才具备 direct GDS 的系统条件，仍需结合 backend 日志验收。

## 11. 单独启动 Go 后端做健康检查

正式 benchmark 会自动启动和停止后端。首次安装后可以先单独验证二进制：

```bash
export ENGINE_SMOKE=/mnt/nvfile/cascade-engine-smoke
mkdir -p "$ENGINE_SMOKE/storage" "$ENGINE_SMOKE/metadata"

/mnt/data/vllmtest/cascade-current/bin/disk-cache \
  -listen 127.0.0.1:19205 \
  -cache-path "$ENGINE_SMOKE/storage" \
  -metadata-path "$ENGINE_SMOKE/metadata" \
  -max-size 10GB \
  > /tmp/cascade-engine-smoke.log 2>&1 &
export ENGINE_PID=$!

curl -fsS http://127.0.0.1:19205/stats
kill "$ENGINE_PID"
wait "$ENGINE_PID" 2>/dev/null || true
```

LMCache 没有独立 daemon；它的 LocalCPU、LocalDisk 和 GdsBackend 由 vLLM 进程内部
创建。

## 12. 运行六组单并发 benchmark

runner 会为每组自动执行：启动 Go engine（Cascade 组）、启动 vLLM、等待健康、串行
warmup、query、保存日志和结果、关闭本组进程。

```bash
cd /mnt/data/vllmtest/cascade-current
export TOOLS_DIR=/mnt/data/vllmtest/gds-retest-tools
export RUN_ID=cache-sixway-c1-$(date +%Y%m%d-%H%M%S)

MODE_FILTER=all \
NUM_DOCUMENTS=10 \
CONCURRENCY=1 \
WARMUP_CONCURRENCY=1 \
DOCUMENT_TOKENS=4096 \
MAX_TOKENS=10 \
bash "$TOOLS_DIR/run_remote_gds_retest.sh"
```

六组依次为：

1. `cascade-host`：本轮新增的实验性 4GiB pinned-host 热层；
2. `cascade-posix`：Cascade POSIX 文件路径；
3. `cascade-gds`：Cascade cuFile/GDS compatibility 路径；
4. `lmcache-local-cpu`：LMCache LocalCPU；
5. `lmcache-local-disk`：LMCache LocalDisk；
6. `lmcache-gds`：LMCache GDS，producer 退出并重启后查询持久缓存。

T4 的 BAR1 空间不足以注册 LMCache 恢复完整 4K prompt 所需的 256MiB pool，因此
第 6 组通过仓库中的 opt-in shim 使用 cuFile 官方支持的 unregistered bounce-buffer
路径。它只对该 benchmark 进程生效，不修改已安装的 LMCache 包。

整轮通常需要 20–30 分钟。脚本结束时输出：

```text
RESULTS_DIR=/mnt/data/vllmtest/gds-retest-results/<RUN_ID>
```

## 13. 运行六组 4 并发 benchmark

4 并发只改变 query；warmup 保持串行，保证缓存填充完整且不把并发写入干扰混进读取
成绩。

```bash
cd /mnt/data/vllmtest/cascade-current
export TOOLS_DIR=/mnt/data/vllmtest/gds-retest-tools
export RUN_ID=cache-sixway-c4-$(date +%Y%m%d-%H%M%S)

MODE_FILTER=all \
NUM_DOCUMENTS=10 \
CONCURRENCY=4 \
WARMUP_CONCURRENCY=1 \
DOCUMENT_TOKENS=4096 \
MAX_TOKENS=10 \
bash "$TOOLS_DIR/run_remote_gds_retest.sh"
```

## 14. 查看成绩

```bash
export RUN_DIR=/mnt/data/vllmtest/gds-retest-results/<RUN_ID>
cat "$RUN_DIR/summary.md"
```

`summary.md` 包含 Mean/P50/P95 TTFT、吞吐量、请求命中率、逻辑 token lookup 和物理
KV 恢复覆盖率。原始 JSON、环境快照、GDS 检查和日志都保存在同一目录。

查看本次固定环境：

```bash
cat "$RUN_DIR/environment.txt"
cat "$RUN_DIR/gdscheck.txt"
```

## 15. 成绩有效性检查

```bash
grep -a 'compiled_transfer=True' "$RUN_DIR"/cascade-*/vllm.log
grep -a 'GDS read-ahead:.*workers=6.*registered=True' \
  "$RUN_DIR/cascade-gds/vllm.log"
grep -a 'Retrieved 4088 out of 4088 required tokens' \
  "$RUN_DIR/lmcache-gds/restart-vllm.log"
```

正式六组必须同时满足：

- 每组 query 成功 10/10；
- 每组请求命中 10/10；
- 逻辑 lookup 为 40,880/40,880 tokens；
- 物理恢复为 40,870/40,870 tokens；
- 三个 Cascade 组为 160/160 chunks；
- LMCache GDS 重启日志有 10 条 `Retrieved 4088/4088`；
- 输出 hash 不一致可以单独报告，但 TTFT 样本不能删除；
- `gdscheck` 是 compatibility 时，报告名称必须保留 compatibility。

已验证参考成绩见 `benchmark-report.md`。

## 16. 常见问题

### 端口被占用

```bash
ss -ltnp | grep -E ':(18205|19205)$'
```

runner 只停止自己记录的进程组，不会杀死其他服务。清理或换端口后重新使用新的
`RUN_ID`。

### `compiled_transfer=False`

检查：

```bash
/opt/rh/gcc-toolset-13/root/usr/bin/g++ --version
/usr/local/cuda-13.2/bin/nvcc --version
ls /mnt/data/vllmtest/cascade-current/csrc/kv_transfer/multi_layer_kv_transfer.cu
```

正式测试设置 `CASCADE_COMPILED_TRANSFER=1` 时，runner 会直接失败，而不是静默使用
Python fallback。

### LMCache 出现约 46–50ms，但没有 `Retrieved`

这是 FULL CUDA graph 下的逻辑假命中，不是有效 KV 恢复。正式 runner 强制
PIECEWISE，并要求真实 `Retrieved` 日志。

### GDS 安装完成但仍是 compatibility

安装 `nvidia-fs` 只是必要条件。还需要文件系统、块设备或供应商客户端被 cuFile 识别
为 direct path。将 `gdscheck.txt`、`findmnt`、nvfile client 版本和网络/RDMA 配置交给
存储供应商处理；在 `use_compat_mode=false` 前不要修改报告名称。

### 中断后确认没有残留进程

```bash
pgrep -af 'vllm serve.*18205|disk-cache.*19205' || true
```

runner 使用 trap 清理自己启动的进程。重新执行时必须使用新的 `RUN_ID`，不要覆盖旧
结果目录。
