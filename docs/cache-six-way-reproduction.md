# Cascade / LMCache 六组性能复现

测试机环境已经部署完成。日常复测只需要进入目录、运行脚本、查看汇总成绩。
首次装机、编译和环境配置见 `docs/cache-six-way-full-reproduction.md`。

## 1. 登录并进入目录

```bash
ssh root@192.168.56.29

cd /mnt/data/vllmtest/cascade-current
export TOOLS_DIR=/mnt/data/vllmtest/gds-retest-tools
```

## 2. 运行六组测试

```bash
export RUN_ID=cache-sixway-$(date +%Y%m%d-%H%M%S)

MODE_FILTER=all \
NUM_DOCUMENTS=10 \
bash "$TOOLS_DIR/run_remote_gds_retest.sh"
```

脚本会依次测试 Cascade host、Cascade POSIX、Cascade cuFile、LMCache LocalCPU、
LMCache LocalDisk 和 LMCache GDS 六组。

整轮通常需要 20–30 分钟。完成后终端会输出：

```text
RESULTS_DIR=/mnt/data/vllmtest/gds-retest-results/<RUN_ID>
```

## 3. 查看成绩

```bash
export RUN_DIR=/mnt/data/vllmtest/gds-retest-results/$RUN_ID
cat "$RUN_DIR/summary.md"
```

`summary.md` 直接给出六组的 Mean TTFT、P50、P95、请求命中率和 token 恢复命中率。
原始结果和日志都在同一个 `$RUN_DIR` 下。

需要重新查看以前某一轮时，先列出历史结果：

```bash
ls -1dt /mnt/data/vllmtest/gds-retest-results/* | head
```

然后指定对应目录：

```bash
export RUN_DIR=/mnt/data/vllmtest/gds-retest-results/<RUN_ID>
cat "$RUN_DIR/summary.md"
```
