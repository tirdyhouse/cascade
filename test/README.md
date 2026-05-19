# 磁盘缓存对比测试

> 对比 vLLM 原生 / vLLM+LMCache / vLLM+DiskCache 三种方案的性能和显存占用。

## 测试环境

| 项目 | 规格 |
|------|------|
| GPU | x |
| 磁盘 | x |
| 模型 | Qwen 27B FP8 |

## 测试项

| # | 场景 | prompt | gen | 说明 |
|---|------|--------|-----|------|
| 1 | 短 prompt | 15 tok | 32 tok | 基础推理性能 |
| 2 | 短 prompt + 长生成 | 15 tok | 256 tok | 生成吞吐 |
| 3 | 长 prompt | 4k tok | 128 tok | prefill 性能 |
| 4 | 前缀命中 | 4k 相同前缀 | 128 tok | 磁盘缓存命中场景 |

## 运行

```bash
# 1. 原生 vLLM
python3 test/scripts/run_bench.py --mode native

# 2. vLLM + LMCache
python3 test/scripts/run_bench.py --mode lmcache

# 3. vLLM + DiskCache
python3 test/scripts/run_bench.py --mode diskcache

# 4. 汇总报告
python3 test/scripts/gen_report.py
```

## 结果

测试结果保存在 `test/results/` 目录下。
