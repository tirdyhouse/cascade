# Cascade 与 LMCache 最终对比报告

> 测试日期：2026-07-25
> 当前结论：单并发时 Cascade 已在内存层追到 LMCache 的约 1ms 范围内；4 并发时
> LMCache LocalCPU 领先 10.70%，但 Cascade 在 cuFile compatibility 与纯 POSIX
> 文件两组中仍然反超 LMCache。

## 1. 测试口径

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA Tesla T4 15,360 MiB |
| Driver / CUDA | 595.71.05 / CUDA 13.2 |
| GDS 组件 | `nvidia_fs 2.28`，GDS release 1.17.1.22 |
| 存储 | `/mnt/nvfile`，`nvfile` 文件系统 |
| 模型 | Qwen2.5-7B-Instruct-AWQ |
| vLLM | 0.25.1 |
| LMCache | 0.4.3 系列，当前测试环境安装版本 |
| 文档 | 10 份互异 prompt，每份 API 实测 4,088 tokens |
| 输出 | 10 tokens |
| 并发 | 1，串行 |
| Cache chunk | 256 tokens，每份 prompt 16 chunks |
| vLLM prefix cache | 关闭 |
| CUDA graph | 强制/保持 PIECEWISE，确保 connector 真正执行 KV 恢复 |

每组均先写入 10 份缓存，再查询相同的 10 份 prompt。TTFT 是客户端 streaming
API 从发请求到收到首 token 的端到端时间。输出 hash 不一致单独统计，但不会从
TTFT 样本中剔除。

逻辑命中统一按完整的 4,088 tokens 计算。vLLM 为生成首 token logits，需要让最后
一个 prompt token 再经过前向，因此调度器要求物理注入 4,087 tokens；Cascade 和
LMCache 使用同一口径，没有人为少匹配一个 token。

## 2. 六组有效成绩

| 存储层 | 方案 | Mean TTFT | P50 | P95 | Min–Max | 请求/Token 命中 | 输出 hash |
|---|---|---:|---:|---:|---:|---:|---:|
| 内存 | **LMCache LocalCPU** | **63.706ms** | **60.997ms** | 76.311ms | 60.305–88.483ms | 10/10；40,880/40,880 | 9/10 |
| 内存 | **Cascade 4GiB pinned host tier** | 65.089ms | 61.991ms | 78.727ms | 61.200–91.489ms | 10/10；40,880/40,880；160/160 chunks | **10/10** |
| cuFile compatibility | **Cascade NvFile，6 workers** | **76.740ms** | **72.917ms** | 95.669ms | 71.337–113.516ms | 10/10；40,880/40,880；160/160 chunks | **10/10** |
| cuFile compatibility | LMCache GdsBackend | 89.157ms | 83.167ms | 117.289ms | 81.350–143.471ms | 10/10；40,880/40,880 | 9/10 |
| POSIX 文件 | **Cascade，4 workers + 256MiB pinned staging** | **89.417ms** | **82.122ms** | 121.985ms | 78.074–149.930ms | 10/10；40,880/40,880；160/160 chunks | **10/10** |
| POSIX 文件 | LMCache LocalDisk | 150.181ms | 146.986ms | 164.953ms | 146.124–176.650ms | 10/10；40,880/40,880 | **10/10** |

同层横向差异：

| 对比 | Mean 差异 | P50 差异 | 结论 |
|---|---:|---:|---|
| Cascade host tier vs LMCache LocalCPU | +1.382ms（+2.17%） | +0.994ms（+1.63%） | 基本追平，稳定差距约 1ms |
| Cascade cuFile vs LMCache GdsBackend | **-12.417ms（-13.93%）** | **-10.250ms（-12.32%）** | Cascade 反超 |
| Cascade POSIX vs LMCache LocalDisk | **-60.764ms（-40.46%）** | **-64.864ms（-44.13%）** | Cascade 明显反超 |

LMCache CPU/GDS 的第 5 份文档发生近似数值输出分叉，因此输出 hash 为 9/10；该
样本的 TTFT 仍完整计入 Mean/P50/P95。它不是 cache miss，也不是文件损坏。

### 2.1 四并发复测

在相同 10 份 prompt 上保持 warmup 串行，query 阶段最多同时保持 4 个在途请求。
所有方案都使用 PIECEWISE，输出 10 tokens；吞吐量按完整 query 阶段的 10 个成功请求
计算。

| 存储层 | 方案 | Mean TTFT | P50 | P95 | 吞吐量 | 请求/Token 命中 | 输出 hash |
|---|---|---:|---:|---:|---:|---:|---:|
| 内存 | **LMCache LocalCPU** | **170.822ms** | **151.311ms** | **223.253ms** | **5.974 req/s** | 10/10；40,880/40,880 | 10/10 |
| 内存 | Cascade 4GiB pinned host tier | 189.108ms | 183.724ms | 249.831ms | 5.865 req/s | 10/10；40,880/40,880；160/160 chunks | 10/10 |
| cuFile compatibility | **Cascade NvFile，6 workers** | **206.994ms** | **178.805ms** | **283.796ms** | **5.809 req/s** | 10/10；40,880/40,880；160/160 chunks | 10/10 |
| cuFile compatibility | LMCache GdsBackend，重启后持久命中 | 235.504ms | 222.513ms | 323.092ms | 5.545 req/s | 10/10；40,880/40,880 | 10/10 |
| POSIX 文件 | **Cascade，4 workers + 256MiB pinned staging** | **238.679ms** | **195.948ms** | **338.744ms** | **5.481 req/s** | 10/10；40,880/40,880；160/160 chunks | 10/10 |
| POSIX 文件 | LMCache LocalDisk | 457.456ms | 463.235ms | 551.144ms | 4.041 req/s | 10/10；40,880/40,880 | 10/10 |

同层横向差异：

| 对比 | Mean 差异 | P50 差异 | 吞吐量差异 | 结论 |
|---|---:|---:|---:|---|
| Cascade host tier vs LMCache LocalCPU | +18.286ms（+10.70%） | +32.413ms（+21.42%） | -1.82% | 并发恢复成为下一阶段主要差距 |
| Cascade cuFile vs LMCache GdsBackend | **-28.510ms（-12.11%）** | **-43.708ms（-19.64%）** | **+4.76%** | Cascade 继续领先 |
| Cascade POSIX vs LMCache LocalDisk | **-218.777ms（-47.82%）** | **-267.287ms（-57.70%）** | **+35.63%** | Cascade 明显领先 |

六组均为 query 并发 4、warmup 并发 1，且物理恢复覆盖均为
40,870/40,870 tokens。LMCache GDS 在 producer 退出并重启后有 10 条真实
`Retrieved 4088 out of 4088 required tokens`，没有把进程内状态或逻辑假命中计入
成绩。

## 3. 为什么以前 LMCache 是约 46–50ms

该成绩不能作为有效恢复结果。FULL CUDA graph 下，LMCache 的非 layerwise 路径在
`start_load_kv` 收到 `attn_metadata=None`，日志虽然显示逻辑命中 4,088 tokens，
却没有任何对应的 `Retrieved ...` 记录，KV payload 实际未注入。

有效复测强制 PIECEWISE 后，每次请求均出现：

```text
LMCache hit tokens: 4088, need to load: 4087
Retrieved 4088 out of 4088 required tokens
```

因此 LMCache LocalCPU 的有效 Mean 是 63.706ms，而不是约 46–50ms。无恢复的
FULL-graph 数字只能作为“首 token 固定开销”的诊断值，不能列入六组排名。

## 4. GDS 状态说明

本机已经安装并加载 `nvidia_fs 2.28`，平台检查也通过；但 `gdscheck.py -p` 明确显示：

```text
properties.use_compat_mode : true
NVMe / NVMeOF / SCSI / NFS / BeeGFS ... : compat
```

所以表里的两组 GDS 都应称为 **cuFile compatibility path**，不能宣传为已经验证的
GPU Direct Storage 直通路径。Cascade 的 192MiB GPU staging pool 确实完成了
`cuFileBufRegister`，但底层 nvfile 文件系统仍由 cuFile compatibility mode 处理。
供应商修复文件系统识别/直通后，需要原样再跑一次六组脚本；当前结果可以比较两套
软件在同一兼容环境中的效率，但不能外推 direct-GDS 的绝对性能。

T4 的 BAR1 只有 256MiB。实测 192MiB 注册稳定，224MiB 注册失败；最终采用 6 个
I/O workers、192MiB pool、13 个注册 slot。4 workers 暴露约 26–30ms I/O wait，
8 workers 则超过全注册双缓冲深度并产生临时未注册 buffer；6 workers 是当前机器的
最佳稳定点。

## 5. Cascade 本轮完成的优化

1. **跨 28 层 compiled paged-KV scatter**
   原路径是 16 chunks × 28 layers，即 448 次 Python/paged-KV 注入。现在每个
   chunk 一次 all-layer CUDA kernel，共 16 次 launch。GDS profile 的注入调度从
   约 21ms 降到约 3.2ms。

2. **修复异步 source-pointer 表生命周期**
   每个在途指针表使用独立 pinned host tensor，并在 CUDA Event 完成后才释放，避免
   CPU 提前覆盖同一 224B 指针表。修复前速度快但输出 0/10 正确；修复后 10/10 正确。

3. **完整 host-cache 快路径**
   16/16 对象都命中 pinned LRU 时，不再为每个请求创建 `ThreadPoolExecutor` 和 16 个
   Future；直接读取 LRU 并完成 H2D/scatter。

4. **match + resolve 融合**
   `/v2/chunks/match` 可同时返回 `matched_objects`，热路径不再追加一次 `/resolve`
   HTTP。旧引擎未返回对象时仍自动回退，协议保持兼容。

5. **异步 retrieval 统计**
   `/v2/chunks/retrieved` 由一个持久 daemon reporter 合并上报，不再阻塞 TTFT，也不
   再为每个请求新建线程。

6. **GDS read-ahead + 注册 staging pool**
   cuFile worker 只负责把数据读到固定 GPU slot；vLLM 执行线程负责校验和注入。最终
   选用 6 workers / 192MiB / 13 slots。

7. **POSIX read-ahead + 启动期 pinned staging 预热**
   4 个 CPU workers 合并读取每个 `.cobj`，一次 H2D 搬运完整 object。启动时预热
   256MiB caching pinned allocator（18 个 object-sized blocks），避免首个查询承担
   OS page pinning。10 文档 Mean 从 96.554ms 降到 89.417ms，P50 从 86.560ms
   降到 82.122ms。

8. **verified path/layout LRU**
   已验证 immutable object path 与 64KiB header/layout 进入有界 LRU；失效时同步
   删除，避免远端文件系统上反复 `Path.resolve()` 和 header 解析。

9. **基准正确性修复**
   输出 hash 不一致不再从 TTFT 中剔除；独立报告 `matching_outputs`。Runner 还会在
   `CASCADE_COMPILED_TRANSFER=1` 时强制核对日志中的 `compiled_transfer=True`，防止
   CUDA 扩展构建失败后静默拿 Python fallback 成绩。

## 6. 剩余差距在哪里

内存组已不存在结构性的几十毫秒差距。Cascade host profile 的稳态恢复为：

| 阶段 | 稳态耗时 |
|---|---:|
| fused resolve | 0.000–0.001ms |
| setup/layout | 约 0.11ms |
| Python/H2D/kernel dispatch | 约 3.6ms |
| GPU 完成 H2D + scatter | 约 16.5ms |
| 恢复总计 | 约 20.7ms |

单请求需要搬运约 223.56MiB KV；T4 上约 16.5ms 已接近 PCIe H2D 带宽下限。
LMCache LocalCPU 日志中的内部 retrieve 为约 18.27–18.48ms。两者端到端 P50 只差
0.994ms，主要是 Cascade 的 Go match/协议对象构造和少量 Python 校验，不再是
448 次 layer 注入或 token 命中率问题。

文件组中 Cascade 已经领先。当前更值得做的下一阶段工作是：

- 将 CUDA extension 作为预编译构件发布，消除生产环境 JIT/GCC 依赖；
- direct-GDS 可用后重新调 staging 大小、worker 数和 cuFile async/batch API；
- 做并发、多请求、不同 prompt 长度和冷文件页缓存压力测试；
- 若还要压低内存组最后约 1ms，需要设计带失效版本的进程内 match cache，不能用
  不安全的永久缓存绕过 Go 权威元数据；
- layerwise I/O/模型计算重叠可作为长上下文方案，但必须继续核验真实恢复，不能再次
  出现 FULL graph“逻辑命中但未注入”的假快结果。

## 7. 原始结果

| 结果 | 远端目录 |
|---|---|
| Cascade host tier，10 docs | `/mnt/data/vllmtest/gds-retest-results/cascade-final-hostcpu-10doc-20260725/` |
| Cascade cuFile，10 docs | `/mnt/data/vllmtest/gds-retest-results/cascade-final-gds-w6-10doc-20260725/` |
| Cascade POSIX，10 docs | `/mnt/data/vllmtest/gds-retest-results/cascade-final-posix-pinnedwarm-10doc-20260725/` |
| LMCache LocalCPU 有效 PIECEWISE | `/mnt/data/vllmtest/gds-retest-results/lmcache-piecewise-valid-20260725-1457/` |
| LMCache LocalDisk 有效 PIECEWISE | `/mnt/data/vllmtest/gds-retest-results/lmcache-localdisk-piecewise-20260725-1450/` |
| LMCache cuFile 有效 PIECEWISE | `/mnt/data/vllmtest/gds-retest-results/lmcache-gds-unregistered-valid-20260725-1520/` |
| 六组 4 并发复测 | `/mnt/data/vllmtest/gds-retest-results/cache-sixway-c4-d06674f/` |
| Cascade GDS 4/6/7/8-worker profiles | `/mnt/data/vllmtest/gds-retest-results/cascade-gds-*-profile-3doc-20260725/` |

统一 runner：`scripts/run_remote_gds_retest.sh`。单组可通过 `MODE_FILTER` 选择，
诊断 breakdown 使用 `CASCADE_LOAD_PROFILE=1`；正式成绩必须保持 profile 关闭。

## 8. 验证

- Python：176 passed，7 skipped；
- Go：`go test ./engine/pkg/cache ./engine/cmd/disk-cache` 通过；
- `black`、`bash -n`、`git diff --check` 通过；
- 三个 Cascade 最终组均为 10/10 输出一致、10/10 请求命中、160/160 chunks 恢复。
