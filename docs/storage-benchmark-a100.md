# A100 Storage Backend Benchmark

This document records a reproducible POSIX vs GDS storage-backend benchmark run
for Predict. Regenerate on the A100 validation host with:

```bash
STORAGE_BENCH_BACKENDS=posix,gds \
STORAGE_BENCH_DEVICE=cuda:0 \
STORAGE_BENCH_SHAPE=4096,4096 \
STORAGE_BENCH_DTYPE=float16 \
STORAGE_BENCH_ITERATIONS=3 \
STORAGE_BENCH_WARMUP=1 \
STORAGE_BENCH_MARKDOWN=docs/storage-benchmark-a100.md \
PYTHON=/root/cascade/.venv-cascade/bin/python \
make bench-storage
# The script updates only the benchmark-results marker block when markers exist.
```

> Results below were generated on the project A100 validation host.

<!-- benchmark-results:start -->
# Storage Backend Benchmark

- Device: `cuda:0`
- Shape: `[4096, 4096]`
- Dtype: `torch.float16`
- Tensor bytes: `33554432`
- Iterations: `3` after `1` warmup
- GDS binding: `{'available': True, 'name': 'cuda.bindings.cufile', 'error': None}`

| Requested | Selected | Save median (s) | Save median (MiB/s) | Load median (s) | Load median (MiB/s) |
|---|---|---:|---:|---:|---:|
| posix | PosixBackend | 0.066639 | 480.20 | 0.009055 | 3534.14 |
| gds | NvFileBackend | 0.020430 | 1566.36 | 0.005160 | 6202.14 |

<!-- benchmark-results:end -->
