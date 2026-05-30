# A100 Storage Backend Benchmark

This document records a reproducible POSIX vs GDS storage-backend benchmark run
for Predict. Regenerate with:

```bash
PYTHONPATH=. /root/cascade/.venv-cascade/bin/python scripts/benchmark_storage_backend.py \
  --backends posix,gds \
  --device cuda:0 \
  --shape 4096,4096 \
  --dtype float16 \
  --iterations 3 \
  --warmup 1 \
  --markdown docs/storage-benchmark-a100.md
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
| posix | PosixBackend | 0.067491 | 474.14 | 0.009099 | 3516.77 |
| gds | NvFileBackend | 0.019749 | 1620.33 | 0.005058 | 6326.58 |
<!-- benchmark-results:end -->
