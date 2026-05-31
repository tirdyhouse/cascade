# Release Notes: Disk-cache validation hardening

## Highlights

- Validated the current disk-cache + vLLM flow on the project A100 host with vLLM 0.21.0 and Qwen2.5-7B-Instruct.
- Refreshed storage-backend benchmark numbers for POSIX and GDS on A100.
- Added regression coverage for connector device selection so multi-GPU examples can safely use `target_device=auto` while explicit device overrides still work.

## Validation summary

Environment:

- GPU: NVIDIA A100-PCIE-40GB
- vLLM: 0.21.0
- Python env: `/root/cascade/.venv-cascade`
- Model: `/tmp/models/Qwen2.5-7B-Instruct`

Commands/results:

```bash
# Go packages and disk-cache HTTP smoke
make test-go
make test-smoke
cd engine && go test -race -count=1 ./pkg/cache

# Storage backend smoke on A100
make test-storage

# POSIX/GDS benchmark refresh
STORAGE_BENCH_MARKDOWN=docs/storage-benchmark-a100.md make bench-storage

# Real vLLM + disk-cache validation
MODEL_PATH=/tmp/models/Qwen2.5-7B-Instruct \
VLLM_EXTRA_ARGS='--tensor-parallel-size 1 --max-model-len 8192' \
REQUEST_REPETITIONS=200 \
MAX_TOKENS=8 \
make test-vllm-cache
```

Observed vLLM validation summary:

- first request retrieved delta: `0`
- second request retrieved delta: `28`
- second request cached tokens: `6624`
- first request latency: `1.704s`
- second request latency: `0.199s`

## Notes

- For an 8k context model, set `REQUEST_REPETITIONS=200` (or increase `--max-model-len`) to avoid vLLM rejecting the generated validation prompt for exceeding the model context window.
- The benchmark doc uses marker-aware regeneration; only the `benchmark-results` block is replaced when refreshing A100 numbers.
