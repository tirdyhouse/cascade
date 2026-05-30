#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark Predict storage backends.

The benchmark saves and loads deterministic tensors through one or more storage
backends, measures wall-clock latency, and reports approximate throughput.
It is intended for A100/GPU validation but can also run POSIX/CPU locally when
adapter dependencies are installed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
from contextlib import nullcontext
from collections.abc import Iterator
import time
from pathlib import Path
from typing import Any

import torch


_DTYPE_CHOICES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
}


def _parse_shape(value: str) -> tuple[int, ...]:
    try:
        dims = tuple(int(part) for part in value.replace("x", ",").split(",") if part)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid shape {value!r}") from exc
    if not dims or any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return dims


def _parse_backends(value: str) -> list[str]:
    backends = [part.strip().lower() for part in value.split(",") if part.strip()]
    allowed = {"posix", "gds", "nvfile", "cufile", "auto"}
    invalid = sorted(set(backends) - allowed)
    if invalid:
        raise argparse.ArgumentTypeError(f"invalid backend(s): {', '.join(invalid)}")
    if not backends:
        raise argparse.ArgumentTypeError("at least one backend is required")
    return backends


def _make_tensor(shape: tuple[int, ...], dtype: torch.dtype, device: str) -> torch.Tensor:
    numel = math.prod(shape)
    if dtype.is_floating_point:
        base = torch.arange(numel, dtype=torch.float32, device=device).reshape(shape)
        return (base / max(numel, 1)).to(dtype)
    return (torch.arange(numel, dtype=torch.int64, device=device) % 127).reshape(shape).to(dtype)


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
    }


def _mb_per_s(nbytes: int, seconds: float) -> float:
    return 0.0 if seconds <= 0 else (nbytes / seconds) / (1024 * 1024)


def _compare_tensors(expected: torch.Tensor, actual: torch.Tensor) -> bool:
    if expected.dtype != actual.dtype or tuple(expected.shape) != tuple(actual.shape):
        return False
    return torch.equal(expected.detach().cpu(), actual.detach().cpu())


def _binding_status() -> dict[str, Any]:
    try:
        from adapter.storage.gds import detect_binding

        binding = detect_binding()
    except Exception as exc:
        return {"available": False, "name": None, "error": repr(exc)}
    return {
        "available": binding is not None,
        "name": getattr(binding, "name", None) if binding is not None else None,
        "error": None,
    }


def _create_storage_backend(prefer: str):
    try:
        from adapter.storage.backend import create_storage_backend
    except ModuleNotFoundError as exc:
        if exc.name == "safetensors":
            raise RuntimeError(
                "missing Python dependency 'safetensors'; install adapter dependencies "
                "or run in the project venv before benchmarking storage backends"
            ) from exc
        raise
    return create_storage_backend(prefer=prefer)


def benchmark_backend(
    backend_name: str,
    root: Path,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: str,
    iterations: int,
    warmup: int,
    keep_files: bool,
) -> dict[str, Any]:
    backend = _create_storage_backend(backend_name)
    selected = backend.__class__.__name__
    tensor = _make_tensor(shape, dtype, device)
    nbytes = tensor.nbytes

    save_times: list[float] = []
    load_times: list[float] = []
    save_bandwidths: list[float] = []
    load_bandwidths: list[float] = []
    paths: list[Path] = []
    total_runs = warmup + iterations

    for index in range(total_runs):
        path = root / f"storage-bench-{backend_name}-{index}.kvcache"
        paths.append(path)

        _sync(device)
        start = time.perf_counter()
        backend.save(path, tensor)
        _sync(device)
        save_elapsed = time.perf_counter() - start

        _sync(device)
        start = time.perf_counter()
        loaded = backend.load(path, device=device)
        _sync(device)
        load_elapsed = time.perf_counter() - start

        if not _compare_tensors(tensor, loaded):
            raise RuntimeError(f"loaded tensor mismatch for backend {backend_name}")

        if index >= warmup:
            save_times.append(save_elapsed)
            load_times.append(load_elapsed)
            save_bandwidths.append(_mb_per_s(nbytes, save_elapsed))
            load_bandwidths.append(_mb_per_s(nbytes, load_elapsed))

    if not keep_files:
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    save_summary = _summarize(save_times)
    load_summary = _summarize(load_times)
    save_bw_summary = _summarize(save_bandwidths)
    load_bw_summary = _summarize(load_bandwidths)
    return {
        "requested_backend": backend_name,
        "selected_backend": selected,
        "backend_repr": repr(backend),
        "shape": list(shape),
        "dtype": str(dtype),
        "device": device,
        "nbytes": nbytes,
        "iterations": iterations,
        "warmup": warmup,
        "save_seconds": save_summary,
        "load_seconds": load_summary,
        "save_mib_per_s": save_bw_summary,
        "load_mib_per_s": load_bw_summary,
    }


def _markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# Storage Backend Benchmark",
        "",
        f"- Device: `{result['device']}`",
        f"- Shape: `{result['shape']}`",
        f"- Dtype: `{result['dtype']}`",
        f"- Tensor bytes: `{result['nbytes']}`",
        f"- Iterations: `{result['iterations']}` after `{result['warmup']}` warmup",
        f"- GDS binding: `{result['gds_binding']}`",
        "",
        "| Requested | Selected | Save median (s) | Save median (MiB/s) | Load median (s) | Load median (MiB/s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in result["results"]:
        lines.append(
            "| {requested_backend} | {selected_backend} | {save_s:.6f} | {save_bw:.2f} | {load_s:.6f} | {load_bw:.2f} |".format(
                requested_backend=row["requested_backend"],
                selected_backend=row["selected_backend"],
                save_s=row["save_seconds"]["median"],
                save_bw=row["save_mib_per_s"]["median"],
                load_s=row["load_seconds"]["median"],
                load_bw=row["load_mib_per_s"]["median"],
            )
        )
    lines.append("")
    return "\n".join(lines)


_RESULTS_START = "<!-- benchmark-results:start -->"
_RESULTS_END = "<!-- benchmark-results:end -->"


def _write_markdown_report(path: str, report: str) -> None:
    target = Path(path)
    if target.exists():
        existing = target.read_text()
        start = existing.find(_RESULTS_START)
        end = existing.find(_RESULTS_END)
        if start != -1 and end != -1 and start < end:
            replacement = f"{_RESULTS_START}\n{report}\n{_RESULTS_END}"
            updated = existing[:start] + replacement + existing[end + len(_RESULTS_END):]
            target.write_text(updated)
            return
    target.write_text(report)


def _benchmark_root(path: str, keep_files: bool) -> tuple[Path, Iterator[None]]:
    if path:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        return root, nullcontext()
    if keep_files:
        root = Path(tempfile.mkdtemp(prefix="predict-storage-bench-"))
        return root, nullcontext()
    tmp = tempfile.TemporaryDirectory(prefix="predict-storage-bench-")
    return Path(tmp.name), tmp


def run(args: argparse.Namespace) -> dict[str, Any]:
    dtype = _DTYPE_CHOICES[args.dtype]
    root, root_context = _benchmark_root(args.path, args.keep_files)
    with root_context:
        results = [
            benchmark_backend(
                backend_name=backend,
                root=root,
                shape=args.shape,
                dtype=dtype,
                device=args.device,
                iterations=args.iterations,
                warmup=args.warmup,
                keep_files=args.keep_files,
            )
            for backend in args.backends
        ]

        return {
            "ok": True,
            "device": args.device,
            "shape": list(args.shape),
            "dtype": str(dtype),
            "nbytes": math.prod(args.shape) * torch.empty((), dtype=dtype).element_size(),
            "iterations": args.iterations,
            "warmup": args.warmup,
            "path": str(root),
            "gds_binding": _binding_status(),
            "results": results,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", type=_parse_backends,
                        default=_parse_backends(os.environ.get("STORAGE_BENCH_BACKENDS", "posix,gds")),
                        help="comma-separated backends: posix,gds,nvfile,cufile,auto")
    parser.add_argument("--device", default=os.environ.get("STORAGE_BENCH_DEVICE", "cuda:0"),
                        help="torch device for benchmark tensors")
    parser.add_argument("--shape", type=_parse_shape,
                        default=_parse_shape(os.environ.get("STORAGE_BENCH_SHAPE", "4096,4096")),
                        help="tensor shape, comma- or x-separated")
    parser.add_argument("--dtype", default=os.environ.get("STORAGE_BENCH_DTYPE", "float16"),
                        choices=sorted(_DTYPE_CHOICES))
    parser.add_argument("--iterations", type=int, default=int(os.environ.get("STORAGE_BENCH_ITERATIONS", "3")))
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("STORAGE_BENCH_WARMUP", "1")))
    parser.add_argument("--path", default=os.environ.get("STORAGE_BENCH_DIR", ""),
                        help="directory for benchmark files; default: temporary directory")
    parser.add_argument("--keep-files", action="store_true",
                        help="do not remove generated benchmark files")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument("--markdown", default=os.environ.get("STORAGE_BENCH_MARKDOWN", ""),
                        help="optional path to write a Markdown report")
    args = parser.parse_args()

    if args.iterations < 1:
        parser.error("--iterations must be >= 1")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")

    result = run(args)
    report = _markdown_report(result)
    if args.markdown:
        _write_markdown_report(args.markdown, report)

    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
