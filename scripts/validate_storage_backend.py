#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Diagnose and smoke-test Predict storage backends.

This script intentionally does not require vLLM. It validates the storage layer
by saving and loading a small tensor through the selected backend and reports
which backend was chosen plus basic GDS binding availability.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
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


def _make_tensor(shape: tuple[int, ...], dtype: torch.dtype, device: str) -> torch.Tensor:
    numel = math.prod(shape)
    if dtype.is_floating_point:
        base = torch.arange(numel, dtype=torch.float32, device=device).reshape(shape)
        return (base / max(numel, 1)).to(dtype)
    return (torch.arange(numel, dtype=torch.int64, device=device) % 127).reshape(shape).to(dtype)


def _binding_status() -> dict[str, Any]:
    try:
        from adapter.storage.gds import detect_binding

        binding = detect_binding()
    except Exception as exc:  # defensive: diagnostics should not crash here
        return {"available": False, "name": None, "error": repr(exc)}
    return {
        "available": binding is not None,
        "name": getattr(binding, "name", None) if binding is not None else None,
        "error": None,
    }


def _compare_tensors(expected: torch.Tensor, actual: torch.Tensor) -> bool:
    if expected.dtype != actual.dtype or tuple(expected.shape) != tuple(actual.shape):
        return False
    return torch.equal(expected.detach().cpu(), actual.detach().cpu())


def run(args: argparse.Namespace) -> dict[str, Any]:
    dtype = _DTYPE_CHOICES[args.dtype]
    smoke_device = args.device
    tensor = _make_tensor(args.shape, dtype, smoke_device)

    result: dict[str, Any] = {
        "requested_backend": args.backend,
        "device": smoke_device,
        "shape": list(args.shape),
        "dtype": str(dtype),
        "gds_binding": _binding_status(),
    }

    try:
        from adapter.storage.backend import create_storage_backend
    except ModuleNotFoundError as exc:
        if exc.name == "safetensors":
            raise RuntimeError(
                "missing Python dependency 'safetensors'; install adapter dependencies "
                "or run in the project venv before validating storage backends"
            ) from exc
        raise

    backend = create_storage_backend(prefer=args.backend)
    result["selected_backend"] = backend.__class__.__name__
    result["backend_repr"] = repr(backend)

    root = Path(args.path) if args.path else Path(tempfile.mkdtemp(prefix="predict-storage-"))
    root.mkdir(parents=True, exist_ok=True)
    path = root / "storage-smoke.kvcache"
    result["path"] = str(path)

    start = time.perf_counter()
    backend.save(path, tensor)
    save_seconds = time.perf_counter() - start

    start = time.perf_counter()
    loaded = backend.load(path, device=smoke_device)
    load_seconds = time.perf_counter() - start

    ok = _compare_tensors(tensor, loaded)
    result.update({
        "ok": ok,
        "file_size": path.stat().st_size if path.exists() else 0,
        "save_seconds": save_seconds,
        "load_seconds": load_seconds,
        "loaded_device": str(loaded.device),
        "loaded_dtype": str(loaded.dtype),
        "loaded_shape": list(loaded.shape),
    })
    if not ok:
        raise RuntimeError("loaded tensor does not match saved tensor")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=os.environ.get("STORAGE_BACKEND", "posix"),
                        choices=("posix", "gds", "nvfile", "cufile", "auto"),
                        help="backend preference; default: %(default)s")
    parser.add_argument("--device", default=os.environ.get("STORAGE_DEVICE", "cpu"),
                        help="torch device for the smoke tensor; default: %(default)s")
    parser.add_argument("--shape", type=_parse_shape,
                        default=_parse_shape(os.environ.get("STORAGE_SHAPE", "2,4,8")),
                        help="tensor shape, comma- or x-separated; default: %(default)s")
    parser.add_argument("--dtype", default=os.environ.get("STORAGE_DTYPE", "float32"),
                        choices=sorted(_DTYPE_CHOICES),
                        help="tensor dtype; default: %(default)s")
    parser.add_argument("--path", default=os.environ.get("STORAGE_TEST_DIR", ""),
                        help="directory for the test file; default: temporary directory")
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable JSON only")
    args = parser.parse_args()

    try:
        result = run(args)
    except Exception as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": repr(exc)}, sort_keys=True))
        else:
            print(f"Storage backend validation failed: {exc!r}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print("Storage backend validation passed")
        print(f"  requested_backend: {result['requested_backend']}")
        print(f"  selected_backend:  {result['selected_backend']} ({result['backend_repr']})")
        print(f"  gds_binding:       {result['gds_binding']}")
        print(f"  tensor:            shape={result['shape']} dtype={result['dtype']} device={result['device']}")
        print(f"  file_size:         {result['file_size']} bytes")
        print(f"  save_seconds:      {result['save_seconds']:.6f}")
        print(f"  load_seconds:      {result['load_seconds']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
