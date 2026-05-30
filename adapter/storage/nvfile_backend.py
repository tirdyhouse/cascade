# SPDX-License-Identifier: Apache-2.0
"""GDS storage backend via cuFile / nvfile / hipfile.

Provides GPU↔NVMe direct I/O.  Supports three backends:

1. ``cuda.bindings.cufile`` — NVIDIA CUDA Python bindings (``pip install cuda-python``).
2. ``cufile`` — NVIDIA GDS Python SDK (separate package).
3. ``nvfile`` / ``hipfile`` — vendor / AMD equivalents.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch

from adapter.storage.backend import (
    StorageBackend,
    _HEADER_SIZE,
    _pack_header,
    _unpack_header,
)

logger = logging.getLogger(__name__)


# ── cuFile bindings abstraction ───────────────────────────────────

from adapter.storage.gds import (
    CuFileBinding as _CuFileBinding,
    CudaBindingsCufile as _CudaBindingsCufile,
    CuFileHandle as _CuFileHandle,
    PythonCufileModule as _PythonCufileModule,
    VendorModule as _VendorModule,
    detect_binding as _detect_binding,
)


# ── Backend ───────────────────────────────────────────────────────


class NvFileBackend(StorageBackend):
    """GPU Direct Storage backend.

    Writes raw tensor data preceded by a small JSON header (4 KB).
    The header is written via POSIX; the tensor payload is written via
    cuFile (GPU↔NVMe direct DMA).
    """

    def __init__(self) -> None:
        self._binding = _detect_binding()
        if self._binding is None:
            raise RuntimeError(
                "No GDS library found.\n"
                "  Install:  pip install cuda-python\n"
                "  Or:       pip install cufile    (NVIDIA GDS SDK)\n"
                "  Or:       pip install nvfile     (vendor)"
            )
        logger.info("NvFileBackend ready: binding=%s", self._binding.name)

    # ── public API ─────────────────────────────────────────────────

    def save(self, path: Path, tensor: torch.Tensor) -> None:
        path = Path(path)
        if not tensor.is_cuda:
            logger.warning(
                "NvFileBackend.save expects a CUDA tensor; got %s", tensor.device
            )
        self._save_with_gds(path, tensor)

    def load(self, path: Path, device: str = "cuda") -> torch.Tensor:
        return self._load_with_gds(Path(path), device=device)

    @classmethod
    def is_available(cls) -> bool:
        return _detect_binding() is not None

    @classmethod
    def is_supported(cls) -> bool:
        return cls.is_available()

    # ── GDS I/O path ──────────────────────────────────────────────

    def _save_with_gds(self, path: Path, tensor: torch.Tensor) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp" + _rand_suffix(8))
        try:
            # Step 1: metadata header via POSIX (4 KB)
            header = _pack_header(tensor)
            with open(tmp, "wb") as f:
                f.write(header)

            # Step 2: tensor data via GDS
            with _CuFileHandle(self._binding, str(tmp), "r+") as f:
                written = f.write(tensor.data_ptr(), tensor.nbytes,
                                  file_offset=_HEADER_SIZE)
                if written != tensor.nbytes:
                    raise RuntimeError(
                        f"GDS write: expected {tensor.nbytes} bytes, got {written}"
                    )

            os.replace(tmp, path)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            raise

    def _load_with_gds(self, path: Path, device: str = "cuda") -> torch.Tensor:
        # Step 1: read header (POSIX)
        with open(path, "rb") as f:
            header_blob = f.read(_HEADER_SIZE)
        meta = _unpack_header(header_blob)

        dtype = _str_to_dtype(meta["dtype"])
        shape = torch.Size(meta["shape"])
        nbytes = meta["nbytes"]

        # Step 2: allocate GPU tensor
        tensor = torch.empty(shape, dtype=dtype, device=device)

        # Step 3: read tensor data via GDS
        with _CuFileHandle(self._binding, str(path), "r") as f:
            read_bytes = f.read(tensor.data_ptr(), nbytes,
                                file_offset=_HEADER_SIZE)
            if read_bytes != nbytes:
                raise RuntimeError(
                    f"GDS read: expected {nbytes} bytes, got {read_bytes}"
                )
        return tensor

    def __repr__(self) -> str:
        return f"NvFileBackend(binding={self._binding.name})"


# ── helpers ────────────────────────────────────────────────────────


def _rand_suffix(n: int) -> str:
    import uuid
    return uuid.uuid4().hex[:n]


_DTYPE_MAP = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.uint8": torch.uint8,
    "torch.int8": torch.int8,
    "torch.int16": torch.int16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.float8_e4m3fn": torch.float8_e4m3fn,
    "torch.float8_e5m2": torch.float8_e5m2,
}


def _str_to_dtype(s: str) -> torch.dtype:
    if s in _DTYPE_MAP:
        return _DTYPE_MAP[s]
    import re
    m = re.match(r"torch\.(\w+)", s)
    if m:
        return getattr(torch, m.group(1))
    raise ValueError(f"Unknown dtype: {s}")
