# SPDX-License-Identifier: Apache-2.0
"""GDS storage backend via nvfile/cuFile.

Provides GPU↔NVMe direct I/O using NVIDIA GPUDirect Storage (or
AMD hipFile equivalent).

Auto-detection
--------------

The backend probes for a GDS-capable library in this order:

1. ``cufile`` — NVIDIA GPUDirect Storage (import ``cufile``).
2. ``nvfile`` — vendor-provided GDS library with same API shape.
3. ``hipfile`` — AMD ROCm equivalent.

Usage
-----

.. code-block:: python

    from adapter.storage.nvfile_backend import NvFileBackend

    if NvFileBackend.is_supported():
        backend = NvFileBackend()
        backend.save(path, gpu_tensor)
        tensor = backend.load(path)
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import struct
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import torch

from adapter.storage.backend import (
    StorageBackend,
    _HEADER_SIZE,
    _pack_header,
    _unpack_header,
)

logger = logging.getLogger(__name__)

# Types accepted by cuFile.write/read — we only use buffer protocol
# or ctypes void-pointer addresses below, so no special imports needed.


def _probe_library() -> Optional[str]:
    """Return the name of the first available GDS library, or ``None``."""
    for lib in ("cufile", "nvfile", "hipfile"):
        try:
            __import__(lib)
            logger.debug("GDS library found: %s", lib)
            return lib
        except ImportError:
            continue
    return None


class _CuFileHandle:
    """Thin wrapper around a CuFile opened via the GDS driver.

    Both ``cufile`` / ``hipfile`` expose the same API shape:

    .. code-block:: python

        with CuFile(path, mode) as f:
            f.write(gpu_addr, nbytes, file_offset=..., dev_offset=...)
            f.read(gpu_addr, nbytes, file_offset=..., dev_offset=...)
    """

    def __init__(self, lib_name: str, driver: object) -> None:
        self._lib_name = lib_name
        self._module = __import__(lib_name)
        self._driver = driver
        self._handle: Optional[object] = None

    # ── context manager ───────────────────────────────────────────

    def open(self, path: str, mode: str = "r+") -> "_CuFileHandle":
        self._handle = self._module.CuFile(path, mode)
        return self

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception:
                pass
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ── I/O ────────────────────────────────────────────────────────

    def write(
        self,
        gpu_addr: int,
        nbytes: int,
        file_offset: int = 0,
        dev_offset: int = 0,
    ) -> int:
        assert self._handle is not None, "CuFile not opened"
        return self._handle.write(
            ctypes.c_void_p(gpu_addr),
            nbytes,
            file_offset=file_offset,
            dev_offset=dev_offset,
        )

    def read(
        self,
        gpu_addr: int,
        nbytes: int,
        file_offset: int = 0,
        dev_offset: int = 0,
    ) -> int:
        assert self._handle is not None, "CuFile not opened"
        return self._handle.read(
            ctypes.c_void_p(gpu_addr),
            nbytes,
            file_offset=file_offset,
            dev_offset=dev_offset,
        )


class NvFileBackend(StorageBackend):
    """GPU Direct Storage backend.

    Writes raw tensor data preceded by a small JSON header (4 KB).
    The header is written via POSIX; the tensor payload is written via
    nvfile/cuFile's ``CuFile.write()`` (GPU→NVMe direct DMA).
    """

    def __init__(self) -> None:
        lib_name = _probe_library()
        if lib_name is None:
            raise RuntimeError(
                "No GDS library found. Install cufile, nvfile, or hipfile."
            )
        self._lib_name = lib_name

        # Create the driver singleton
        module = __import__(lib_name)
        self._driver = module.CuFileDriver()

        logger.info(
            "NvFileBackend ready: library=%s driver=%s",
            lib_name,
            type(self._driver).__name__,
        )

    # ── public API ─────────────────────────────────────────────────

    def save(self, path: Path, tensor: torch.Tensor) -> None:
        assert tensor.is_cuda, "NvFileBackend.save requires a CUDA tensor"
        self._save_with_gds(path, tensor)

    def load(self, path: Path, device: str = "cuda") -> torch.Tensor:
        return self._load_with_gds(path, device=device)

    @classmethod
    def is_available(cls) -> bool:
        return _probe_library() is not None

    @classmethod
    def is_supported(cls) -> bool:
        """Same as :meth:`is_available` (alias for factory use)."""
        return cls.is_available()

    # ── GDS I/O implementation ─────────────────────────────────────

    def _save_with_gds(self, path: Path, tensor: torch.Tensor) -> None:
        """Write *tensor* to *path* using GDS.

        1. Write JSON metadata header via POSIX (small, 4 KB).
        2. Write raw GPU tensor data via GDS (direct GPU→NVMe).
        3. Atomic rename from temp path to final path.
        """
        # Use a temp file so partial writes are never visible.
        tmp = path.with_suffix(path.suffix + ".tmp" + _rand_suffix(8))

        try:
            # Step 1: write metadata header (POSIX)
            header = _pack_header(tensor)
            with open(tmp, "wb") as f:
                f.write(header)

            # Step 2: write tensor data via GDS
            addr = tensor.data_ptr()
            nbytes = tensor.nbytes
            with _CuFileHandle(self._lib_name, self._driver) as f:
                f.open(str(tmp), "r+")
                written = f.write(addr, nbytes, file_offset=_HEADER_SIZE)
                if written != nbytes:
                    raise RuntimeError(
                        f"GDS write: expected {nbytes} bytes, got {written}"
                    )

            # Atomic commit
            os.replace(tmp, path)
        except Exception:
            # Clean up temp file on failure
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            raise

    def _load_with_gds(
        self,
        path: Path,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Load a tensor previously saved with :meth:`_save_with_gds`.

        1. Read JSON metadata header (POSIX, first 4 KB).
        2. Allocate GPU tensor.
        3. Read raw tensor data via GDS (direct NVMe→GPU).
        """
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
        addr = tensor.data_ptr()
        with _CuFileHandle(self._lib_name, self._driver) as f:
            f.open(str(path), "r")
            read_bytes = f.read(addr, nbytes, file_offset=_HEADER_SIZE)
            if read_bytes != nbytes:
                raise RuntimeError(
                    f"GDS read: expected {nbytes} bytes, got {read_bytes}"
                )

        logger.debug(
            "GDS read %s: shape=%s dtype=%s size=%.1f MB",
            path.name,
            list(shape),
            dtype,
            nbytes / 1e6,
        )
        return tensor

    def __repr__(self) -> str:
        return f"NvFileBackend(library={self._lib_name})"


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
    # fallback: eval torch.<name>
    import re

    m = re.match(r"torch\.(\w+)", s)
    if m:
        return getattr(torch, m.group(1))
    raise ValueError(f"Unknown dtype: {s}")
