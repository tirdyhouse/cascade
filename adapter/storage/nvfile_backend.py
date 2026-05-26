# SPDX-License-Identifier: Apache-2.0
"""GDS storage backend via cuFile/nvfile/hipfile (Python module).

Provides GPU↔NVMe direct I/O using NVIDIA GPUDirect Storage or
AMD hipFile equivalent.

Requires one of:
- ``cufile`` — NVIDIA GDS Python bindings (``pip install cufile``)
- ``nvfile`` — vendor-provided GDS library
- ``hipfile`` — AMD ROCm equivalent
"""

from __future__ import annotations

import ctypes
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


# ── Probe / detect available GDS Python module ────────────────────


def _probe_module() -> Optional[str]:
    """Return the name of the available GDS Python module, or ``None``."""
    for lib in ("cufile", "nvfile", "hipfile"):
        try:
            __import__(lib)
            logger.debug("GDS library found: %s", lib)
            return lib
        except ImportError:
            continue
    return None


# ── CuFile handle wrapper ─────────────────────────────────────────


class _CuFileHandle:
    """Context manager wrapping a ``CuFile`` instance.

    Both ``cufile`` and ``hipfile`` expose:

    .. code-block:: python

        with CuFile(path, mode) as f:
            f.write(gpu_addr, nbytes, file_offset=..., dev_offset=...)
            f.read(gpu_addr, nbytes, file_offset=..., dev_offset=...)
    """

    def __init__(self, lib_name: str) -> None:
        self._module = __import__(lib_name)
        self._handle = None

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

    def write(
        self, gpu_addr: int, nbytes: int,
        file_offset: int = 0, dev_offset: int = 0,
    ) -> int:
        return self._handle.write(
            ctypes.c_void_p(gpu_addr), nbytes,
            file_offset=file_offset, dev_offset=dev_offset,
        )

    def read(
        self, gpu_addr: int, nbytes: int,
        file_offset: int = 0, dev_offset: int = 0,
    ) -> int:
        return self._handle.read(
            ctypes.c_void_p(gpu_addr), nbytes,
            file_offset=file_offset, dev_offset=dev_offset,
        )


# ── Backend ───────────────────────────────────────────────────────


class NvFileBackend(StorageBackend):
    """GPU Direct Storage backend.

    Writes raw tensor data preceded by a small JSON header (4 KB).
    The header is written via POSIX; the tensor payload is written via
    cuFile's ``CuFile.write()`` / ``CuFile.read()`` (GPU↔NVMe direct DMA).

    Requires the Python ``cufile`` package (NVIDIA GDS SDK) or
    a compatible vendor library (``nvfile`` / ``hipfile``).
    """

    def __init__(self) -> None:
        lib_name = _probe_module()
        if lib_name is None:
            raise RuntimeError(
                "No GDS Python module found.\n"
                "  Install NVIDIA GDS:       pip install cufile\n"
                "  Or vendor equivalent:     pip install nvfile\n"
                "  Or AMD ROCm hipFile:      pip install hipfile"
            )
        self._lib_name = lib_name
        module = __import__(lib_name)
        self._driver = module.CuFileDriver()
        logger.info("NvFileBackend ready: library=%s", lib_name)

    # ── public API ─────────────────────────────────────────────────

    def save(self, path: Path, tensor: torch.Tensor) -> None:
        path = Path(path)
        if not tensor.is_cuda:
            logger.warning(
                "NvFileBackend.save expects a CUDA tensor; "
                "got %s on %s", tensor.dtype, tensor.device,
            )
        self._save_with_gds(path, tensor)

    def load(self, path: Path, device: str = "cuda") -> torch.Tensor:
        return self._load_with_gds(Path(path), device=device)

    @classmethod
    def is_available(cls) -> bool:
        return _probe_module() is not None

    @classmethod
    def is_supported(cls) -> bool:
        """Alias for :meth:`is_available` (used by factory)."""
        return cls.is_available()

    # ── GDS I/O implementation ─────────────────────────────────────

    def _save_with_gds(self, path: Path, tensor: torch.Tensor) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp" + _rand_suffix(8))
        try:
            header = _pack_header(tensor)
            with open(tmp, "wb") as f:
                f.write(header)

            addr = tensor.data_ptr()
            nbytes = tensor.nbytes
            with _CuFileHandle(self._lib_name) as f:
                f.open(str(tmp), "r+")
                written = f.write(addr, nbytes, file_offset=_HEADER_SIZE)
                if written != nbytes:
                    raise RuntimeError(
                        f"GDS write: expected {nbytes} bytes, got {written}"
                    )
            os.replace(tmp, path)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            raise

    def _load_with_gds(self, path: Path, device: str = "cuda") -> torch.Tensor:
        with open(path, "rb") as f:
            header_blob = f.read(_HEADER_SIZE)
        meta = _unpack_header(header_blob)

        dtype = _str_to_dtype(meta["dtype"])
        shape = torch.Size(meta["shape"])
        nbytes = meta["nbytes"]

        tensor = torch.empty(shape, dtype=dtype, device=device)

        addr = tensor.data_ptr()
        with _CuFileHandle(self._lib_name) as f:
            f.open(str(path), "r")
            read_bytes = f.read(addr, nbytes, file_offset=_HEADER_SIZE)
            if read_bytes != nbytes:
                raise RuntimeError(
                    f"GDS read: expected {nbytes} bytes, got {read_bytes}"
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
    import re
    m = re.match(r"torch\.(\w+)", s)
    if m:
        return getattr(torch, m.group(1))
    raise ValueError(f"Unknown dtype: {s}")
