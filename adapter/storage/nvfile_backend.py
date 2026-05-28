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


class _CuFileBinding:
    """Abstracts over different cuFile library bindings.

    Each binding must provide:
    - ``driver_open()`` / ``driver_close()``
    - ``handle_register(fd)`` → opaque handle
    - ``handle_deregister(handle)``
    - ``read(handle, gpu_ptr, size, file_offset, dev_offset)`` → bytes read
    - ``write(handle, gpu_ptr, size, file_offset, dev_offset)`` → bytes written
    """


class _CudaBindingsCufile(_CuFileBinding):
    """Binding via ``cuda.bindings.cufile`` (low-level C wrappers).

    Installed with ``pip install cuda-python``.
    """

    name = "cuda.bindings.cufile"

    def __init__(self):
        from cuda.bindings import cufile as C

        # Store references so the module stays importable
        self._C = C

    def driver_open(self):
        return self._C.driver_open()

    def driver_close(self):
        return self._C.driver_close()
    def handle_register(self, fd: int):
        import numpy as np
        from cuda.bindings.cufile import Descr, FileHandleType, descr_dtype

        buf = np.zeros(1, dtype=descr_dtype)
        buf['type'] = FileHandleType.OPAQUE_FD
        buf['handle']['fd'] = fd
        desc = Descr.from_data(buf)
        handle = self._C.handle_register(desc.ptr)
        return handle
    def handle_deregister(self, handle) -> None:
        self._C.handle_deregister(handle)

    def read(self, handle, gpu_ptr, size, file_offset, dev_offset):
        return self._C.read(handle, gpu_ptr, size, file_offset, dev_offset)

    def write(self, handle, gpu_ptr, size, file_offset, dev_offset):
        return self._C.write(handle, gpu_ptr, size, file_offset, dev_offset)


class _PythonCufileModule(_CuFileBinding):
    """Binding via the high-level Python ``cufile`` package.

    Installed as part of the NVIDIA GDS SDK.
    """

    name = "cufile (Python)"

    def __init__(self):
        import cufile as C

        self._driver = C.CuFileDriver()

    def driver_open(self):
        return self._driver  # CuFileDriver is already initialized

    def driver_close(self):
        self._driver = None

    def handle_register(self, fd: int):
        # The high-level module creates CuFile from path, not fd
        # We'll handle this differently — see _CuFileHandle
        raise NotImplementedError("use CuFile path interface")

    def handle_deregister(self, handle) -> None:
        handle.close()

    def read(self, handle, gpu_ptr, size, file_offset, dev_offset):
        return handle.read(gpu_ptr, size, file_offset=file_offset, dev_offset=dev_offset)

    def write(self, handle, gpu_ptr, size, file_offset, dev_offset):
        return handle.write(gpu_ptr, size, file_offset=file_offset, dev_offset=dev_offset)


_VENDOR_NAMES = {"nvfile": "nvfile", "hipfile": "hipfile"}


class _VendorModule(_CuFileBinding):
    """Binding via a vendor-supplied module (nvfile / hipfile)."""

    def __init__(self, mod_name: str):
        self._mod = __import__(mod_name)
        self._driver = self._mod.CuFileDriver()
        self.name = mod_name

    def driver_open(self):
        return self._driver

    def driver_close(self):
        self._driver = None

    def handle_register(self, fd: int):
        raise NotImplementedError("use CuFile path interface")

    def handle_deregister(self, handle) -> None:
        handle.close()

    def read(self, handle, gpu_ptr, size, file_offset, dev_offset):
        return handle.read(gpu_ptr, size, file_offset=file_offset, dev_offset=dev_offset)

    def write(self, handle, gpu_ptr, size, file_offset, dev_offset):
        return handle.write(gpu_ptr, size, file_offset=file_offset, dev_offset=dev_offset)


# ── Probe ─────────────────────────────────────────────────────────


def _detect_binding() -> Optional[_CuFileBinding]:
    """Return the best available cuFile binding, or ``None``."""

    # 1. cuda.bindings.cufile (from pip install cuda-python)
    try:
        from cuda.bindings import cufile

        binding = _CudaBindingsCufile()
        binding.driver_open()
        logger.info("GDS binding: cuda.bindings.cufile")
        return binding
    except Exception as exc:
        logger.debug("cuda.bindings.cufile unavailable: %s", exc)

    # 2. High-level Python cufile module
    try:
        import cufile

        binding = _PythonCufileModule()
        logger.info("GDS binding: cufile (Python)")
        return binding
    except ImportError:
        pass

    # 3. Vendor modules
    for name in ("nvfile", "hipfile"):
        try:
            binding = _VendorModule(name)
            logger.info("GDS binding: %s", name)
            return binding
        except ImportError:
            continue

    return None


# ── CuFile handle ────────────────────────────────────────────────


class _CuFileHandle:
    """Opens a file for GDS read/write.

    For the high-level API (cufile Python package / vendor)::

        with CuFile(path, mode) as f:
            f.write(gpu_ptr, ...)

    For the low-level API (cuda.bindings.cufile)::

        cuFileHandleRegister(fd) → handle
        cuFileRead(handle, gpu_ptr, ...)
    """

    def __init__(self, binding: _CuFileBinding, path: str, mode: str):
        self._binding = binding
        self._path = path
        self._mode = mode
        self._handle = None
        self._fd = None

    def __enter__(self):
        # High-level path: CuFile(path, mode)
        if hasattr(self._binding, "_mod"):
            mod = self._binding._mod
            self._handle = mod.CuFile(self._path, self._mode)
            return self

        # Low-level path: open fd + register handle
        flags = os.O_RDWR if "r+" in self._mode or "w" in self._mode else os.O_RDONLY
        self._fd = os.open(self._path, flags)
        self._handle = self._binding.handle_register(self._fd)
        return self

    def __exit__(self, *args):
        if self._handle is not None:
            try:
                if hasattr(self._handle, "close"):
                    self._handle.close()
                else:
                    self._binding.handle_deregister(self._handle)
            except Exception:
                pass
            self._handle = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def write(self, gpu_addr: int, nbytes: int,
              file_offset: int = 0, dev_offset: int = 0) -> int:
        import ctypes
        ptr = ctypes.c_void_p(gpu_addr)

        # High-level API takes (ptr, nbytes, file_offset=, dev_offset=)
        if hasattr(self._handle, "write"):
            return self._handle.write(ptr, nbytes, file_offset=file_offset,
                                      dev_offset=dev_offset)

        # Low-level API takes (handle, gpu_addr, nbytes, file_offset, dev_offset)
        return self._binding.write(self._handle, gpu_addr, nbytes,
                                   file_offset, dev_offset)

    def read(self, gpu_addr: int, nbytes: int,
             file_offset: int = 0, dev_offset: int = 0) -> int:
        import ctypes
        ptr = ctypes.c_void_p(gpu_addr)

        if hasattr(self._handle, "read"):
            return self._handle.read(ptr, nbytes, file_offset=file_offset,
                                     dev_offset=dev_offset)

        return self._binding.read(self._handle, gpu_addr, nbytes,
                                  file_offset, dev_offset)


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
