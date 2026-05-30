# SPDX-License-Identifier: Apache-2.0
"""cuFile / GDS binding discovery and handle management."""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class CuFileBinding:
    """Abstracts over different cuFile library bindings.

    Each binding must provide:
    - ``driver_open()`` / ``driver_close()``
    - ``handle_register(fd)`` → opaque handle
    - ``handle_deregister(handle)``
    - ``read(handle, gpu_ptr, size, file_offset, dev_offset)`` → bytes read
    - ``write(handle, gpu_ptr, size, file_offset, dev_offset)`` → bytes written
    """


class CudaBindingsCufile(CuFileBinding):
    """Binding via ``cuda.bindings.cufile`` (low-level C wrappers)."""

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


class PythonCufileModule(CuFileBinding):
    """Binding via the high-level Python ``cufile`` package."""

    name = "cufile (Python)"

    def __init__(self):
        import cufile as C

        self._driver = C.CuFileDriver()

    def driver_open(self):
        return self._driver  # CuFileDriver is already initialized

    def driver_close(self):
        self._driver = None

    def handle_register(self, fd: int):
        # The high-level module creates CuFile from path, not fd.
        # CuFileHandle handles this path-oriented API directly.
        raise NotImplementedError("use CuFile path interface")

    def handle_deregister(self, handle) -> None:
        handle.close()

    def read(self, handle, gpu_ptr, size, file_offset, dev_offset):
        return handle.read(gpu_ptr, size, file_offset=file_offset, dev_offset=dev_offset)

    def write(self, handle, gpu_ptr, size, file_offset, dev_offset):
        return handle.write(gpu_ptr, size, file_offset=file_offset, dev_offset=dev_offset)


class VendorModule(CuFileBinding):
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


def detect_binding() -> Optional[CuFileBinding]:
    """Return the best available cuFile binding, or ``None``."""

    # 1. cuda.bindings.cufile (from pip install cuda-python)
    try:
        from cuda.bindings import cufile  # noqa: F401

        binding = CudaBindingsCufile()
        binding.driver_open()
        logger.info("GDS binding: cuda.bindings.cufile")
        return binding
    except Exception as exc:
        logger.debug("cuda.bindings.cufile unavailable: %s", exc)

    # 2. High-level Python cufile module
    try:
        import cufile  # noqa: F401

        binding = PythonCufileModule()
        logger.info("GDS binding: cufile (Python)")
        return binding
    except ImportError:
        pass

    # 3. Vendor modules
    for name in ("nvfile", "hipfile"):
        try:
            binding = VendorModule(name)
            logger.info("GDS binding: %s", name)
            return binding
        except ImportError:
            continue

    return None


class CuFileHandle:
    """Opens a file for GDS read/write.

    For the high-level API (cufile Python package / vendor)::

        with CuFile(path, mode) as f:
            f.write(gpu_ptr, ...)

    For the low-level API (cuda.bindings.cufile)::

        cuFileHandleRegister(fd) → handle
        cuFileRead(handle, gpu_ptr, ...)
    """

    def __init__(self, binding: CuFileBinding, path: str, mode: str):
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
