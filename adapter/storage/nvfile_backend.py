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
from typing import List, Optional

import torch

from adapter.storage.backend import (
    StorageBackend,
    TensorSliceSpec,
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

# ── Alignment helpers ──────────────────────────────────────────────

_GDS_ALIGNMENT = 4096  # 4 KiB alignment for GPU↔NVMe direct I/O


def _align_up(size: int, alignment: int = _GDS_ALIGNMENT) -> int:
    return (size + alignment - 1) // alignment * alignment


def _aligned_device_buffer(
    min_bytes: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Allocate a `*device*` buffer whose ``data_ptr`` is 4 KiB-aligned.

    Returns
    -------
    (backing, aligned_slice, offset)
        *backing* — overall allocation (``min_bytes + alignment - 1``).
        *aligned_slice* — view of *backing* starting at the aligned ptr.
        *offset* — number of bytes skipped to achieve alignment.
    """
    backing = torch.empty(
        min_bytes + _GDS_ALIGNMENT - 1,
        dtype=torch.uint8,
        device=device,
    )
    base_ptr = backing.data_ptr()
    misalign = base_ptr % _GDS_ALIGNMENT
    offset = (_GDS_ALIGNMENT - misalign) % _GDS_ALIGNMENT
    aligned = backing[offset:offset + min_bytes]
    return backing, aligned, offset


# ── Backend ───────────────────────────────────────────────────────


class NvFileBackend(StorageBackend):
    """GPU Direct Storage backend.

    Writes raw tensor data preceded by a small JSON header (4 KB).
    The header is written via POSIX; the tensor payload is written via
    cuFile (GPU↔NVMe direct DMA).

    All GPU buffer pointers, file offsets and transfer sizes are 4 KiB
    aligned to avoid ``EINVAL`` on remote GDS mounts.
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

    # ── Positional I/O (chunk-object support) ──────────────────────

    def write_tensor_at(
        self,
        path: Path,
        tensor: torch.Tensor,
        offset: int,
        stored_nbytes: int,
    ) -> None:
        """Write *tensor* at *offset* via GDS with 4 KiB-aligned I/O."""
        tensor = tensor.contiguous()
        nbytes = tensor.nbytes
        if stored_nbytes < nbytes:
            raise ValueError(
                f"stored_nbytes ({stored_nbytes}) < tensor nbytes ({nbytes})"
            )
        if offset % _GDS_ALIGNMENT != 0:
            raise ValueError(
                f"offset ({offset}) must be {_GDS_ALIGNMENT}-byte aligned"
            )
        if stored_nbytes % _GDS_ALIGNMENT != 0:
            raise ValueError(
                f"stored_nbytes ({stored_nbytes}) must be "
                f"{_GDS_ALIGNMENT}-byte aligned"
            )

        device_str = f"cuda:{tensor.device.index}" if tensor.is_cuda else "cpu"
        stored_aligned = _align_up(stored_nbytes)
        backing, aligned, _ = _aligned_device_buffer(stored_aligned, device_str)

        # Copy tensor bytes into the aligned buffer (uint8 view)
        src_uint8 = tensor.contiguous().view(torch.uint8).reshape(-1)
        if tensor.is_cuda:
            aligned[:nbytes].copy_(src_uint8, non_blocking=False)
        else:
            aligned[:nbytes] = torch.frombuffer(
                bytearray(src_uint8.cpu().numpy().tobytes()), dtype=torch.uint8
            )
        if stored_aligned > nbytes:
            aligned[nbytes:stored_aligned].zero_()

        with _CuFileHandle(self._binding, str(path), "r+") as f:
            written = f.write(
                aligned.data_ptr(), stored_aligned,
                file_offset=offset,
            )
            if written != stored_aligned:
                raise RuntimeError(
                    f"GDS write_tensor_at: expected {stored_aligned} bytes "
                    f"at offset {offset}, got {written}"
                )

    def load_tensor_slices(
        self,
        path: Path,
        specs: List[TensorSliceSpec],
        device: str,
    ) -> List[torch.Tensor]:
        """Load tensor slices with one cuFile read per contiguous group.

        Chunk-object layer payloads are normally adjacent and 4 KiB aligned.
        Reading each layer separately adds a cuFile submission and device
        allocation per layer, so coalesce adjacent or overlapping slices while
        preserving the caller's requested order. Returned tensors are views of
        their group buffer and follow the read-only StorageBackend contract.
        """
        if not specs:
            return []

        indexed_specs = sorted(
            enumerate(specs), key=lambda item: item[1].offset
        )
        groups = []
        group_start = indexed_specs[0][1].offset
        group_end = group_start
        group_specs = []

        for index, spec in indexed_specs:
            if spec.nbytes > spec.stored_nbytes:
                raise ValueError(
                    f"slice nbytes ({spec.nbytes}) exceeds stored_nbytes "
                    f"({spec.stored_nbytes})"
                )
            if spec.offset % _GDS_ALIGNMENT != 0:
                raise ValueError(
                    f"offset ({spec.offset}) must be "
                    f"{_GDS_ALIGNMENT}-byte aligned"
                )
            if spec.stored_nbytes % _GDS_ALIGNMENT != 0:
                raise ValueError(
                    f"stored_nbytes ({spec.stored_nbytes}) must be "
                    f"{_GDS_ALIGNMENT}-byte aligned"
                )

            spec_end = spec.offset + spec.stored_nbytes
            if group_specs and spec.offset > group_end:
                groups.append((group_start, group_end, group_specs))
                group_start = spec.offset
                group_specs = []
            group_end = max(group_end, spec_end)
            group_specs.append((index, spec))
        groups.append((group_start, group_end, group_specs))

        results = [None] * len(specs)
        with _CuFileHandle(self._binding, str(path), "r") as f:
            for start, end, grouped_specs in groups:
                expected_nbytes = end - start
                backing, aligned, _ = _aligned_device_buffer(
                    expected_nbytes, device
                )

                read_bytes = f.read(
                    aligned.data_ptr(),
                    expected_nbytes,
                    file_offset=start,
                )
                if read_bytes != expected_nbytes:
                    raise RuntimeError(
                        "GDS load_tensor_slices: expected "
                        f"{expected_nbytes} bytes for slice group at offset "
                        f"{start}, got {read_bytes}"
                    )

                for index, spec in grouped_specs:
                    relative_offset = spec.offset - start
                    tensor = aligned.narrow(
                        0, relative_offset, spec.nbytes
                    )
                    tensor = tensor.view(dtype=spec.dtype).reshape(spec.shape)
                    results[index] = tensor

        tensors = []
        for tensor in results:
            if tensor is None:
                raise RuntimeError(
                    "GDS load_tensor_slices returned incomplete results"
                )
            tensors.append(tensor)
        return tensors

    # ── GDS I/O path ──────────────────────────────────────────────

    def _save_with_gds(self, path: Path, tensor: torch.Tensor) -> None:
        tensor = tensor.contiguous()
        nbytes = tensor.nbytes
        stored_nbytes = _align_up(nbytes)
        device_str = f"cuda:{tensor.device.index}" if tensor.is_cuda else "cpu"

        tmp = path.with_suffix(path.suffix + ".tmp" + _rand_suffix(8))
        try:
            # Step 1: metadata header via POSIX (4 KB, already aligned)
            header = _pack_header(tensor)
            with open(tmp, "wb") as f:
                f.write(header)

            # Step 2: allocate aligned buffer, copy tensor into it
            backing, aligned, _ = _aligned_device_buffer(
                stored_nbytes, device_str,
            )
            src_uint8 = tensor.view(torch.uint8).reshape(-1)
            if tensor.is_cuda:
                aligned[:nbytes].copy_(src_uint8, non_blocking=False)
            else:
                aligned[:nbytes] = torch.frombuffer(
                    bytearray(src_uint8.cpu().numpy().tobytes()), dtype=torch.uint8
                )
            if stored_nbytes > nbytes:
                aligned[nbytes:stored_nbytes].zero_()

            # Step 3: write aligned data via GDS
            with _CuFileHandle(self._binding, str(tmp), "r+") as f:
                written = f.write(
                    aligned.data_ptr(), stored_nbytes,
                    file_offset=_HEADER_SIZE,
                )
                if written != stored_nbytes:
                    raise RuntimeError(
                        f"GDS write: expected {stored_nbytes} bytes, "
                        f"got {written}"
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
        stored_nbytes = _align_up(nbytes)

        # Step 2: allocate aligned device buffer
        backing, aligned, _ = _aligned_device_buffer(stored_nbytes, device)

        # Step 3: read aligned data via GDS
        with _CuFileHandle(self._binding, str(path), "r") as f:
            read_bytes = f.read(
                aligned.data_ptr(), stored_nbytes,
                file_offset=_HEADER_SIZE,
            )
            if read_bytes != stored_nbytes:
                raise RuntimeError(
                    f"GDS read: expected {stored_nbytes} bytes, "
                    f"got {read_bytes}"
                )

        # Extract logical tensor from first nbytes
        raw = aligned[:nbytes].clone().reshape(-1)
        return raw.view(dtype=dtype).reshape(shape)

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
