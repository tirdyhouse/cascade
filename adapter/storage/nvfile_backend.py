# SPDX-License-Identifier: Apache-2.0
"""GDS storage backend via cuFile / nvfile / hipfile.

Provides GPU↔NVMe direct I/O.  Supports three backends:

1. ``cuda.bindings.cufile`` — NVIDIA CUDA Python bindings (``pip install cuda-python``).
2. ``cufile`` — NVIDIA GDS Python SDK (separate package).
3. ``nvfile`` / ``hipfile`` — vendor / AMD equivalents.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any, List, Optional

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


@dataclass
class _GdsReadGroup:
    """One coalesced, aligned cuFile read within a chunk object."""

    start: int
    end: int
    specs: list[tuple[int, TensorSliceSpec]]


@dataclass
class _PreparedGdsReadGroup:
    """A read group with an already-reserved device destination buffer."""

    group: _GdsReadGroup
    backing: torch.Tensor | None
    buffer: torch.Tensor
    pool_slot: int | None = None


@dataclass
class PreparedGdsTensorSlices:
    """Main-thread allocation state for a later cuFile read.

    CUDA allocation and buffer registration happen on vLLM's execution thread.
    A GDS I/O worker only submits cuFile reads into these fixed addresses.
    """

    device: str
    groups: list[_PreparedGdsReadGroup]
    pool: "_GdsStagingPool | None" = None
    released: bool = False


class _GdsStagingPool:
    """A reusable, cuFile-registered GPU buffer split into fixed I/O slots."""

    def __init__(
        self,
        backend: "NvFileBackend",
        *,
        device: str,
        buffer_bytes: int,
        slot_count: int,
    ) -> None:
        if buffer_bytes <= 0 or buffer_bytes % _GDS_ALIGNMENT:
            raise ValueError("buffer_bytes must be a positive 4 KiB multiple")
        if slot_count <= 0:
            raise ValueError("slot_count must be positive")

        self._backend = backend
        self._device = device
        self.buffer_bytes = buffer_bytes
        self.slot_count = slot_count
        self.slot_bytes = (buffer_bytes // slot_count) // _GDS_ALIGNMENT
        self.slot_bytes *= _GDS_ALIGNMENT
        if self.slot_bytes <= 0:
            raise ValueError("GDS staging pool slot is empty")

        self._backing, self._aligned, _ = _aligned_device_buffer(
            buffer_bytes, device
        )
        self._available: deque[int] = deque(range(slot_count))
        self._pending_releases: deque[tuple[int, Any]] = deque()
        self._registered = False
        self._closed = False
        self._try_register()

    def _try_register(self) -> None:
        try:
            self._backend._register_gds_buffer(
                self._aligned.data_ptr(), self.buffer_bytes
            )
        except Exception as exc:
            # Reusing the allocation still avoids per-object cudaMalloc calls.
            # cuFile may fall back to its own registration or bounce buffers.
            logger.warning(
                "cuFileBufRegister failed for %d MiB staging pool: %s; "
                "continuing without explicit registration",
                self.buffer_bytes // (1024 * 1024),
                exc,
            )
        else:
            self._registered = True
            logger.info(
                "Registered %d MiB cuFile staging pool with %d slots",
                self.buffer_bytes // (1024 * 1024), self.slot_count,
            )

    def _reap_completed(self) -> None:
        if not self._pending_releases:
            return
        pending: deque[tuple[int, Any]] = deque()
        while self._pending_releases:
            slot, event = self._pending_releases.popleft()
            if event.query():
                self._available.append(slot)
            else:
                pending.append((slot, event))
        self._pending_releases = pending

    def acquire(self, nbytes: int) -> tuple[int, torch.Tensor] | None:
        """Reserve a slot, waiting only when prior CUDA use still owns it."""
        if self._closed or nbytes > self.slot_bytes:
            return None
        self._reap_completed()
        while not self._available:
            if not self._pending_releases:
                return None
            _, event = self._pending_releases[0]
            event.synchronize()
            self._reap_completed()
        slot = self._available.popleft()
        offset = slot * self.slot_bytes
        return slot, self._aligned.narrow(0, offset, nbytes)

    def release(self, slot: int, *, after_cuda_use: bool) -> None:
        if self._closed:
            return
        if not after_cuda_use:
            self._available.append(slot)
            return
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device=self._device))
        self._pending_releases.append((slot, event))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            while self._pending_releases:
                _, event = self._pending_releases.popleft()
                event.synchronize()
        except Exception:
            pass
        if self._registered:
            try:
                self._backend._deregister_gds_buffer(self._aligned.data_ptr())
            except Exception:
                pass
        self._available.clear()


def _coalesced_read_groups(specs: List[TensorSliceSpec]) -> list[_GdsReadGroup]:
    """Validate slice descriptors and group overlapping/adjacent ranges."""
    if not specs:
        return []

    indexed_specs = sorted(
        enumerate(specs), key=lambda item: item[1].offset
    )
    groups: list[_GdsReadGroup] = []
    group_start = indexed_specs[0][1].offset
    group_end = group_start
    group_specs: list[tuple[int, TensorSliceSpec]] = []

    for index, spec in indexed_specs:
        if spec.nbytes > spec.stored_nbytes:
            raise ValueError(
                f"slice nbytes ({spec.nbytes}) exceeds stored_nbytes "
                f"({spec.stored_nbytes})"
            )
        if spec.offset % _GDS_ALIGNMENT != 0:
            raise ValueError(
                f"offset ({spec.offset}) must be {_GDS_ALIGNMENT}-byte aligned"
            )
        if spec.stored_nbytes % _GDS_ALIGNMENT != 0:
            raise ValueError(
                f"stored_nbytes ({spec.stored_nbytes}) must be "
                f"{_GDS_ALIGNMENT}-byte aligned"
            )

        spec_end = spec.offset + spec.stored_nbytes
        if group_specs and spec.offset > group_end:
            groups.append(_GdsReadGroup(group_start, group_end, group_specs))
            group_start = spec.offset
            group_specs = []
        group_end = max(group_end, spec_end)
        group_specs.append((index, spec))
    groups.append(_GdsReadGroup(group_start, group_end, group_specs))
    return groups


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
        self._gds_staging_pool: _GdsStagingPool | None = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        """Release an optional registered staging pool."""
        pool = getattr(self, "_gds_staging_pool", None)
        if pool is not None:
            pool.close()
            self._gds_staging_pool = None

    def _register_gds_buffer(self, ptr: int, nbytes: int) -> None:
        """Register a persistent CUDA allocation when the binding supports it."""
        if isinstance(self._binding, _CudaBindingsCufile):
            self._binding._C.buf_register(ptr, nbytes, 0)
            return
        try:
            import ctypes
            from cufile.bindings import cuFileBufRegister

            cuFileBufRegister(ctypes.c_void_p(ptr), nbytes, flags=0)
        except ImportError as exc:
            raise RuntimeError(
                "the active cuFile binding does not expose buffer registration"
            ) from exc

    def _deregister_gds_buffer(self, ptr: int) -> None:
        if isinstance(self._binding, _CudaBindingsCufile):
            self._binding._C.buf_deregister(ptr)
            return
        import ctypes
        from cufile.bindings import cuFileBufDeregister

        cuFileBufDeregister(ctypes.c_void_p(ptr))

    def get_gds_staging_pool(
        self,
        *,
        device: str,
        max_read_nbytes: int,
        worker_count: int,
        buffer_size_bytes: int,
    ) -> _GdsStagingPool | None:
        """Return a reusable cuFile buffer pool sized for bounded read-ahead.

        LMCache uses one registered GPU allocation and several GDS I/O workers.
        Mirror that structure here while degrading safely to individual buffers
        when registration or capacity is unavailable.
        """
        if (
            not device.startswith("cuda")
            or max_read_nbytes <= 0
            or worker_count <= 0
            or buffer_size_bytes <= 0
        ):
            return None
        buffer_size_bytes = _align_up(buffer_size_bytes)
        max_slots = buffer_size_bytes // max_read_nbytes
        desired_slots = worker_count * 2 + 1
        slot_count = min(max_slots, desired_slots)
        while slot_count >= worker_count:
            slot_bytes = (buffer_size_bytes // slot_count) // _GDS_ALIGNMENT
            slot_bytes *= _GDS_ALIGNMENT
            if slot_bytes >= max_read_nbytes:
                break
            slot_count -= 1
        if slot_count < worker_count:
            logger.warning(
                "GDS staging pool (%d MiB) cannot hold %d concurrent %d MiB "
                "reads; using per-read buffers",
                buffer_size_bytes // (1024 * 1024),
                worker_count,
                (max_read_nbytes + 1024 * 1024 - 1) // (1024 * 1024),
            )
            return None

        pool = self._gds_staging_pool
        if (
            pool is not None
            and not pool._closed
            and pool._device == device
            and pool.buffer_bytes == buffer_size_bytes
            and pool.slot_count == slot_count
            and pool.slot_bytes >= max_read_nbytes
        ):
            return pool

        if pool is not None:
            pool.close()
        try:
            pool = _GdsStagingPool(
                self,
                device=device,
                buffer_bytes=buffer_size_bytes,
                slot_count=slot_count,
            )
        except Exception as exc:
            logger.warning(
                "Unable to create GDS staging pool (%d MiB): %s; "
                "using per-read buffers",
                buffer_size_bytes // (1024 * 1024), exc,
            )
            return None
        self._gds_staging_pool = pool
        return pool

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
        prepared = self.prepare_tensor_slices_load(specs, device)
        try:
            self.read_prepared_tensor_slices(path, prepared)
            return self.materialize_prepared_tensor_slices(prepared)
        except Exception:
            self.release_prepared_tensor_slices(
                prepared, after_cuda_use=False
            )
            raise

    def max_coalesced_read_nbytes(
        self, specs: List[TensorSliceSpec]
    ) -> int:
        """Return the largest contiguous range this read would submit."""
        return max(
            (group.end - group.start for group in _coalesced_read_groups(specs)),
            default=0,
        )

    def prepare_tensor_slices_load(
        self,
        specs: List[TensorSliceSpec],
        device: str,
        *,
        pool: _GdsStagingPool | None = None,
    ) -> PreparedGdsTensorSlices:
        """Allocate output buffers on the caller thread before GDS I/O."""
        prepared_groups: list[_PreparedGdsReadGroup] = []
        try:
            for group in _coalesced_read_groups(specs):
                read_nbytes = group.end - group.start
                slot: int | None = None
                buffer: torch.Tensor | None = None
                backing: torch.Tensor | None = None
                if pool is not None:
                    reservation = pool.acquire(read_nbytes)
                    if reservation is not None:
                        slot, buffer = reservation
                if buffer is None:
                    backing, buffer, _ = _aligned_device_buffer(
                        read_nbytes, device
                    )
                prepared_groups.append(_PreparedGdsReadGroup(
                    group=group,
                    backing=backing,
                    buffer=buffer,
                    pool_slot=slot,
                ))
        except Exception:
            for prepared_group in prepared_groups:
                if pool is not None and prepared_group.pool_slot is not None:
                    pool.release(prepared_group.pool_slot, after_cuda_use=False)
            raise
        return PreparedGdsTensorSlices(
            device=device,
            groups=prepared_groups,
            pool=pool,
        )

    def read_prepared_tensor_slices(
        self,
        path: Path,
        prepared: PreparedGdsTensorSlices,
    ) -> None:
        """Issue cuFile reads into buffers allocated by ``prepare_*``.

        This method intentionally avoids tensor allocation and reshaping so it
        can run in a GDS worker thread, matching LMCache's I/O structure.
        """
        if prepared.device.startswith("cuda") and torch.cuda.is_available():
            cuda_device = torch.device(prepared.device)
            if cuda_device.index is not None:
                torch.cuda.set_device(cuda_device)
        with _CuFileHandle(self._binding, str(path), "r") as f:
            for prepared_group in prepared.groups:
                group = prepared_group.group
                expected_nbytes = group.end - group.start
                read_bytes = f.read(
                    prepared_group.buffer.data_ptr(),
                    expected_nbytes,
                    file_offset=group.start,
                )
                if read_bytes != expected_nbytes:
                    raise RuntimeError(
                        "GDS load_tensor_slices: expected "
                        f"{expected_nbytes} bytes for slice group at offset "
                        f"{group.start}, got {read_bytes}"
                    )

    def materialize_prepared_tensor_slices(
        self,
        prepared: PreparedGdsTensorSlices,
    ) -> List[torch.Tensor]:
        """Return read-only tensor views in the caller's requested order."""
        result_count = sum(
            len(prepared_group.group.specs)
            for prepared_group in prepared.groups
        )
        results: list[torch.Tensor | None] = [None] * result_count
        for prepared_group in prepared.groups:
            group = prepared_group.group
            for index, spec in group.specs:
                relative_offset = spec.offset - group.start
                tensor = prepared_group.buffer.narrow(
                    0, relative_offset, spec.nbytes
                )
                results[index] = tensor.view(dtype=spec.dtype).reshape(spec.shape)

        tensors: list[torch.Tensor] = []
        for tensor in results:
            if tensor is None:
                raise RuntimeError(
                    "GDS load_tensor_slices returned incomplete results"
                )
            tensors.append(tensor)
        return tensors

    def release_prepared_tensor_slices(
        self,
        prepared: PreparedGdsTensorSlices,
        *,
        after_cuda_use: bool,
    ) -> None:
        """Return pooled buffers only after queued KV injection has consumed them."""
        if prepared.released:
            return
        prepared.released = True
        pool = prepared.pool
        if pool is None:
            return
        for prepared_group in prepared.groups:
            if prepared_group.pool_slot is not None:
                pool.release(
                    prepared_group.pool_slot,
                    after_cuda_use=after_cuda_use,
                )

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
