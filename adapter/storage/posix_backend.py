# SPDX-License-Identifier: Apache-2.0
"""POSIX storage backend (CPU bounce buffer + safetensors file I/O).

This is the universal fallback path — it works on any system and does
not require any special hardware or drivers.

Data path::

    GPU tensor → cudaMemcpy (DeviceToHost) → CPU buffer → safetensors file
    safetensors file → CPU buffer → cudaMemcpy (HostToDevice) → GPU tensor
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import safetensors.torch
import torch

from adapter.storage.backend import StorageBackend, TensorSliceSpec

logger = logging.getLogger(__name__)


class PosixBackend(StorageBackend):
    """POSIX + CPU bounce buffer backend.

    This is the **fallback** path: it copies GPU tensors through CPU
    memory before writing to / reading from disk.  Slower than GDS but
    universally compatible.
    """

    def save(self, path: Path, tensor: torch.Tensor) -> None:
        if tensor.is_cuda:
            # Bounce through CPU
            cpu_tensor = tensor.detach().cpu()
        else:
            cpu_tensor = tensor
        safetensors.torch.save_file({"kv_cache": cpu_tensor}, str(path))

    def load(self, path: Path, device: str = "cuda") -> torch.Tensor:
        data = safetensors.torch.load_file(str(path))
        tensor = data["kv_cache"]
        if device.startswith("cuda") and not tensor.is_cuda:
            tensor = tensor.to(device, non_blocking=True)
        return tensor

    def is_available(self) -> bool:
        return True  # always available

    # ── Positional I/O ─────────────────────────────────────────────

    def write_tensor_at(
        self,
        path: Path,
        tensor: torch.Tensor,
        offset: int,
        stored_nbytes: int,
    ) -> None:
        tensor = tensor.contiguous()
        nbytes = tensor.nbytes
        if stored_nbytes < nbytes:
            raise ValueError(
                f"stored_nbytes ({stored_nbytes}) < tensor nbytes ({nbytes})"
            )

        # Make a CPU byte view
        cpu_tensor = tensor.detach().cpu() if tensor.is_cuda else tensor.detach()
        # Use uint8 view for bfloat16 compat (numpy does not support bfloat16)
        raw = cpu_tensor.view(dtype=torch.uint8).numpy().tobytes()
        if stored_nbytes > nbytes:
            raw = raw.ljust(stored_nbytes, b"\x00")

        with open(path, "r+b") as f:
            f.seek(offset)
            written = f.write(raw)
        if written != stored_nbytes:
            raise RuntimeError(
                f"POSIX write_tensor_at: expected {stored_nbytes} bytes, "
                f"wrote {written}"
            )

    def load_tensor_slices(
        self,
        path: Path,
        specs: List[TensorSliceSpec],
        device: str,
    ) -> List[torch.Tensor]:
        """Load tensor slices to CPU or CUDA.

        CUDA loads use a pinned host buffer so their H2D copy can be queued
        asynchronously.  ``load_tensor_slices_to_pinned_cpu`` exposes the
        same read half of this path for the connector's bounded read-ahead
        pipeline: worker threads perform POSIX I/O only, while the vLLM
        thread remains responsible for CUDA work and KV injection.
        """
        return self._load_tensor_slices(path, specs, device)

    def load_tensor_slices_to_pinned_cpu(
        self,
        path: Path,
        specs: List[TensorSliceSpec],
    ) -> List[torch.Tensor]:
        """Load slices into pinned CPU memory when available.

        This is intentionally POSIX-specific.  It lets a CPU worker prefetch
        disk data without issuing CUDA operations on a foreign thread.  If
        pinned allocation is unavailable, safely return regular CPU tensors.
        """
        return self._load_tensor_slices(
            path, specs, "cpu", prefer_pinned_cpu=True
        )

    def move_pinned_tensor_slices_to_device(
        self,
        tensors: List[torch.Tensor],
        device: str,
    ) -> List[torch.Tensor]:
        """Move host slice views to CUDA while preserving coalesced groups.

        ``load_tensor_slices_to_pinned_cpu`` returns views of one pinned byte
        buffer for each contiguous I/O group.  Copying those views one at a
        time would turn a coalesced H2D transfer back into one transfer per
        layer.  Rebuild each backing byte view, transfer it once, then recreate
        the original tensor views on the destination.
        """
        if not tensors:
            return []
        if not device.startswith("cuda"):
            return list(tensors)

        result: list[torch.Tensor | None] = [None] * len(tensors)
        groups: dict[tuple[int, int], list[tuple[int, torch.Tensor, int]]] = {}
        storages = {}

        for index, tensor in enumerate(tensors):
            if tensor.is_cuda:
                result[index] = tensor
                continue
            storage = tensor.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            byte_offset = tensor.storage_offset() * tensor.element_size()
            groups.setdefault(key, []).append((index, tensor, byte_offset))
            storages[key] = storage

        for key, group in groups.items():
            storage = storages[key]
            host_bytes = torch.empty(0, dtype=torch.uint8).set_(
                storage, 0, (storage.nbytes(),)
            )
            device_bytes = host_bytes.to(device, non_blocking=True)
            for index, tensor, byte_offset in group:
                result[index] = device_bytes.narrow(
                    0, byte_offset, tensor.nbytes
                ).view(dtype=tensor.dtype).reshape(tensor.shape)

        if any(tensor is None for tensor in result):
            raise RuntimeError(
                "POSIX pinned slice transfer returned incomplete results"
            )
        return [tensor for tensor in result if tensor is not None]

    def _load_tensor_slices(
        self,
        path: Path,
        specs: List[TensorSliceSpec],
        device: str,
        *,
        prefer_pinned_cpu: bool = False,
    ) -> List[torch.Tensor]:
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
            spec_end = spec.offset + spec.stored_nbytes
            if group_specs and spec.offset > group_end:
                groups.append((group_start, group_end, group_specs))
                group_start = spec.offset
                group_specs = []
            group_end = max(group_end, spec_end)
            group_specs.append((index, spec))
        groups.append((group_start, group_end, group_specs))

        results = [None] * len(specs)
        with open(path, "rb") as f:
            for start, end, grouped_specs in groups:
                expected_nbytes = end - start
                f.seek(start)

                # A pageable bytearray makes ``Tensor.to(...,
                # non_blocking=True)`` synchronise the H2D copy.  When the
                # caller targets CUDA, read directly into pinned host memory
                # instead, so the copy can be queued on the current stream.
                # Keep the bytearray fallback for CPU loads and for builds
                # where pinned allocations are unavailable.
                buffer_tensor: torch.Tensor
                target_cuda = device.startswith("cuda") and device != "cpu"
                use_pinned_host_buffer = target_cuda or prefer_pinned_cpu
                if use_pinned_host_buffer:
                    try:
                        host_buffer = torch.empty(
                            expected_nbytes,
                            dtype=torch.uint8,
                            pin_memory=True,
                        )
                        if (
                            host_buffer.device.type != "cpu"
                            or not host_buffer.is_pinned()
                        ):
                            raise RuntimeError(
                                "Pinned POSIX read buffer is not CPU pinned"
                            )
                        host_array = host_buffer.numpy()
                    except (RuntimeError, TypeError) as exc:
                        logger.debug(
                            "Pinned POSIX read buffer unavailable; falling "
                            "back to pageable memory: %s",
                            exc,
                        )
                        raw = bytearray(expected_nbytes)
                        f.seek(start)
                        bytes_read = f.readinto(raw)
                        buffer_tensor = torch.frombuffer(
                            raw, dtype=torch.uint8
                        )
                        if target_cuda:
                            buffer_tensor = buffer_tensor.to(
                                device, non_blocking=True
                            )
                        else:
                            buffer_tensor = buffer_tensor.clone()
                    else:
                        bytes_read = f.readinto(host_array)
                        if target_cuda:
                            buffer_tensor = host_buffer.to(
                                device, non_blocking=True
                            )
                        else:
                            buffer_tensor = host_buffer
                else:
                    raw = bytearray(expected_nbytes)
                    bytes_read = f.readinto(raw)
                    buffer_tensor = torch.frombuffer(raw, dtype=torch.uint8)

                if bytes_read != expected_nbytes:
                    raise RuntimeError(
                        "POSIX load_tensor_slices: expected "
                        f"{expected_nbytes} bytes for slice group at offset "
                        f"{start}, got {bytes_read}"
                    )

                if not target_cuda and not use_pinned_host_buffer:
                    buffer_tensor = buffer_tensor.clone()

                for index, spec in grouped_specs:
                    relative_offset = spec.offset - start
                    tensor = buffer_tensor.narrow(
                        0, relative_offset, spec.nbytes
                    )
                    tensor = tensor.view(dtype=spec.dtype).reshape(spec.shape)
                    results[index] = tensor

        if any(tensor is None for tensor in results):
            raise RuntimeError(
                "POSIX load_tensor_slices returned incomplete results"
            )
        return results
