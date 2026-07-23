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
        # Single open for all slices
        with open(path, "rb") as f:
            results: List[torch.Tensor] = []
            for spec in specs:
                f.seek(spec.offset)
                raw = f.read(spec.stored_nbytes)
                if len(raw) != spec.stored_nbytes:
                    raise RuntimeError(
                        f"POSIX load_tensor_slices: expected "
                        f"{spec.stored_nbytes} bytes for slice at offset "
                        f"{spec.offset}, got {len(raw)}"
                    )

                # Use torch.frombuffer with uint8 view for bfloat16 compat
                # (numpy does not understand bfloat16)
                raw_bytes = raw[:spec.nbytes]
                buf = bytearray(raw_bytes)
                tensor = torch.frombuffer(
                    buf, dtype=torch.uint8
                ).clone()
                # view changes element size (e.g. 256 uint8 → 128 bfloat16)
                tensor = tensor.view(dtype=spec.dtype).reshape(spec.shape)

                if device.startswith("cuda") and device != "cpu":
                    tensor = tensor.to(device, non_blocking=True)
                results.append(tensor)

        return results
