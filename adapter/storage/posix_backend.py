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

import safetensors.torch
import torch

from adapter.storage.backend import StorageBackend

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
