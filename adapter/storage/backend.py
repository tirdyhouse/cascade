# SPDX-License-Identifier: Apache-2.0
"""Storage backend abstraction + factory.

The factory (:func:`create_storage_backend`) auto-selects the best
available backend — GDS when nvfile/cufile is present, POSIX otherwise.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# ── Metadata header for raw-tensor format ──────────────────────────
# When GDS is active we bypass safetensors and write raw GPU tensor
# data prefixed with a small JSON header (padded to 4 KB).
_HEADER_SIZE = 4096  # bytes, same as LMCache
_HEADER_VERSION = 1


def _pack_header(tensor: torch.Tensor) -> bytes:
    """Pack tensor shape / dtype into a fixed-size header."""
    import json
    import struct

    meta = {
        "version": _HEADER_VERSION,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "nbytes": tensor.nbytes,
    }
    blob = json.dumps(meta, separators=(",", ":")).encode()
    assert len(blob) < _HEADER_SIZE, f"metadata too large: {len(blob)}"
    # Pad to _HEADER_SIZE
    blob = blob.ljust(_HEADER_SIZE, b"\x00")
    return blob


def _unpack_header(blob: bytes) -> dict:
    """Parse the fixed-size header."""
    import json

    # Strip trailing nulls
    payload = blob.rstrip(b"\x00")
    return json.loads(payload)


# ── Abstract base ──────────────────────────────────────────────────


class StorageBackend(ABC):
    """Pluggable storage backend for GPU tensor persistence.

    Two concrete implementations exist:

    * :class:`NvFileBackend` — GPU↔NVMe direct via nvfile/cufile (GDS).
    * :class:`PosixBackend` — CPU bounce buffer + POSIX file I/O.
    """

    @abstractmethod
    def save(self, path: Path, tensor: torch.Tensor) -> None:
        """Persist *tensor* (which **must** reside on GPU) to *path*.

        The caller is responsible for ensuring the parent directory exists.
        """

    @abstractmethod
    def load(self, path: Path, device: str = "cuda") -> torch.Tensor:
        """Load a GPU tensor previously written by :meth:`save`.

        Returns a tensor on *device*.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` when this backend can be used right now."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ── Factory ────────────────────────────────────────────────────────


def create_storage_backend(
    prefer: Optional[str] = None,
) -> StorageBackend:
    """Auto-select the best storage backend.

    Resolution order:
    1. If *prefer* is ``"gds"`` or ``"nvfile"`` → try :class:`NvFileBackend`.
    2. If *prefer* is ``"posix"`` → use :class:`PosixBackend`.
    3. If *prefer* is ``None`` (default) → try GDS first, fall back to POSIX.

    *prefer* is case-insensitive.
    """
    if prefer is not None:
        prefer = prefer.lower()

    # Forced backend
    if prefer == "posix":
        logger.info("Storage backend: PosixBackend (explicit)")
        return _POSIX

    # Try GDS first
    if prefer in (None, "gds", "nvfile", "cufile", "auto"):
        gds = _try_gds()
        if gds is not None:
            return gds
        if prefer is not None and prefer != "auto":
            logger.warning(
                "Requested GDS backend (%s) but it is unavailable; "
                "falling back to PosixBackend",
                prefer,
            )

    logger.info("Storage backend: PosixBackend")
    return _POSIX


def _try_gds() -> Optional[StorageBackend]:
    """Return an :class:`NvFileBackend` instance if available."""
    try:
        from adapter.storage.nvfile_backend import NvFileBackend

        if NvFileBackend.is_supported():
            logger.info("Storage backend: NvFileBackend (GDS)")
            return NvFileBackend()
    except Exception as exc:
        logger.debug("NvFileBackend init failed: %s", exc)
    return None


# Lazily-imported singleton for the POSIX path
from adapter.storage.posix_backend import PosixBackend as _PosixBackendImpl

_POSIX = _PosixBackendImpl()
