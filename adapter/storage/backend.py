# SPDX-License-Identifier: Apache-2.0
"""Storage backend abstraction + factory.

The factory (:func:`create_storage_backend`) auto-selects the best
available backend — GDS when nvfile/cufile is present, POSIX otherwise.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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


# ── Tensor-slice descriptor ─────────────────────────────────────────


@dataclass(frozen=True)
class TensorSliceSpec:
    """Describes a contiguous byte slice within a file.

    Attributes:
        offset:        Byte offset from the start of the file.
        nbytes:        Logical (uncompressed) tensor size in bytes.
        stored_nbytes: Number of bytes physically stored on disk
                       (may be larger than *nbytes* due to alignment).
        shape:         Logical tensor shape.
        dtype:         Logical tensor data type.
    """
    offset: int
    nbytes: int
    stored_nbytes: int
    shape: tuple[int, ...]
    dtype: torch.dtype


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

    # ── Positional I/O (chunk-object support) ──────────────────────

    @abstractmethod
    def write_tensor_at(
        self,
        path: Path,
        tensor: torch.Tensor,
        offset: int,
        stored_nbytes: int,
    ) -> None:
        """Write *tensor* bytes at the given *offset* in *path*.

        ``stored_nbytes`` specifies the on-disk padded size (≥ tensor
        logical size).  The tensor is made contiguous before writing.
        The file **must** already exist and be large enough.

        Raises
        ------
        RuntimeError
            If the actual number of bytes written does not match
            *stored_nbytes*.
        """

    @abstractmethod
    def load_tensor_slices(
        self,
        path: Path,
        specs: List[TensorSliceSpec],
        device: str,
    ) -> List[torch.Tensor]:
        """Load multiple tensor slices from *path* in a single pass.

        Parameters
        ----------
        path:
            Path to the file.
        specs:
            Ordered list of slice descriptors.
        device:
            Target device (``"cpu"`` or ``"cuda"``).

        Returns
        -------
        list[torch.Tensor]
            Tensors in the same order as *specs*. Tensors loaded in one I/O
            group may share underlying storage and must be treated as
            read-only by callers.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ── Factory ────────────────────────────────────────────────────────


def create_storage_backend(
    prefer: Optional[str] = None,
    *,
    strict: bool = False,
) -> StorageBackend:
    """Select the requested storage backend.

    ``auto`` and the default preference try GDS first and fall back to POSIX.
    Explicit GDS requests retain that compatibility unless ``strict`` is true;
    strict mode is intended for validation and benchmarks where silently
    measuring POSIX as GDS would invalidate the result.
    """
    if prefer is not None:
        prefer = prefer.lower()

    known_backends = {"posix", "gds", "nvfile", "cufile", "auto"}
    if prefer is not None and prefer not in known_backends:
        message = f"Unknown storage backend: {prefer}"
        if strict:
            raise ValueError(message)
        logger.warning("%s; falling back to PosixBackend", message)
        return _POSIX

    if prefer == "posix":
        logger.info("Storage backend: PosixBackend (explicit)")
        return _POSIX

    if prefer in (None, "gds", "nvfile", "cufile", "auto"):
        gds = _try_gds()
        if gds is not None:
            return gds
        if prefer is not None and prefer != "auto":
            message = (
                f"Requested GDS backend ({prefer}) but it is unavailable"
            )
            if strict:
                raise RuntimeError(message)
            logger.warning("%s; falling back to PosixBackend", message)

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
