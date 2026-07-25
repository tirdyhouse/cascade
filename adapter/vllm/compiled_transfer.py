"""Optional compiled multi-layer paged-KV transfer.

The current chunk-object layout stores each layer as a contiguous
``[tokens, 2, ...]`` view.  This helper builds tiny GPU pointer tables and
dispatches one CUDA kernel across every layer in an object, instead of
performing one Python advanced-index scatter per layer.

The extension is deliberately optional.  Callers can keep the established
PyTorch path when a CUDA toolchain is unavailable or the active KV layout is
not the Triton ``[blocks, 2, block_size, ...]`` layout.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Any, Optional, Sequence

import torch


logger = logging.getLogger(__name__)

_EXTENSION_LOCK = Lock()
_EXTENSION: Any = None
_EXTENSION_ERROR: Optional[BaseException] = None


def load_compiled_transfer_extension() -> Any:
    """Build or reuse the Cascade CUDA transfer extension."""
    global _EXTENSION, _EXTENSION_ERROR

    if _EXTENSION is not None:
        return _EXTENSION
    if _EXTENSION_ERROR is not None:
        raise RuntimeError("Cascade CUDA transfer extension is unavailable") from (
            _EXTENSION_ERROR
        )

    with _EXTENSION_LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        if _EXTENSION_ERROR is not None:
            raise RuntimeError(
                "Cascade CUDA transfer extension is unavailable"
            ) from _EXTENSION_ERROR

        source = (
            Path(__file__).resolve().parents[2]
            / "csrc"
            / "kv_transfer"
            / "multi_layer_kv_transfer.cu"
        )
        if not source.is_file():
            raise RuntimeError(f"CUDA transfer source is missing: {source}")

        try:
            from torch.utils.cpp_extension import load

            _EXTENSION = load(
                name="cascade_multi_layer_kv_transfer_v1",
                sources=[str(source)],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3"],
                with_cuda=True,
                verbose=os.environ.get("CASCADE_CUDA_BUILD_VERBOSE", "0") == "1",
            )
        except BaseException as exc:
            _EXTENSION_ERROR = exc
            raise
        return _EXTENSION


class CompiledMultiLayerKVTransfer:
    """Reusable pointer tables for the compiled layer-major scatter."""

    def __init__(self, extension: Any):
        self._extension = extension
        self._device: Optional[torch.device] = None
        self._layer_count = 0
        self._source_ptrs_gpu: Optional[torch.Tensor] = None
        self._destination_ptrs_gpu: Optional[torch.Tensor] = None
        self._destination_signature: Optional[tuple[int, ...]] = None
        # Pointer-table H2D copies are deliberately asynchronous.  A host
        # table must not be mutated or returned to the pinned allocator until
        # its copy has completed; otherwise a copy queued behind a large KV
        # H2D can observe the following chunk's pointers.  Keep the tiny host
        # tables alive behind completion events and retire them lazily.
        self._pending_host_pointer_tables: deque[
            tuple[torch.cuda.Event, tuple[torch.Tensor, ...]]
        ] = deque()
        self._debug_logged = False

    def _ensure_pointer_tables(
        self,
        layer_count: int,
        device: torch.device,
    ) -> None:
        if (
            self._device == device
            and self._layer_count == layer_count
            and self._source_ptrs_gpu is not None
        ):
            return

        self._device = device
        self._layer_count = layer_count
        self._source_ptrs_gpu = torch.empty(
            layer_count, dtype=torch.int64, device=device
        )
        self._destination_ptrs_gpu = torch.empty(
            layer_count, dtype=torch.int64, device=device
        )
        self._destination_signature = None

    def _retain_host_pointer_tables(
        self,
        tables: tuple[torch.Tensor, ...],
        device: torch.device,
    ) -> None:
        """Protect pinned pointer tables until their queued H2D completes."""
        while (
            self._pending_host_pointer_tables
            and self._pending_host_pointer_tables[0][0].query()
        ):
            self._pending_host_pointer_tables.popleft()

        completed_copy = torch.cuda.Event()
        completed_copy.record(torch.cuda.current_stream(device))
        self._pending_host_pointer_tables.append((completed_copy, tables))

    def inject(
        self,
        sources: Sequence[torch.Tensor],
        destinations: Sequence[torch.Tensor],
        slot_mapping: torch.Tensor,
        block_size: int,
    ) -> None:
        """Inject one object's tensors into every destination KV layer.

        Raises ``ValueError`` before launching a kernel when a tensor does not
        satisfy the compiled path's exact layout contract.  The caller may
        safely retain the established PyTorch fallback for those cases.
        """
        if not sources or len(sources) != len(destinations):
            raise ValueError("source and destination layer counts must match")
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        first_source = sources[0]
        first_destination = destinations[0]
        if not first_source.is_cuda or not first_destination.is_cuda:
            raise ValueError("compiled transfer requires CUDA tensors")
        if first_source.device != first_destination.device:
            raise ValueError("source and destination devices must match")
        if first_source.ndim < 3 or first_source.shape[1] != 2:
            raise ValueError("source layout must be [tokens, 2, ...]")
        if (
            first_destination.ndim < 4
            or first_destination.shape[1] != 2
            or first_destination.shape[2] != block_size
        ):
            raise ValueError("destination layout must be [blocks, 2, block_size, ...]")
        if not first_source.is_contiguous() or not first_destination.is_contiguous():
            raise ValueError("compiled transfer requires contiguous tensors")
        if first_source.dtype != first_destination.dtype:
            raise ValueError("source and destination dtypes must match")

        token_count = first_source.shape[0]
        if token_count <= 0 or slot_mapping.numel() != token_count:
            raise ValueError("slot mapping length must equal source token count")
        hidden_elements = first_source.numel() // (token_count * 2)
        destination_hidden = first_destination.numel() // (
            first_destination.shape[0] * 2 * block_size
        )
        if hidden_elements != destination_hidden:
            raise ValueError("source and destination hidden sizes must match")

        source_shape = tuple(first_source.shape)
        destination_shape = tuple(first_destination.shape)
        for source, destination in zip(sources, destinations):
            if (
                source.device != first_source.device
                or destination.device != first_source.device
                or source.dtype != first_source.dtype
                or destination.dtype != first_source.dtype
                or tuple(source.shape) != source_shape
                or tuple(destination.shape) != destination_shape
                or not source.is_contiguous()
                or not destination.is_contiguous()
            ):
                raise ValueError("all KV layers must share one contiguous layout")

        device = first_source.device
        self._ensure_pointer_tables(len(sources), device)
        assert self._source_ptrs_gpu is not None
        assert self._destination_ptrs_gpu is not None

        source_ptrs_cpu = torch.empty(len(sources), dtype=torch.int64, pin_memory=True)
        source_ptrs_cpu.numpy()[:] = [tensor.data_ptr() for tensor in sources]
        self._source_ptrs_gpu.copy_(source_ptrs_cpu, non_blocking=True)
        host_pointer_tables = [source_ptrs_cpu]

        destination_signature = tuple(tensor.data_ptr() for tensor in destinations)
        if destination_signature != self._destination_signature:
            destination_ptrs_cpu = torch.empty(
                len(destinations), dtype=torch.int64, pin_memory=True
            )
            destination_ptrs_cpu.numpy()[:] = destination_signature
            self._destination_ptrs_gpu.copy_(destination_ptrs_cpu, non_blocking=True)
            host_pointer_tables.append(destination_ptrs_cpu)
            self._destination_signature = destination_signature

        self._retain_host_pointer_tables(tuple(host_pointer_tables), device)

        slots = slot_mapping.to(
            device=device,
            dtype=torch.int64,
            non_blocking=True,
        ).contiguous()
        self._extension.scatter_layer_major(
            self._source_ptrs_gpu,
            self._destination_ptrs_gpu,
            slots,
            hidden_elements * first_source.element_size(),
            block_size,
        )

        if os.environ.get("CASCADE_COMPILED_TRANSFER_DEBUG", "0") == "1":
            # Diagnostic-only exactness check against the established Triton
            # advanced-indexing contract.  Synchronizing here is intentionally
            # too expensive for serving, but makes it possible to distinguish
            # a native-kernel indexing bug from corruption elsewhere in the
            # connector pipeline on a real vLLM allocation.
            torch.cuda.synchronize(device=device)
            block_indices = slots // block_size
            block_offsets = slots % block_size
            mismatched_layers = []
            for index, (source, destination) in enumerate(zip(sources, destinations)):
                restored = destination[block_indices, :, block_offsets]
                if not torch.equal(restored, source):
                    mismatched_layers.append(index)
            if not self._debug_logged or mismatched_layers:
                logger.info(
                    "Compiled KV transfer debug: source_shape=%s "
                    "source_stride=%s source_storage_offset=%d "
                    "destination_shape=%s destination_stride=%s "
                    "destination_storage_offset=%d slots=%d slot_min=%d "
                    "slot_max=%d stream=%s mismatched_layers=%s",
                    tuple(first_source.shape),
                    first_source.stride(),
                    first_source.storage_offset(),
                    tuple(first_destination.shape),
                    first_destination.stride(),
                    first_destination.storage_offset(),
                    slots.numel(),
                    int(slots.min().item()),
                    int(slots.max().item()),
                    torch.cuda.current_stream(device),
                    mismatched_layers,
                )
                self._debug_logged = True


def create_compiled_multi_layer_transfer() -> CompiledMultiLayerKVTransfer:
    """Create a transfer helper, compiling the extension if necessary."""
    return CompiledMultiLayerKVTransfer(load_compiled_transfer_extension())
