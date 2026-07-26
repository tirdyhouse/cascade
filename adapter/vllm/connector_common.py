"""Disk-cache vLLM connector — v2 chunk-object architecture.

This module replaces the old block-hash / per-layer-safetensors connector
with a chunk-object-based design that uses:

* :class:`ChunkKeyStrategy`  (e.g. chain-hash) for deterministic chunk keys.
* :class:`ChunkObjectWriter` / :func:`load_chunk_object` for multi-layer
  chunk storage.
* The Go engine's ``match_chunks`` / ``resolve_chunks`` / ``commit_chunks``
  v2 HTTP API for coordination.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Optional

import torch

from adapter.storage import create_storage_backend
from adapter.storage.nvfile_backend import NvFileBackend
from adapter.storage.posix_backend import PosixBackend
from adapter.vllm.compiled_transfer import create_compiled_multi_layer_transfer
from adapter.vllm.chunk_keys import (
    ChunkDescriptor,
    create_chunk_key_strategy,
)
from adapter.vllm.go_client import DiskCacheGoClient
from adapter.storage.chunk_object import (
    ChunkObjectWriter,
    load_chunk_object,
    read_chunk_object_layout,
    tensor_slice_specs_from_header,
)
from adapter.vllm.tensor_ops import extract_kv_from_layer, inject_kv_into_layer

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import init_logger

logger = init_logger("vllm.disk_cache")


def _config_bool(value: Any, name: str) -> bool:
    """Parse a connector boolean from either JSON or agent string config."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _validate_shared_cache_marker(cache_root: Path, shared_cache_id: str) -> None:
    """Verify that this node mounted the same cache root as the metadata host."""
    marker_path = cache_root / ".cascade-shared-cache.json"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"cannot read shared cache marker {marker_path}: {exc}"
        ) from exc
    if not isinstance(marker, dict):
        raise RuntimeError(f"invalid shared cache marker {marker_path}")
    if marker.get("format_version") != 1:
        raise RuntimeError(
            f"unsupported shared cache marker format at {marker_path}: "
            f"{marker.get('format_version')!r}"
        )
    if marker.get("shared_cache_id") != shared_cache_id:
        raise RuntimeError(
            f"shared cache marker id at {marker_path} does not match "
            f"disk_cache_shared_id {shared_cache_id!r}"
        )


# ── Public metadata container ──────────────────────────────────────────


@dataclass
class DiskCacheMeta(KVConnectorMetadata):
    """Metadata container passed from scheduler to connector layers."""

    requests: list["_ReqMeta"] = field(default_factory=list)

    def add(self, req_meta: "_ReqMeta") -> None:
        self.requests.append(req_meta)


# ── Internal data types ────────────────────────────────────────────────


@dataclass
class _ReqMeta:
    """Per-request metadata for a single store or load operation.

    Attributes
    ----------
    request_id:
        vLLM request identifier.
    token_ids:
        Full token-id sequence for the request.
    block_ids:
        Allocated block IDs for the new tokens (store) or load target.
    block_size:
        vLLM block size.
    is_store:
        ``True`` when this meta describes a store operation.
    is_load:
        ``True`` when this meta describes a load operation.
    store_tokens:
        Number of tokens to save (only complete logical chunks).
    load_tokens:
        Number of tokens to load (from matched prefix).
    namespace:
        Stable cache namespace for this model + configuration.
    shard:
        Shard string for this rank (e.g. ``"tp0-pp0"``).
    expected_layers:
        Ordered list of layer names expected in the chunk objects.
    descriptors:
        Chunk descriptors covering the stored range or the physical KV range
        to inject for a load.
    source_descriptors:
        Original on-disk descriptors for a load.  A full prompt hit needs one
        final token recomputed for logits, so its final source object can be
        one token longer than the physical range injected into vLLM.
    """

    request_id: str
    token_ids: list[int]
    block_ids: list[int]
    block_size: int
    is_store: bool
    is_load: bool
    store_tokens: int
    load_tokens: int
    namespace: str
    shard: str
    expected_layers: list[str]
    descriptors: list[ChunkDescriptor]
    source_descriptors: list[ChunkDescriptor] = field(default_factory=list)
    resolved_objects: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _PendingLoadSpec:
    """Saved between :meth:`_get_num_new_matched_tokens` and
    :meth:`update_state_after_alloc` / :meth:`build_connector_meta`."""

    matched_tokens: int
    descriptors: list[ChunkDescriptor]
    local_cached: int
    # Logical lookup coverage may include the final prompt token. vLLM still
    # executes that token to produce first-token logits, so it is omitted from
    # the physical KV restore request.
    load_tokens: Optional[int] = None
    can_load: bool = False
    allocated_block_ids: list[int] = field(default_factory=list)
    resolved_objects: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _WriterState:
    """Per-(request_id, chunk_key) writer and tracking state."""

    writer: ChunkObjectWriter
    key: str
    index: int
    start: int
    end: int
    finalized_layers: set[str] = field(default_factory=set)
    aborted: bool = False
    committed: bool = False
    owner_request_id: str = ""


@dataclass
class _RequestTracker:
    """Per-request bookkeeping across scheduler iterations."""

    token_ids: list[int] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    store_chunks: set[str] = field(default_factory=set)
    committed_chunk_keys: set[str] = field(default_factory=set)
    committed_tokens: int = 0
    prompt_len: int = 0
    num_saved_tokens: int = 0
    num_computed_tokens: int = 0
    external_cached_tokens: int = 0
    # Full prompt lookup coverage can be one token ahead of the physical KV
    # restore count because vLLM forwards the final token for logits. Keep it
    # separately so that forward does not rewrite an already-persisted tail.
    logical_cached_tokens: int = 0


# ── Mixin ──────────────────────────────────────────────────────────────


class DiskCacheConnectorCommonMixin:
    """Shared logic for vLLM disk-cache connectors (v2 chunk-object path).

    Intended to be mixed with ``KVConnectorBase_V1``.
    DiskCacheConnector delegates ``get_num_new_matched_tokens``
    to ``_get_num_new_matched_tokens`` and ``save_kv_layer``
    to ``_save_request_kv`` — both defined here.
    """

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        """Keep Cascade's per-layer Python hooks outside graph replay.

        Full graphs are experimental and opt-in.  They are safe for the
        benchmark's full-hit consumer step because KV loading finishes before
        model execution and that step has nothing new to persist.  Large
        producer prefills still dispatch through the piecewise graph.  Keep
        the conservative default for workloads that may cross a persistence
        boundary while replaying a full graph, since ``save_kv_layer`` is a
        Python hook and would be skipped by replay.
        """
        return not bool(extra_config.get("disk_cache_allow_full_cudagraph", False))

    @staticmethod
    def _pending_load_token_count(pending: _PendingLoadSpec) -> int:
        """Return the physical KV-token count for a pending lookup.

        ``None`` preserves the interpretation of pending specs created by
        older callers and tests: their match count was also their load count.
        """
        if pending.load_tokens is None:
            return pending.matched_tokens
        return pending.load_tokens

    # ── Private hooks (called by connector subclasses) ─────────────────

    def _get_num_new_matched_tokens(
        self,
        request: Any,
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """Determine how many new tokens can be loaded from the disk cache.

        Lookup candidates cover the complete prompt, including a partial tail
        chunk.  This matches LMCache's logical hit accounting.  vLLM must
        still execute the final prompt token to produce first-token logits, so
        a full logical hit asks the scheduler to restore only ``N - 1`` KV
        tokens.  That mandatory forward is shared serving work, not a cache
        miss.

        The match response is accepted only when its token count, chunk count,
        and ordered keys all describe the same candidate prefix.
        """
        request_id = self._get_req_id(request)
        if request_id is not None:
            self._pending_loads.pop(request_id, None)
        if not self._connected or request_id is None:
            return (0, False)

        tokens = self._get_token_ids(request)
        if len(tokens) < 2:
            return (0, False)

        # Match the complete prompt so a full hit has the same logical token
        # coverage as LMCache. The scheduler receives one less token below.
        end = len(tokens)
        if num_computed_tokens >= end - 1:
            return (0, False)

        descriptors = self._strategy.describe(tokens, self._namespace, end=end)
        candidates = [{"key": d.key, "end_tokens": d.end} for d in descriptors]

        try:
            result = self._go.match_chunks(
                self._namespace,
                candidates,
                self._required_shards,
                resolve_shard=self._shard,
            )
        except Exception:
            logger.debug("match_chunks failed (will retry)", exc_info=True)
            return (0, False)

        if not isinstance(result, dict):
            logger.warning(
                "match_chunks returned non-object response %r; discarding",
                result,
            )
            return (0, False)
        required_fields = {"matched_chunks", "matched_tokens", "matched_keys"}
        if not required_fields.issubset(result):
            logger.warning(
                "match_chunks response missing fields %s; discarding",
                sorted(required_fields - set(result)),
            )
            return (0, False)

        matched_chunks = result["matched_chunks"]
        matched_tokens = result["matched_tokens"]
        matched_keys = result["matched_keys"]
        if type(matched_chunks) is not int or type(matched_tokens) is not int:
            logger.warning(
                "match_chunks returned invalid counts chunks=%r tokens=%r; "
                "discarding",
                matched_chunks,
                matched_tokens,
            )
            return (0, False)
        if matched_chunks < 0 or matched_tokens < 0 or matched_tokens > end:
            logger.warning(
                "match_chunks returned out-of-range counts chunks=%r "
                "tokens=%r; discarding",
                matched_chunks,
                matched_tokens,
            )
            return (0, False)

        matching_descs = [d for d in descriptors if d.end <= matched_tokens]
        if matched_tokens > 0 and (
            not matching_descs or matching_descs[-1].end != matched_tokens
        ):
            logger.warning(
                "match_chunks returned unmatched boundary %r; discarding",
                matched_tokens,
            )
            return (0, False)
        if matched_chunks != len(matching_descs):
            logger.warning(
                "match_chunks count mismatch: returned %d chunks for %d "
                "descriptor-prefix entries; discarding",
                matched_chunks,
                len(matching_descs),
            )
            return (0, False)

        if matched_keys is None and matched_chunks == 0:
            matched_keys = []
        if not isinstance(matched_keys, list) or not all(
            isinstance(key, str) for key in matched_keys
        ):
            logger.warning(
                "match_chunks returned invalid matched_keys %r; discarding",
                matched_keys,
            )
            return (0, False)
        expected_keys = [d.key for d in matching_descs]
        if matched_keys != expected_keys:
            logger.warning(
                "match_chunks returned key prefix %r, expected %r; discarding",
                matched_keys,
                expected_keys,
            )
            return (0, False)

        resolved_objects = result.get("matched_objects", [])
        if resolved_objects is None:
            resolved_objects = []
        fused_objects_valid = (
            isinstance(resolved_objects, list)
            and len(resolved_objects) == matched_chunks
            and all(isinstance(obj, dict) for obj in resolved_objects)
            and [obj.get("key") for obj in resolved_objects] == expected_keys
            and all(obj.get("shard") == self._shard for obj in resolved_objects)
        )
        if resolved_objects and not fused_objects_valid:
            logger.warning(
                "match_chunks returned invalid fused objects; falling back "
                "to resolve_chunks"
            )
            resolved_objects = []

        load_tokens = matched_tokens
        if matched_tokens == end:
            # This is the same full-hit adjustment in LMCache's vLLM adapter:
            # the final prompt token must be forwarded to obtain logits for
            # the first generated token.
            load_tokens -= 1
        external = max(0, load_tokens - num_computed_tokens)
        if external <= 0:
            return (0, False)
        self._pending_loads[request_id] = _PendingLoadSpec(
            matched_tokens=matched_tokens,
            descriptors=matching_descs,
            local_cached=num_computed_tokens,
            load_tokens=load_tokens,
            resolved_objects=resolved_objects,
        )
        return (external, False)

    def _save_request_kv(
        self,
        req: _ReqMeta,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
    ) -> None:
        """Save one KV-cache layer for one store request.

        Called by the connector's :meth:`save_kv_layer` for each
        (request, layer) pair.

        Extracts the canonical ``[T, 2, ...]`` tensor from the KV cache
        and writes each complete chunk to its :class:`ChunkObjectWriter`.
        """
        if req.request_id in self._failed_load_request_ids:
            return

        # Build slot mapping for this request's store tokens
        slot_mapping = self._build_slot_mapping(
            block_ids=req.block_ids,
            block_size=req.block_size,
            num_tokens=req.store_tokens,
        )
        if slot_mapping.numel() == 0:
            return

        extracted = extract_kv_from_layer(
            kv_layer, slot_mapping, attn_metadata, self._block_size
        )
        # extracted shape: [T, 2, ...]

        # Process each complete chunk descriptor within store_tokens
        for desc in req.descriptors:
            if desc.end > req.store_tokens:
                continue
            wkey = (req.request_id, desc.key)

            if wkey not in self._active_writers:
                writer = ChunkObjectWriter(
                    work_dir=self.cache_root,
                    namespace=req.namespace,
                    key=desc.key,
                    shard=self._shard,
                    index=desc.index,
                    start=desc.start,
                    end=desc.end,
                    expected_layers=set(self._expected_layers),
                    backend=self._storage,
                )
                self._active_writers[wkey] = _WriterState(
                    writer=writer,
                    key=desc.key,
                    index=desc.index,
                    start=desc.start,
                    end=desc.end,
                    owner_request_id=req.request_id,
                )

            ws = self._active_writers[wkey]
            if ws.aborted:
                continue
            if layer_name not in ws.finalized_layers:
                # Slice the exact token range for this chunk
                chunk_tensor = extracted[desc.start : desc.end]
                expected_len = desc.end - desc.start
                if chunk_tensor.size(0) != expected_len:
                    logger.warning(
                        "_save_request_kv: chunk %s expected %d tokens, "
                        "got %d; skipping",
                        desc.key,
                        expected_len,
                        chunk_tensor.size(0),
                    )
                    continue

                try:
                    ws.writer.add_layer(layer_name, chunk_tensor)
                    ws.finalized_layers.add(layer_name)
                except Exception as exc:
                    logger.error(
                        "Failed to add layer %s to chunk %s: %s",
                        layer_name,
                        desc.key,
                        exc,
                    )
                    ws.aborted = True
                    ws.writer.abort()

    # ── Lifecycle hooks (called by vLLM) ──────────────────────────────

    def update_state_after_alloc(
        self,
        request: Any,
        blocks: Any,
        num_external_tokens: int,
    ) -> None:
        """Record block IDs for a pending load.

        Only acts when ``num_external_tokens > 0`` and a pending spec
        exists for this request.
        """
        if num_external_tokens <= 0:
            return
        pending = self._pending_loads.get(request.request_id)
        if pending is None:
            return

        # Validate: num_external_tokens should match pending minus local
        expected = self._pending_load_token_count(pending) - pending.local_cached
        if expected > 0 and num_external_tokens != expected:
            logger.warning(
                "update_state_after_alloc: expected %d external tokens "
                "but got %d for request %s",
                expected,
                num_external_tokens,
                request.request_id,
            )
            return

        # Get block IDs — try different access patterns for compatibility
        block_ids = self._extract_block_ids(blocks)
        if not block_ids:
            logger.warning(
                "update_state_after_alloc: no block IDs for request %s",
                request.request_id,
            )
            return

        pending.allocated_block_ids = block_ids
        pending.can_load = True

    def build_connector_meta(
        self,
        scheduler_output: Any,
        finished_req_ids: Optional[list[str]] = None,
    ) -> DiskCacheMeta:
        """Translate one vLLM scheduler step into load/store operations.

        The scheduler reports the number of tokens computed *before* the
        current step in ``CachedRequestData.num_computed_tokens`` and the
        number scheduled in ``SchedulerOutput.num_scheduled_tokens``. Keeping
        those values separate prevents chunked-prefill, rollback, and resume
        paths from using committed disk bytes as a proxy for vLLM progress.
        """
        if not hasattr(self, "_unfinished_requests"):
            self._unfinished_requests = {}

        if finished_req_ids is None:
            finished_req_ids = getattr(scheduler_output, "finished_req_ids", None)
        for req_id in finished_req_ids or ():
            self._pending_loads.pop(req_id, None)
            self._request_trackers.pop(req_id, None)
            getattr(self, "_unfinished_requests", {}).pop(req_id, None)
            for writer_key in [key for key in self._active_writers if key[0] == req_id]:
                writer = self._active_writers.pop(writer_key)
                if not writer.committed:
                    writer.aborted = True
                    writer.writer.abort()

        meta = DiskCacheMeta()

        def add_load_meta(
            request_id: str,
            tracker: _RequestTracker,
            pending: Optional[_PendingLoadSpec],
        ) -> None:
            if pending is None or not pending.can_load:
                return
            load_tokens = self._pending_load_token_count(pending)
            if load_tokens <= 0 or not pending.allocated_block_ids:
                return
            source_descriptors = [
                desc for desc in pending.descriptors if desc.start < load_tokens
            ]
            descriptors = [
                ChunkDescriptor(
                    index=desc.index,
                    start=desc.start,
                    end=min(desc.end, load_tokens),
                    key=desc.key,
                )
                for desc in source_descriptors
            ]
            if not descriptors:
                return
            meta.add(
                _ReqMeta(
                    request_id=request_id,
                    token_ids=tracker.token_ids,
                    block_ids=pending.allocated_block_ids,
                    block_size=self._block_size,
                    is_store=False,
                    is_load=True,
                    store_tokens=0,
                    load_tokens=load_tokens,
                    namespace=self._namespace,
                    shard=self._shard,
                    expected_layers=list(self._expected_layers),
                    descriptors=descriptors,
                    source_descriptors=source_descriptors,
                    resolved_objects=list(pending.resolved_objects),
                )
            )

        def add_store_meta(request_id: str, tracker: _RequestTracker) -> None:
            store_end = self._store_end_for_step(
                tracker.num_computed_tokens, tracker.prompt_len
            )
            descriptors = self._new_store_descriptors(tracker, store_end)
            if not descriptors:
                return
            meta.add(
                _ReqMeta(
                    request_id=request_id,
                    token_ids=tracker.token_ids,
                    block_ids=tracker.block_ids,
                    block_size=self._block_size,
                    is_store=True,
                    is_load=False,
                    store_tokens=store_end,
                    load_tokens=0,
                    namespace=self._namespace,
                    shard=self._shard,
                    expected_layers=list(self._expected_layers),
                    descriptors=descriptors,
                )
            )
            tracker.store_chunks.update(desc.key for desc in descriptors)

        for new_req in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
            req_id = self._get_req_id(new_req)
            prompt_tokens = self._get_token_ids(new_req)
            if req_id is None or not prompt_tokens:
                continue

            prompt_len = len(
                getattr(new_req, "prompt_token_ids", None) or prompt_tokens
            )
            scheduled = max(0, self._get_num_scheduled(scheduler_output, req_id))
            computed_before = max(
                0, int(getattr(new_req, "num_computed_tokens", 0) or 0)
            )
            pending = self._pending_loads.pop(req_id, None)
            external_cached = (
                self._pending_load_token_count(pending)
                if pending is not None and pending.can_load
                else 0
            )
            logical_cached = (
                pending.matched_tokens
                if pending is not None and pending.can_load
                else 0
            )
            computed_after = min(
                max(computed_before, external_cached) + scheduled,
                prompt_len,
            )

            block_ids = self._extract_block_ids(getattr(new_req, "block_ids", None))
            tracker = self._request_trackers.get(req_id)
            if tracker is None:
                tracker = _RequestTracker()
                self._request_trackers[req_id] = tracker
            tracker.token_ids = list(prompt_tokens[:computed_after])
            tracker.block_ids = block_ids
            tracker.prompt_len = prompt_len
            tracker.num_computed_tokens = computed_after
            tracker.external_cached_tokens = external_cached
            tracker.logical_cached_tokens = logical_cached
            tracker.num_saved_tokens = min(tracker.num_saved_tokens, computed_after)
            self._unfinished_requests.setdefault(req_id, new_req)

            add_load_meta(req_id, tracker, pending)
            add_store_meta(req_id, tracker)

        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached is None or isinstance(cached, list):
            return meta

        req_ids = getattr(cached, "req_ids", ()) or ()
        new_block_ids = getattr(cached, "new_block_ids", ()) or ()
        all_token_ids = getattr(cached, "all_token_ids", {}) or {}
        computed_before_values = getattr(cached, "num_computed_tokens", ()) or ()
        new_token_values = getattr(cached, "new_token_ids", ()) or ()
        resumed_ids = set(getattr(cached, "resumed_req_ids", ()) or ())
        resumed_ids.update(getattr(cached, "preempted_req_ids", ()) or ())

        for index, req_id in enumerate(req_ids):
            tracker = self._request_trackers.get(req_id)
            if tracker is None:
                logger.warning("Skipping cached request %s without a tracker", req_id)
                continue

            request = self._unfinished_requests.get(req_id)
            full_tokens = all_token_ids.get(req_id)
            if full_tokens is None and request is not None:
                full_tokens = self._get_token_ids(request)
            if full_tokens is None:
                full_tokens = list(tracker.token_ids)
                if index < len(new_token_values):
                    full_tokens.extend(new_token_values[index] or [])
            full_tokens = list(full_tokens or [])
            if not full_tokens:
                logger.warning("Skipping cached request %s without token IDs", req_id)
                continue

            prompt_len = tracker.prompt_len
            if request is not None:
                prompt_ids = getattr(request, "prompt_token_ids", None)
                if prompt_ids is not None:
                    prompt_len = len(prompt_ids)
            if prompt_len <= 0:
                prompt_len = len(full_tokens)
            prompt_len = min(prompt_len, len(full_tokens))

            if index < len(computed_before_values):
                computed_before = max(0, int(computed_before_values[index]))
            else:
                computed_before = tracker.num_computed_tokens
            scheduled = max(0, self._get_num_scheduled(scheduler_output, req_id))
            pending = self._pending_loads.pop(req_id, None)
            has_pending_load = pending is not None and pending.can_load
            external_cached = (
                self._pending_load_token_count(pending)
                if has_pending_load
                else min(tracker.external_cached_tokens, computed_before)
            )
            logical_cached = (
                pending.matched_tokens
                if has_pending_load
                else tracker.logical_cached_tokens
            )

            if computed_before < tracker.num_computed_tokens:
                tracker.token_ids = list(
                    full_tokens[: min(computed_before, prompt_len)]
                )
                tracker.num_saved_tokens = min(
                    tracker.num_saved_tokens, computed_before
                )
                tracker.external_cached_tokens = min(
                    tracker.external_cached_tokens, computed_before
                )
                tracker.logical_cached_tokens = min(
                    tracker.logical_cached_tokens, computed_before
                )
                logical_cached = min(logical_cached, computed_before)

            block_update = self._get_block_ids_for_cached(list(new_block_ids), index)
            if req_id in resumed_ids:
                if block_update:
                    tracker.block_ids = block_update
                else:
                    logger.warning(
                        "Resumed request %s supplied no replacement block IDs",
                        req_id,
                    )
            elif block_update:
                tracker.block_ids = tracker.block_ids + block_update

            computed_after = min(
                max(computed_before, external_cached) + scheduled,
                prompt_len,
            )
            tracker.token_ids = list(full_tokens[:computed_after])
            tracker.prompt_len = prompt_len
            tracker.num_computed_tokens = computed_after
            tracker.external_cached_tokens = external_cached
            tracker.logical_cached_tokens = logical_cached

            add_load_meta(req_id, tracker, pending)
            add_store_meta(req_id, tracker)

        return meta

    def start_load_kv(
        self,
        forward_context: Any,
        **kwargs: Any,
    ) -> None:
        """Load saved KV without letting I/O failures abort model execution.

        vLLM consumes invalid block IDs after the forward, discards output
        backed by those blocks, and reschedules the affected requests.
        """
        self._load_error = None
        self._failed_load_request_ids.clear()

        meta = self._get_connector_metadata()
        if not self._is_disk_cache_meta(meta):
            return
        load_reqs = [req for req in meta.requests if req.is_load]
        if not load_reqs:
            return

        def fail_all(message: str) -> None:
            logger.error("%s", message)
            for req in load_reqs:
                self._mark_load_failed(req, message)

        if not self._connected:
            fail_all("Cannot load chunk objects while disk-cache is disconnected")
            return
        if not self._expected_layers:
            fail_all("Cannot load chunk objects without configured KV layer names")
            return

        attn_metadata = getattr(forward_context, "attn_metadata", None)
        if attn_metadata is None and getattr(self, "_compiled_transfer", None) is None:
            fail_all(
                "Cannot load chunk objects: forward context has no "
                "attention metadata and compiled transfer is unavailable"
            )
            return

        layer_cache = self._collect_available_kv_layers(
            getattr(forward_context, "no_compile_layers", None)
        )
        for layer_name, kv_cache in self._registered_kv_caches.items():
            layer_cache.setdefault(layer_name, kv_cache)

        missing_layers = [
            layer_name
            for layer_name in self._expected_layers
            if layer_name not in layer_cache
        ]
        if missing_layers:
            fail_all(
                "Missing expected layers in forward context for load: "
                f"{missing_layers}"
            )
            return

        for req in load_reqs:
            try:
                self._load_request_kv(req, forward_context, attn_metadata, layer_cache)
            except Exception as exc:
                self._mark_load_failed(req, exc)
                logger.exception(
                    "Failed to load disk-cache KV for request %s; "
                    "scheduler will recompute invalid blocks",
                    req.request_id,
                )

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        """Save one KV-cache layer for every eligible store request."""
        if not self._connected:
            return
        meta = self._get_connector_metadata()
        if not self._is_disk_cache_meta(meta):
            return

        for req in meta.requests:
            if req.is_store:
                self._save_request_kv(req, layer_name, kv_layer, attn_metadata)

    def wait_for_save(self) -> None:
        """Best-effort finalize and commit for the current forward.

        All layer and duplicate-identity barriers are checked before any writer
        is finalized. Save failures must never fail model execution: temporary
        state is discarded and the error is logged. Immutable final objects are
        retained so a later deterministic writer can validate and retry an
        idempotent metadata commit, including after a lost HTTP response.
        """
        for request_id in set(self._failed_load_request_ids):
            self._abort_request_writers(request_id)

        candidates: list[tuple[tuple[str, str], _WriterState]] = []
        for writer_key, writer_state in list(self._active_writers.items()):
            if writer_state.committed:
                self._active_writers.pop(writer_key, None)
                continue
            candidates.append((writer_key, writer_state))
        if not candidates:
            return

        def discard_writer_states() -> None:
            for writer_key, writer_state in candidates:
                if not writer_state.committed:
                    writer_state.aborted = True
                    writer_state.writer.abort()
                self._active_writers.pop(writer_key, None)

        expected_layers = set(self._expected_layers)
        projected: list[dict[str, Any]] = []
        unique_projected: dict[tuple[str, str, str], dict[str, Any]] = {}
        barrier_error: Optional[str] = None

        for writer_key, writer_state in candidates:
            if writer_state.aborted:
                barrier_error = f"writer {writer_key} was already aborted"
                break
            if writer_state.finalized_layers != expected_layers:
                barrier_error = (
                    f"writer {writer_key} is incomplete: expected "
                    f"{sorted(expected_layers)}, got "
                    f"{sorted(writer_state.finalized_layers)}"
                )
                break
            try:
                relative_path = writer_state.writer.final_path.relative_to(
                    self.cache_root
                )
            except Exception as exc:
                barrier_error = f"writer {writer_key} has invalid final path: {exc}"
                break

            owner_request_id = writer_state.owner_request_id or writer_key[0]
            obj = {
                "file_path": str(relative_path),
                "namespace": self._namespace,
                "key": writer_state.key,
                "shard": self._shard,
                "index": writer_state.index,
                "start_tokens": writer_state.start,
                "end_tokens": writer_state.end,
                "_owner_request_id": owner_request_id,
                "_writer_key": writer_key,
            }
            identity = (obj["namespace"], obj["key"], obj["shard"])
            existing = unique_projected.get(identity)
            if existing is not None:
                for field_name in (
                    "file_path",
                    "index",
                    "start_tokens",
                    "end_tokens",
                ):
                    if existing[field_name] != obj[field_name]:
                        barrier_error = (
                            "duplicate chunk identity has conflicting "
                            f"{field_name}: owners "
                            f"{existing['_owner_request_id']!r} and "
                            f"{owner_request_id!r}"
                        )
                        break
                if barrier_error is not None:
                    break
            else:
                unique_projected[identity] = obj
            projected.append(obj)

        if barrier_error is not None:
            logger.error("wait_for_save barrier failed: %s", barrier_error)
            discard_writer_states()
            return

        objects_to_commit: list[dict[str, Any]] = []
        try:
            for obj in projected:
                writer_state = self._active_writers[obj["_writer_key"]]
                header = writer_state.writer.finalize()
                committed_obj = dict(obj)
                committed_obj["size"] = writer_state.writer.final_path.stat().st_size
                committed_obj["_header"] = header
                objects_to_commit.append(committed_obj)
        except Exception:
            logger.exception(
                "Failed to finalize chunk-object writer; skipping cache save"
            )
            discard_writer_states()
            return

        unique_objects: dict[tuple[str, str, str], dict[str, Any]] = {}
        for obj in objects_to_commit:
            identity = (obj["namespace"], obj["key"], obj["shard"])
            existing = unique_objects.get(identity)
            if existing is not None:
                if existing["size"] != obj["size"]:
                    logger.error(
                        "wait_for_save: duplicate chunk identity has "
                        "conflicting size; skipping cache save"
                    )
                    discard_writer_states()
                    return
                continue
            unique_objects[identity] = obj

        commit_payload = [
            {key: value for key, value in obj.items() if not key.startswith("_")}
            for obj in unique_objects.values()
        ]
        try:
            self._go.commit_chunks(commit_payload)
        except Exception as exc:
            logger.error(
                "commit_chunks failed; retaining finalized objects for " "retry: %s",
                exc,
            )
            discard_writer_states()
            return

        # The writers above created these immutable objects under our own
        # cache root, and Go has now atomically published their metadata.
        # Populate the containment-checked path LRU here rather than waiting
        # for a second load of the same object.  That keeps the first
        # warmup->query retrieval off the remote filesystem's Path.resolve()
        # critical path.  Objects discovered from Go still go through the
        # full check on their first observation.
        if (
            getattr(self, "_resolved_object_path_cache_capacity", 0) > 0
            or getattr(self, "_object_layout_cache_capacity", 0) > 0
            or getattr(self, "_posix_host_cache_capacity_bytes", 0) > 0
        ):
            for obj in unique_objects.values():
                try:
                    full_path = self._resolve_cache_object_path(obj["file_path"])
                    _, specs = self._remember_object_layout(full_path, obj["_header"])
                    if getattr(
                        self, "_posix_host_cache_capacity_bytes", 0
                    ) > 0 and isinstance(self._storage, PosixBackend):
                        loaded = self._storage.load_tensor_slices_to_pinned_cpu(
                            full_path, specs
                        )
                        if len(loaded) != len(self._expected_layers):
                            raise RuntimeError(
                                "POSIX host-cache seed returned "
                                f"{len(loaded)} tensors for "
                                f"{len(self._expected_layers)} layers"
                            )
                        self._remember_posix_host_object(
                            full_path,
                            dict(zip(self._expected_layers, loaded)),
                        )
                except Exception:
                    # These are performance caches only.  A failure here
                    # must not roll back an already-published immutable
                    # object; the normal load path retains full validation.
                    logger.warning(
                        "Could not seed object metadata caches for %s",
                        obj["file_path"],
                        exc_info=True,
                    )
            if getattr(self, "_posix_host_cache_capacity_bytes", 0) > 0:
                with self._posix_host_cache_lock:
                    logger.info(
                        "POSIX host cache resident: entries=%d bytes=%d "
                        "capacity_bytes=%d",
                        len(self._posix_host_objects),
                        self._posix_host_cache_bytes,
                        self._posix_host_cache_capacity_bytes,
                    )

        for obj in objects_to_commit:
            request_id = obj["_owner_request_id"]
            tracker = self._request_trackers.get(request_id)
            if tracker is not None:
                tracker.committed_chunk_keys.add(obj["key"])
                tracker.committed_tokens = max(
                    tracker.committed_tokens,
                    obj["end_tokens"],
                )
            writer_state = self._active_writers.get(obj["_writer_key"])
            if writer_state is not None:
                writer_state.committed = True

        for writer_key, _ in candidates:
            self._active_writers.pop(writer_key, None)

    def request_finished(self, request: Any, block_ids: Any) -> tuple[bool, Any]:
        """Clean up per-request state after a request finishes."""
        rid = self._get_req_id(request)
        if rid:
            self._pending_loads.pop(rid, None)
            self._request_trackers.pop(rid, None)
            getattr(self, "_unfinished_requests", {}).pop(rid, None)
            # Clean up active writers belonging to this request
            writers_to_remove = [
                wkey for wkey in self._active_writers if wkey[0] == rid
            ]
            for wkey in writers_to_remove:
                ws = self._active_writers.pop(wkey, None)
                if ws is not None and not ws.committed:
                    ws.aborted = True
                    ws.writer.abort()
        return False, None

    def take_events(self) -> list[Any]:
        return []

    def get_finished(self, finished_req_ids: list[str] | set[str]) -> tuple[Any, Any]:
        return None, None

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Synchronous no-op; load failures are reported as invalid blocks."""
        return

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> None:
        """Register worker KV tensors while preserving vLLM's base contract."""
        if not isinstance(kv_caches, dict):
            raise TypeError("kv_caches must be a dict of layer name to tensor")
        self._registered_kv_caches = dict(kv_caches)
        seen = set(self._expected_layers)
        for layer_name, cache in kv_caches.items():
            if not isinstance(layer_name, str) or cache is None:
                continue
            if layer_name not in seen:
                self._expected_layers.append(layer_name)
                seen.add(layer_name)
        self._warm_posix_pinned_staging_allocator(kv_caches)

    def _warm_posix_pinned_staging_allocator(
        self,
        kv_caches: dict[str, torch.Tensor],
    ) -> None:
        """Pre-size the caching pinned allocator for first-hit POSIX reads.

        Read workers still own independent tensors and the CUDA caching host
        allocator remains responsible for stream-safe reuse.  Allocating and
        immediately releasing object-sized blocks during worker setup merely
        moves the expensive OS page-pinning work out of request TTFT.
        """
        if getattr(self, "_posix_pinned_staging_warmed", False):
            return
        capacity = getattr(self, "_posix_pinned_staging_capacity_bytes", 0)
        if capacity <= 0:
            return
        if not isinstance(getattr(self, "_storage", None), PosixBackend):
            self._posix_pinned_staging_warmed = True
            return
        if getattr(self, "_posix_host_cache_capacity_bytes", 0) > 0:
            self._posix_pinned_staging_warmed = True
            return

        object_nbytes = 0
        for layer_name in self._expected_layers:
            cache = kv_caches.get(layer_name)
            if (
                cache is None
                or not cache.is_cuda
                or cache.ndim < 4
                or cache.shape[0] <= 0
                or cache.shape[2] != self._block_size
            ):
                # vLLM may register cache groups incrementally.  Wait until a
                # later call supplies every expected layer before sizing the
                # object blocks; warming a partial-layer size would not help
                # the real coalesced reads.
                return
            cache_tokens = cache.shape[0] * self._block_size
            bytes_per_token = cache.numel() * cache.element_size() // cache_tokens
            layer_nbytes = bytes_per_token * self._chunk_size
            object_nbytes += (layer_nbytes + 4095) // 4096 * 4096

        if object_nbytes <= 0 or object_nbytes > capacity:
            self._posix_pinned_staging_warmed = True
            logger.warning(
                "POSIX pinned staging warm-up skipped: object_bytes=%d "
                "capacity_bytes=%d",
                object_nbytes,
                capacity,
            )
            return

        slot_count = capacity // object_nbytes
        self._posix_pinned_staging_warmed = True
        try:
            buffers = [
                torch.empty(object_nbytes, dtype=torch.uint8, pin_memory=True)
                for _ in range(slot_count)
            ]
            del buffers
        except (RuntimeError, TypeError) as exc:
            logger.warning(
                "POSIX pinned staging warm-up failed for %d slots of %d " "bytes: %s",
                slot_count,
                object_nbytes,
                exc,
            )
            return
        logger.info(
            "POSIX pinned staging allocator warmed: slots=%d "
            "object_bytes=%d capacity_mib=%d",
            slot_count,
            object_nbytes,
            capacity // 1024**2,
        )

    def _collect_available_kv_layers(
        self,
        no_compile_layers: Any,
    ) -> dict[str, torch.Tensor]:
        """Collect KV cache tensors from *no_compile_layers*.

        Supports both dict and object-attribute access patterns.
        """
        layer_cache: dict[str, torch.Tensor] = {}
        if no_compile_layers is None:
            return layer_cache
        if isinstance(no_compile_layers, dict):
            items = no_compile_layers.items()
        else:
            items = (
                (lname, getattr(no_compile_layers, lname, None))
                for lname in self._expected_layers
            )
        for lname, layer_obj in items:
            if layer_obj is None:
                continue
            kv = getattr(layer_obj, "kv_cache", None)
            if kv is not None:
                layer_cache[lname] = kv
        return layer_cache

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Return and clear the set of block IDs that had load errors."""
        result = set(self._block_ids_with_load_errors)
        self._block_ids_with_load_errors.clear()
        return result

    # ── Initialisation ──────────────────────────────────────────────

    def __init__(
        self,
        vllm_config: Any,
        role: str,
        kv_cache_config: Any,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}

        # Paths and Go engine
        self.cache_root = Path(extra.get("disk_cache_path", "/tmp/disk-cache"))
        self.go_addr = extra.get("disk_cache_engine_addr", "http://localhost:9100")
        self._go = DiskCacheGoClient(self.go_addr)
        self._shared_cache = _config_bool(
            extra.get("disk_cache_shared", False),
            "disk_cache_shared",
        )
        self._shared_cache_id = str(extra.get("disk_cache_shared_id", "")).strip()
        if self._shared_cache:
            if not self.cache_root.is_absolute():
                raise ValueError(
                    "disk_cache_path must be absolute when disk_cache_shared=true"
                )
            if not self._shared_cache_id:
                raise ValueError(
                    "disk_cache_shared_id is required when disk_cache_shared=true"
                )
            if not self.cache_root.is_dir():
                raise RuntimeError(
                    f"shared cache root is not mounted or accessible: {self.cache_root}"
                )
            _validate_shared_cache_marker(
                self.cache_root,
                self._shared_cache_id,
            )

        # Block and chunk configuration
        self._block_size = vllm_config.cache_config.block_size

        chunk_size_tokens = int(extra.get("disk_cache_chunk_size_tokens", 256))
        if chunk_size_tokens <= 0:
            raise ValueError(
                f"disk_cache_chunk_size_tokens must be > 0, " f"got {chunk_size_tokens}"
            )
        if chunk_size_tokens % self._block_size != 0:
            raise ValueError(
                f"disk_cache_chunk_size_tokens ({chunk_size_tokens}) must "
                f"be an integer multiple of block_size ({self._block_size})"
            )
        self._chunk_size = chunk_size_tokens

        # Chunk key strategy
        strategy_name = extra.get("disk_cache_key_strategy", "chain")
        self._strategy = create_chunk_key_strategy(
            strategy_name,
            tokens_per_chunk=chunk_size_tokens,
            block_size=self._block_size,
        )

        # Storage backend. Strict mode is opt-in so existing explicit GDS
        # configurations retain their historical POSIX fallback behavior.
        storage_prefer = extra.get("storage_backend", "auto")
        storage_strict = bool(extra.get("storage_backend_strict", False))
        self._storage = create_storage_backend(
            prefer=storage_prefer,
            strict=storage_strict,
        )

        # Device resolution
        self.target_device = extra.get("target_device", "auto")

        # POSIX object reads can safely run in CPU worker threads, while CUDA
        # copies and KV injection stay on vLLM's execution thread.  Keep the
        # queue bounded to avoid pinning one host buffer per prompt chunk.
        try:
            self._posix_load_workers = int(
                extra.get("disk_cache_posix_load_workers", 4)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "disk_cache_posix_load_workers must be an integer"
            ) from exc
        if self._posix_load_workers < 1:
            raise ValueError("disk_cache_posix_load_workers must be >= 1")

        # Match the four POSIX read-ahead workers by default.  This batches
        # adjacent chunk objects into one paged-KV scatter per layer, reducing
        # Python/CUDA dispatches for long cache hits; set one for the legacy
        # per-object behavior.
        try:
            self._posix_inject_batch_chunks = int(
                extra.get("disk_cache_posix_inject_batch_chunks", 4)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "disk_cache_posix_inject_batch_chunks must be an integer"
            ) from exc
        if self._posix_inject_batch_chunks < 1:
            raise ValueError("disk_cache_posix_inject_batch_chunks must be >= 1")

        # Experimental all-layer CUDA scatter.  Keep it opt-in until the
        # native extension has passed correctness and performance validation
        # for the active vLLM/Triton KV layout.  A build failure safely leaves
        # the established PyTorch injection path active.
        self._compiled_transfer_requested = bool(
            extra.get("disk_cache_compiled_transfer", False)
        )

        # A successful full-hit repeatedly resolves the same immutable object
        # paths.  ``Path.resolve()`` follows every component on the backing
        # filesystem and was measurable on remote mounts.  Cache only paths
        # that have already passed the containment check below; invalidation
        # removes the entry, and chunk objects are write-once by contract.
        try:
            self._resolved_object_path_cache_capacity = int(
                extra.get("disk_cache_resolved_object_path_cache_entries", 4096)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "disk_cache_resolved_object_path_cache_entries must be an integer"
            ) from exc
        if self._resolved_object_path_cache_capacity < 0:
            raise ValueError(
                "disk_cache_resolved_object_path_cache_entries must be >= 0"
            )

        try:
            self._object_layout_cache_capacity = int(
                extra.get("disk_cache_object_layout_cache_entries", 4096)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "disk_cache_object_layout_cache_entries must be an integer"
            ) from exc
        if self._object_layout_cache_capacity < 0:
            raise ValueError("disk_cache_object_layout_cache_entries must be >= 0")

        # Optional same-process hot tier for a like-for-like comparison with
        # LMCache LocalCPU.  Objects remain durably stored through the POSIX
        # backend; this bounded LRU only retains their immutable payloads in
        # host memory so a repeated hit can skip the page-cache -> pinned-RAM
        # copy.  Keep it disabled by default because pinned memory is a
        # capacity decision the serving operator must make explicitly.
        try:
            self._posix_host_cache_capacity_bytes = int(
                float(extra.get("disk_cache_posix_host_cache_gib", 0.0)) * 1024**3
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "disk_cache_posix_host_cache_gib must be a number"
            ) from exc
        if self._posix_host_cache_capacity_bytes < 0:
            raise ValueError("disk_cache_posix_host_cache_gib must be >= 0")

        # Optional startup warm-up for PyTorch's caching pinned-host allocator.
        # POSIX retrieval otherwise pays the cost of pinning an entire prompt's
        # worth of read buffers on the first hit.  LMCache LocalDisk reserves a
        # comparable pinned staging tier up front; expose the same capacity
        # choice here without retaining a second copy when the host-object LRU
        # above is already enabled.
        try:
            self._posix_pinned_staging_capacity_bytes = int(
                float(extra.get("disk_cache_posix_pinned_staging_mib", 0.0)) * 1024**2
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "disk_cache_posix_pinned_staging_mib must be a number"
            ) from exc
        if self._posix_pinned_staging_capacity_bytes < 0:
            raise ValueError("disk_cache_posix_pinned_staging_mib must be >= 0")

        # Several cuFile I/O workers write into a persistent, registered GPU
        # staging allocation; this connector still performs all KV-cache
        # injection on its own vLLM execution thread.  Six workers keep I/O
        # covered by the 192 MiB pool's 13 registered slots on the reference
        # T4 setup.  Operators can lower this for slower storage or override
        # it upward when BAR1 permits a deeper fully registered pool.
        try:
            self._gds_load_workers = int(extra.get("disk_cache_gds_load_workers", 6))
        except (TypeError, ValueError) as exc:
            raise ValueError("disk_cache_gds_load_workers must be an integer") from exc
        if self._gds_load_workers < 1:
            raise ValueError("disk_cache_gds_load_workers must be >= 1")
        try:
            self._gds_staging_buffer_mib = int(
                extra.get("disk_cache_gds_staging_buffer_mib", 192)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "disk_cache_gds_staging_buffer_mib must be an integer"
            ) from exc
        if self._gds_staging_buffer_mib < 0:
            raise ValueError("disk_cache_gds_staging_buffer_mib must be >= 0")

        # Optional request-level timing breakdown for cache-hit investigation.
        # It is deliberately disabled by default: the connector sits on the
        # inference critical path, so collecting per-stage clocks is only for
        # benchmark and diagnosis runs.
        self._load_profile = bool(extra.get("disk_cache_load_profile", False))
        # Successful-retrieval reporting only increments a Go-side metric; it
        # does not participate in cache validity or eviction.  Keep that HTTP
        # round trip off TTFT and coalesce bursts on one daemon reporter.
        self._async_retrieval_metrics = bool(
            extra.get("disk_cache_async_retrieval_metrics", True)
        )

        # ── Stable namespace ────────────────────────────────────────
        model_config = getattr(vllm_config, "model_config", None)
        if model_config is not None:
            model_hash = getattr(model_config, "compute_hash", None)
            if model_hash is not None:
                model_id = model_hash()
            else:
                model_id = str(getattr(model_config, "model", "unknown"))
        else:
            model_id = "unknown"

        parallel_config = getattr(vllm_config, "parallel_config", None)
        tp_size = getattr(parallel_config, "tensor_parallel_size", 1)
        pp_size = getattr(parallel_config, "pipeline_parallel_size", 1)
        world_size = tp_size * pp_size

        cache_dtype = str(
            getattr(
                getattr(vllm_config, "cache_config", None),
                "cache_dtype",
                "auto",
            )
        )

        ns_parts = {
            "version": "v2",
            "strategy": strategy_name,
            "model": model_id,
            "dtype": cache_dtype,
            "block_size": self._block_size,
            "chunk_size": self._chunk_size,
            "tp": tp_size,
            "pp": pp_size,
            "world_size": world_size,
        }
        self._namespace = hashlib.sha256(
            json.dumps(ns_parts, sort_keys=True).encode()
        ).hexdigest()

        # ── Shard identity ──────────────────────────────────────────
        rank = getattr(parallel_config, "rank", 0)
        tp_rank = getattr(parallel_config, "tensor_parallel_rank", rank)
        pp_rank = getattr(parallel_config, "pipeline_parallel_rank", 0)
        self._shard = f"tp{tp_rank}-pp{pp_rank}"

        # Required shards for cache hits
        required_str = extra.get("disk_cache_required_shards", None)
        if required_str:
            self._required_shards = [
                s.strip() for s in required_str.split(",") if s.strip()
            ]
        else:
            self._required_shards = [self._shard]

        # ── Layer tracking ──────────────────────────────────────────
        self._expected_layers: list[str] = []
        groups = (
            list(getattr(kv_cache_config, "kv_cache_groups", []) or [])
            if kv_cache_config is not None
            else []
        )
        if len(groups) > 1:
            raise ValueError(
                "DiskCacheConnector v2 currently supports exactly one "
                "kv_cache_group; multi-group KV slot mapping is not supported"
            )
        if groups:
            seen: set[str] = set()
            for lname in getattr(groups[0], "layer_names", []) or []:
                if lname not in seen:
                    seen.add(lname)
                    self._expected_layers.append(lname)

        # ── Runtime state ───────────────────────────────────────────
        self._vllm_config = vllm_config
        self._connected = self._health_check()
        if self._shared_cache and not self._connected:
            raise RuntimeError(
                "shared cache metadata service is unavailable or does not match "
                f"shared cache id {self._shared_cache_id!r}"
            )

        # Per-request pending loads (keyed by request_id)
        self._pending_loads: dict[str, _PendingLoadSpec] = {}
        # Per-request trackers for chunk key management
        self._request_trackers: dict[str, _RequestTracker] = {}
        self._unfinished_requests: dict[str, Any] = {}
        # Active chunk-object writers (keyed by (req_id, chunk_key))
        self._active_writers: dict[tuple[str, str], _WriterState] = {}
        # Block IDs with load errors
        self._block_ids_with_load_errors: set[int] = set()
        self._registered_kv_caches: dict[str, torch.Tensor] = {}
        # Diagnostics for the most recent recoverable load failure.
        self._load_error: Optional[str] = None
        self._failed_load_request_ids: set[str] = set()
        self._resolved_object_paths: OrderedDict[str, Path] = OrderedDict()
        self._object_layouts: OrderedDict[Path, tuple[Any, list[Any]]] = OrderedDict()
        self._object_layout_lock = Lock()
        self._posix_host_objects: OrderedDict[
            Path, tuple[dict[str, torch.Tensor], int]
        ] = OrderedDict()
        self._posix_host_cache_bytes = 0
        self._posix_host_cache_lock = Lock()
        self._posix_pinned_staging_warmed = False
        self._retrieval_report_lock = Lock()
        self._pending_retrieved_chunks = 0
        self._retrieval_report_wakeup = Event()
        self._retrieval_report_thread: Optional[Thread] = None
        self._compiled_transfer = None
        if self._compiled_transfer_requested:
            try:
                self._compiled_transfer = create_compiled_multi_layer_transfer()
            except Exception as exc:
                logger.warning(
                    "Compiled multi-layer KV transfer unavailable; using "
                    "PyTorch fallback: %s",
                    exc,
                    exc_info=True,
                )
        self.node_id = socket.gethostname()
        if not self._shared_cache:
            self.cache_root.mkdir(parents=True, exist_ok=True)
        # Resolve this once.  Per-request loading validates every object path
        # against it; resolving the root again for every 256-token object was
        # visible on remote filesystems in the cache-hit critical path.
        self._cache_root_resolved = self.cache_root.resolve()

        if self._connected:
            logger.info(
                "DiskCacheConnector v2 ready: cache=%s engine=%s "
                "bs=%d chunk=%dtok strategy=%s namespace=%.16s "
                "shard=%s required=%s backend=%s posix_load_workers=%d "
                "posix_inject_batch_chunks=%d resolved_object_path_cache_entries=%d "
                "object_layout_cache_entries=%d "
                "posix_host_cache_gib=%.3f "
                "posix_pinned_staging_mib=%d "
                "compiled_transfer=%s "
                "gds_load_workers=%d "
                "gds_staging_buffer_mib=%d load_profile=%s "
                "async_retrieval_metrics=%s",
                self.cache_root,
                self.go_addr,
                self._block_size,
                self._chunk_size,
                strategy_name,
                self._namespace,
                self._shard,
                self._required_shards,
                type(self._storage).__name__,
                self._posix_load_workers,
                self._posix_inject_batch_chunks,
                self._resolved_object_path_cache_capacity,
                self._object_layout_cache_capacity,
                self._posix_host_cache_capacity_bytes / 1024**3,
                self._posix_pinned_staging_capacity_bytes // 1024**2,
                self._compiled_transfer is not None,
                self._gds_load_workers,
                self._gds_staging_buffer_mib,
                self._load_profile,
                self._async_retrieval_metrics,
            )

    # ── Internal helpers ─────────────────────────────────────────────

    def _is_disk_cache_meta(self, meta: Any) -> bool:
        return isinstance(meta, DiskCacheMeta)

    def _resolve_device(self, target_tensor: torch.Tensor) -> str:
        if self.target_device != "auto":
            return self.target_device
        return str(target_tensor.device)

    def _build_slot_mapping(
        self,
        block_ids: list[int],
        block_size: int,
        num_tokens: int,
    ) -> torch.Tensor:
        """Build a slot-mapping tensor from block IDs and token count.

        Validates that the allocated blocks can accommodate the requested
        number of tokens.  Raises ``ValueError`` on invalid inputs.
        """
        if num_tokens < 0:
            raise ValueError(f"num_tokens must be >= 0, got {num_tokens}")
        if not block_ids or num_tokens == 0:
            return torch.empty(0, dtype=torch.long)
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")
        max_slots = len(block_ids) * block_size
        if max_slots < num_tokens:
            raise ValueError(
                f"Not enough slots: {len(block_ids)} blocks * {block_size} "
                f"block_size = {max_slots} slots, but {num_tokens} tokens "
                f"requested"
            )
        block_offsets = torch.arange(0, block_size)
        mapping = (
            block_offsets.view(1, block_size)
            + torch.tensor(block_ids).view(len(block_ids), 1) * block_size
        ).flatten()[:num_tokens]
        return mapping

    def _get_token_ids(self, request: Any) -> list[int]:
        """Return the complete request token sequence when available."""
        ids = getattr(request, "all_token_ids", None)
        if ids is not None:
            return list(ids)
        ids = getattr(request, "prompt_token_ids", None)
        if ids is not None:
            return list(ids)
        return []

    def _get_req_id(self, request: Any) -> Optional[str]:
        """Extract request ID from a scheduler or worker request object."""
        rid = getattr(request, "request_id", None)
        if rid is not None:
            return rid
        rid = getattr(request, "req_id", None)
        if rid is not None:
            return rid
        return None

    def on_new_request(self, request: Any) -> None:
        """Retain the scheduler request for later cached-request updates."""
        req_id = self._get_req_id(request)
        if req_id is not None:
            self._unfinished_requests[req_id] = request

    def _extract_block_ids(self, blocks: Any) -> list[int]:
        """Extract integer block IDs for the first KV-cache group.

        vLLM 0.25.1 represents block IDs as ``tuple[list[int], ...]``;
        the outer dimension is the KV-cache group. Cascade currently maps one
        request object per group-0 paged cache, so deliberately selects the
        first group instead of flattening groups into invalid physical slots.
        """
        if blocks is None:
            return []

        get_ids = getattr(blocks, "get_block_ids", None)
        if callable(get_ids):
            try:
                blocks = get_ids()
            except Exception:
                logger.warning("Failed to get vLLM block IDs", exc_info=True)
                return []

        if not isinstance(blocks, (list, tuple)) or not blocks:
            return []

        first = blocks[0]
        group = first if isinstance(first, (list, tuple)) else blocks
        if not all(isinstance(block_id, int) for block_id in group):
            logger.warning("Ignoring non-integer vLLM block IDs: %r", group)
            return []
        return list(group)

    def _get_num_scheduled(
        self,
        scheduler_output: Any,
        req_id: str,
        default: int = 0,
    ) -> int:
        """Get the number of scheduled tokens for a request.

        Tries ``num_scheduled_tokens`` dict first, then falls back
        to a default.
        """
        ns = getattr(scheduler_output, "num_scheduled_tokens", None)
        if ns is None:
            return default
        if isinstance(ns, dict):
            return ns.get(req_id, default)
        return default

    def _get_block_ids_for_cached(
        self,
        new_block_ids: list[Any],
        index: int,
    ) -> list[int]:
        """Return group-0 block IDs for one cached scheduler request."""
        if index < 0 or index >= len(new_block_ids):
            return []
        return self._extract_block_ids(new_block_ids[index])

    def _store_end_for_step(
        self,
        computed_after_step: int,
        prompt_len: int,
    ) -> int:
        """Return the prefix length that is safe to publish after this step."""
        if computed_after_step <= 0 or prompt_len <= 1:
            return 0
        computed_after_step = min(computed_after_step, prompt_len)
        if computed_after_step >= prompt_len:
            # Persist the full prompt. On a full hit the last token remains
            # logically cached, while the scheduler intentionally restores
            # only the preceding KV state before executing it for logits.
            return prompt_len
        return computed_after_step // self._chunk_size * self._chunk_size

    def _new_store_descriptors(
        self,
        tracker: _RequestTracker,
        store_end: int,
    ) -> list[ChunkDescriptor]:
        if store_end <= 0:
            return []
        descriptors = self._strategy.describe(
            tracker.token_ids, self._namespace, end=store_end
        )
        # ``logical_cached_tokens`` is populated by the current connector for
        # full-prompt hits. Retain the physical boundary as a fallback for
        # resumed/legacy tracker state that predates that field.
        cached_boundary = max(
            tracker.logical_cached_tokens, tracker.external_cached_tokens
        )
        return [
            desc
            for desc in descriptors
            if desc.end > cached_boundary
            and desc.key not in tracker.committed_chunk_keys
        ]

    def _find_req_id_for_key(self, key: str) -> Optional[str]:
        """Search active writers for a request ID by chunk key."""
        for (req_id, ck), ws in self._active_writers.items():
            if ck == key:
                return req_id
        for req_id, tracker in self._request_trackers.items():
            if key in tracker.store_chunks:
                return req_id
        return None

    def _report_chunks_retrieved_async(self, count: int) -> None:
        """Coalesce metric-only retrieval reports outside the TTFT path."""
        if count <= 0:
            return

        start_thread = False
        with self._retrieval_report_lock:
            self._pending_retrieved_chunks += count
            if self._retrieval_report_thread is None:
                self._retrieval_report_thread = Thread(
                    target=self._drain_retrieval_reports,
                    name="cascade-retrieval-metrics",
                    daemon=True,
                )
                start_thread = True
        if start_thread:
            self._retrieval_report_thread.start()
        self._retrieval_report_wakeup.set()

    def _drain_retrieval_reports(self) -> None:
        """Run one persistent metric reporter for this connector instance."""
        while True:
            self._retrieval_report_wakeup.wait()
            while True:
                with self._retrieval_report_lock:
                    pending = self._pending_retrieved_chunks
                    self._pending_retrieved_chunks = 0
                    if pending <= 0:
                        # Clearing under the same lock used by producers
                        # prevents a set/clear race from losing a wakeup.
                        self._retrieval_report_wakeup.clear()
                        break
                try:
                    self._go.chunks_retrieved(pending)
                except Exception as exc:
                    # This endpoint only feeds counters.  Match/resolve and
                    # object durability remain authoritative if it is down.
                    logger.debug("chunks_retrieved call failed: %s", exc)

    # ── Load implementation ──────────────────────────────────────────

    def _load_request_kv(
        self,
        req: _ReqMeta,
        forward_context: Any,
        attn_metadata: Any,
        layer_cache: dict[str, torch.Tensor],
    ) -> None:
        """Load KV for one request from chunk objects on disk."""
        if req.load_tokens <= 0 or not req.descriptors:
            return

        # Unit-test and third-party connector shims sometimes construct this
        # mixin without running its concrete connector initializer.
        profile_enabled = bool(getattr(self, "_load_profile", False))
        profile_started = time.perf_counter() if profile_enabled else 0.0
        resolve_seconds = 0.0
        setup_seconds = 0.0
        layout_seconds = 0.0
        prepare_seconds = 0.0
        io_wait_seconds = 0.0
        io_worker_seconds = 0.0
        materialize_seconds = 0.0
        inject_seconds = 0.0
        gpu_sync_seconds = 0.0
        report_seconds = 0.0
        posix_host_hits = 0

        # Resolve chunk paths via Go engine — preserve descriptor order
        seen_keys: set[str] = set()
        keys_to_resolve: list[str] = []
        for d in req.descriptors:
            if d.end <= req.load_tokens and d.key not in seen_keys:
                seen_keys.add(d.key)
                keys_to_resolve.append(d.key)
        if not keys_to_resolve:
            return

        try:
            resolve_started = time.perf_counter() if profile_enabled else 0.0
            objects = list(req.resolved_objects)
            if not objects:
                objects = self._go.resolve_chunks(
                    req.namespace,
                    keys_to_resolve,
                    req.shard,
                )
            if profile_enabled:
                resolve_seconds = time.perf_counter() - resolve_started
        except Exception as exc:
            logger.error(
                "resolve_chunks failed for request %s: %s",
                req.request_id,
                exc,
            )
            self._record_load_error(req)
            self._load_error = str(exc)
            raise RuntimeError(f"resolve_chunks failed for {req.request_id}: {exc}")

        setup_started = time.perf_counter() if profile_enabled else 0.0

        # Build key→object map; reject duplicate or unexpected entries.
        expected_keys = set(keys_to_resolve)
        obj_by_key: dict[str, dict[str, Any]] = {}
        for obj in objects:
            k = obj.get("key", "")
            if k not in expected_keys:
                self._record_load_error(req)
                self._load_error = f"unexpected resolve key {k}"
                raise RuntimeError(
                    f"resolve_chunks returned unexpected key {k} "
                    f"for {req.request_id}"
                )
            if k in obj_by_key:
                logger.error(
                    "resolve_chunks returned duplicate key %s for request %s",
                    k,
                    req.request_id,
                )
                self._record_load_error(req)
                self._load_error = f"duplicate resolve key {k}"
                raise RuntimeError(
                    f"resolve_chunks returned duplicate key {k} "
                    f"for {req.request_id}"
                )
            obj_by_key[k] = obj

        if set(obj_by_key) != expected_keys:
            missing_keys = sorted(expected_keys - set(obj_by_key))
            self._record_load_error(req)
            self._load_error = f"missing resolve keys {missing_keys}"
            raise RuntimeError(
                f"No resolved object for keys {missing_keys} "
                f"in request {req.request_id}"
            )

        # Build slot mapping for injection
        slot_mapping = self._build_slot_mapping(
            block_ids=req.block_ids,
            block_size=req.block_size,
            num_tokens=req.load_tokens,
        )
        if slot_mapping.numel() == 0:
            logger.error("Empty slot mapping for load request %s", req.request_id)
            self._record_load_error(req)
            self._load_error = f"empty slot mapping for {req.request_id}"
            raise RuntimeError(f"Empty slot mapping for load request {req.request_id}")

        num_loaded_objects = 0
        target_dtype: Optional[torch.dtype] = None
        load_device = "cpu"

        # Determine target dtype and device from the first available layer.
        for lname in self._expected_layers:
            kv = layer_cache.get(lname)
            if kv is not None:
                target_dtype = kv.dtype
                load_device = self._resolve_device(kv)
                break

        # ``descriptors`` can clip the final source object by one token on a
        # full logical hit. Keep the original descriptor for header and shape
        # validation, then inject only the physically requested slice.
        source_by_key = {
            desc.key: desc for desc in (req.source_descriptors or req.descriptors)
        }
        cache_root_resolved = getattr(self, "_cache_root_resolved", None)
        if cache_root_resolved is None:
            cache_root_resolved = self.cache_root.resolve()

        # Validate all resolved paths before issuing any disk I/O.  Besides
        # keeping error behavior deterministic, this lets POSIX prefetches
        # safely overlap only the read half of the load path.
        resolved_entries: list[tuple[ChunkDescriptor, ChunkDescriptor, str, Path]] = []
        for desc in req.descriptors:
            if desc.end > req.load_tokens:
                continue
            key = desc.key
            source_desc = source_by_key.get(key)
            if (
                source_desc is None
                or source_desc.index != desc.index
                or source_desc.start != desc.start
                or source_desc.end < desc.end
            ):
                self._record_load_error(req)
                self._load_error = f"invalid source descriptor for key {key}"
                raise RuntimeError(
                    f"Invalid source descriptor for key={key} "
                    f"request={req.request_id}"
                )
            obj_info = obj_by_key.get(key)
            if obj_info is None:
                logger.error(
                    "resolve_chunks: no object for key=%s request=%s",
                    key,
                    req.request_id,
                )
                self._record_load_error(req)
                self._load_error = f"missing object for key {key}"
                raise RuntimeError(
                    f"No resolved object for key={key} request={req.request_id}"
                )

            file_path = obj_info.get("file_path", "")
            if not file_path:
                logger.error(
                    "resolve_chunks: empty file_path for key=%s request=%s",
                    key,
                    req.request_id,
                )
                self._invalidate_chunk(req.namespace, key, req.shard)
                self._record_load_error(req)
                self._load_error = f"empty file_path for key {key}"
                raise RuntimeError(
                    f"Empty file_path for key={key} request={req.request_id}"
                )
            try:
                full_path = self._resolve_cache_object_path(
                    file_path,
                    cache_root_resolved,
                )
            except ValueError as exc:
                self._invalidate_chunk(req.namespace, key, req.shard)
                self._record_load_error(req)
                self._load_error = f"file_path escapes cache root: {file_path}"
                raise RuntimeError(
                    f"Resolved file_path escapes cache root for key={key}"
                ) from exc

            resolved_entries.append((desc, source_desc, key, full_path))

        if profile_enabled:
            setup_seconds = time.perf_counter() - setup_started

        def fail_load(
            desc: ChunkDescriptor,
            key: str,
            full_path: Path,
            exc: Exception,
        ) -> None:
            logger.error(
                "load_chunk_object failed: %s (key=%s): %s",
                full_path,
                key,
                exc,
            )
            if isinstance(exc, (OSError, ValueError)):
                self._invalidate_chunk(req.namespace, key, req.shard, full_path)
            self._record_load_error(req)
            self._load_error = str(exc)
            raise RuntimeError(f"load_chunk_object failed for key={key}: {exc}")

        def validate_and_prepare_tensors(
            desc: ChunkDescriptor,
            source_desc: ChunkDescriptor,
            key: str,
            full_path: Path,
            header: Any,
            tensors: dict[str, torch.Tensor],
            *,
            host_prefetched: bool,
        ) -> tuple[dict[str, torch.Tensor], int]:
            # Validate the immutable object contract before injecting any
            # layer.  Read-ahead failures therefore never cause a partial
            # object injection.
            try:
                self._validate_chunk_header(header, key, source_desc)
            except Exception:
                self._invalidate_chunk(req.namespace, key, req.shard, full_path)
                raise

            stored_tokens = source_desc.end - source_desc.start
            requested_tokens = desc.end - desc.start
            if requested_tokens <= 0 or requested_tokens > stored_tokens:
                err_msg = (
                    f"Invalid requested range [{desc.start}, {desc.end}) "
                    f"for source [{source_desc.start}, {source_desc.end})"
                )
                self._record_load_error(req)
                self._load_error = err_msg
                raise RuntimeError(err_msg)

            # First validate the immutable on-disk object shape.  For a full
            # prompt hit, then discard its final KV token before H2D/injection
            # so vLLM can recompute it for the first-token logits.
            for lname in self._expected_layers:
                tensor = tensors.get(lname)
                if tensor is None:
                    err_msg = f"Layer {lname} not found in chunk {key}"
                    self._record_load_error(req)
                    self._load_error = err_msg
                    raise RuntimeError(err_msg)
                if tensor.size(0) != stored_tokens:
                    err_msg = (
                        f"Layer {lname} in chunk {key}: expected stored "
                        f"{stored_tokens} tokens, got {tensor.size(0)}"
                    )
                    self._record_load_error(req)
                    self._load_error = err_msg
                    raise RuntimeError(err_msg)

            if requested_tokens != stored_tokens:
                tensors = {
                    lname: tensors[lname][:requested_tokens]
                    for lname in self._expected_layers
                }

            if host_prefetched:
                tensors = dict(
                    zip(
                        self._expected_layers,
                        self._storage.move_pinned_tensor_slices_to_device(
                            [tensors[lname] for lname in self._expected_layers],
                            load_device,
                        ),
                    )
                )

            return tensors, requested_tokens

        def inject_prepared_tensors(
            desc: ChunkDescriptor,
            key: str,
            tensors: dict[str, torch.Tensor],
            requested_tokens: int,
        ) -> None:
            """Scatter one already-validated chunk object into paged KV."""

            if try_compiled_inject([(desc, key, tensors, requested_tokens)]):
                return
            if attn_metadata is None:
                raise RuntimeError(
                    "Full-graph KV restore requires a compatible compiled "
                    "Triton transfer layout"
                )

            # Validate and inject each layer.
            for lname in self._expected_layers:
                tensor = tensors.get(lname)
                if tensor is None:
                    err_msg = f"Layer {lname} not found in chunk {key}"
                    self._record_load_error(req)
                    self._load_error = err_msg
                    raise RuntimeError(err_msg)
                if tensor.size(0) != requested_tokens:
                    err_msg = (
                        f"Layer {lname} in chunk {key}: expected "
                        f"{requested_tokens} tokens, got {tensor.size(0)}"
                    )
                    self._record_load_error(req)
                    self._load_error = err_msg
                    raise RuntimeError(err_msg)

                kv_tensor = layer_cache.get(lname)
                if kv_tensor is None:
                    err_msg = f"Layer {lname} not in forward context cache"
                    self._record_load_error(req)
                    self._load_error = err_msg
                    raise RuntimeError(err_msg)

                if target_dtype is not None and tensor.dtype != target_dtype:
                    tensor = tensor.to(target_dtype)

                # Slot mapping for this chunk's token range.
                chunk_slot_mapping = slot_mapping[desc.start : desc.end]
                if chunk_slot_mapping.numel() != tensor.size(0):
                    err_msg = (
                        f"Slot mapping size mismatch for layer {lname}: "
                        f"{chunk_slot_mapping.numel()} != {tensor.size(0)}"
                    )
                    self._record_load_error(req)
                    self._load_error = err_msg
                    raise RuntimeError(err_msg)

                layer_attn = self._get_layer_attn_meta(attn_metadata, lname)
                inject_kv_into_layer(
                    kv_tensor,
                    tensor,
                    chunk_slot_mapping,
                    layer_attn,
                    self._block_size,
                )

        compiled_transfer = getattr(self, "_compiled_transfer", None)
        compiled_triton_layout = False
        if compiled_transfer is not None and load_device.startswith("cuda"):
            if attn_metadata is None:
                # Full CUDA graph replay does not expose per-request attention
                # metadata to connector hooks.  Registered KV tensors still
                # carry the exact Triton layout contract, and the compiled
                # transfer performs a second complete validation before it
                # launches.  Never use the metadata-dependent PyTorch fallback
                # in this mode.
                compiled_triton_layout = all(
                    kv is not None
                    and kv.ndim >= 4
                    and kv.shape[1] == 2
                    and kv.shape[2] == self._block_size
                    and kv.is_contiguous()
                    for kv in (
                        layer_cache.get(lname) for lname in self._expected_layers
                    )
                )
            else:
                try:
                    from vllm.v1.attention.backends.triton_attn import (
                        TritonAttentionMetadata,
                    )

                    compiled_triton_layout = all(
                        isinstance(
                            self._get_layer_attn_meta(attn_metadata, lname),
                            TritonAttentionMetadata,
                        )
                        for lname in self._expected_layers
                    )
                except (ImportError, TypeError):
                    compiled_triton_layout = False

        def try_compiled_inject(
            prepared_objects: list[
                tuple[
                    ChunkDescriptor,
                    str,
                    dict[str, torch.Tensor],
                    int,
                ]
            ],
        ) -> bool:
            """Use one native all-layer scatter per prepared object."""
            if not compiled_triton_layout or compiled_transfer is None:
                return False

            for desc, key, tensors, requested_tokens in prepared_objects:
                sources = []
                destinations = []
                for lname in self._expected_layers:
                    source = tensors.get(lname)
                    destination = layer_cache.get(lname)
                    if source is None or destination is None:
                        return False
                    if source.size(0) != requested_tokens:
                        return False
                    sources.append(source)
                    destinations.append(destination)

                chunk_slot_mapping = slot_mapping[desc.start : desc.end]
                if chunk_slot_mapping.numel() != requested_tokens:
                    return False
                try:
                    compiled_transfer.inject(
                        sources,
                        destinations,
                        chunk_slot_mapping,
                        self._block_size,
                    )
                except ValueError as exc:
                    logger.debug(
                        "Compiled transfer layout rejected chunk %s; using "
                        "PyTorch fallback: %s",
                        key,
                        exc,
                    )
                    return False
            return True

        def validate_and_inject(
            desc: ChunkDescriptor,
            source_desc: ChunkDescriptor,
            key: str,
            full_path: Path,
            header: Any,
            tensors: dict[str, torch.Tensor],
            *,
            host_prefetched: bool,
        ) -> None:
            tensors, requested_tokens = validate_and_prepare_tensors(
                desc,
                source_desc,
                key,
                full_path,
                header,
                tensors,
                host_prefetched=host_prefetched,
            )
            inject_prepared_tensors(desc, key, tensors, requested_tokens)

        def validate_and_inject_profiled(
            desc: ChunkDescriptor,
            source_desc: ChunkDescriptor,
            key: str,
            full_path: Path,
            header: Any,
            tensors: dict[str, torch.Tensor],
            *,
            host_prefetched: bool,
        ) -> None:
            """Include H2D and paged-KV injection in an optional timer."""
            nonlocal inject_seconds
            if not profile_enabled:
                validate_and_inject(
                    desc,
                    source_desc,
                    key,
                    full_path,
                    header,
                    tensors,
                    host_prefetched=host_prefetched,
                )
                return
            inject_started = time.perf_counter()
            try:
                validate_and_inject(
                    desc,
                    source_desc,
                    key,
                    full_path,
                    header,
                    tensors,
                    host_prefetched=host_prefetched,
                )
            finally:
                inject_seconds += time.perf_counter() - inject_started

        def validate_and_inject_posix_batch(
            entries: list[
                tuple[
                    ChunkDescriptor,
                    ChunkDescriptor,
                    str,
                    Path,
                    Any,
                    dict[str, torch.Tensor],
                ]
            ],
        ) -> None:
            """Validate adjacent POSIX objects, then scatter per layer.

            The normal object layout keeps every layer of a 256-token chunk
            together for a coalesced disk read/H2D.  Once a bounded group is
            resident on GPU, flattening it layer-major changes  ``chunks ×
            layers`` paged-KV scatters into just ``layers`` scatters while
            preserving the original object validation barrier.
            """
            prepared: list[
                tuple[ChunkDescriptor, str, dict[str, torch.Tensor], int]
            ] = []
            for desc, source_desc, key, full_path, header, tensors in entries:
                prepared_tensors, requested_tokens = validate_and_prepare_tensors(
                    desc,
                    source_desc,
                    key,
                    full_path,
                    header,
                    tensors,
                    host_prefetched=True,
                )
                prepared.append((desc, key, prepared_tensors, requested_tokens))

            if try_compiled_inject(prepared):
                return
            if attn_metadata is None:
                raise RuntimeError(
                    "Full-graph KV restore requires a compatible compiled "
                    "Triton transfer layout"
                )

            if len(prepared) == 1:
                desc, key, tensors, requested_tokens = prepared[0]
                inject_prepared_tensors(desc, key, tensors, requested_tokens)
                return

            # Cache hits are contiguous prefix descriptors in the common case;
            # retain a concatenation fallback so the microbatch stays correct
            # if a future key strategy supplies gaps.
            first_desc = prepared[0][0]
            last_desc = prepared[-1][0]
            contiguous = all(
                previous[0].end == current[0].start
                for previous, current in zip(prepared, prepared[1:])
            )
            if contiguous:
                batch_slot_mapping = slot_mapping[first_desc.start : last_desc.end]
            else:
                batch_slot_mapping = torch.cat(
                    [slot_mapping[desc.start : desc.end] for desc, *_ in prepared],
                    dim=0,
                )
            expected_tokens = sum(item[3] for item in prepared)
            if batch_slot_mapping.numel() != expected_tokens:
                err_msg = (
                    "Batch slot mapping size mismatch: "
                    f"{batch_slot_mapping.numel()} != {expected_tokens}"
                )
                self._record_load_error(req)
                self._load_error = err_msg
                raise RuntimeError(err_msg)

            batch_keys = ",".join(key for _, key, _, _ in prepared)
            for lname in self._expected_layers:
                layer_tensors = [tensors[lname] for _, _, tensors, _ in prepared]
                tensor = torch.cat(layer_tensors, dim=0)
                if tensor.size(0) != expected_tokens:
                    err_msg = (
                        f"Layer {lname} in chunk batch {batch_keys}: expected "
                        f"{expected_tokens} tokens, got {tensor.size(0)}"
                    )
                    self._record_load_error(req)
                    self._load_error = err_msg
                    raise RuntimeError(err_msg)

                kv_tensor = layer_cache.get(lname)
                if kv_tensor is None:
                    err_msg = f"Layer {lname} not in forward context cache"
                    self._record_load_error(req)
                    self._load_error = err_msg
                    raise RuntimeError(err_msg)
                if target_dtype is not None and tensor.dtype != target_dtype:
                    tensor = tensor.to(target_dtype)

                layer_attn = self._get_layer_attn_meta(attn_metadata, lname)
                inject_kv_into_layer(
                    kv_tensor,
                    tensor,
                    batch_slot_mapping,
                    layer_attn,
                    self._block_size,
                )

        def validate_and_inject_posix_batch_profiled(
            entries: list[
                tuple[
                    ChunkDescriptor,
                    ChunkDescriptor,
                    str,
                    Path,
                    Any,
                    dict[str, torch.Tensor],
                ]
            ],
        ) -> None:
            nonlocal inject_seconds
            if not profile_enabled:
                validate_and_inject_posix_batch(entries)
                return
            inject_started = time.perf_counter()
            try:
                validate_and_inject_posix_batch(entries)
            finally:
                inject_seconds += time.perf_counter() - inject_started

        use_posix_read_ahead = (
            isinstance(self._storage, PosixBackend)
            and load_device.startswith("cuda")
            and len(resolved_entries) > 1
            and getattr(self, "_posix_load_workers", 4) > 1
        )
        use_gds_read_ahead = (
            isinstance(self._storage, NvFileBackend)
            and load_device.startswith("cuda")
            and len(resolved_entries) > 1
            and getattr(self, "_gds_load_workers", 6) > 1
        )

        if use_posix_read_ahead:
            # A bounded producer/consumer pipeline gives the next reads time
            # to run while this thread queues H2D copies and injects the
            # current object.  Do not submit every chunk at once: long
            # prompts would otherwise pin an unbounded amount of host memory.
            def read_posix_object(
                path: Path,
            ) -> tuple[Any, dict[str, torch.Tensor], float, float, bool]:
                layout_started = time.perf_counter() if profile_enabled else 0.0
                header, specs = self._get_object_layout(path)
                worker_layout_seconds = (
                    time.perf_counter() - layout_started if profile_enabled else 0.0
                )
                cached_tensors = self._get_posix_host_object(path)
                if cached_tensors is not None:
                    return (
                        header,
                        cached_tensors,
                        0.0,
                        worker_layout_seconds,
                        True,
                    )
                read_started = time.perf_counter() if profile_enabled else 0.0
                loaded = self._storage.load_tensor_slices_to_pinned_cpu(path, specs)
                if len(loaded) != len(specs):
                    raise RuntimeError(
                        "Storage backend returned "
                        f"{len(loaded)} tensors for {len(specs)} specs"
                    )
                tensors = dict(
                    zip(
                        self._expected_layers,
                        loaded,
                    )
                )
                tensors = self._remember_posix_host_object(path, tensors)
                read_seconds = (
                    time.perf_counter() - read_started if profile_enabled else 0.0
                )
                return header, tensors, read_seconds, worker_layout_seconds, False

            worker_count = min(
                getattr(self, "_posix_load_workers", 4),
                len(resolved_entries),
            )
            inject_batch_size = min(
                getattr(self, "_posix_inject_batch_chunks", 1),
                len(resolved_entries),
            )
            # A fully resident host-cache hit has no blocking I/O to overlap.
            # Sending its 16 tiny LRU lookups through a newly-created thread
            # pool costs more than doing them directly on the caller thread.
            # Probe the complete request first and retain the worker pipeline
            # unchanged for every cold or mixed hit.
            hot_entries: list[
                tuple[
                    ChunkDescriptor,
                    ChunkDescriptor,
                    str,
                    Path,
                    Any,
                    dict[str, torch.Tensor],
                ]
            ] = []
            if getattr(self, "_posix_host_cache_capacity_bytes", 0) > 0:
                for desc, source_desc, key, full_path in resolved_entries:
                    try:
                        layout_started = time.perf_counter() if profile_enabled else 0.0
                        header, _ = self._get_object_layout(full_path)
                        if profile_enabled:
                            layout_seconds += time.perf_counter() - layout_started
                        cached_tensors = self._get_posix_host_object(full_path)
                    except Exception as exc:
                        fail_load(desc, key, full_path, exc)
                    if cached_tensors is None:
                        hot_entries = []
                        break
                    hot_entries.append(
                        (
                            desc,
                            source_desc,
                            key,
                            full_path,
                            header,
                            cached_tensors,
                        )
                    )

            if len(hot_entries) == len(resolved_entries):
                posix_host_hits += len(hot_entries)
                for batch_start in range(0, len(hot_entries), inject_batch_size):
                    batch_entries = hot_entries[
                        batch_start : batch_start + inject_batch_size
                    ]
                    if inject_batch_size == 1:
                        (
                            desc,
                            source_desc,
                            key,
                            full_path,
                            header,
                            tensors,
                        ) = batch_entries[0]
                        validate_and_inject_profiled(
                            desc,
                            source_desc,
                            key,
                            full_path,
                            header,
                            tensors,
                            host_prefetched=True,
                        )
                    else:
                        validate_and_inject_posix_batch_profiled(batch_entries)
                    num_loaded_objects += len(batch_entries)
            else:
                entry_iter = iter(resolved_entries)
                with ThreadPoolExecutor(
                    max_workers=worker_count,
                    thread_name_prefix="cascade-posix-read",
                ) as executor:
                    pending = deque()
                    for _ in range(worker_count):
                        entry = next(entry_iter, None)
                        if entry is None:
                            break
                        desc, source_desc, key, full_path = entry
                        future = executor.submit(
                            read_posix_object,
                            full_path,
                        )
                        pending.append((entry, future))

                    batch_entries = []
                    while pending:
                        (
                            desc,
                            source_desc,
                            key,
                            full_path,
                        ), future = pending.popleft()
                        try:
                            wait_started = (
                                time.perf_counter() if profile_enabled else 0.0
                            )
                            (
                                header,
                                tensors,
                                worker_seconds,
                                worker_layout_seconds,
                                host_cache_hit,
                            ) = future.result()
                            if profile_enabled:
                                io_wait_seconds += time.perf_counter() - wait_started
                                io_worker_seconds += worker_seconds
                                layout_seconds += worker_layout_seconds
                                posix_host_hits += int(host_cache_hit)
                        except Exception as exc:
                            fail_load(desc, key, full_path, exc)

                        # Submit before injecting so the worker can overlap the
                        # following POSIX read with this object's GPU work.
                        entry = next(entry_iter, None)
                        if entry is not None:
                            next_path = entry[3]
                            next_future = executor.submit(
                                read_posix_object,
                                next_path,
                            )
                            pending.append((entry, next_future))

                        if inject_batch_size == 1:
                            validate_and_inject_profiled(
                                desc,
                                source_desc,
                                key,
                                full_path,
                                header,
                                tensors,
                                host_prefetched=True,
                            )
                            num_loaded_objects += 1
                        else:
                            batch_entries.append(
                                (
                                    desc,
                                    source_desc,
                                    key,
                                    full_path,
                                    header,
                                    tensors,
                                )
                            )
                            if len(batch_entries) >= inject_batch_size or not pending:
                                validate_and_inject_posix_batch_profiled(batch_entries)
                                num_loaded_objects += len(batch_entries)
                                batch_entries = []
        elif use_gds_read_ahead:
            # LMCache's GDS backend allocates GPU destinations first, then
            # submits cuFile reads through four I/O workers.  Keep that split
            # here: workers never mutate vLLM KV cache; this thread validates
            # each immutable object and injects it in descriptor order.
            worker_count = min(
                getattr(self, "_gds_load_workers", 6),
                len(resolved_entries),
            )
            gds_entries: list[
                tuple[tuple[ChunkDescriptor, ChunkDescriptor, str, Path], Any, Any]
            ] = []
            max_read_nbytes = 0
            for entry in resolved_entries:
                desc, source_desc, key, full_path = entry
                try:
                    layout_started = time.perf_counter() if profile_enabled else 0.0
                    header, specs = self._get_object_layout(full_path)
                    if profile_enabled:
                        layout_seconds += time.perf_counter() - layout_started
                    max_read_nbytes = max(
                        max_read_nbytes,
                        self._storage.max_coalesced_read_nbytes(specs),
                    )
                except Exception as exc:
                    fail_load(desc, key, full_path, exc)
                gds_entries.append((entry, header, specs))

            pool = self._storage.get_gds_staging_pool(
                device=load_device,
                max_read_nbytes=max_read_nbytes,
                worker_count=worker_count,
                buffer_size_bytes=(
                    getattr(self, "_gds_staging_buffer_mib", 192) * 1024 * 1024
                ),
            )
            if pool is None:
                logger.info(
                    "GDS read-ahead: objects=%d workers=%d staging_pool=disabled",
                    len(gds_entries),
                    worker_count,
                )
            else:
                logger.info(
                    "GDS read-ahead: objects=%d workers=%d staging_pool=%dMiB "
                    "slots=%d registered=%s",
                    len(gds_entries),
                    worker_count,
                    getattr(pool, "buffer_bytes", 0) // (1024 * 1024),
                    getattr(pool, "slot_count", 0),
                    getattr(pool, "_registered", False),
                )
            prefetch_depth = min(len(gds_entries), worker_count * 2)
            entry_iter = iter(gds_entries)
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="cascade-gds-read",
            ) as executor:
                pending = deque()

                def read_gds_object(path: Path, prepared: Any) -> float:
                    read_started = time.perf_counter() if profile_enabled else 0.0
                    self._storage.read_prepared_tensor_slices(path, prepared)
                    return (
                        time.perf_counter() - read_started if profile_enabled else 0.0
                    )

                def submit_next() -> bool:
                    nonlocal prepare_seconds
                    item = next(entry_iter, None)
                    if item is None:
                        return False
                    entry, header, specs = item
                    desc, source_desc, key, full_path = entry
                    try:
                        prepare_started = (
                            time.perf_counter() if profile_enabled else 0.0
                        )
                        prepared = self._storage.prepare_tensor_slices_load(
                            specs, load_device, pool=pool
                        )
                        if profile_enabled:
                            prepare_seconds += time.perf_counter() - prepare_started
                    except Exception as exc:
                        fail_load(desc, key, full_path, exc)
                    future = executor.submit(
                        read_gds_object,
                        full_path,
                        prepared,
                    )
                    pending.append((entry, header, prepared, future))
                    return True

                try:
                    for _ in range(prefetch_depth):
                        if not submit_next():
                            break

                    while pending:
                        (
                            (desc, source_desc, key, full_path),
                            header,
                            prepared,
                            future,
                        ) = pending.popleft()
                        try:
                            wait_started = (
                                time.perf_counter() if profile_enabled else 0.0
                            )
                            worker_seconds = future.result()
                            if profile_enabled:
                                io_wait_seconds += time.perf_counter() - wait_started
                                io_worker_seconds += worker_seconds
                            materialize_started = (
                                time.perf_counter() if profile_enabled else 0.0
                            )
                            tensors = dict(
                                zip(
                                    self._expected_layers,
                                    self._storage.materialize_prepared_tensor_slices(
                                        prepared
                                    ),
                                )
                            )
                            if profile_enabled:
                                materialize_seconds += (
                                    time.perf_counter() - materialize_started
                                )
                        except Exception as exc:
                            self._storage.release_prepared_tensor_slices(
                                prepared, after_cuda_use=False
                            )
                            fail_load(desc, key, full_path, exc)

                        try:
                            validate_and_inject_profiled(
                                desc,
                                source_desc,
                                key,
                                full_path,
                                header,
                                tensors,
                                host_prefetched=False,
                            )
                        except Exception as exc:
                            del tensors
                            self._storage.release_prepared_tensor_slices(
                                prepared, after_cuda_use=True
                            )
                            fail_load(desc, key, full_path, exc)
                        else:
                            del tensors
                            self._storage.release_prepared_tensor_slices(
                                prepared, after_cuda_use=True
                            )
                            num_loaded_objects += 1
                            # The initial bounded queue keeps GDS reads in
                            # flight while this thread injects the current
                            # object. Refill it after this source buffer is
                            # protected by a CUDA completion event.
                            submit_next()
                finally:
                    # The executor waits for active cuFile calls before the
                    # pool slots are made reusable after an error path.
                    while pending:
                        _, _, prepared, future = pending.popleft()
                        try:
                            future.result()
                        except Exception:
                            pass
                        self._storage.release_prepared_tensor_slices(
                            prepared, after_cuda_use=False
                        )
        else:
            # CPU targets and explicitly single-worker GDS retain serial I/O.
            for desc, source_desc, key, full_path in resolved_entries:
                try:
                    read_started = time.perf_counter() if profile_enabled else 0.0
                    header, tensors = load_chunk_object(
                        full_path,
                        layer_names=self._expected_layers,
                        device=load_device,
                        backend=self._storage,
                    )
                    if profile_enabled:
                        read_seconds = time.perf_counter() - read_started
                        io_wait_seconds += read_seconds
                        io_worker_seconds += read_seconds
                except Exception as exc:
                    fail_load(desc, key, full_path, exc)

                validate_and_inject_profiled(
                    desc,
                    source_desc,
                    key,
                    full_path,
                    header,
                    tensors,
                    host_prefetched=False,
                )
                num_loaded_objects += 1

        # CUDA copies and paged-KV scatters are normally enqueued.  In profile
        # mode synchronize once so the request breakdown includes their real
        # device time instead of only Python dispatch time.  This is
        # diagnostic-only and mirrors the dependency the following forward has
        # on the same cache contents.
        if profile_enabled and load_device.startswith("cuda"):
            gpu_sync_started = time.perf_counter()
            torch.cuda.synchronize(device=load_device)
            gpu_sync_seconds = time.perf_counter() - gpu_sync_started

        # Report successful retrieval count to Go engine — only after ALL succeed
        if num_loaded_objects > 0:
            try:
                report_started = time.perf_counter() if profile_enabled else 0.0
                if getattr(self, "_async_retrieval_metrics", False):
                    self._report_chunks_retrieved_async(num_loaded_objects)
                else:
                    self._go.chunks_retrieved(num_loaded_objects)
                if profile_enabled:
                    report_seconds = time.perf_counter() - report_started
            except Exception as exc:
                logger.debug("chunks_retrieved call failed: %s", exc)

        if profile_enabled:
            total_seconds = time.perf_counter() - profile_started
            logger.info(
                "DiskCache load profile request=%s backend=%s objects=%d "
                "layers=%d tokens=%d resolve_ms=%.3f setup_ms=%.3f "
                "layout_ms=%.3f prepare_ms=%.3f io_wait_ms=%.3f "
                "io_worker_ms=%.3f materialize_ms=%.3f inject_ms=%.3f "
                "gpu_sync_ms=%.3f report_ms=%.3f posix_host_hits=%d "
                "total_ms=%.3f",
                req.request_id,
                type(self._storage).__name__,
                num_loaded_objects,
                len(self._expected_layers),
                req.load_tokens,
                resolve_seconds * 1000,
                setup_seconds * 1000,
                layout_seconds * 1000,
                prepare_seconds * 1000,
                io_wait_seconds * 1000,
                io_worker_seconds * 1000,
                materialize_seconds * 1000,
                inject_seconds * 1000,
                gpu_sync_seconds * 1000,
                report_seconds * 1000,
                posix_host_hits,
                total_seconds * 1000,
            )

    def _validate_chunk_header(
        self,
        header: Any,
        expected_key: str,
        desc: ChunkDescriptor,
    ) -> None:
        """Validate a chunk-object header matches expectations."""
        if getattr(header, "version", None) != 2:
            raise RuntimeError(
                f"Unsupported chunk header version: {getattr(header, 'version', None)}"
            )
        if not getattr(header, "complete", False):
            raise RuntimeError("Chunk header is incomplete")
        if header.namespace != self._namespace:
            raise RuntimeError(
                f"Chunk header namespace mismatch: "
                f"{header.namespace} != {self._namespace}"
            )
        if header.key != expected_key:
            raise RuntimeError(
                f"Chunk header key mismatch: " f"{header.key} != {expected_key}"
            )
        if header.shard != self._shard:
            raise RuntimeError(
                f"Chunk header shard mismatch: " f"{header.shard} != {self._shard}"
            )
        if header.index != desc.index:
            raise RuntimeError(
                f"Chunk header index mismatch: " f"{header.index} != {desc.index}"
            )
        if header.start != desc.start:
            raise RuntimeError(
                f"Chunk header start mismatch: " f"{header.start} != {desc.start}"
            )
        if header.end != desc.end:
            raise RuntimeError(
                f"Chunk header end mismatch: " f"{header.end} != {desc.end}"
            )

    def _invalidate_chunk(
        self,
        namespace: str,
        key: str,
        shard: str,
        path: Optional[Path] = None,
    ) -> None:
        """Remove a proven bad local object before invalidating metadata."""
        if path is not None:
            self._forget_resolved_object_path(path)
            self._forget_object_layout(path)
            self._forget_posix_host_object(path)
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning(
                    "Failed to remove bad chunk object %s; invalidating metadata anyway: %s",
                    path,
                    exc,
                )
        try:
            self._go.invalidate_chunk(namespace, key, shard)
        except Exception as exc:
            logger.warning(
                "Failed to invalidate bad chunk %s/%s/%s: %s",
                namespace,
                key,
                shard,
                exc,
            )

    def _get_object_layout(self, path: Path) -> tuple[Any, list[Any]]:
        """Return a verified header/slice layout from the bounded LRU.

        An object first discovered through Go is always opened, parsed, and
        size-checked by :func:`read_chunk_object_layout`.  Chunk objects are
        immutable after publication, so subsequent loads can safely reuse
        that result until an explicit load failure invalidates the object.
        Only the small OrderedDict critical sections are locked: independent
        first-time header reads may still run concurrently on POSIX workers.
        """
        capacity = getattr(self, "_object_layout_cache_capacity", 0)
        cache = getattr(self, "_object_layouts", None)
        lock = getattr(self, "_object_layout_lock", None)
        if capacity > 0 and cache is not None and lock is not None:
            with lock:
                cached = cache.get(path)
                if cached is not None:
                    cache.move_to_end(path)
                    return cached

        header, specs = read_chunk_object_layout(path, self._expected_layers)
        return self._remember_object_layout(path, header, specs)

    def _remember_object_layout(
        self,
        path: Path,
        header: Any,
        specs: Optional[list[Any]] = None,
    ) -> tuple[Any, list[Any]]:
        """Insert one already-verified immutable layout into the LRU."""
        if specs is None:
            specs = tensor_slice_specs_from_header(header, self._expected_layers)

        capacity = getattr(self, "_object_layout_cache_capacity", 0)
        cache = getattr(self, "_object_layouts", None)
        lock = getattr(self, "_object_layout_lock", None)
        if capacity <= 0 or cache is None or lock is None:
            return header, specs

        with lock:
            cache[path] = (header, specs)
            cache.move_to_end(path)
            while len(cache) > capacity:
                cache.popitem(last=False)
        return header, specs

    def _forget_object_layout(self, path: Path) -> None:
        """Drop a cached header/layout before removing an invalid object."""
        cache = getattr(self, "_object_layouts", None)
        lock = getattr(self, "_object_layout_lock", None)
        if cache is None or lock is None:
            return
        with lock:
            cache.pop(path, None)

    @staticmethod
    def _host_object_nbytes(tensors: dict[str, torch.Tensor]) -> int:
        """Count unique backing storages for one host-cached object."""
        storages: dict[tuple[int, int], int] = {}
        for tensor in tensors.values():
            if tensor.device.type != "cpu":
                raise ValueError("POSIX host cache accepts CPU tensors only")
            storage = tensor.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            storages[key] = storage.nbytes()
        return sum(storages.values())

    def _get_posix_host_object(
        self,
        path: Path,
    ) -> Optional[dict[str, torch.Tensor]]:
        """Return one immutable payload from the bounded host-memory LRU."""
        capacity = getattr(self, "_posix_host_cache_capacity_bytes", 0)
        cache = getattr(self, "_posix_host_objects", None)
        lock = getattr(self, "_posix_host_cache_lock", None)
        if capacity <= 0 or cache is None or lock is None:
            return None
        with lock:
            cached = cache.get(path)
            if cached is None:
                return None
            cache.move_to_end(path)
            return cached[0]

    def _remember_posix_host_object(
        self,
        path: Path,
        tensors: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Retain an immutable POSIX object while respecting the byte cap."""
        capacity = getattr(self, "_posix_host_cache_capacity_bytes", 0)
        cache = getattr(self, "_posix_host_objects", None)
        lock = getattr(self, "_posix_host_cache_lock", None)
        if capacity <= 0 or cache is None or lock is None:
            return tensors

        object_nbytes = self._host_object_nbytes(tensors)
        if object_nbytes <= 0 or object_nbytes > capacity:
            return tensors

        with lock:
            current_bytes = getattr(self, "_posix_host_cache_bytes", 0)
            replaced = cache.pop(path, None)
            if replaced is not None:
                current_bytes -= replaced[1]
            cache[path] = (tensors, object_nbytes)
            cache.move_to_end(path)
            current_bytes += object_nbytes
            while current_bytes > capacity and cache:
                _, (_, evicted_nbytes) = cache.popitem(last=False)
                current_bytes -= evicted_nbytes
            self._posix_host_cache_bytes = current_bytes
        return tensors

    def _forget_posix_host_object(self, path: Path) -> None:
        """Drop resident bytes when their durable object is invalidated."""
        cache = getattr(self, "_posix_host_objects", None)
        lock = getattr(self, "_posix_host_cache_lock", None)
        if cache is None or lock is None:
            return
        with lock:
            cached = cache.pop(path, None)
            if cached is not None:
                self._posix_host_cache_bytes = max(
                    0,
                    getattr(self, "_posix_host_cache_bytes", 0) - cached[1],
                )

    def _resolve_cache_object_path(
        self,
        file_path: str,
        cache_root_resolved: Optional[Path] = None,
    ) -> Path:
        """Return an already containment-checked immutable object path.

        The first occurrence always retains the full ``Path.resolve`` plus
        ``relative_to`` hardening check.  Later occurrences of that exact
        relative path use the process-local LRU because a chunk object is
        immutable once it is published.  Failed loads call
        :meth:`_invalidate_chunk`, which removes the cached entry before
        invalidating Go metadata.
        """
        if not isinstance(file_path, str) or not file_path:
            raise ValueError("file_path must be a non-empty string")

        cache = getattr(self, "_resolved_object_paths", None)
        if cache is not None:
            cached = cache.get(file_path)
            if cached is not None:
                cache.move_to_end(file_path)
                return cached

        root = cache_root_resolved
        if root is None:
            root = getattr(self, "_cache_root_resolved", None)
        if root is None:
            root = self.cache_root.resolve()
        try:
            full_path = (root / file_path).resolve()
            full_path.relative_to(root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(
                f"Resolved file_path escapes cache root: {file_path}"
            ) from exc

        capacity = getattr(self, "_resolved_object_path_cache_capacity", 0)
        if cache is not None and capacity > 0:
            cache[file_path] = full_path
            cache.move_to_end(file_path)
            while len(cache) > capacity:
                cache.popitem(last=False)
        return full_path

    def _forget_resolved_object_path(self, path: Path) -> None:
        """Drop cached containment results for a removed chunk object."""
        cache = getattr(self, "_resolved_object_paths", None)
        if not cache:
            return
        for relative_path, cached_path in list(cache.items()):
            if cached_path == path:
                cache.pop(relative_path, None)

    def _record_load_error(self, req: _ReqMeta) -> None:
        """Record block IDs associated with a load error."""
        for bid in req.block_ids:
            self._block_ids_with_load_errors.add(bid)

    def _abort_request_writers(self, request_id: str) -> None:
        """Abort and discard all uncommitted writers owned by one request."""
        for writer_key in [key for key in self._active_writers if key[0] == request_id]:
            writer_state = self._active_writers.pop(writer_key)
            if not writer_state.committed:
                writer_state.aborted = True
                writer_state.writer.abort()

    def _mark_load_failed(self, req: _ReqMeta, error: Exception | str) -> None:
        """Record a recoverable load failure and isolate its pending writes."""
        self._record_load_error(req)
        self._failed_load_request_ids.add(req.request_id)
        self._load_error = str(error)
        self._abort_request_writers(req.request_id)

    def _get_layer_attn_meta(
        self,
        attn_metadata: Any,
        layer_name: str,
    ) -> Any:
        """Get attention metadata for a specific layer."""
        if isinstance(attn_metadata, dict):
            return attn_metadata.get(layer_name, attn_metadata)
        return attn_metadata

    # ── Health ───────────────────────────────────────────────────────

    def _health_check(self) -> bool:
        try:
            if not self._go.health_check():
                return False
            if not getattr(self, "_shared_cache", False):
                return True
            info = self._go.cluster_info()
            return (
                info.get("api_version") == 1
                and info.get("metadata_mode") == "shared"
                and info.get("shared_cache_id")
                == getattr(self, "_shared_cache_id", "")
                and info.get("published_object_verified") is True
                and info.get("eviction_enabled") is False
            )
        except Exception:
            return False
