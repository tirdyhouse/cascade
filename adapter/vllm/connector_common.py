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

import hashlib
import json
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

from adapter.storage import create_storage_backend
from adapter.vllm.chunk_keys import (
    ChunkDescriptor,
    create_chunk_key_strategy,
)
from adapter.vllm.go_client import DiskCacheGoClient
from adapter.storage.chunk_object import (
    ChunkObjectWriter,
    load_chunk_object,
)
from adapter.vllm.tensor_ops import extract_kv_from_layer, inject_kv_into_layer

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import init_logger

logger = init_logger("vllm.disk_cache")


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
        Chunk descriptors covering the stored / loaded token range.
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


@dataclass
class _PendingLoadSpec:
    """Saved between :meth:`_get_num_new_matched_tokens` and
    :meth:`update_state_after_alloc` / :meth:`build_connector_meta`."""
    matched_tokens: int
    descriptors: list[ChunkDescriptor]
    local_cached: int
    can_load: bool = False
    allocated_block_ids: list[int] = field(default_factory=list)


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

# ── Mixin ──────────────────────────────────────────────────────────────


class DiskCacheConnectorCommonMixin:
    """Shared logic for vLLM disk-cache connectors (v2 chunk-object path).

    Intended to be mixed with ``KVConnectorBase_V1``.
    DiskCacheConnector delegates ``get_num_new_matched_tokens``
    to ``_get_num_new_matched_tokens`` and ``save_kv_layer``
    to ``_save_request_kv`` — both defined here.
    """

    @classmethod
    def requires_piecewise_for_cudagraph(
        cls, extra_config: dict[str, Any]
    ) -> bool:
        """Keep Cascade's per-layer Python hooks outside CUDA graph replay."""
        return True

    # ── Private hooks (called by connector subclasses) ─────────────────

    def _get_num_new_matched_tokens(
        self,
        request: Any,
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """Determine how many new tokens can be loaded from the disk cache.

        Match candidates cover ``tokens[:len(tokens)-1]`` (all tokens except
        the last one), including a partial tail chunk. The match response is
        accepted only when its token count, chunk count, and ordered keys all
        describe the same candidate prefix.
        """
        request_id = self._get_req_id(request)
        if request_id is not None:
            self._pending_loads.pop(request_id, None)
        if not self._connected or request_id is None:
            return (0, False)

        tokens = self._get_token_ids(request)
        if len(tokens) < 2:
            return (0, False)

        # Cover all tokens except the last one, including a partial tail chunk.
        end = len(tokens) - 1
        if end <= 0 or num_computed_tokens >= end:
            return (0, False)

        descriptors = self._strategy.describe(tokens, self._namespace, end=end)
        candidates = [
            {"key": d.key, "end_tokens": d.end}
            for d in descriptors
        ]

        try:
            result = self._go.match_chunks(
                self._namespace,
                candidates,
                self._required_shards,
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

        external = max(0, matched_tokens - num_computed_tokens)
        self._pending_loads[request_id] = _PendingLoadSpec(
            matched_tokens=matched_tokens,
            descriptors=matching_descs,
            local_cached=num_computed_tokens,
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
                chunk_tensor = extracted[desc.start:desc.end]
                expected_len = desc.end - desc.start
                if chunk_tensor.size(0) != expected_len:
                    logger.warning(
                        "_save_request_kv: chunk %s expected %d tokens, "
                        "got %d; skipping",
                        desc.key, expected_len, chunk_tensor.size(0),
                    )
                    continue

                try:
                    ws.writer.add_layer(layer_name, chunk_tensor)
                    ws.finalized_layers.add(layer_name)
                except Exception as exc:
                    logger.error(
                        "Failed to add layer %s to chunk %s: %s",
                        layer_name, desc.key, exc,
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
        expected = pending.matched_tokens - pending.local_cached
        if expected > 0 and num_external_tokens != expected:
            logger.warning(
                "update_state_after_alloc: expected %d external tokens "
                "but got %d for request %s",
                expected, num_external_tokens, request.request_id,
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
            finished_req_ids = getattr(
                scheduler_output, "finished_req_ids", None
            )
        for req_id in finished_req_ids or ():
            self._pending_loads.pop(req_id, None)
            self._request_trackers.pop(req_id, None)
            getattr(self, "_unfinished_requests", {}).pop(req_id, None)
            for writer_key in [
                key for key in self._active_writers if key[0] == req_id
            ]:
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
            load_tokens = pending.matched_tokens
            if load_tokens <= 0 or not pending.allocated_block_ids:
                return
            descriptors = [
                desc for desc in pending.descriptors if desc.end <= load_tokens
            ]
            if not descriptors:
                return
            meta.add(_ReqMeta(
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
            ))

        def add_store_meta(request_id: str, tracker: _RequestTracker) -> None:
            store_end = self._store_end_for_step(
                tracker.num_computed_tokens, tracker.prompt_len
            )
            descriptors = self._new_store_descriptors(tracker, store_end)
            if not descriptors:
                return
            meta.add(_ReqMeta(
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
            ))
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
                pending.matched_tokens
                if pending is not None and pending.can_load
                else 0
            )
            computed_after = min(
                max(computed_before, external_cached) + scheduled,
                prompt_len,
            )

            block_ids = self._extract_block_ids(
                getattr(new_req, "block_ids", None)
            )
            tracker = self._request_trackers.get(req_id)
            if tracker is None:
                tracker = _RequestTracker()
                self._request_trackers[req_id] = tracker
            tracker.token_ids = list(prompt_tokens[:computed_after])
            tracker.block_ids = block_ids
            tracker.prompt_len = prompt_len
            tracker.num_computed_tokens = computed_after
            tracker.external_cached_tokens = external_cached
            tracker.num_saved_tokens = min(
                tracker.num_saved_tokens, computed_after
            )
            self._unfinished_requests.setdefault(req_id, new_req)

            add_load_meta(req_id, tracker, pending)
            add_store_meta(req_id, tracker)

        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached is None or isinstance(cached, list):
            return meta

        req_ids = getattr(cached, "req_ids", ()) or ()
        new_block_ids = getattr(cached, "new_block_ids", ()) or ()
        all_token_ids = getattr(cached, "all_token_ids", {}) or {}
        computed_before_values = getattr(
            cached, "num_computed_tokens", ()
        ) or ()
        new_token_values = getattr(cached, "new_token_ids", ()) or ()
        resumed_ids = set(getattr(cached, "resumed_req_ids", ()) or ())
        resumed_ids.update(
            getattr(cached, "preempted_req_ids", ()) or ()
        )

        for index, req_id in enumerate(req_ids):
            tracker = self._request_trackers.get(req_id)
            if tracker is None:
                logger.warning(
                    "Skipping cached request %s without a tracker", req_id
                )
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
                logger.warning(
                    "Skipping cached request %s without token IDs", req_id
                )
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
            scheduled = max(
                0, self._get_num_scheduled(scheduler_output, req_id)
            )
            pending = self._pending_loads.pop(req_id, None)
            has_pending_load = pending is not None and pending.can_load
            external_cached = (
                pending.matched_tokens
                if has_pending_load
                else min(tracker.external_cached_tokens, computed_before)
            )

            if computed_before < tracker.num_computed_tokens:
                tracker.token_ids = list(
                    full_tokens[:min(computed_before, prompt_len)]
                )
                tracker.num_saved_tokens = min(
                    tracker.num_saved_tokens, computed_before
                )
                tracker.external_cached_tokens = min(
                    tracker.external_cached_tokens, computed_before
                )

            block_update = self._get_block_ids_for_cached(
                list(new_block_ids), index
            )
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
            fail_all(
                "Cannot load chunk objects while disk-cache is disconnected"
            )
            return
        if not self._expected_layers:
            fail_all(
                "Cannot load chunk objects without configured KV layer names"
            )
            return

        attn_metadata = getattr(forward_context, "attn_metadata", None)
        if attn_metadata is None:
            fail_all(
                "Cannot load chunk objects: forward context has no "
                "attention metadata"
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
                self._load_request_kv(
                    req, forward_context, attn_metadata, layer_cache
                )
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
                self._save_request_kv(
                    req, layer_name, kv_layer, attn_metadata
                )

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
        unique_projected: dict[
            tuple[str, str, str], dict[str, Any]
        ] = {}
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
                barrier_error = (
                    f"writer {writer_key} has invalid final path: {exc}"
                )
                break

            owner_request_id = (
                writer_state.owner_request_id or writer_key[0]
            )
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
                writer_state.writer.finalize()
                committed_obj = dict(obj)
                committed_obj["size"] = (
                    writer_state.writer.final_path.stat().st_size
                )
                objects_to_commit.append(committed_obj)
        except Exception:
            logger.exception(
                "Failed to finalize chunk-object writer; skipping cache save"
            )
            discard_writer_states()
            return

        unique_objects: dict[
            tuple[str, str, str], dict[str, Any]
        ] = {}
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
            {
                key: value
                for key, value in obj.items()
                if not key.startswith("_")
            }
            for obj in unique_objects.values()
        ]
        try:
            self._go.commit_chunks(commit_payload)
        except Exception as exc:
            logger.error(
                "commit_chunks failed; retaining finalized objects for "
                "retry: %s",
                exc,
            )
            discard_writer_states()
            return

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

    def request_finished(
        self, request: Any, block_ids: Any
    ) -> tuple[bool, Any]:
        """Clean up per-request state after a request finishes."""
        rid = self._get_req_id(request)
        if rid:
            self._pending_loads.pop(rid, None)
            self._request_trackers.pop(rid, None)
            getattr(self, "_unfinished_requests", {}).pop(rid, None)
            # Clean up active writers belonging to this request
            writers_to_remove = [
                wkey for wkey in self._active_writers
                if wkey[0] == rid
            ]
            for wkey in writers_to_remove:
                ws = self._active_writers.pop(wkey, None)
                if ws is not None and not ws.committed:
                    ws.aborted = True
                    ws.writer.abort()
        return False, None

    def take_events(self) -> list[Any]:
        return []

    def get_finished(
        self, finished_req_ids: list[str] | set[str]
    ) -> tuple[Any, Any]:
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
            items = ((lname, getattr(no_compile_layers, lname, None))
                     for lname in self._expected_layers)
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
        self.cache_root = Path(
            extra.get("disk_cache_path", "/tmp/disk-cache")
        )
        self.go_addr = extra.get(
            "disk_cache_engine_addr", "http://localhost:9100"
        )
        self._go = DiskCacheGoClient(self.go_addr)

        # Block and chunk configuration
        self._block_size = vllm_config.cache_config.block_size

        chunk_size_tokens = int(
            extra.get("disk_cache_chunk_size_tokens", 256)
        )
        if chunk_size_tokens <= 0:
            raise ValueError(
                f"disk_cache_chunk_size_tokens must be > 0, "
                f"got {chunk_size_tokens}"
            )
        if chunk_size_tokens % self._block_size != 0:
            raise ValueError(
                f"disk_cache_chunk_size_tokens ({chunk_size_tokens}) must "
                f"be an integer multiple of block_size ({self._block_size})"
            )
        self._chunk_size = chunk_size_tokens

        # Chunk key strategy
        strategy_name = extra.get(
            "disk_cache_key_strategy", "chain"
        )
        self._strategy = create_chunk_key_strategy(
            strategy_name,
            tokens_per_chunk=chunk_size_tokens,
            block_size=self._block_size,
        )

        # Storage backend
        storage_prefer = extra.get("storage_backend", "auto")
        self._storage = create_storage_backend(prefer=storage_prefer)

        # Device resolution
        self.target_device = extra.get("target_device", "auto")

        # ── Stable namespace ────────────────────────────────────────
        model_config = getattr(vllm_config, "model_config", None)
        if model_config is not None:
            model_hash = getattr(model_config, "compute_hash", None)
            if model_hash is not None:
                model_id = model_hash()
            else:
                model_id = str(
                    getattr(model_config, "model", "unknown")
                )
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
        groups = list(
            getattr(kv_cache_config, "kv_cache_groups", []) or []
        ) if kv_cache_config is not None else []
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
        self.node_id = socket.gethostname()
        self.cache_root.mkdir(parents=True, exist_ok=True)

        if self._connected:
            logger.info(
                "DiskCacheConnector v2 ready: cache=%s engine=%s "
                "bs=%d chunk=%dtok strategy=%s namespace=%.16s "
                "shard=%s required=%s backend=%s",
                self.cache_root,
                self.go_addr,
                self._block_size,
                self._chunk_size,
                strategy_name,
                self._namespace,
                self._shard,
                self._required_shards,
                type(self._storage).__name__,
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
            raise ValueError(
                f"num_tokens must be >= 0, got {num_tokens}"
            )
        if not block_ids or num_tokens == 0:
            return torch.empty(0, dtype=torch.long)
        if block_size <= 0:
            raise ValueError(
                f"block_size must be > 0, got {block_size}"
            )
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
            return prompt_len - 1
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
        return [
            desc for desc in descriptors
            if desc.end > tracker.external_cached_tokens
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
            objects = self._go.resolve_chunks(
                req.namespace, keys_to_resolve, req.shard
            )
        except Exception as exc:
            logger.error(
                "resolve_chunks failed for request %s: %s",
                req.request_id, exc,
            )
            self._record_load_error(req)
            self._load_error = str(exc)
            raise RuntimeError(
                f"resolve_chunks failed for {req.request_id}: {exc}"
            )

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
                    k, req.request_id,
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
            logger.error(
                "Empty slot mapping for load request %s", req.request_id
            )
            self._record_load_error(req)
            self._load_error = f"empty slot mapping for {req.request_id}"
            raise RuntimeError(
                f"Empty slot mapping for load request {req.request_id}"
            )

        num_loaded_objects = 0
        target_dtype: Optional[torch.dtype] = None

        # Determine target dtype from first available layer in cache
        for lname in self._expected_layers:
            kv = layer_cache.get(lname)
            if kv is not None:
                target_dtype = kv.dtype
                break

        for desc in req.descriptors:
            if desc.end > req.load_tokens:
                continue
            key = desc.key
            obj_info = obj_by_key.get(key)
            if obj_info is None:
                logger.error(
                    "resolve_chunks: no object for key=%s request=%s",
                    key, req.request_id,
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
                    key, req.request_id,
                )
                self._invalidate_chunk(req.namespace, key, req.shard)
                self._record_load_error(req)
                self._load_error = f"empty file_path for key {key}"
                raise RuntimeError(
                    f"Empty file_path for key={key} request={req.request_id}"
                )
            full_path = (self.cache_root / file_path).resolve()
            try:
                full_path.relative_to(self.cache_root.resolve())
            except ValueError as exc:
                self._invalidate_chunk(req.namespace, key, req.shard)
                self._record_load_error(req)
                self._load_error = f"file_path escapes cache root: {file_path}"
                raise RuntimeError(
                    f"Resolved file_path escapes cache root for key={key}"
                ) from exc

            try:
                # Determine device from first available layer
                load_device = self.target_device
                if load_device == "auto":
                    for lname in self._expected_layers:
                        if lname in layer_cache:
                            load_device = str(
                                layer_cache[lname].device
                            )
                            break
                    if load_device == "auto":
                        load_device = "cpu"

                header, tensors = load_chunk_object(
                    full_path,
                    layer_names=self._expected_layers,
                    device=load_device,
                    backend=self._storage,
                )
            except Exception as exc:
                logger.error(
                    "load_chunk_object failed: %s (key=%s): %s",
                    full_path, key, exc,
                )
                if isinstance(exc, (OSError, ValueError)):
                    self._invalidate_chunk(
                        req.namespace, key, req.shard, full_path
                    )
                self._record_load_error(req)
                self._load_error = str(exc)
                raise RuntimeError(
                    f"load_chunk_object failed for key={key}: {exc}"
                )

            # Validate the immutable object contract before injecting any layer.
            try:
                self._validate_chunk_header(header, key, desc)
            except Exception:
                self._invalidate_chunk(req.namespace, key, req.shard, full_path)
                raise

            # Validate and inject each layer
            for lname in self._expected_layers:
                tensor = tensors.get(lname)
                if tensor is None:
                    err_msg = f"Layer {lname} not found in chunk {key}"
                    self._record_load_error(req)
                    self._load_error = err_msg
                    raise RuntimeError(err_msg)
                expected_tokens = desc.end - desc.start
                if tensor.size(0) != expected_tokens:
                    err_msg = (
                        f"Layer {lname} in chunk {key}: expected "
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

                # Slot mapping for this chunk's token range
                chunk_slot_mapping = slot_mapping[
                    desc.start:desc.end
                ]
                if chunk_slot_mapping.numel() != tensor.size(0):
                    err_msg = (
                        f"Slot mapping size mismatch for layer {lname}: "
                        f"{chunk_slot_mapping.numel()} != {tensor.size(0)}"
                    )
                    self._record_load_error(req)
                    self._load_error = err_msg
                    raise RuntimeError(err_msg)

                layer_attn = self._get_layer_attn_meta(
                    attn_metadata, lname
                )
                inject_kv_into_layer(
                    kv_tensor,
                    tensor,
                    chunk_slot_mapping,
                    layer_attn,
                    self._block_size,
                )

            num_loaded_objects += 1

        # Report successful retrieval count to Go engine — only after ALL succeed
        if num_loaded_objects > 0:
            try:
                self._go.chunks_retrieved(num_loaded_objects)
            except Exception as exc:
                logger.debug("chunks_retrieved call failed: %s", exc)

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
                f"Chunk header key mismatch: "
                f"{header.key} != {expected_key}"
            )
        if header.shard != self._shard:
            raise RuntimeError(
                f"Chunk header shard mismatch: "
                f"{header.shard} != {self._shard}"
            )
        if header.index != desc.index:
            raise RuntimeError(
                f"Chunk header index mismatch: "
                f"{header.index} != {desc.index}"
            )
        if header.start != desc.start:
            raise RuntimeError(
                f"Chunk header start mismatch: "
                f"{header.start} != {desc.start}"
            )
        if header.end != desc.end:
            raise RuntimeError(
                f"Chunk header end mismatch: "
                f"{header.end} != {desc.end}"
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

    def _record_load_error(self, req: _ReqMeta) -> None:
        """Record block IDs associated with a load error."""
        for bid in req.block_ids:
            self._block_ids_with_load_errors.add(bid)

    def _abort_request_writers(self, request_id: str) -> None:
        """Abort and discard all uncommitted writers owned by one request."""
        for writer_key in [
            key for key in self._active_writers if key[0] == request_id
        ]:
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
            return self._go.health_check()
        except Exception:
            return False
