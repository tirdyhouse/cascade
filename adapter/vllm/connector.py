import hashlib
import json
import socket
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import safetensors.torch
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from collections.abc import Iterable
    from vllm.distributed.kv_events import KVCacheEvent
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger('vllm.disk_cache')


def align_to_block_size(num_tokens: int, block_size: int) -> int:
    if num_tokens < 1:
        return 0
    return (num_tokens - 1) // block_size * block_size


def hash_token_count(num_tokens: int, block_size: int) -> int:
    aligned = align_to_block_size(num_tokens, block_size)
    return max(aligned, 1)


def compute_prompt_hash(token_ids, num_tokens, mm_hashes):
    import struct
    h = hashlib.sha256()
    for tid in token_ids[:num_tokens]:
        h.update(struct.pack(">I", tid))
    for mh in mm_hashes:
        h.update(mh.encode())
    return h.hexdigest()[:32]


def inject_kv_into_layer(dst, src, slot_mapping, attn_metadata, block_size):
    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
    if isinstance(attn_metadata, TritonAttentionMetadata):
        block_idxs = slot_mapping // block_size
        offsets = slot_mapping % block_size
        dst[block_idxs, :, offsets] = src
    else:
        num_pages = dst.shape[1]
        page_size = dst.shape[2]
        dst_flat = dst.reshape(2, num_pages * page_size, -1)
        dst_flat[:, slot_mapping, ...] = src


def extract_kv_from_layer(layer, slot_mapping, attn_metadata, block_size):
    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
    if isinstance(attn_metadata, TritonAttentionMetadata):
        block_idxs = slot_mapping // block_size
        offsets = slot_mapping % block_size
        return layer[block_idxs, :, offsets]
    num_pages, page_size = layer.shape[1], layer.shape[2]
    return layer.reshape(2, num_pages * page_size, -1)[:, slot_mapping, ...]


@dataclass
class DiskCacheMeta(KVConnectorMetadata):
    requests: list["_ReqMeta"] = field(default_factory=list)

    def add(self, token_ids, block_ids, block_size, is_store, mm_hashes, num_tokens, prompt_hash):
        self.requests.append(_ReqMeta(
            token_ids=token_ids, block_ids=block_ids, block_size=block_size,
            is_store=is_store, mm_hashes=mm_hashes,
            num_tokens=num_tokens, prompt_hash=prompt_hash,
        ))


@dataclass
class _ReqMeta:
    token_ids: list[int]
    block_ids: list[int]
    block_size: int
    is_store: bool
    mm_hashes: list[str]
    num_tokens: int
    prompt_hash: str


class DiskCacheConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        self.cache_root = Path(extra.get("disk_cache_path", "/tmp/disk-cache"))
        self.go_addr = extra.get("disk_cache_engine_addr", "http://localhost:9100")
        self.target_device = extra.get("target_device", "auto")
        self._block_size = vllm_config.cache_config.block_size
        self.node_id = socket.gethostname()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._connected = self._health_check()
        self._requests_need_load = {}
        if self._connected:
            logger.info("DiskCacheConnector ready: cache=%s engine=%s bs=%d",
                        self.cache_root, self.go_addr, self._block_size)
        else:
            logger.warning("DiskCacheConnector: Go engine not reachable at %s", self.go_addr)

    def _resolve_device(self, target_tensor):
        if self.target_device != "auto":
            return self.target_device
        return str(target_tensor.device)

    def start_load_kv(self, forward_context, **kwargs):
        meta = self._get_connector_metadata()
        if not isinstance(meta, DiskCacheMeta) or not meta.requests:
            return
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return
        for req in meta.requests:
            if req.is_store:
                continue
            num_blocks = len(req.block_ids)
            block_offsets = torch.arange(0, req.block_size)
            slot_mapping = (
                block_offsets.view(1, req.block_size)
                + torch.tensor(req.block_ids).view(num_blocks, 1) * req.block_size
            ).flatten()[:req.num_tokens]
            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]
                kv_cache_layer = getattr(layer, "kv_cache", None)
                if kv_cache_layer is None:
                    continue
                layer_hash = self._layer_hash(req.prompt_hash, layer_name)
                file_path = self._cached_file_path(layer_hash)
                if not file_path.exists():
                    continue
                # Notify Go engine of block retrieval for stats tracking
                self._go_get(int(layer_hash[:16], 16))
                try:
                    loaded = safetensors.torch.load_file(str(file_path))
                    kv_cache = loaded["kv_cache"]
                    # Match the target layer's dtype (bfloat16 vs float16)
                    target_dtype = kv_cache_layer.dtype
                    if kv_cache.dtype != target_dtype:
                        kv_cache = kv_cache.to(target_dtype)
                    device = self._resolve_device(kv_cache_layer)
                    kv_cache = kv_cache.to(device, non_blocking=True)
                    layer_attn = attn_metadata.get(layer_name, attn_metadata) if isinstance(attn_metadata, dict) else attn_metadata
                    inject_kv_into_layer(kv_cache_layer, kv_cache, slot_mapping, layer_attn, self._block_size)
                except Exception as e:
                    logger.warning("Failed to load KV for %s: %s", layer_name, e)

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        if not self._connected:
            return
        meta = self._get_connector_metadata()
        if not isinstance(meta, DiskCacheMeta):
            return
        for req in meta.requests:
            if not req.is_store:
                continue
            num_blocks = len(req.block_ids)
            block_offsets = torch.arange(0, req.block_size)
            slot_mapping = (
                block_offsets.view(1, req.block_size)
                + torch.tensor(req.block_ids).view(num_blocks, 1) * req.block_size
            ).flatten()[:req.num_tokens]
            kv_cache = extract_kv_from_layer(kv_layer, slot_mapping, attn_metadata, self._block_size)
            layer_hash = self._layer_hash(req.prompt_hash, layer_name)
            file_path = self._cached_file_path(layer_hash)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            cpu_kv = kv_cache.detach().cpu()
            safetensors.torch.save_file({"kv_cache": cpu_kv}, str(file_path))
            file_size = file_path.stat().st_size
            go_hash = int(layer_hash[:16], 16)
            self._go_put(go_hash, str(file_path.relative_to(self.cache_root)), file_size)

    def wait_for_save(self):
        """Called after all layers are written. Records all sub-block sentinels
        via /record_batch for prefix caching across different prompt lengths."""
        meta = self._get_connector_metadata()
        if isinstance(meta, DiskCacheMeta):
            for req in meta.requests:
                if req.is_store:
                    self._go_record_batch(req.token_ids, req.mm_hashes, req.num_tokens)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        token_ids = request.prompt_token_ids or []
        if len(token_ids) < 2:
            return 0, False
        num_to_check = align_to_block_size(len(token_ids) - 1, self._block_size)
        if num_to_check <= num_computed_tokens:
            return 0, False
        mm_hashes = [f.identifier for f in request.mm_features]
        result = self._go_match(token_ids, mm_hashes)
        if result and result.get("matched_tokens", 0) > 0:
            matched = result["matched_tokens"]
            logger.info("Disk cache HIT for request %s (%d tokens)", request.request_id, matched)
            return matched - num_computed_tokens, False
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(self, scheduler_output):
        meta = DiskCacheMeta()
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            mm_hashes = [f.identifier for f in new_req.mm_features]
            num_to_check = align_to_block_size(len(token_ids) - 1, self._block_size) if len(token_ids) > 1 else 0
            hash_n = hash_token_count(len(token_ids) - 1, self._block_size) if len(token_ids) > 1 else 0
            prompt_hash = compute_prompt_hash(token_ids, hash_n, mm_hashes) if hash_n > 0 else ""
            if new_req.req_id in self._requests_need_load:
                meta.add(token_ids=token_ids, block_ids=new_req.block_ids[0],
                         block_size=self._block_size, is_store=False,
                         mm_hashes=mm_hashes, num_tokens=num_to_check, prompt_hash=prompt_hash)
            elif num_to_check > 0:
                meta.add(token_ids=token_ids, block_ids=new_req.block_ids[0],
                         block_size=self._block_size, is_store=True,
                         mm_hashes=mm_hashes, num_tokens=num_to_check, prompt_hash=prompt_hash)
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            if req_id in self._requests_need_load:
                request = self._requests_need_load[req_id]
                token_ids = list(request.all_token_ids) if hasattr(request.all_token_ids, "__getitem__") else []
                mm_hashes = [f.identifier for f in request.mm_features]
                num_to_check = align_to_block_size(len(token_ids) - 1, self._block_size) if len(token_ids) > 1 else 0
                hash_n = hash_token_count(len(token_ids) - 1, self._block_size) if len(token_ids) > 1 else 0
                prompt_hash = compute_prompt_hash(token_ids, hash_n, mm_hashes) if hash_n > 0 else ""
                new_block_ids = cached_reqs.new_block_ids[i]
                if new_block_ids is not None:
                    meta.add(token_ids=token_ids, block_ids=new_block_ids[0],
                             block_size=self._block_size, is_store=False,
                             mm_hashes=mm_hashes, num_tokens=num_to_check, prompt_hash=prompt_hash)
        self._requests_need_load.clear()
        return meta

    def request_finished(self, request, block_ids):
        return False, None

    def take_events(self):
        return []

    def get_finished(self, finished_req_ids):
        return None, None

    def _layer_hash(self, prompt_hash, layer_name):
        h = hashlib.sha256()
        h.update(prompt_hash.encode())
        h.update(layer_name.encode())
        return h.hexdigest()[:32]

    def _cached_file_path(self, layer_hash):
        return self.cache_root / layer_hash[:2] / layer_hash[2:4] / f"{layer_hash}.safetensors"

    def _go_match(self, token_ids, mm_hashes):
        """Send token IDs to Go engine for cache hit detection (parallel)."""
        try:
            req = urllib.request.Request(
                f"{self.go_addr}/match",
                data=json.dumps({
                    "token_ids": token_ids,
                    "mm_hashes": mm_hashes,
                    "block_size": self._block_size,
                }).encode(),
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=5)
            return json.loads(resp.read())
        except Exception as e:
            logger.debug("Go Match failed: %s", e)
            return None

    def _go_record(self, prompt_hash, num_tokens):
        """Record a cache-complete sentinel in Go engine metadata."""
        try:
            req = urllib.request.Request(
                f"{self.go_addr}/record",
                data=json.dumps({
                    "prompt_hash": prompt_hash,
                    "num_tokens": num_tokens,
                }).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            logger.debug("Go Record failed: %s", e)

    def _go_record_batch(self, token_ids, mm_hashes, num_tokens):
        """Record all sub-block sentinels via /record_batch.
        Go engine computes incremental cumulative hashes and stores all prefix lengths."""
        try:
            req = urllib.request.Request(
                f"{self.go_addr}/record_batch",
                data=json.dumps({
                    "token_ids": token_ids[:num_tokens],
                    "mm_hashes": mm_hashes,
                    "block_size": self._block_size,
                }).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            logger.debug("Go RecordBatch failed: %s", e)

    def _go_put(self, hash_val, file_path, size):
        try:
            req = urllib.request.Request(
                f"{self.go_addr}/put",
                data=json.dumps({"hash": hash_val, "file_path": file_path, "size": size}).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            logger.debug("Go Put failed: %s", e)

    def _go_get(self, hash_val):
        try:
            resp = urllib.request.urlopen(f"{self.go_addr}/get?hash={hash_val:016x}", timeout=5)
            return json.loads(resp.read())
        except Exception:
            return None

    def _health_check(self):
        try:
            urllib.request.urlopen(f"{self.go_addr}/stats", timeout=3)
            return True
        except Exception:
            return False


KVConnectorClass = DiskCacheConnector
