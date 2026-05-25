"""
Disk Cache Connector for vLLM 0.21+.

Architecture:
  - Tensor data: written to/read from .bin files on local disk
  - Metadata: managed by Go engine (Pebble + LRU) via HTTP API
"""

import json
import socket
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

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

logger = init_logger(__name__)


@dataclass
class DiskCacheMeta(KVConnectorMetadata):
    """Minimal metadata for disk cache."""
    pass


class DiskCacheConnector(KVConnectorBase_V1):
    """
    vLLM 0.21+ connector for disk-backed KV cache.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        self.cache_root = Path(extra.get("disk_cache_path", "/tmp/disk-cache"))
        self.go_addr = extra.get("disk_cache_engine_addr", "http://localhost:9100")
        self.target_device = extra.get("target_device", "auto")
        self.node_id = socket.gethostname()

        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._connected = self._health_check()

        if self._connected:
            logger.info(
                "DiskCacheConnector ready: cache=%s engine=%s",
                self.cache_root, self.go_addr,
            )
        else:
            logger.warning(
                "DiskCacheConnector: Go engine not reachable at %s", self.go_addr,
            )

    # ── Device ──

    def _resolve_device(self, target_tensor) -> str:
        if self.target_device != "auto":
            return self.target_device
        return str(target_tensor.device)

    # ── Worker-side (required by abstract) ──

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Start loading KV from disk."""
        # vLLM 0.21 uses forward_context to pass KV metadata
        # For disk cache, actual loading happens synchronously in save/load
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Block until KV layer is loaded."""
        # Disk reads are synchronous, no waiting needed
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """Save KV layer to disk."""
        if not self._connected:
            return

        # HACK: simple hash based on layer name and data
        block_hash = hash(layer_name) ^ hash(str(kv_layer.shape))

        # Write tensor to disk file
        cpu_tensor = kv_layer.cpu()
        # Convert BFloat16 to float16 for numpy compatibility
        if cpu_tensor.dtype == torch.bfloat16:
            cpu_tensor = cpu_tensor.to(torch.float16)
        data = cpu_tensor.numpy().tobytes()

        file_path = self._file_path(block_hash)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(data)

        # Notify Go engine
        self._go_put(
            block_hash,
            str(file_path.relative_to(self.cache_root)),
            len(data),
        )

    def wait_for_save(self) -> None:
        """Block until all saves complete."""
        pass

    # ── Scheduler-side (required by abstract) ──

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[Optional[int], bool]:
        """No cache hits yet (TODO: real hash lookup)."""
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """No post-alloc state needed."""
        pass

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> KVConnectorMetadata:
        """Build connector metadata."""
        return DiskCacheMeta()

    # ── Overrides with defaults ──

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """Request finished. Blocks will be freed by vLLM scheduler."""
        return False, None

    def take_events(self) -> "Iterable[KVCacheEvent]":
        return []

    def get_finished(self, finished_req_ids):
        """No async transfers, nothing finished."""
        return None, None

    # ── File helpers ──

    def _file_path(self, block_hash: int) -> Path:
        h = f"{block_hash:016x}"
        return self.cache_root / h[:2] / h[2:4] / f"{h}.bin"

    # ── Go engine RPC ──

    def _go_put(self, hash: int, file_path: str, size: int):
        try:
            req = urllib.request.Request(
                f"{self.go_addr}/put",
                data=json.dumps({
                    "hash": hash,
                    "file_path": file_path,
                    "size": size,
                }).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            logger.debug("Go Put failed: %s", e)

    def _go_get(self, hash: int) -> Optional[dict]:
        try:
            resp = urllib.request.urlopen(
                f"{self.go_addr}/get?hash={hash:016x}", timeout=5
            )
            return json.loads(resp.read())
        except Exception:
            return None

    def _health_check(self) -> bool:
        try:
            urllib.request.urlopen(f"{self.go_addr}/stats", timeout=3)
            return True
        except Exception:
            return False


KVConnectorClass = DiskCacheConnector
