"""
Disk Cache Connector for vLLM.

Architecture:
  - Python: writes/reads tensor data as files, calls Go for metadata
  - Go:     manages Pebble metadata + LRU eviction, served via HTTP

Usage:
    # 1. Start Go engine
    disk-cache -cache-path /mnt/nvme/kv-cache -max-size 1TB

    # 2. Start vLLM with connector
    vllm serve model --kv-connector disk-cache
"""

import ctypes
import json
import logging
import os
import socket
import time
import urllib.request
from pathlib import Path
from typing import Optional

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase
from vllm.logger import init_logger

logger = init_logger(__name__)


class DiskCacheConnector(KVConnectorBase):
    """
    vLLM connector for disk-backed KV cache.

    - Write path:  save_kv_layer() → file → Go Put()
    - Read path:   start_load_kv() → Go Get() → read file → GPU
    - Eviction:    Go LRU policy, triggered before file writes
    """

    def __init__(self, vllm_config: VllmConfig, *args, **kwargs):
        super().__init__(vllm_config, *args, **kwargs)

        extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        self.cache_root = Path(extra.get("disk_cache_path", "/tmp/disk-cache"))
        self.go_addr = extra.get("disk_cache_engine_addr", "http://localhost:9100")
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
                "DiskCacheConnector: Go engine not reachable at %s. "
                "Run 'disk-cache' first.", self.go_addr,
            )

    # ── Worker-side hooks ──

    def save_kv_layer(self, layer_idx: int, kv_tensor, *args, **kwargs):
        """Write a single layer's KV tensor to disk."""
        if not self._connected:
            return

        block_hash = hash(layer_idx, kv_tensor)  # TODO: use real vLLM block hash

        # GPU → CPU (PyTorch async copy, doesn't block inference)
        cpu_tensor = kv_tensor.cpu()
        data = cpu_tensor.numpy().tobytes()

        # Write to file
        file_path = self._file_path(block_hash)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(data)

        # Tell Go: file is written, record metadata
        self._go_put(block_hash, str(file_path.relative_to(self.cache_root)), len(data))

    def start_load_kv(self, blocks, *args, **kwargs):
        """Load KV blocks from disk into GPU."""
        if not self._connected:
            return

        for block in blocks:
            block_hash = block.block_hash
            meta = self._go_get(block_hash)
            if meta is None:
                continue

            # Read file
            file_path = self.cache_root / meta["file_path"]
            if not file_path.exists():
                continue

            data = file_path.read_bytes()
            # TODO: copy data to GPU tensor (block.tensor)

    def wait_for_layer_load(self, layer_idx: int):
        pass

    def wait_for_save(self):
        pass

    # ── Scheduler-side hooks ──

    def get_num_new_matched_tokens(self, request) -> int:
        return 0  # TODO: check disk cache for existing KV

    def request_finished(self, request, blocks) -> bool:
        return False

    def take_events(self):
        return []

    def get_finished(self, request_ids: list[str]) -> list[str]:
        return []

    # ── Private helpers ──

    def _file_path(self, block_hash: int) -> Path:
        h = f"{block_hash:016x}"
        return self.cache_root / h[:2] / h[2:4] / f"{h}.bin"

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
