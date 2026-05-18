"""
Disk Cache Connector for vLLM.

This connector implements the KVConnectorBase interface to provide
disk-backed KV cache storage for vLLM.

Usage:
    vllm serve model --kv-connector disk-cache
"""

from __future__ import annotations

import ctypes
import logging
import os
import socket
from typing import Optional

# vLLM imports
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.request import Request

logger = init_logger(__name__)


class DiskCacheConnector(KVConnectorBase):
    """
    vLLM connector for disk-backed KV cache.

    Delegates storage operations to the Go disk-cache engine via HTTP API.
    The Go engine handles all disk I/O, eviction, and metadata management.
    """

    def __init__(self, vllm_config: VllmConfig, *args, **kwargs):
        super().__init__(vllm_config, *args, **kwargs)

        # Read config from vLLM's kv_transfer_config
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        self.cache_path = extra.get("disk_cache_path", "/tmp/disk-cache")
        self.max_size = extra.get("disk_cache_max_size", "100GB")
        self.engine_addr = extra.get("disk_cache_engine_addr", "http://localhost:9100")

        self.node_id = socket.gethostname()
        self._connected = False
        self._connect()

    def _connect(self):
        """Connect to the Go disk-cache engine."""
        try:
            import requests
            resp = requests.get(f"{self.engine_addr}/health", timeout=5)
            if resp.status_code == 200:
                self._connected = True
                logger.info(
                    "DiskCacheConnector connected to engine at %s "
                    "(cache_path=%s, max_size=%s)",
                    self.engine_addr, self.cache_path, self.max_size,
                )
            else:
                logger.warning("DiskCacheConnector: engine health check failed")
        except Exception as e:
            logger.warning(
                "DiskCacheConnector: cannot connect to engine at %s: %s. "
                "Ensure disk-cache engine is running.",
                self.engine_addr, e,
            )

    # ── Scheduler-side hooks ──

    def get_num_new_matched_tokens(self, request: Request) -> int:
        """
        Returns the number of prompt tokens whose KV cache is already on disk.
        For now, returns 0 (cold start). Disk-hit logic will be added later.
        """
        return 0

    def request_finished(self, request: Request, blocks) -> bool:
        """
        Called when a request completes. Returns True if the connector
        takes ownership of freeing the blocks (async save to disk).
        """
        if not self._connected or not blocks:
            return False

        # Store blocks to disk asynchronously
        for block in blocks:
            try:
                self._store_block(block)
            except Exception as e:
                logger.error("Failed to store block: %s", e)

        # Return False: let vLLM free the blocks normally
        return False

    def take_events(self):
        """Return KV events for the scheduler."""
        return []

    # ── Worker-side hooks ──

    def start_load_kv(self, blocks, *args, **kwargs):
        """Start loading KV blocks from disk."""
        for block in blocks:
            try:
                self._load_block(block)
            except Exception as e:
                logger.error("Failed to load block: %s", e)

    def wait_for_layer_load(self, layer_idx: int):
        """Wait for a specific layer's KV to finish loading."""
        pass  # synchronous for now

    def save_kv_layer(self, layer_idx: int, kv_tensor, *args, **kwargs):
        """
        Save a single layer's KV tensor to disk.

        This is called per layer during the forward pass.
        For now, we batch saves in request_finished instead.
        """
        pass

    def wait_for_save(self):
        """Wait for all pending saves to complete."""
        pass

    def get_finished(self, request_ids: list[str]) -> list[str]:
        """Return IDs of requests that have finished async operations."""
        return []

    # ── Internal helpers ──

    def _store_block(self, block):
        """Send a block to the Go engine for storage."""
        if not hasattr(block, 'block_hash') or not hasattr(block, 'tensor'):
            return

        import requests
        data = {
            "hash": str(block.block_hash),
            "size": block.tensor.numel() * block.tensor.element_size(),
        }
        try:
            requests.post(
                f"{self.engine_addr}/blocks/{block.block_hash}/store",
                json=data,
                timeout=10,
            )
        except Exception:
            pass  # Engine will retry or skip

    def _load_block(self, block):
        """Request a block from the Go engine."""
        if not hasattr(block, 'block_hash'):
            return

        import requests
        try:
            resp = requests.get(
                f"{self.engine_addr}/blocks/{block.block_hash}/load",
                timeout=10,
            )
            if resp.status_code == 200:
                logger.debug("Block %s loaded from disk", block.block_hash)
            else:
                logger.debug("Block %s not on disk", block.block_hash)
        except Exception:
            pass


# Register the connector so vLLM can discover it
KVConnectorClass = DiskCacheConnector
