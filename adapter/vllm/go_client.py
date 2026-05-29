"""Small HTTP client for the Go disk-cache engine."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any


class DiskCacheGoClient:
    """Best-effort client used by the vLLM connector.

    Existing connector behavior is intentionally preserved: operational calls log
    or return fallback values at the call site rather than raising into vLLM.
    """

    def __init__(self, base_url: str, timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def post(self, path: str, payload: dict[str, Any], timeout: float | None = None) -> bytes:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
            return resp.read()

    def get_json(self, path: str, query: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"
        with urllib.request.urlopen(url, timeout=timeout or self.timeout) as resp:
            return json.loads(resp.read())

    def health_check(self, timeout: float = 3.0) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/stats", timeout=timeout):
                return True
        except Exception:
            return False

    def chunk_put(self, prefix_key: str, layer_name: str, chunk_idx: int, num_tokens: int) -> None:
        self.post(
            "/chunk_put",
            {
                "prefix_key": prefix_key,
                "layer_name": layer_name,
                "chunk_index": chunk_idx,
                "num_tokens": num_tokens,
            },
        )

    def chunk_list(self, prefix_key: str, layer_name: str) -> list[int]:
        data = self.get_json(
            "/chunk_list",
            {"prefix_key": prefix_key, "layer_name": layer_name},
        )
        return data.get("chunks", [])

    def match(self, token_ids: list[int], mm_hashes: list[str], block_size: int) -> dict[str, Any]:
        data = self.post(
            "/match",
            {"token_ids": token_ids, "mm_hashes": mm_hashes, "block_size": block_size},
        )
        return json.loads(data)

    def record(self, prompt_hash: str, num_tokens: int) -> None:
        self.post("/record", {"prompt_hash": prompt_hash, "num_tokens": num_tokens})

    def record_batch(self, token_ids: list[int], mm_hashes: list[str], block_size: int) -> None:
        self.post(
            "/record_batch",
            {"token_ids": token_ids, "mm_hashes": mm_hashes, "block_size": block_size},
        )

    def put(self, hash_val: int, file_path: str, size: int) -> None:
        self.post("/put", {"hash": hash_val, "file_path": file_path, "size": size})

    def get(self, hash_val: int) -> Any:
        return self.get_json("/get", {"hash": f"{hash_val:016x}"})
