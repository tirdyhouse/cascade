"""Chunk path/range helpers shared by vLLM disk-cache connectors."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path


def chunk_ranges(num_tokens: int, tokens_per_chunk: int) -> Iterator[tuple[int, int, int]]:
    """Generate ``(chunk_idx, start, end)`` ranges for token chunking."""
    if num_tokens <= 0 or tokens_per_chunk <= 0:
        return
    chunk_idx = 0
    while chunk_idx * tokens_per_chunk < num_tokens:
        start = chunk_idx * tokens_per_chunk
        end = min(start + tokens_per_chunk, num_tokens)
        yield chunk_idx, start, end
        chunk_idx += 1


def chunk_file_path(
    cache_root: str | Path,
    prefix_key: str,
    layer_name: str,
    chunk_idx: int,
) -> Path:
    """Return ``{root}/{pk[0:2]}/{pk[2:4]}/{pk}/{layer}/{idx}.safetensors``."""
    root = Path(cache_root)
    return root / prefix_key[:2] / prefix_key[2:4] / prefix_key / layer_name / f"{chunk_idx}.safetensors"


def cached_file_path(cache_root: str | Path, layer_hash: str) -> Path:
    return Path(cache_root) / layer_hash[:2] / layer_hash[2:4] / f"{layer_hash}.safetensors"
