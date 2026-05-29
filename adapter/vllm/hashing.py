"""Hash helpers shared by vLLM disk-cache connectors."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterable, Sequence


def align_to_block_size(num_tokens: int, block_size: int) -> int:
    if num_tokens < 1:
        return 0
    return (num_tokens - 1) // block_size * block_size


def hash_token_count(num_tokens: int, block_size: int) -> int:
    aligned = align_to_block_size(num_tokens, block_size)
    return max(aligned, 1)


def compute_prompt_hash(
    token_ids: Sequence[int],
    num_tokens: int,
    mm_hashes: Iterable[str],
) -> str:
    h = hashlib.sha256()
    for tid in token_ids[:num_tokens]:
        h.update(struct.pack(">I", tid))
    for mh in mm_hashes:
        h.update(mh.encode())
    return h.hexdigest()[:32]


def prefix_key(token_ids: Sequence[int], block_size: int) -> str:
    """Prefix shared across same-prefix requests."""
    n = min(block_size, len(token_ids))
    h = hashlib.sha256()
    for tid in token_ids[:n]:
        h.update(struct.pack(">I", tid))
    return h.hexdigest()[:32]


def layer_hash(prompt_hash: str, layer_name: str) -> str:
    h = hashlib.sha256()
    h.update(prompt_hash.encode())
    h.update(layer_name.encode())
    return h.hexdigest()[:32]
