"""Pluggable chunk-key strategies for vLLM disk-cache connectors.

A chunk-key strategy determines how each chunk of token IDs is assigned
a unique key for caching.  Different strategies produce different key
structures (e.g. chain-hash, prefix-hash, identity, …).
"""

from __future__ import annotations

import hashlib
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
from urllib.parse import quote


# ── ChunkDescriptor ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ChunkDescriptor:
    """Describes a single chunk within a token sequence.

    Attributes:
        index: 0-based chunk index.
        start: Start token position (inclusive).
        end:   End token position (exclusive).
        key:   Unique key string for this chunk.
    """
    index: int
    start: int
    end: int
    key: str


# ── Abstract strategy ───────────────────────────────────────────────────


class ChunkKeyStrategy(ABC):
    """Pluggable strategy for computing chunk keys.

    Subclasses must implement :meth:`describe`.
    """

    @abstractmethod
    def describe(
        self,
        token_ids: Sequence[int],
        namespace: str,
        end: Optional[int] = None,
    ) -> list[ChunkDescriptor]:
        """Split *token_ids* into chunks and return their descriptors.

        Parameters
        ----------
        token_ids:
            Full sequence of token IDs.
        namespace:
            Logical namespace for key isolation (e.g. model name or prefix).
        end:
            Optional explicit end position (exclusive).  When ``None`` the
            full sequence is used.

        Returns
        -------
        list[ChunkDescriptor]:
            One descriptor per chunk, in order.
        """
        ...


# ── Chain (chained hash) strategy ──────────────────────────────────────


_DOMAIN_SEPARATOR = b"predict-chunk-key-chain:v1"


class ChainChunkKeyStrategy(ChunkKeyStrategy):
    """Chained-hash chunk key strategy.

    Each chunk's key is a **full** 64-hex-digit SHA-256 of
    ``SHA256(previous_digest || token_bytes)``, producing a cryptographic
    chain: key ``i`` depends on **all** tokens in chunks ``0 .. i``.

    Configuration
    -------------
    ``tokens_per_chunk``:
        Number of tokens per chunk (default 256).  **Must** be > 0 and
        an integer multiple of ``block_size``, otherwise ``ValueError``.

    Key derivation
    --------------
    ::

        seed = SHA256(b"predict-chunk-key-chain:v1" || namespace)
        digest_0 = SHA256(seed || pack(>I, tokens[0:chunk]))
        key_0    = hex(digest_0)        # full 64 hex chars
        digest_1 = SHA256(digest_0_bytes || pack(>I, tokens[chunk:2*chunk]))
        key_1    = hex(digest_1)
        …
    """

    def __init__(self, tokens_per_chunk: int = 256, block_size: int = 1):
        if tokens_per_chunk <= 0:
            raise ValueError(
                f"tokens_per_chunk must be > 0, got {tokens_per_chunk}"
            )
        if tokens_per_chunk % block_size != 0:
            raise ValueError(
                f"tokens_per_chunk ({tokens_per_chunk}) must be an integer "
                f"multiple of block_size ({block_size})"
            )
        self._tokens_per_chunk = tokens_per_chunk
        self._block_size = block_size

    @property
    def tokens_per_chunk(self) -> int:
        return self._tokens_per_chunk

    def describe(
        self,
        token_ids: Sequence[int],
        namespace: str,
        end: Optional[int] = None,
    ) -> list[ChunkDescriptor]:
        if end is None:
            end = len(token_ids)

        # Seed: domain-separated namespace hash
        seed = hashlib.sha256(_DOMAIN_SEPARATOR)
        seed.update(namespace.encode())
        prev_digest: bytes = seed.digest()  # 32 raw bytes

        descriptors: list[ChunkDescriptor] = []
        chunk_idx = 0
        pos = 0

        while pos < end:
            chunk_end = min(pos + self._tokens_per_chunk, end)
            h = hashlib.sha256()
            h.update(prev_digest)  # 32 raw bytes from previous round
            for tid in token_ids[pos:chunk_end]:
                h.update(struct.pack(">I", tid))
            digest: bytes = h.digest()       # 32 raw bytes → next prev
            key: str = h.hexdigest()         # full 64 hex chars → chunk key

            descriptors.append(ChunkDescriptor(
                index=chunk_idx,
                start=pos,
                end=chunk_end,
                key=key,
            ))

            prev_digest = digest
            chunk_idx += 1
            pos = chunk_end

        return descriptors


# ── Factory ─────────────────────────────────────────────────────────────


def create_chunk_key_strategy(
    name: str,
    tokens_per_chunk: int = 256,
    block_size: int = 1,
) -> ChunkKeyStrategy:
    """Factory: return a :class:`ChunkKeyStrategy` by *name*.

    Supported names (case-insensitive):

    * ``"chain"``, ``"chained_prefix"`` → :class:`ChainChunkKeyStrategy`
    """
    key = name.lower().replace("-", "_")

    if key in ("chain", "chained_prefix"):
        return ChainChunkKeyStrategy(
            tokens_per_chunk=tokens_per_chunk,
            block_size=block_size,
        )

    raise ValueError(
        f"Unknown chunk-key strategy: {name!r}. "
        f"Supported: chain, chained_prefix"
    )


# ── File path helper ────────────────────────────────────────────────────


def chunk_object_path(
    cache_root: str | Path,
    namespace: str,
    key: str,
    shard: int,
) -> Path:
    """Construct a chunk-object file path.

    Format: ``{root}/v2/chain/{namespace}/{key}/{shard}.cobj``

    The ``v2/chain`` prefix provides version isolation; *key* and *shard*
    are safely joined to prevent directory traversal.
    """
    root = Path(cache_root).resolve()

    if not namespace or not key:
        raise ValueError("namespace and key must be non-empty")
    ns = quote(namespace, safe="")
    encoded_key = quote(key, safe="")
    encoded_shard = quote(str(shard), safe="")
    return root / "v2" / "chain" / ns / encoded_key / f"{encoded_shard}.cobj"
