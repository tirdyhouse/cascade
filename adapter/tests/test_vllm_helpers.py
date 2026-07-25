from __future__ import annotations

import hashlib
import json
import struct
from unittest import mock

import pytest
import torch

from adapter.vllm.chunking import cached_file_path, chunk_file_path, chunk_ranges
from adapter.vllm.go_client import DiskCacheGoClient
from adapter.vllm.hashing import (
    align_to_block_size,
    compute_prompt_hash,
    hash_token_count,
    layer_hash,
    prefix_key,
)


def _expected_hash(token_ids, mm_hashes=()):
    h = hashlib.sha256()
    for tid in token_ids:
        h.update(struct.pack(">I", tid))
    for mh in mm_hashes:
        h.update(mh.encode())
    return h.hexdigest()[:32]


class TestHashingHelpers:
    def test_align_to_block_size_preserves_existing_contract(self):
        assert align_to_block_size(0, 16) == 0
        assert align_to_block_size(1, 16) == 0
        assert align_to_block_size(16, 16) == 0
        assert align_to_block_size(17, 16) == 16
        assert align_to_block_size(33, 16) == 32

    def test_hash_token_count_keeps_at_least_one_token(self):
        assert hash_token_count(0, 16) == 1
        assert hash_token_count(1, 16) == 1
        assert hash_token_count(17, 16) == 16

    def test_compute_prompt_hash_matches_connector_wire_format(self):
        token_ids = [11, 22, 33, 44]
        mm_hashes = ["image-a", "image-b"]
        assert compute_prompt_hash(token_ids, 3, mm_hashes) == _expected_hash(
            token_ids[:3], mm_hashes
        )

    def test_prefix_key_uses_first_block_only(self):
        token_ids = [11, 22, 33, 44]
        assert prefix_key(token_ids, 2) == _expected_hash([11, 22])
        assert prefix_key(token_ids, 16) == _expected_hash(token_ids)

    def test_layer_hash_combines_prompt_hash_and_layer_name(self):
        prompt_hash = "a" * 32
        h = hashlib.sha256()
        h.update(prompt_hash.encode())
        h.update(b"layer.0")
        assert layer_hash(prompt_hash, "layer.0") == h.hexdigest()[:32]


class TestChunkingHelpers:
    def test_chunk_ranges(self):
        assert list(chunk_ranges(0, 4)) == []
        assert list(chunk_ranges(10, 4)) == [
            (0, 0, 4),
            (1, 4, 8),
            (2, 8, 10),
        ]

    def test_chunk_file_path_partitions_by_prefix(self, tmp_path):
        prefix = "abcdef0123456789"
        assert chunk_file_path(tmp_path, prefix, "layer.0", 3) == (
            tmp_path / "ab" / "cd" / prefix / "layer.0" / "3.safetensors"
        )

    def test_cached_file_path_partitions_by_layer_hash(self, tmp_path):
        lh = "1234567890abcdef"
        assert (
            cached_file_path(tmp_path, lh)
            == tmp_path / "12" / "34" / f"{lh}.safetensors"
        )


class TestDiskCacheGoClient:
    def test_post_sends_json_payload(self):
        client = DiskCacheGoClient("http://example.test/")
        with mock.patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = b"ok"
            result = client.post("/put", {"hash": 1, "file_path": "a", "size": 2})

        assert result == b"ok"
        req = urlopen.call_args.args[0]
        assert req.full_url == "http://example.test/put"
        assert json.loads(req.data.decode()) == {"hash": 1, "file_path": "a", "size": 2}
        assert req.headers["Content-type"] == "application/json"

    def test_get_json_encodes_query_parameters(self):
        client = DiskCacheGoClient("http://example.test")
        with mock.patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = (
                b'{"chunks":[0,1]}'
            )
            result = client.get_json(
                "/chunk_list", {"prefix_key": "a b", "layer_name": "layer/0"}
            )

        assert result == {"chunks": [0, 1]}
        assert urlopen.call_args.args[0] == (
            "http://example.test/chunk_list?prefix_key=a+b&layer_name=layer%2F0"
        )

    def test_high_level_methods_preserve_endpoint_payloads(self):
        client = DiskCacheGoClient("http://engine")
        with mock.patch.object(
            client, "post", return_value=b'{"matched_tokens":4}'
        ) as post:
            assert client.match([1, 2], ["mm"], 16) == {"matched_tokens": 4}
            post.assert_called_once_with(
                "/match",
                {"token_ids": [1, 2], "mm_hashes": ["mm"], "block_size": 16},
            )

        with mock.patch.object(
            client, "get_json", return_value={"chunks": [2, 0]}
        ) as get_json:
            assert client.chunk_list("prefix", "layer.0") == [2, 0]
            get_json.assert_called_once_with(
                "/chunk_list",
                {"prefix_key": "prefix", "layer_name": "layer.0"},
            )

        with mock.patch.object(client, "post") as post:
            client.record_retrieved(3)
            post.assert_called_once_with("/retrieved", {"count": 3})

        with mock.patch.object(
            client,
            "post",
            return_value=b'{"matched_chunks":1,"matched_tokens":8,"matched_keys":["k"]}',
        ) as post:
            client.match_chunks(
                "namespace",
                [{"key": "k", "end_tokens": 8}],
                ["tp0-pp0"],
                resolve_shard="tp0-pp0",
            )
            post.assert_called_once_with(
                "/v2/chunks/match",
                {
                    "namespace": "namespace",
                    "candidates": [{"key": "k", "end_tokens": 8}],
                    "required_shards": ["tp0-pp0"],
                    "resolve_shard": "tp0-pp0",
                },
            )


# ═══════════════════════════════════════════════════════════════════
# ChunkKeyStrategy tests
# ═══════════════════════════════════════════════════════════════════


class TestChainChunkKeyStrategy:
    """ChainChunkKeyStrategy: deterministic chain-hash chunk keys."""

    def test_same_tokens_same_keys(self):
        from adapter.vllm.chunk_keys import ChainChunkKeyStrategy

        strategy = ChainChunkKeyStrategy(tokens_per_chunk=4, block_size=1)
        desc1 = strategy.describe([1, 2, 3, 4, 5, 6], "test")
        desc2 = strategy.describe([1, 2, 3, 4, 5, 6], "test")
        assert len(desc1) == len(desc2)
        for d1, d2 in zip(desc1, desc2):
            assert d1.key == d2.key
            assert d1.start == d2.start
            assert d1.end == d2.end

    def test_exact_wire_format_vectors(self):
        """Keys remain compatible with objects written by earlier versions."""
        from adapter.vllm.chunk_keys import ChainChunkKeyStrategy

        strategy = ChainChunkKeyStrategy(tokens_per_chunk=4, block_size=1)
        desc = strategy.describe(
            [0, 1, 2**31, 2**32 - 1, 17],
            "model/ns:1",
        )
        assert [item.key for item in desc] == [
            "48b79fa63431a0a518f1f7789db95f708fbe8bb0de2fc75686504f5c26289f65",
            "d614fd6bad544dd9292fb509fa92b9211889d60e2d1f4a870d4a069e9ee22bf4",
        ]

    def test_different_namespace_different_keys(self):
        from adapter.vllm.chunk_keys import ChainChunkKeyStrategy

        strategy = ChainChunkKeyStrategy(tokens_per_chunk=4, block_size=1)
        desc_a = strategy.describe([1, 2, 3, 4], "ns-a")
        desc_b = strategy.describe([1, 2, 3, 4], "ns-b")
        assert desc_a[0].key != desc_b[0].key

    def test_shared_prefix_same_first_chunk(self):
        """Same prefix (first chunk) should match across longer sequences."""
        from adapter.vllm.chunk_keys import ChainChunkKeyStrategy

        strategy = ChainChunkKeyStrategy(tokens_per_chunk=4, block_size=1)
        short = strategy.describe([1, 2, 3, 4], "test")
        long_ = strategy.describe([1, 2, 3, 4, 5, 6, 7, 8], "test")
        assert short[0].key == long_[0].key

    def test_partial_tail_chunk(self):
        """Last chunk with fewer tokens still produces a valid 64-char key."""
        from adapter.vllm.chunk_keys import ChainChunkKeyStrategy

        strategy = ChainChunkKeyStrategy(tokens_per_chunk=4, block_size=1)
        desc = strategy.describe([1, 2, 3, 4, 5], "test")
        assert len(desc) == 2
        assert desc[0].start == 0
        assert desc[0].end == 4
        assert desc[1].start == 4
        assert desc[1].end == 5
        assert len(desc[0].key) == 64  # full SHA-256 hex
        assert len(desc[1].key) == 64

    def test_chain_dependency(self):
        """Second chunk key depends on first chunk's tokens (chain)."""
        from adapter.vllm.chunk_keys import ChainChunkKeyStrategy

        strategy = ChainChunkKeyStrategy(tokens_per_chunk=4, block_size=1)
        desc1 = strategy.describe([1, 2, 3, 4, 5, 6, 7, 8], "test")
        desc2 = strategy.describe([9, 10, 11, 12, 5, 6, 7, 8], "test")
        # First chunks differ
        assert desc1[0].key != desc2[0].key
        # Second chunks also differ due to chain dependency
        assert desc1[1].key != desc2[1].key

    def test_config_validation(self):
        """tokens_per_chunk must be > 0 and block_size multiple."""
        from adapter.vllm.chunk_keys import ChainChunkKeyStrategy

        with pytest.raises(ValueError, match="must be > 0"):
            ChainChunkKeyStrategy(tokens_per_chunk=0, block_size=1)
        with pytest.raises(ValueError, match="must be an integer multiple"):
            ChainChunkKeyStrategy(tokens_per_chunk=5, block_size=2)

    def test_explicit_end(self):
        """Passing explicit end limits the described range."""
        from adapter.vllm.chunk_keys import ChainChunkKeyStrategy

        strategy = ChainChunkKeyStrategy(tokens_per_chunk=4, block_size=1)
        desc = strategy.describe([1, 2, 3, 4, 5, 6, 7, 8], "test", end=5)
        assert len(desc) == 2
        assert desc[0].start == 0
        assert desc[0].end == 4
        assert desc[1].start == 4
        assert desc[1].end == 5


class TestCreateChunkKeyStrategy:
    """Factory function."""

    def test_chain(self):
        from adapter.vllm.chunk_keys import (
            ChainChunkKeyStrategy,
            create_chunk_key_strategy,
        )

        strategy = create_chunk_key_strategy("chain", tokens_per_chunk=8)
        assert isinstance(strategy, ChainChunkKeyStrategy)
        assert strategy.tokens_per_chunk == 8


class TestChunkObjectPathHelper:

    def test_path_format(self, tmp_path):
        from adapter.vllm.chunk_keys import chunk_object_path

        path = chunk_object_path(tmp_path, "mymodel", "key123", 0)
        assert path == tmp_path / "v2" / "chain" / "mymodel" / "key123" / "0.cobj"

    def test_safe_sanitises(self, tmp_path):
        from adapter.vllm.chunk_keys import chunk_object_path

        path = chunk_object_path(tmp_path, "../escape", "key", 0)
        parts = path.relative_to(tmp_path).parts
        # no ".." component allowed; dots filtered out
        assert ".." not in parts


class TestCanonicalTensorLayout:
    """Verify the canonical [T, 2, ...] layout contract."""

    @staticmethod
    def _ensure_vllm_mock():
        """Ensure vllm mock modules exist so tensor_ops imports succeed."""
        import sys

        if "vllm.v1.attention.backends.triton_attn" in sys.modules:
            return
        from unittest import mock

        _triton = mock.MagicMock()
        _triton.TritonAttentionMetadata = type("TritonAttentionMetadata", (), {})
        for _mod in (
            "vllm",
            "vllm.v1",
            "vllm.v1.attention",
            "vllm.v1.attention.backends",
            "vllm.v1.attention.backends.triton_attn",
        ):
            if _mod not in sys.modules:
                sys.modules[_mod] = mock.MagicMock()
        sys.modules["vllm.v1.attention.backends.triton_attn"] = _triton

    def test_extract_non_triton(self):
        self._ensure_vllm_mock()
        from adapter.vllm.tensor_ops import extract_kv_from_layer

        class MockAttnMeta:
            pass

        num_pages = 8
        page_size = 4
        hidden_dim = 64
        layer = torch.randn(2, num_pages, page_size, hidden_dim)
        slot_mapping = torch.tensor([0, 5, 10])
        result = extract_kv_from_layer(layer, slot_mapping, MockAttnMeta(), page_size)
        assert result.shape == (3, 2, hidden_dim)
        assert result.is_contiguous()
        for i, slot in enumerate(slot_mapping):
            page = slot // page_size
            offset = slot % page_size
            assert torch.equal(result[i, 0], layer[0, page, offset])
            assert torch.equal(result[i, 1], layer[1, page, offset])

    def test_inject_non_triton(self):
        self._ensure_vllm_mock()
        from adapter.vllm.tensor_ops import extract_kv_from_layer, inject_kv_into_layer

        class MockAttnMeta:
            pass

        num_pages = 8
        page_size = 4
        hidden_dim = 64
        layer = torch.zeros(2, num_pages, page_size, hidden_dim)
        slot_mapping = torch.tensor([0, 5, 10])
        src = torch.randn(3, 2, hidden_dim)
        inject_kv_into_layer(layer, src, slot_mapping, MockAttnMeta(), page_size)
        extracted = extract_kv_from_layer(
            layer, slot_mapping, MockAttnMeta(), page_size
        )
        assert torch.equal(extracted, src)

    def test_non_triton_preserves_multiple_tail_dimensions(self):
        self._ensure_vllm_mock()
        from adapter.vllm.tensor_ops import (
            extract_kv_from_layer,
            inject_kv_into_layer,
        )

        class MockAttnMeta:
            pass

        layer = torch.zeros(2, 3, 4, 5, 7)
        slot_mapping = torch.tensor([0, 5, 10])
        src = torch.randn(3, 2, 5, 7)

        inject_kv_into_layer(layer, src, slot_mapping, MockAttnMeta(), block_size=4)
        extracted = extract_kv_from_layer(
            layer, slot_mapping, MockAttnMeta(), block_size=4
        )

        assert extracted.shape == (3, 2, 5, 7)
        assert torch.equal(extracted, src)

    def test_extract_non_triton_contiguity_guarantee(self):
        self._ensure_vllm_mock()
        from adapter.vllm.tensor_ops import extract_kv_from_layer

        class MockAttnMeta:
            pass

        layer = torch.randn(2, 4, 4, 32)
        slot_mapping = torch.tensor([0, 1, 2, 3, 4, 5])
        result = extract_kv_from_layer(layer, slot_mapping, MockAttnMeta(), 4)
        assert result.is_contiguous()
        assert result.shape[0] == len(slot_mapping)
        assert result.shape[1] == 2

    def test_triton_stubbed_contiguous(self):
        self._ensure_vllm_mock()
        from adapter.vllm.tensor_ops import extract_kv_from_layer
        from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata

        class MockTritonMeta(TritonAttentionMetadata):
            pass

        num_blocks = 4
        num_heads = 8
        head_dim = 128
        layer = torch.randn(num_blocks, num_heads, 16, head_dim)
        block_size = 16
        slot_mapping = torch.tensor([0, 16, 32])
        result = extract_kv_from_layer(
            layer, slot_mapping, MockTritonMeta(), block_size
        )
        assert result.shape == (3, num_heads, head_dim)
        assert result.is_contiguous()
