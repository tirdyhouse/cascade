from __future__ import annotations

import hashlib
import json
import struct
from unittest import mock

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
        assert compute_prompt_hash(token_ids, 3, mm_hashes) == _expected_hash(token_ids[:3], mm_hashes)

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
        assert cached_file_path(tmp_path, lh) == tmp_path / "12" / "34" / f"{lh}.safetensors"


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
            urlopen.return_value.__enter__.return_value.read.return_value = b'{"chunks":[0,1]}'
            result = client.get_json("/chunk_list", {"prefix_key": "a b", "layer_name": "layer/0"})

        assert result == {"chunks": [0, 1]}
        assert urlopen.call_args.args[0] == (
            "http://example.test/chunk_list?prefix_key=a+b&layer_name=layer%2F0"
        )

    def test_high_level_methods_preserve_endpoint_payloads(self):
        client = DiskCacheGoClient("http://engine")
        with mock.patch.object(client, "post", return_value=b'{"matched_tokens":4}') as post:
            assert client.match([1, 2], ["mm"], 16) == {"matched_tokens": 4}
            post.assert_called_once_with(
                "/match",
                {"token_ids": [1, 2], "mm_hashes": ["mm"], "block_size": 16},
            )

        with mock.patch.object(client, "get_json", return_value={"chunks": [2, 0]}) as get_json:
            assert client.chunk_list("prefix", "layer.0") == [2, 0]
            get_json.assert_called_once_with(
                "/chunk_list",
                {"prefix_key": "prefix", "layer_name": "layer.0"},
            )

        with mock.patch.object(client, "post") as post:
            client.record_retrieved(3)
            post.assert_called_once_with("/retrieved", {"count": 3})
