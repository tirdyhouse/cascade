from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


def _stub_module(name: str, *, package: bool = False):
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=package)
    if package:
        module.__path__ = []
    return module


def _install_vllm_stubs_if_needed():
    try:
        has_vllm = importlib.util.find_spec("vllm") is not None
    except ValueError:
        has_vllm = False
    if has_vllm:
        return

    class KVConnectorMetadata:
        pass

    class KVConnectorBase_V1:
        def __init__(self, *args, **kwargs):
            pass

    base = _stub_module("vllm.distributed.kv_transfer.kv_connector.v1.base")
    base.KVConnectorMetadata = KVConnectorMetadata
    base.KVConnectorBase_V1 = KVConnectorBase_V1
    logger_mod = _stub_module("vllm.logger")
    logger_mod.init_logger = lambda name: mock.Mock(name=name)

    modules = {
        "vllm": _stub_module("vllm", package=True),
        "vllm.distributed": _stub_module("vllm.distributed", package=True),
        "vllm.distributed.kv_transfer": _stub_module("vllm.distributed.kv_transfer", package=True),
        "vllm.distributed.kv_transfer.kv_connector": _stub_module(
            "vllm.distributed.kv_transfer.kv_connector", package=True
        ),
        "vllm.distributed.kv_transfer.kv_connector.v1": _stub_module(
            "vllm.distributed.kv_transfer.kv_connector.v1", package=True
        ),
        "vllm.distributed.kv_transfer.kv_connector.v1.base": base,
        "vllm.logger": logger_mod,
    }
    for name, module in modules.items():
        sys.modules[name] = module


_install_vllm_stubs_if_needed()

from adapter.vllm import connector, connector_v21
from adapter.vllm.connector_common import DiskCacheConnectorCommonMixin


@dataclass
class Req:
    token_ids: list[int]
    block_ids: list[int]
    block_size: int
    is_store: bool
    mm_hashes: list[str]
    num_tokens: int
    prompt_hash: str = ""


class DummyConnector(DiskCacheConnectorCommonMixin):
    def __init__(self, tmp_path: Path):
        self.cache_root = tmp_path
        self._block_size = 4
        self._tokens_per_chunk = 3
        self._connected = True
        self._requests_need_load = {}
        self._go_chunk_put_calls = []
        self._go_put_calls = []
        self._go_chunk_list_result = []
        self._retrieved = 0
        self.target_device = "auto"
        self._storage = mock.Mock()
        self._storage.save.side_effect = lambda path, _tensor: path.write_bytes(b"saved")

    def _go_chunk_list(self, prefix_key, layer_name):
        return list(self._go_chunk_list_result)

    def _go_chunk_put(self, prefix_key, layer_name, chunk_idx, num_tokens):
        self._go_chunk_put_calls.append((prefix_key, layer_name, chunk_idx, num_tokens))

    def _go_put(self, hash_val, file_path, size):
        self._go_put_calls.append((hash_val, file_path, size))

    def _go_record_retrieved(self, count=1):
        self._retrieved += count

    def _prefix_key(self, token_ids):
        return "abcdef0123456789abcdef0123456789"


def test_build_slot_mapping_matches_block_layout(tmp_path):
    conn = DummyConnector(tmp_path)
    req = Req(token_ids=list(range(10)), block_ids=[2, 5], block_size=4, is_store=False, mm_hashes=[], num_tokens=6)

    assert conn._build_slot_mapping(req).tolist() == [8, 9, 10, 11, 20, 21]


def test_save_layer_chunks_skips_existing_full_chunks_but_saves_partial(tmp_path):
    conn = DummyConnector(tmp_path)
    kv_cache = torch.arange(2 * 7 * 2).reshape(2, 7, 2)
    conn._go_chunk_list_result = [0]

    conn._save_layer_chunks(
        "abcdef0123456789abcdef0123456789",
        "layer.0",
        kv_cache,
        num_tokens=7,
        existing_chunks={0},
    )

    assert conn._storage.save.call_count == 2
    saved_paths = [call.args[0] for call in conn._storage.save.call_args_list]
    assert saved_paths == [
        tmp_path / "ab" / "cd" / "abcdef0123456789abcdef0123456789" / "layer.0" / "1.safetensors",
        tmp_path / "ab" / "cd" / "abcdef0123456789abcdef0123456789" / "layer.0" / "2.safetensors",
    ]
    assert torch.equal(conn._storage.save.call_args_list[0].args[1], kv_cache[:, 3:6, :])
    assert torch.equal(conn._storage.save.call_args_list[1].args[1], kv_cache[:, 6:7, :])
    assert conn._go_chunk_put_calls == [
        ("abcdef0123456789abcdef0123456789", "layer.0", 1, 3),
        ("abcdef0123456789abcdef0123456789", "layer.0", 2, 1),
    ]
    assert conn._go_put_calls == [
        (int("abcdef0123456789", 16), "ab/cd/abcdef0123456789abcdef0123456789/layer.0/1.safetensors", 5),
        (int("abcdef0123456789", 16), "ab/cd/abcdef0123456789abcdef0123456789/layer.0/2.safetensors", 5),
    ]


def test_load_layer_chunks_records_only_successful_loads(tmp_path):
    conn = DummyConnector(tmp_path)
    prefix = "abcdef0123456789abcdef0123456789"
    existing = conn._chunk_file_path(prefix, "layer.0", 0)
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"placeholder")
    conn._go_chunk_list_result = [0, 1]
    loaded = torch.ones(2, 3, 2, dtype=torch.float32)
    conn._storage.load.return_value = loaded
    kv_cache_layer = torch.zeros(8, 2, dtype=torch.float16)
    slot_mapping = torch.arange(2)
    attn_metadata = {"layer.0": "layer-attn"}

    with mock.patch("adapter.vllm.connector_common.inject_kv_into_layer") as inject:
        conn._load_layer_chunks(prefix, "layer.0", kv_cache_layer, slot_mapping, 2, attn_metadata)

    conn._storage.load.assert_called_once_with(existing, device="cpu")
    assert conn._retrieved == 1
    injected_kv = inject.call_args.args[1]
    assert injected_kv.dtype == kv_cache_layer.dtype
    assert injected_kv.shape == (2, 2, 2)
    inject.assert_called_once_with(kv_cache_layer, injected_kv, slot_mapping, "layer-attn", conn._block_size)


def test_build_connector_meta_adds_store_and_load_then_clears_for_partial_hit(tmp_path):
    conn = DummyConnector(tmp_path)
    conn._requests_need_load["req-1"] = SimpleNamespace(
        all_token_ids=[1, 2, 3, 4, 5, 6],
        mm_features=[],
        disk_cache_loaded_aligned_tokens=2,
    )
    new_req = SimpleNamespace(
        req_id="req-1",
        prompt_token_ids=[1, 2, 3, 4, 5, 6],
        mm_features=[],
        block_ids=[[7, 8]],
    )
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[new_req],
        scheduled_cached_reqs=SimpleNamespace(req_ids=["req-1"], new_block_ids=[[[9, 10]]]),
    )

    meta = conn.build_connector_meta(scheduler_output)

    assert [(req.is_store, req.block_ids, req.num_tokens) for req in meta.requests] == [
        (True, [7, 8], 4),
        (False, [7, 8], 4),
        (False, [9, 10], 4),
    ]
    assert conn._requests_need_load == {}


def test_build_connector_meta_skips_store_when_disk_cache_fully_covers_request(tmp_path):
    conn = DummyConnector(tmp_path)
    conn._requests_need_load["req-1"] = SimpleNamespace(
        all_token_ids=[1, 2, 3, 4, 5, 6],
        mm_features=[],
        disk_cache_loaded_aligned_tokens=4,
    )
    new_req = SimpleNamespace(
        req_id="req-1",
        prompt_token_ids=[1, 2, 3, 4, 5, 6],
        mm_features=[],
        block_ids=[[7, 8]],
    )
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[new_req],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
    )

    meta = conn.build_connector_meta(scheduler_output)

    assert [(req.is_store, req.block_ids, req.num_tokens) for req in meta.requests] == [
        (False, [7, 8], 4),
    ]
    assert conn._requests_need_load == {}


def test_connector_variants_preserve_match_token_differences():
    common_attrs = {
        "_block_size": 4,
        "_go_match": mock.Mock(return_value={"matched_tokens": 12}),
    }
    request = SimpleNamespace(
        prompt_token_ids=list(range(20)),
        mm_features=[],
        request_id="req",
    )
    regular = object.__new__(connector.DiskCacheConnector)
    regular.__dict__.update(common_attrs)
    v21 = object.__new__(connector_v21.DiskCacheConnector)
    v21.__dict__.update(common_attrs)

    assert regular.get_num_new_matched_tokens(request, 4) == (8, False)
    assert regular.get_num_new_matched_tokens(request, 12) == (0, False)
    assert v21.get_num_new_matched_tokens(request, 4) == (8, False)
    assert v21.get_num_new_matched_tokens(request, 12) == (0, False)

    common_attrs["_go_match"] = mock.Mock(return_value={"matched_tokens": 16})
    regular.__dict__.update(common_attrs)
    v21.__dict__.update(common_attrs)
    assert regular.get_num_new_matched_tokens(request, 8) == (4, False)
    assert v21.get_num_new_matched_tokens(request, 8) == (8, False)


def test_v21_save_layer_keeps_connected_guard(tmp_path):
    regular = object.__new__(connector.DiskCacheConnector)
    regular._get_connector_metadata = mock.Mock(return_value=object())
    regular._is_disk_cache_meta = mock.Mock(return_value=False)
    regular.save_kv_layer("layer.0", object(), object())
    regular._get_connector_metadata.assert_called_once()

    v21 = object.__new__(connector_v21.DiskCacheConnector)
    v21._connected = False
    v21._get_connector_metadata = mock.Mock()
    v21.save_kv_layer("layer.0", object(), object())
    v21._get_connector_metadata.assert_not_called()
