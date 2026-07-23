"""Tests for the v2 chunk-object-based connector.

These tests run entirely on CPU with mocked Go engine and storage
backend — no GPU, no GDS driver, no real vLLM required.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import pytest


def _stub_module(name: str, *, package: bool = False):
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(
        name, loader=None, is_package=package
    )
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

    base = _stub_module(
        "vllm.distributed.kv_transfer.kv_connector.v1.base"
    )
    base.KVConnectorMetadata = KVConnectorMetadata
    base.KVConnectorBase_V1 = KVConnectorBase_V1
    logger_mod = _stub_module("vllm.logger")
    logger_mod.init_logger = lambda name: mock.Mock(name=name)

    modules = {
        "vllm": _stub_module("vllm", package=True),
        "vllm.distributed": _stub_module(
            "vllm.distributed", package=True
        ),
        "vllm.distributed.kv_transfer": _stub_module(
            "vllm.distributed.kv_transfer", package=True
        ),
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
from adapter.vllm.connector_common import (
    DiskCacheConnectorCommonMixin,
    DiskCacheMeta,
    _ReqMeta,
    _PendingLoadSpec,
    _RequestTracker,
    _WriterState,
)
from adapter.vllm.chunk_keys import ChunkDescriptor


# ═══════════════════════════════════════════════════════════════════
# Test helpers
# ═══════════════════════════════════════════════════════════════════


class DummyConnector(DiskCacheConnectorCommonMixin):
    """Minimal connector for unit tests.

    Sets only the attributes needed by each test.  Does NOT call
    ``super().__init__`` to avoid depending on vLLM config objects.
    """

    def __init__(self, tmp_path: Path):
        self.cache_root = tmp_path
        self._block_size = 4
        self._chunk_size = 8  # must be multiple of block_size
        self._connected = True
        self.target_device = "auto"
        self._namespace = "test-ns"
        self._shard = "tp0-pp0"
        self._shard_idx = 0
        self._required_shards = ["tp0-pp0"]
        self._expected_layers: list[str] = []
        self._strategy = mock.Mock()
        self._strategy.describe.return_value = []
        self._go = mock.Mock()
        self._go.match_chunks.return_value = {
            "matched_chunks": 0,
            "matched_tokens": 0,
            "matched_keys": None,
        }
        self._go.resolve_chunks.return_value = []
        self._go.commit_chunks.return_value = None
        self._go.chunks_retrieved.return_value = None
        self._storage = mock.Mock()
        self._pending_loads: dict[str, _PendingLoadSpec] = {}
        self._request_trackers: dict[str, _RequestTracker] = {}
        self._active_writers: dict[tuple[str, str], _WriterState] = {}
        self._block_ids_with_load_errors: set[int] = set()
        self._failed_load_request_ids: set[str] = set()
        self._registered_kv_caches: dict[str, torch.Tensor] = {}
        self._load_error: str | None = None
        self._connector_meta: DiskCacheMeta | None = None

    def _get_connector_metadata(self) -> DiskCacheMeta | None:
        return self._connector_meta

    def _is_disk_cache_meta(self, meta) -> bool:
        return isinstance(meta, DiskCacheMeta)


def _make_chunk_descriptors(token_ids, chunk_size):
    """Create chain-hash descriptors for test assertions.

    Uses the real ``ChainChunkKeyStrategy``.
    """
    from adapter.vllm.chunk_keys import ChainChunkKeyStrategy

    strategy = ChainChunkKeyStrategy(
        tokens_per_chunk=chunk_size, block_size=1
    )
    return strategy.describe(token_ids, "test-ns")


def _make_scheduler_output(
    new_reqs=None,
    cached_req_ids=None,
    cached_new_block_ids=None,
    num_scheduled_tokens=None,
):
    """Build a SimpleNamespace mimicking vLLM scheduler output."""
    if new_reqs is None:
        new_reqs = []
    if cached_req_ids is None:
        cached_req_ids = []
    if cached_new_block_ids is None:
        cached_new_block_ids = []
    if num_scheduled_tokens is None:
        num_scheduled_tokens = {}
    return SimpleNamespace(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=cached_req_ids,
            new_block_ids=cached_new_block_ids,
        ),
        num_scheduled_tokens=num_scheduled_tokens,
    )


def _make_new_req(
    req_id="req-1",
    token_ids=None,
    block_ids=None,
    num_computed_tokens=0,
    mm_features=None,
):
    if token_ids is None:
        token_ids = [1, 2, 3, 4, 5]
    if block_ids is None:
        block_ids = [[7, 8]]
    if mm_features is None:
        mm_features = []
    return SimpleNamespace(
        req_id=req_id,
        request_id=req_id,
        all_token_ids=list(token_ids),
        prompt_token_ids=list(token_ids),
        block_ids=block_ids,
        num_computed_tokens=num_computed_tokens,
        mm_features=mm_features,
    )


# ═══════════════════════════════════════════════════════════════════
# _build_slot_mapping tests
# ═══════════════════════════════════════════════════════════════════


def test_build_slot_mapping_matches_block_layout(tmp_path):
    conn = DummyConnector(tmp_path)
    mapping = conn._build_slot_mapping(
        block_ids=[2, 5], block_size=4, num_tokens=6
    )
    assert mapping.tolist() == [8, 9, 10, 11, 20, 21]


def test_build_slot_mapping_empty_on_no_tokens(tmp_path):
    conn = DummyConnector(tmp_path)
    mapping = conn._build_slot_mapping(
        block_ids=[0], block_size=4, num_tokens=0
    )
    assert mapping.numel() == 0


# ═══════════════════════════════════════════════════════════════════
# _get_num_new_matched_tokens tests
# ═══════════════════════════════════════════════════════════════════


def test_get_num_new_matched_tokens_partial_match(tmp_path):
    """Partial match returns correct external token count."""
    conn = DummyConnector(tmp_path)
    conn._chunk_size = 8

    tokens = list(range(20))
    conn._strategy.describe.return_value = _make_chunk_descriptors(
        tokens[:8], conn._chunk_size
    )
    descs = conn._strategy.describe.return_value
    conn._go.match_chunks.return_value = {
        "matched_chunks": len(descs),
        "matched_tokens": 8,
        "matched_keys": [desc.key for desc in descs],
    }

    request = SimpleNamespace(
        request_id="req-1",
        all_token_ids=tokens,
    )
    external, _ = conn._get_num_new_matched_tokens(request, 0)
    # matched=8, local=0 => external=8
    assert external == 8

    pending = conn._pending_loads.get("req-1")
    assert pending is not None
    assert pending.matched_tokens == 8
    assert pending.local_cached == 0


def test_get_num_new_matched_tokens_full_match_leaves_one_token(tmp_path):
    """Full match (all except last token) returns len-1-local."""
    conn = DummyConnector(tmp_path)
    conn._chunk_size = 8

    # tokens length 17, end = (17-1)//8*8 = 16
    tokens = list(range(17))
    conn._strategy.describe.return_value = _make_chunk_descriptors(
        tokens[:16], conn._chunk_size
    )
    descs = conn._strategy.describe.return_value
    conn._go.match_chunks.return_value = {
        "matched_chunks": len(descs),
        "matched_tokens": 16,
        "matched_keys": [desc.key for desc in descs],
    }

    request = SimpleNamespace(
        request_id="req-1",
        all_token_ids=tokens,
    )
    # local=0: external = 16 - 0 = 16
    external, _ = conn._get_num_new_matched_tokens(request, 0)
    assert external == 16

    # local=8: external = 16 - 8 = 8
    external, _ = conn._get_num_new_matched_tokens(request, 8)
    assert external == 8

    # local=16: no new tokens
    external, _ = conn._get_num_new_matched_tokens(request, 16)
    assert external == 0


def test_get_num_new_matched_tokens_no_match_when_too_short(tmp_path):
    """Less than 2 tokens returns 0."""
    conn = DummyConnector(tmp_path)
    request = SimpleNamespace(
        request_id="req-1",
        all_token_ids=[42],
    )
    external, _ = conn._get_num_new_matched_tokens(request, 0)
    assert external == 0
    assert "req-1" not in conn._pending_loads


def test_get_num_new_matched_tokens_go_failure_returns_zero(tmp_path):
    """Go/HTTP failure returns 0, no false hit."""
    conn = DummyConnector(tmp_path)
    conn._chunk_size = 8
    conn._strategy.describe.return_value = _make_chunk_descriptors(
        list(range(20))[:8], conn._chunk_size
    )
    conn._go.match_chunks.side_effect = RuntimeError("engine down")

    request = SimpleNamespace(
        request_id="req-1",
        all_token_ids=list(range(20)),
    )
    external, _ = conn._get_num_new_matched_tokens(request, 0)
    assert external == 0


def test_get_num_new_matched_tokens_invalid_boundary_discarded(tmp_path):
    """Match returning unmatched boundary is discarded."""
    conn = DummyConnector(tmp_path)
    conn._chunk_size = 8
    descs = _make_chunk_descriptors(list(range(20))[:8], conn._chunk_size)
    conn._strategy.describe.return_value = descs
    # matched_tokens=5 is not any descriptor.end (descs have end=8)
    conn._go.match_chunks.return_value = {
        "matched_chunks": 1,
        "matched_tokens": 5,
        "matched_keys": [descs[0].key],
    }

    request = SimpleNamespace(
        request_id="req-1",
        all_token_ids=list(range(20)),
    )
    external, _ = conn._get_num_new_matched_tokens(request, 0)
    assert external == 0


@pytest.mark.parametrize(
    "response",
    [
        {"matched_chunks": 1, "matched_tokens": 8},
        {"matched_chunks": True, "matched_tokens": 8, "matched_keys": ["x"]},
        {"matched_chunks": 1, "matched_tokens": 8, "matched_keys": ["wrong"]},
        {"matched_chunks": 2, "matched_tokens": 8, "matched_keys": ["x"]},
    ],
)
def test_get_num_new_matched_tokens_rejects_inconsistent_response(
    tmp_path, response
):
    conn = DummyConnector(tmp_path)
    descs = _make_chunk_descriptors(list(range(20))[:8], 8)
    conn._strategy.describe.return_value = descs
    conn._go.match_chunks.return_value = response
    request = SimpleNamespace(
        request_id="req-1", all_token_ids=list(range(20))
    )

    assert conn._get_num_new_matched_tokens(request, 0) == (0, False)
    assert "req-1" not in conn._pending_loads


# ═══════════════════════════════════════════════════════════════════
# update_state_after_alloc tests
# ═══════════════════════════════════════════════════════════════════


def test_update_state_after_alloc_marks_can_load(tmp_path):
    conn = DummyConnector(tmp_path)
    # Pre-set a pending spec
    conn._pending_loads["req-1"] = _PendingLoadSpec(
        matched_tokens=8, descriptors=[], local_cached=0
    )
    request = SimpleNamespace(request_id="req-1")

    class MockBlocks:
        def get_block_ids(self):
            return [3, 4]

    conn.update_state_after_alloc(request, MockBlocks(), 8)

    pending = conn._pending_loads.get("req-1")
    assert pending is not None
    assert pending.can_load
    assert pending.allocated_block_ids == [3, 4]


def test_update_state_after_alloc_rejects_mismatch(tmp_path):
    conn = DummyConnector(tmp_path)
    conn._pending_loads["req-1"] = _PendingLoadSpec(
        matched_tokens=8, descriptors=[], local_cached=0
    )
    request = SimpleNamespace(request_id="req-1")

    conn.update_state_after_alloc(request, [3, 4], 5)
    # num_external_tokens(5) != expected(8) => reject
    pending = conn._pending_loads.get("req-1")
    assert pending is not None
    assert not pending.can_load


# ═══════════════════════════════════════════════════════════════════
# build_connector_meta tests
# ═══════════════════════════════════════════════════════════════════


def test_build_connector_meta_new_req_store_tokens(tmp_path):
    conn = DummyConnector(tmp_path)
    conn._chunk_size = 8

    # A request with 16 computed tokens + 8 scheduled = 24 total
    # store_tokens = 24 // 8 * 8 = 24 (3 complete chunks)
    new_req = _make_new_req(
        req_id="req-1",
        token_ids=list(range(30)),
        block_ids=[[7, 8]],
        num_computed_tokens=16,
    )
    scheduler_output = _make_scheduler_output(
        new_reqs=[new_req],
        num_scheduled_tokens={"req-1": 8},
    )
    conn._strategy.describe.return_value = _make_chunk_descriptors(
        list(range(30))[:24], conn._chunk_size
    )

    meta = conn.build_connector_meta(scheduler_output)

    store_reqs = [r for r in meta.requests if r.is_store]
    assert len(store_reqs) == 1
    assert store_reqs[0].store_tokens == 24
    assert store_reqs[0].block_ids == [7, 8]
    assert "req-1" in conn._request_trackers


def test_build_connector_meta_load_tokens(tmp_path):
    conn = DummyConnector(tmp_path)
    conn._chunk_size = 8

    # Pre-set a pending load with can_load=True
    descs = _make_chunk_descriptors(list(range(16)), conn._chunk_size)
    conn._pending_loads["req-1"] = _PendingLoadSpec(
        matched_tokens=16,
        descriptors=descs,
        local_cached=0,
        can_load=True,
        allocated_block_ids=[3, 4],
    )

    new_req = _make_new_req(
        req_id="req-1",
        token_ids=list(range(20)),
        block_ids=[[7, 8]],
    )
    scheduler_output = _make_scheduler_output(
        new_reqs=[new_req],
        num_scheduled_tokens={"req-1": 4},
    )
    conn._strategy.describe.return_value = _make_chunk_descriptors(
        list(range(20))[:4], conn._chunk_size
    )

    meta = conn.build_connector_meta(scheduler_output)

    load_reqs = [r for r in meta.requests if r.is_load]
    assert len(load_reqs) == 1
    assert load_reqs[0].load_tokens == 16
    assert load_reqs[0].block_ids == [3, 4]  # from alloc, not original


def test_build_connector_meta_cleans_finished(tmp_path):
    conn = DummyConnector(tmp_path)
    conn._request_trackers["done-req"] = _RequestTracker()
    conn._pending_loads["done-req"] = _PendingLoadSpec(
        matched_tokens=8, descriptors=[], local_cached=0
    )

    scheduler_output = _make_scheduler_output()
    conn.build_connector_meta(
        scheduler_output, finished_req_ids=["done-req"]
    )

    assert "done-req" not in conn._request_trackers
    assert "done-req" not in conn._pending_loads


def test_build_connector_meta_chunked_prefill_no_uncomputed(tmp_path):
    """Chunked prefill does NOT save uncomputed tokens."""
    conn = DummyConnector(tmp_path)
    conn._chunk_size = 8

    # Request with 3 computed + 5 scheduled = 8 total compute
    # store_tokens = 8 // 8 * 8 = 8 (1 complete chunk)
    new_req = _make_new_req(
        req_id="req-1",
        token_ids=list(range(20)),
        block_ids=[[7, 8]],
        num_computed_tokens=3,
    )
    scheduler_output = _make_scheduler_output(
        new_reqs=[new_req],
        num_scheduled_tokens={"req-1": 5},
    )
    conn._strategy.describe.return_value = _make_chunk_descriptors(
        list(range(20))[:8], conn._chunk_size
    )

    meta = conn.build_connector_meta(scheduler_output)

    store_reqs = [r for r in meta.requests if r.is_store]
    assert len(store_reqs) == 1
    # Only complete chunks: 8 tokens (1 chunk)
    assert store_reqs[0].store_tokens == 8
    for d in store_reqs[0].descriptors:
        assert d.end <= 8


# ═══════════════════════════════════════════════════════════════════
# get_block_ids_with_load_errors tests
# ═══════════════════════════════════════════════════════════════════


def test_get_block_ids_with_load_errors(tmp_path):
    conn = DummyConnector(tmp_path)
    conn._block_ids_with_load_errors.update([3, 7])

    result = conn.get_block_ids_with_load_errors()
    assert result == {3, 7}
    assert conn.get_block_ids_with_load_errors() == set()


# ═══════════════════════════════════════════════════════════════════
# save_kv_layer + wait_for_save integration tests
# ═══════════════════════════════════════════════════════════════════


def test_save_and_commit_with_real_chunk_object(tmp_path):
    """Save via save_kv_layer and commit via wait_for_save with real
    ChunkObjectWriter and mocked Go client."""
    conn = DummyConnector(tmp_path)
    conn._namespace = "test-ns"
    conn._expected_layers = ["k_layer", "v_layer"]

    from adapter.vllm.chunk_keys import ChainChunkKeyStrategy

    strategy = ChainChunkKeyStrategy(
        tokens_per_chunk=8, block_size=1
    )
    conn._strategy = strategy

    token_ids = list(range(16))
    descs = strategy.describe(token_ids, conn._namespace)
    assert len(descs) == 2  # two 8-token chunks

    req_meta = _ReqMeta(
        request_id="req-1",
        token_ids=token_ids,
        block_ids=[0, 1],
        block_size=8,
        is_store=True,
        is_load=False,
        store_tokens=16,
        load_tokens=0,
        namespace=conn._namespace,
        shard=conn._shard,
        expected_layers=conn._expected_layers,
        descriptors=descs,
    )

    meta = DiskCacheMeta()
    meta.add(req_meta)
    conn._connector_meta = meta

    attn_meta = SimpleNamespace()
    kv_tensor = torch.randn(16, 2, 32)
    kv_layer = SimpleNamespace(kv_cache=kv_tensor)

    with mock.patch(
        "adapter.vllm.connector_common.extract_kv_from_layer",
        return_value=kv_tensor,
    ):
        # Save k_layer
        conn.save_kv_layer("k_layer", kv_layer, attn_meta)
        # Save v_layer
        conn.save_kv_layer("v_layer", kv_layer, attn_meta)

    # Two writers should exist (one per chunk)
    assert len(conn._active_writers) == 2

    # Wait for save: should finalize and commit
    conn.wait_for_save()

    # After commit, writers should be cleaned up
    assert len(conn._active_writers) == 0

    conn._go.commit_chunks.assert_called_once()
    call_args = conn._go.commit_chunks.call_args[0][0]
    assert len(call_args) == 2  # two objects committed


def test_missing_layer_aborts_writer(tmp_path):
    """When one layer fails, the writer is aborted."""
    conn = DummyConnector(tmp_path)
    conn._namespace = "test-ns"
    conn._expected_layers = ["k_layer", "v_layer"]
    conn._strategy.describe.return_value = [
        ChunkDescriptor(index=0, start=0, end=8, key="chunk0")
    ]

    req_meta = _ReqMeta(
        request_id="req-1",
        token_ids=list(range(8)),
        block_ids=[0],
        block_size=8,
        is_store=True,
        is_load=False,
        store_tokens=8,
        load_tokens=0,
        namespace=conn._namespace,
        shard=conn._shard,
        expected_layers=conn._expected_layers,
        descriptors=[
            ChunkDescriptor(index=0, start=0, end=8, key="chunk0")
        ],
    )
    meta = DiskCacheMeta()
    meta.add(req_meta)
    conn._connector_meta = meta

    mock_writer = mock.Mock()
    mock_writer.add_layer.side_effect = RuntimeError("write failed")
    mock_writer.final_path = tmp_path / "dummy.cobj"

    with (
        mock.patch(
            "adapter.vllm.connector_common.ChunkObjectWriter",
            return_value=mock_writer,
        ),
        mock.patch(
            "adapter.vllm.connector_common.extract_kv_from_layer",
            return_value=torch.randn(8, 2, 32),
        ),
    ):
        conn.save_kv_layer(
            "k_layer",
            SimpleNamespace(kv_cache=torch.randn(8, 2, 32)),
            SimpleNamespace(),
        )

    assert mock_writer.abort.called


def test_load_failure_records_block_ids(tmp_path):
    """Load failure records block IDs and raises."""
    conn = DummyConnector(tmp_path)
    conn._namespace = "test-ns"
    conn._expected_layers = ["k_layer"]
    conn._shard_idx = 0

    req_meta = _ReqMeta(
        request_id="req-1",
        token_ids=list(range(16)),
        block_ids=[3, 4],
        block_size=8,
        is_store=False,
        is_load=True,
        store_tokens=0,
        load_tokens=8,
        namespace=conn._namespace,
        shard=conn._shard,
        expected_layers=conn._expected_layers,
        descriptors=[
            ChunkDescriptor(index=0, start=0, end=8, key="chunk0")
        ],
    )

    conn._go.resolve_chunks.return_value = [
        {"key": "chunk0", "file_path": "nonexistent.cobj"}
    ]

    layer_cache = {"k_layer": torch.zeros(8, 2, 32)}

    with pytest.raises(RuntimeError, match="load_chunk_object failed"):
        conn._load_request_kv(
            req_meta,
            SimpleNamespace(attn_metadata=SimpleNamespace()),
            SimpleNamespace(),
            layer_cache,
        )

    assert 3 in conn._block_ids_with_load_errors
    assert 4 in conn._block_ids_with_load_errors


def test_load_missing_object_removes_file_and_invalidates_metadata(tmp_path):
    conn = DummyConnector(tmp_path)
    conn._expected_layers = ["k_layer"]
    req_meta = _ReqMeta(
        request_id="req-1",
        token_ids=list(range(8)),
        block_ids=[1, 2],
        block_size=4,
        is_store=False,
        is_load=True,
        load_tokens=8,
        store_tokens=0,
        namespace="test-ns",
        shard="tp0-pp0",
        expected_layers=["k_layer"],
        descriptors=[ChunkDescriptor(index=0, start=0, end=8, key="chunk0")],
    )
    path = tmp_path / "missing.cobj"
    path.write_bytes(b"corrupt")
    conn._go.resolve_chunks.return_value = [
        {"key": "chunk0", "file_path": "missing.cobj"}
    ]
    conn._storage.load_tensor_slices.side_effect = ValueError("bad object")

    with pytest.raises(RuntimeError, match="load_chunk_object failed"):
        conn._load_request_kv(
            req_meta,
            SimpleNamespace(attn_metadata=SimpleNamespace()),
            SimpleNamespace(),
            {"k_layer": torch.zeros(8, 2, 32)},
        )

    assert not path.exists()
    conn._go.invalidate_chunk.assert_called_once_with(
        "test-ns", "chunk0", "tp0-pp0"
    )


def test_load_runtime_error_does_not_invalidate_metadata(tmp_path):
    conn = DummyConnector(tmp_path)
    conn._expected_layers = ["k_layer"]
    req_meta = _ReqMeta(
        request_id="req-1",
        token_ids=list(range(8)),
        block_ids=[1, 2],
        block_size=4,
        is_store=False,
        is_load=True,
        load_tokens=8,
        store_tokens=0,
        namespace="test-ns",
        shard="tp0-pp0",
        expected_layers=["k_layer"],
        descriptors=[ChunkDescriptor(index=0, start=0, end=8, key="chunk0")],
    )
    path = tmp_path / "healthy.cobj"
    path.write_bytes(b"not parsed because backend fails first")
    conn._go.resolve_chunks.return_value = [
        {"key": "chunk0", "file_path": "healthy.cobj"}
    ]
    with mock.patch(
        "adapter.vllm.connector_common.load_chunk_object",
        side_effect=RuntimeError("cuda busy"),
    ):
        with pytest.raises(RuntimeError, match="load_chunk_object failed"):
            conn._load_request_kv(
                req_meta,
                SimpleNamespace(attn_metadata=SimpleNamespace()),
                SimpleNamespace(),
                {"k_layer": torch.zeros(8, 2, 32)},
            )

    assert path.exists()
    conn._go.invalidate_chunk.assert_not_called()

def test_invalidate_chunk_still_drops_metadata_when_unlink_fails(tmp_path, monkeypatch):
    conn = DummyConnector(tmp_path)
    path = tmp_path / "bad.cobj"
    path.write_bytes(b"bad")
    monkeypatch.setattr(Path, "unlink", mock.Mock(side_effect=OSError("busy")))

    conn._invalidate_chunk("test-ns", "chunk0", "tp0-pp0", path)

    assert path.exists()
    conn._go.invalidate_chunk.assert_called_once_with(
        "test-ns", "chunk0", "tp0-pp0"
    )


def test_validate_chunk_header_rejects_version_and_incomplete(tmp_path):
    conn = DummyConnector(tmp_path)
    desc = ChunkDescriptor(index=0, start=0, end=8, key="chunk0")
    base = {
        "namespace": "test-ns",
        "key": "chunk0",
        "shard": "tp0-pp0",
        "index": 0,
        "start": 0,
        "end": 8,
    }

    with pytest.raises(RuntimeError, match="version"):
        conn._validate_chunk_header(
            SimpleNamespace(version=1, complete=True, **base), "chunk0", desc
        )
    with pytest.raises(RuntimeError, match="incomplete"):
        conn._validate_chunk_header(
            SimpleNamespace(version=2, complete=False, **base), "chunk0", desc
        )

def test_load_failure_does_not_count_retrieved(tmp_path):
    """Failed load does NOT call chunks_retrieved."""
    conn = DummyConnector(tmp_path)
    conn._namespace = "test-ns"
    conn._expected_layers = ["k_layer"]
    conn._shard_idx = 0

    req_meta = _ReqMeta(
        request_id="req-1",
        token_ids=list(range(16)),
        block_ids=[3],
        block_size=8,
        is_store=False,
        is_load=True,
        store_tokens=0,
        load_tokens=8,
        namespace=conn._namespace,
        shard=conn._shard,
        expected_layers=conn._expected_layers,
        descriptors=[
            ChunkDescriptor(index=0, start=0, end=8, key="chunk0")
        ],
    )

    conn._go.resolve_chunks.return_value = [
        {"key": "chunk0", "file_path": "missing.cobj"}
    ]

    layer_cache = {"k_layer": torch.zeros(8, 2, 32)}

    with pytest.raises(RuntimeError):
        conn._load_request_kv(
            req_meta,
            SimpleNamespace(attn_metadata=SimpleNamespace()),
            SimpleNamespace(),
            layer_cache,
        )

    conn._go.chunks_retrieved.assert_not_called()


def test_start_load_kv_isolates_request_failure_and_reports_invalid_blocks(
    tmp_path,
):
    conn = DummyConnector(tmp_path)
    conn._expected_layers = ["k_layer"]
    failed = _ReqMeta(
        request_id="failed",
        token_ids=list(range(8)),
        block_ids=[3],
        block_size=8,
        is_store=False,
        is_load=True,
        store_tokens=0,
        load_tokens=8,
        namespace=conn._namespace,
        shard=conn._shard,
        expected_layers=conn._expected_layers,
        descriptors=[],
    )
    successful = _ReqMeta(
        request_id="successful",
        token_ids=list(range(8)),
        block_ids=[7],
        block_size=8,
        is_store=False,
        is_load=True,
        store_tokens=0,
        load_tokens=8,
        namespace=conn._namespace,
        shard=conn._shard,
        expected_layers=conn._expected_layers,
        descriptors=[],
    )
    conn._connector_meta = DiskCacheMeta(requests=[failed, successful])
    forward_context = SimpleNamespace(
        attn_metadata=SimpleNamespace(),
        no_compile_layers={
            "k_layer": SimpleNamespace(kv_cache=torch.zeros(8, 2, 4))
        },
    )

    with mock.patch.object(
        conn,
        "_load_request_kv",
        side_effect=[RuntimeError("corrupt object"), None],
    ) as load_request:
        conn.start_load_kv(forward_context)

    assert load_request.call_count == 2
    assert conn._failed_load_request_ids == {"failed"}
    assert conn.get_block_ids_with_load_errors() == {3}


def test_start_load_kv_global_failure_returns_invalid_blocks(tmp_path):
    conn = DummyConnector(tmp_path)
    conn._expected_layers = ["k_layer"]
    req = _ReqMeta(
        request_id="req-1",
        token_ids=list(range(8)),
        block_ids=[3, 4],
        block_size=8,
        is_store=False,
        is_load=True,
        store_tokens=0,
        load_tokens=8,
        namespace=conn._namespace,
        shard=conn._shard,
        expected_layers=conn._expected_layers,
        descriptors=[],
    )
    conn._connector_meta = DiskCacheMeta(requests=[req])

    conn.start_load_kv(SimpleNamespace(attn_metadata=None))

    assert conn._failed_load_request_ids == {"req-1"}
    assert conn.get_block_ids_with_load_errors() == {3, 4}


def test_failed_load_request_is_not_saved(tmp_path):
    conn = DummyConnector(tmp_path)
    conn._expected_layers = ["k_layer"]
    conn._failed_load_request_ids.add("req-1")
    req = _ReqMeta(
        request_id="req-1",
        token_ids=list(range(8)),
        block_ids=[0],
        block_size=8,
        is_store=True,
        is_load=False,
        store_tokens=8,
        load_tokens=0,
        namespace=conn._namespace,
        shard=conn._shard,
        expected_layers=conn._expected_layers,
        descriptors=[
            ChunkDescriptor(index=0, start=0, end=8, key="chunk0")
        ],
    )
    conn._connector_meta = DiskCacheMeta(requests=[req])

    conn.save_kv_layer("k_layer", object(), SimpleNamespace())

    assert conn._active_writers == {}


# ═══════════════════════════════════════════════════════════════════
# Chain namespace descriptor tests
# ═══════════════════════════════════════════════════════════════════


def test_chain_strategy_produces_correct_descriptors():
    """Real ChainChunkKeyStrategy produces correct descriptors."""
    from adapter.vllm.chunk_keys import ChainChunkKeyStrategy

    strategy = ChainChunkKeyStrategy(
        tokens_per_chunk=8, block_size=1
    )
    descs = strategy.describe(list(range(20)), "test-ns")
    # 20 tokens, 8 per chunk => 2 full + 1 partial
    assert len(descs) == 3
    assert descs[0].start == 0
    assert descs[0].end == 8
    assert descs[1].start == 8
    assert descs[1].end == 16
    assert descs[2].start == 16
    assert descs[2].end == 20
    for d in descs:
        assert len(d.key) == 64


# ═══════════════════════════════════════════════════════════════════
# Old v1 helpers are NOT called tests
# ═══════════════════════════════════════════════════════════════════


def test_old_v1_helpers_not_available(tmp_path):
    """The new connector should NOT expose old v1 helper methods."""
    conn = DummyConnector(tmp_path)
    old_v1_methods = [
        "_go_chunk_put", "_go_chunk_list", "_go_match",
        "_go_record", "_go_record_batch", "_go_put",
        "_go_record_retrieved", "_chunk_file_path", "_prefix_key",
        "_cached_file_path", "_layer_hash", "_chunk_ranges",
        "_requests_need_load", "_save_layer_chunks", "_save_chunk",
        "_load_layer_chunks", "_load_layer_chunks_optimized",
        "_load_layer_chunks_with_block_hash",
        "_save_layer_chunks_with_block_hash", "_kv_tokensize",
    ]
    for m in old_v1_methods:
        assert not hasattr(conn, m), f"{m} should not exist"


# ═══════════════════════════════════════════════════════════════════
# Connector/V21 delegate same full-hit logic
# ═══════════════════════════════════════════════════════════════════


def test_connector_variants_use_common_get_num_new_matched_tokens():
    """Both connector variants delegate to the same
    ``_get_num_new_matched_tokens`` from the mixin."""
    common_attrs = {
        "_block_size": 4,
        "_chunk_size": 8,
        "_namespace": "test-ns",
        "_shard": "tp0-pp0",
        "_required_shards": ["tp0-pp0"],
        "_strategy": mock.Mock(),
        "_go": mock.Mock(),
        "_pending_loads": {},
        "_connected": True,
        "_get_connector_metadata": mock.Mock(
            return_value=DiskCacheMeta()
        ),
        "_is_disk_cache_meta": mock.Mock(return_value=False),
        "_expected_layers": [],
        "_request_trackers": {},
        "_active_writers": {},
        "_block_ids_with_load_errors": set(),
        "cache_root": Path(tempfile.mkdtemp()),
        "target_device": "auto",
        "node_id": "test",
    }

    request = SimpleNamespace(
        request_id="req",
        all_token_ids=list(range(10)),
    )

    regular = object.__new__(connector.DiskCacheConnector)
    regular.__dict__.update(common_attrs)
    regular._strategy.describe.return_value = []
    regular._go.match_chunks.return_value = {}
    r1, _ = regular.get_num_new_matched_tokens(request, 0)

    v21 = object.__new__(connector_v21.DiskCacheConnector)
    v21.__dict__.update(common_attrs)
    v21._strategy.describe.return_value = []
    v21._go.match_chunks.return_value = {}
    r2, _ = v21.get_num_new_matched_tokens(request, 0)

    assert r1 == r2


# ═══════════════════════════════════════════════════════════════════
# request_finished / take_events / get_finished
# ═══════════════════════════════════════════════════════════════════


def test_request_finished_cleans_up(tmp_path):
    conn = DummyConnector(tmp_path)
    conn._pending_loads["req-1"] = _PendingLoadSpec(
        matched_tokens=8, descriptors=[], local_cached=0
    )
    conn._request_trackers["req-1"] = _RequestTracker()

    request = SimpleNamespace(request_id="req-1")
    conn.request_finished(request, None)

    assert "req-1" not in conn._pending_loads
    assert "req-1" not in conn._request_trackers


def test_lifecycle_returns_defaults(tmp_path):
    conn = DummyConnector(tmp_path)
    assert conn.take_events() == []
    assert conn.get_finished([]) == (None, None)
