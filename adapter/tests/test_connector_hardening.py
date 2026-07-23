"""Tests for connector hardening — v2 chunk-object connector.

Tests cover:
* Nested tuple block IDs (NewRequestData/KVCacheBlocks format)
* Slot mapping validation (insufficient slots raises)
* Dict no_compile_layers discovery
* wait_for_layer_load exists and is no-op
* Missing resolve object raises and chunks_retrieved not called
* Incomplete writer abort/pop in wait_for_save
* Commit payload uses start_tokens/end_tokens
* Resume/preempt tracker replaces block IDs
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
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

    Same pattern as test_connector_common's DummyConnector but with
    extra fields needed by hardening tests.
    """

    def __init__(self, tmp_path: Path):
        self.cache_root = tmp_path
        self._block_size = 4
        self._chunk_size = 8
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
        self._connector_meta: DiskCacheMeta | None = None
        self._load_error: str | None = None

    def _get_connector_metadata(self) -> DiskCacheMeta | None:
        return self._connector_meta

    def _is_disk_cache_meta(self, meta) -> bool:
        return isinstance(meta, DiskCacheMeta)


def test_connector_requires_piecewise_cudagraph():
    """FULL graph replay must not skip Cascade's per-layer Python hooks."""
    assert DummyConnector.requires_piecewise_for_cudagraph({}) is True
    assert DummyConnector.requires_piecewise_for_cudagraph(
        {"use_layerwise": False}
    ) is True


# ═══════════════════════════════════════════════════════════════════
# 1. Nested tuple block IDs
# ═══════════════════════════════════════════════════════════════════


def test_extract_block_ids_nested_tuple(tmp_path):
    """_extract_block_ids handles tuple[list[int], ...]."""
    conn = DummyConnector(tmp_path)
    block_ids = ([3, 7], [9, 11])
    result = conn._extract_block_ids(block_ids)
    assert result == [3, 7]


def test_extract_block_ids_flat_tuple(tmp_path):
    """_extract_block_ids with tuple of lists, group 0 selected."""
    conn = DummyConnector(tmp_path)
    # Simulate get_block_ids() returning tuple
    class MockBlocks:
        def get_block_ids(self):
            return ([5, 6], [7, 8])
    result = conn._extract_block_ids(MockBlocks())
    assert result == [5, 6]


def test_extract_block_ids_empty_tuple(tmp_path):
    """_extract_block_ids with empty tuple returns []."""
    conn = DummyConnector(tmp_path)
    assert conn._extract_block_ids(()) == []


def test_extract_block_ids_nested_list(tmp_path):
    """_extract_block_ids handles nested list."""
    conn = DummyConnector(tmp_path)
    assert conn._extract_block_ids([[10, 20]]) == [10, 20]


def test_get_block_ids_for_cached_nested_tuple(tmp_path):
    """_get_block_ids_for_cached handles tuple[list[int], ...]."""
    conn = DummyConnector(tmp_path)
    result = conn._get_block_ids_for_cached(
        [([1, 2], [3, 4]), ([5, 6],)], 0
    )
    assert result == [1, 2]


def test_get_block_ids_for_cached_flat_list(tmp_path):
    """_get_block_ids_for_cached handles flat list."""
    conn = DummyConnector(tmp_path)
    result = conn._get_block_ids_for_cached([[7, 8], [9, 10]], 1)
    assert result == [9, 10]


def test_extract_block_ids_get_block_ids_tuple(tmp_path):
    """KVCacheBlocks.get_block_ids() returning tuple."""
    conn = DummyConnector(tmp_path)

    class MockKVCacheBlocks:
        def get_block_ids(self):
            return ([1, 2, 3], [4, 5, 6])

    result = conn._extract_block_ids(MockKVCacheBlocks())
    assert result == [1, 2, 3]


# ═══════════════════════════════════════════════════════════════════
# 2. Slot mapping validation
# ═══════════════════════════════════════════════════════════════════


def test_build_slot_mapping_insufficient_slots_raises(tmp_path):
    """_build_slot_mapping raises ValueError when slots < num_tokens."""
    conn = DummyConnector(tmp_path)
    with pytest.raises(ValueError, match="Not enough slots"):
        conn._build_slot_mapping(
            block_ids=[0, 1], block_size=4, num_tokens=20
        )


def test_build_slot_mapping_negative_tokens_raises(tmp_path):
    conn = DummyConnector(tmp_path)
    with pytest.raises(ValueError, match="num_tokens must be >= 0"):
        conn._build_slot_mapping(
            block_ids=[0], block_size=4, num_tokens=-1
        )


def test_build_slot_mapping_zero_block_size_raises(tmp_path):
    conn = DummyConnector(tmp_path)
    with pytest.raises(ValueError, match="block_size must be > 0"):
        conn._build_slot_mapping(
            block_ids=[0], block_size=0, num_tokens=4
        )


def test_build_slot_mapping_exact_fit_ok(tmp_path):
    """Exact fit does not raise."""
    conn = DummyConnector(tmp_path)
    mapping = conn._build_slot_mapping(
        block_ids=[0, 1], block_size=4, num_tokens=8
    )
    assert mapping.numel() == 8
    assert mapping.tolist() == list(range(8))


# ═══════════════════════════════════════════════════════════════════
# 3. Dict no_compile_layers discovery
# ═══════════════════════════════════════════════════════════════════


def test_collect_kv_layers_dict(tmp_path):
    """_collect_available_kv_layers with dict no_compile_layers."""
    conn = DummyConnector(tmp_path)
    conn._expected_layers = ["layer_0", "layer_1", "layer_2"]

    kv_tensor = torch.zeros(4, 2, 32)
    no_compile = {
        "layer_0": SimpleNamespace(kv_cache=kv_tensor),
        "layer_1": SimpleNamespace(kv_cache=kv_tensor),
        # layer_2 missing
    }
    result = conn._collect_available_kv_layers(no_compile)
    assert "layer_0" in result
    assert "layer_1" in result
    assert "layer_2" not in result


def test_collect_kv_layers_object(tmp_path):
    """_collect_available_kv_layers with object attribute access."""
    conn = DummyConnector(tmp_path)
    conn._expected_layers = ["k_layer", "v_layer"]

    kv_tensor = torch.zeros(4, 2, 32)
    no_compile = SimpleNamespace(
        k_layer=SimpleNamespace(kv_cache=kv_tensor),
        v_layer=SimpleNamespace(kv_cache=kv_tensor),
    )
    result = conn._collect_available_kv_layers(no_compile)
    assert "k_layer" in result
    assert "v_layer" in result


def test_collect_kv_layers_none(tmp_path):
    """_collect_available_kv_layers with None returns empty dict."""
    conn = DummyConnector(tmp_path)
    result = conn._collect_available_kv_layers(None)
    assert result == {}


# ═══════════════════════════════════════════════════════════════════
# 4. wait_for_layer_load
# ═══════════════════════════════════════════════════════════════════


def test_wait_for_layer_load_exists(tmp_path):
    """wait_for_layer_load is callable and does not raise when no error."""
    conn = DummyConnector(tmp_path)
    # Should not raise
    conn.wait_for_layer_load("k_layer")
    conn.wait_for_layer_load("v_layer")


def test_wait_for_layer_load_is_noop_after_error(tmp_path):
    """Invalid blocks, not an attention-layer exception, drive recovery."""
    conn = DummyConnector(tmp_path)
    conn._load_error = "test error"
    conn.wait_for_layer_load("k_layer")
    assert conn._load_error == "test error"


# ═══════════════════════════════════════════════════════════════════
# 5. Missing resolve object raises and chunks_retrieved not called
# ═══════════════════════════════════════════════════════════════════


def test_load_missing_object_raises(tmp_path):
    """Missing resolve object raises RuntimeError, no chunks_retrieved."""
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

    # resolve_chunks returns empty — no object for chunk0
    conn._go.resolve_chunks.return_value = []

    layer_cache = {"k_layer": torch.zeros(8, 2, 32)}

    with pytest.raises(RuntimeError, match="No resolved object"):
        conn._load_request_kv(
            req_meta,
            SimpleNamespace(attn_metadata=SimpleNamespace()),
            SimpleNamespace(),
            layer_cache,
        )

    conn._go.chunks_retrieved.assert_not_called()


def test_load_empty_file_path_raises(tmp_path):
    """Empty file_path in resolve result raises RuntimeError."""
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
        {"key": "chunk0", "file_path": ""}
    ]

    layer_cache = {"k_layer": torch.zeros(8, 2, 32)}

    with pytest.raises(RuntimeError, match="Empty file_path"):
        conn._load_request_kv(
            req_meta,
            SimpleNamespace(attn_metadata=SimpleNamespace()),
            SimpleNamespace(),
            layer_cache,
        )

    conn._go.chunks_retrieved.assert_not_called()
    conn._go.invalidate_chunk.assert_called_once_with(
        "test-ns", "chunk0", conn._shard
    )


def test_load_duplicate_resolve_key_raises(tmp_path):
    """Duplicate key in resolve response raises RuntimeError."""
    conn = DummyConnector(tmp_path)
    conn._namespace = "test-ns"
    conn._expected_layers = ["k_layer"]
    conn._shard_idx = 0

    req_meta = _ReqMeta(
        request_id="req-1",
        token_ids=list(range(8)),
        block_ids=[0],
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
        {"key": "chunk0", "file_path": "a.cobj"},
        {"key": "chunk0", "file_path": "b.cobj"},
    ]

    layer_cache = {"k_layer": torch.zeros(8, 2, 32)}

    with pytest.raises(RuntimeError, match="duplicate"):
        conn._load_request_kv(
            req_meta,
            SimpleNamespace(attn_metadata=SimpleNamespace()),
            SimpleNamespace(),
            layer_cache,
        )

    conn._go.chunks_retrieved.assert_not_called()


# ═══════════════════════════════════════════════════════════════════
# 6. Incomplete writer abort/pop in wait_for_save
# ═══════════════════════════════════════════════════════════════════


def test_incomplete_writer_aborted_and_popped(tmp_path):
    """wait_for_save aborts and pops writers missing layers."""
    conn = DummyConnector(tmp_path)

    mock_writer = mock.Mock()
    mock_writer.final_path = tmp_path / "dummy.cobj"

    # Writer missing expected layer
    wkey = ("req-1", "chunk0")
    conn._active_writers[wkey] = _WriterState(
        writer=mock_writer,
        key="chunk0",
        index=0,
        start=0,
        end=8,
        finalized_layers={"k_layer"},  # missing "v_layer"
    )
    conn._expected_layers = ["k_layer", "v_layer"]

    conn.wait_for_save()

    mock_writer.finalize.assert_not_called()

    assert mock_writer.abort.called
    assert wkey not in conn._active_writers


# ═══════════════════════════════════════════════════════════════════
# 7. Commit payload uses start_tokens/end_tokens
# ═══════════════════════════════════════════════════════════════════


def test_commit_payload_uses_start_end_tokens(tmp_path):
    """wait_for_save commit payload uses start_tokens/end_tokens fields."""
    conn = DummyConnector(tmp_path)
    conn._namespace = "test-ns"
    conn._expected_layers = ["k_layer", "v_layer"]

    # Create a writer that will be finalized
    mock_writer = mock.Mock()
    mock_final_path = tmp_path / "some" / "path.cobj"
    mock_final_path.parent.mkdir(parents=True, exist_ok=True)
    mock_final_path.write_bytes(b"test")
    mock_writer.final_path = mock_final_path
    mock_writer.finalize.return_value = None

    wkey = ("req-1", "chunk0")
    conn._active_writers[wkey] = _WriterState(
        writer=mock_writer,
        key="chunk0",
        index=0,
        start=0,
        end=8,
        finalized_layers={"k_layer", "v_layer"},
    )

    conn.wait_for_save()

    # Check commit payload had start_tokens/end_tokens, not start/end
    call_args = conn._go.commit_chunks.call_args[0][0]
    assert len(call_args) == 1
    obj = call_args[0]
    assert "start_tokens" in obj
    assert "end_tokens" in obj
    assert obj["start_tokens"] == 0
    assert obj["end_tokens"] == 8
    # Old-style fields should not be present
    assert "start" not in obj
    assert "end" not in obj


def _complete_writer_state(
    conn: DummyConnector,
    tmp_path: Path,
    *,
    request_id: str,
    key: str,
    index: int = 0,
    start: int = 0,
    end: int = 8,
    size: int = 4,
) -> tuple[tuple[str, str], mock.Mock, Path]:
    """Install a complete mock writer and return its key, mock, and path."""
    writer = mock.Mock()
    final_path = tmp_path / request_id / f"{key}.cobj"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_bytes(b"x" * size)
    writer.final_path = final_path
    writer_key = (request_id, key)
    conn._active_writers[writer_key] = _WriterState(
        writer=writer,
        key=key,
        index=index,
        start=start,
        end=end,
        finalized_layers=set(conn._expected_layers),
        owner_request_id=request_id,
    )
    return writer_key, writer, final_path


def test_commit_failure_keeps_final_and_tracker_retryable(tmp_path):
    """A transient metadata failure must not fail the model step."""
    conn = DummyConnector(tmp_path)
    conn._expected_layers = ["k_layer", "v_layer"]
    conn._request_trackers["req-1"] = _RequestTracker()
    writer_key, writer, final_path = _complete_writer_state(
        conn,
        tmp_path,
        request_id="req-1",
        key="chunk0",
    )
    conn._go.commit_chunks.side_effect = RuntimeError("engine unavailable")

    conn.wait_for_save()

    assert final_path.exists()
    assert writer.abort.called
    assert writer_key not in conn._active_writers
    tracker = conn._request_trackers["req-1"]
    assert tracker.committed_chunk_keys == set()
    assert tracker.committed_tokens == 0

    conn._go.commit_chunks.side_effect = None
    retry_key, retry_writer, _ = _complete_writer_state(
        conn,
        tmp_path,
        request_id="req-1",
        key="chunk0",
    )

    conn.wait_for_save()

    assert retry_writer.finalize.called
    assert retry_key not in conn._active_writers
    assert tracker.committed_chunk_keys == {"chunk0"}
    assert tracker.committed_tokens == 8
    assert conn._go.commit_chunks.call_count == 2


def test_duplicate_identity_conflict_stops_before_finalize(tmp_path):
    conn = DummyConnector(tmp_path)
    conn._expected_layers = ["k_layer", "v_layer"]
    _, first, _ = _complete_writer_state(
        conn,
        tmp_path,
        request_id="req-1",
        key="chunk0",
        end=8,
    )
    _, second, _ = _complete_writer_state(
        conn,
        tmp_path,
        request_id="req-2",
        key="chunk0",
        end=16,
    )

    conn.wait_for_save()

    first.finalize.assert_not_called()
    second.finalize.assert_not_called()
    assert first.abort.called
    assert second.abort.called
    conn._go.commit_chunks.assert_not_called()


def test_finalization_failure_keeps_published_sibling_and_no_commit(tmp_path):
    conn = DummyConnector(tmp_path)
    conn._expected_layers = ["k_layer", "v_layer"]
    first_key, first, first_path = _complete_writer_state(
        conn,
        tmp_path,
        request_id="req-1",
        key="chunk0",
    )
    second_key, second, _ = _complete_writer_state(
        conn,
        tmp_path,
        request_id="req-2",
        key="chunk1",
        index=1,
        start=8,
        end=16,
    )
    second.finalize.side_effect = OSError("publish failed")

    conn.wait_for_save()

    assert first.finalize.called
    assert first_path.exists()
    assert first.abort.called
    assert second.abort.called
    assert first_key not in conn._active_writers
    assert second_key not in conn._active_writers
    conn._go.commit_chunks.assert_not_called()


def test_failed_load_writer_isolated_from_other_request_commit(tmp_path):
    conn = DummyConnector(tmp_path)
    conn._expected_layers = ["k_layer", "v_layer"]
    conn._failed_load_request_ids.add("failed-req")
    failed_key, failed_writer, _ = _complete_writer_state(
        conn,
        tmp_path,
        request_id="failed-req",
        key="failed-chunk",
    )
    good_key, good_writer, _ = _complete_writer_state(
        conn,
        tmp_path,
        request_id="good-req",
        key="good-chunk",
    )

    conn.wait_for_save()

    assert failed_writer.abort.called
    failed_writer.finalize.assert_not_called()
    assert good_writer.finalize.called
    assert failed_key not in conn._active_writers
    assert good_key not in conn._active_writers
    payload = conn._go.commit_chunks.call_args.args[0]
    assert [obj["key"] for obj in payload] == ["good-chunk"]

# ═══════════════════════════════════════════════════════════════════
# 8. Resume/preempt tracker replaces block IDs
# ═══════════════════════════════════════════════════════════════════


def test_build_connector_meta_resumed_replaces_block_ids(tmp_path):
    """Resumed cached request replaces tracker block_ids."""
    conn = DummyConnector(tmp_path)
    conn._chunk_size = 8
    conn._strategy.describe.return_value = []

    conn._request_trackers["req-1"] = _RequestTracker(
        token_ids=list(range(20)),
        block_ids=[0, 1],
        committed_tokens=8,
    )

    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req-1"],
            new_block_ids=[([10, 11], [12, 13])],
            resumed_req_ids={"req-1"},
        ),
        num_scheduled_tokens={"req-1": 4},
        finished_req_ids=None,
    )

    conn.build_connector_meta(scheduler_output)

    tracker = conn._request_trackers.get("req-1")
    assert tracker is not None
    assert tracker.block_ids == [10, 11]


def test_cached_steps_keep_external_hit_boundary_and_append_blocks(tmp_path):
    conn = DummyConnector(tmp_path)
    conn._chunk_size = 8
    conn._block_size = 4
    conn._strategy.describe.side_effect = lambda tokens, namespace, end=None: [
        ChunkDescriptor(i, i * 8, min((i + 1) * 8, end), f"chunk{i}")
        for i in range((end or 0) // 8 + (1 if (end or 0) % 8 else 0))
    ]
    conn._request_trackers["req-1"] = _RequestTracker(
        token_ids=list(range(16)),
        block_ids=[0, 1],
        prompt_len=24,
        num_computed_tokens=16,
        external_cached_tokens=8,
    )
    conn._unfinished_requests = {
        "req-1": SimpleNamespace(
            request_id="req-1",
            prompt_token_ids=list(range(24)),
            all_token_ids=list(range(24)),
        )
    }
    cached = SimpleNamespace(
        req_ids=["req-1"],
        resumed_req_ids=set(),
        new_block_ids=[([2],)],
        all_token_ids={"req-1": list(range(24))},
        num_computed_tokens=[16],
        new_token_ids=[],
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={"req-1": 8},
        finished_req_ids=set(),
    )

    meta = conn.build_connector_meta(output)
    tracker = conn._request_trackers["req-1"]
    assert tracker.external_cached_tokens == 8
    assert tracker.block_ids == [0, 1, 2]
    stores = [item for item in meta.requests if item.is_store]
    assert stores
    assert all(desc.start >= 8 for desc in stores[0].descriptors)


def test_cached_rollback_does_not_restore_stale_external_hit(tmp_path):
    conn = DummyConnector(tmp_path)
    conn._strategy.describe.return_value = []
    conn._request_trackers["req-1"] = _RequestTracker(
        token_ids=list(range(16)),
        block_ids=[0, 1],
        prompt_len=24,
        num_computed_tokens=16,
        external_cached_tokens=8,
        num_saved_tokens=12,
    )
    conn._unfinished_requests = {
        "req-1": SimpleNamespace(
            request_id="req-1",
            prompt_token_ids=list(range(24)),
            all_token_ids=list(range(24)),
        )
    }
    cached = SimpleNamespace(
        req_ids=["req-1"],
        resumed_req_ids=set(),
        new_block_ids=[()],
        all_token_ids={"req-1": list(range(24))},
        num_computed_tokens=[4],
        new_token_ids=[],
    )
    output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={"req-1": 4},
        finished_req_ids=set(),
    )

    conn.build_connector_meta(output)

    tracker = conn._request_trackers["req-1"]
    assert tracker.external_cached_tokens == 4
    assert tracker.num_computed_tokens == 8
    assert tracker.num_saved_tokens == 4


def test_build_connector_meta_preempted_replaces_block_ids(tmp_path):
    """Preempted cached request replaces tracker block_ids."""
    conn = DummyConnector(tmp_path)
    conn._chunk_size = 8
    conn._strategy.describe.return_value = []

    conn._request_trackers["req-2"] = _RequestTracker(
        token_ids=list(range(20)),
        block_ids=[3, 4],
        committed_tokens=4,
    )

    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req-2"],
            new_block_ids=[([7, 8],)],
            preempted_req_ids={"req-2"},
        ),
        num_scheduled_tokens={"req-2": 4},
        finished_req_ids=None,
    )

    conn.build_connector_meta(scheduler_output)

    tracker = conn._request_trackers.get("req-2")
    assert tracker is not None
    assert tracker.block_ids == [7, 8]


# ═══════════════════════════════════════════════════════════════════
# 9. Lifecycle: request_finished cleans active_writers
# ═══════════════════════════════════════════════════════════════════


def test_request_finished_cleans_active_writers(tmp_path):
    """request_finished removes active writers for the request."""
    conn = DummyConnector(tmp_path)
    mock_writer = mock.Mock()

    conn._active_writers[("req-1", "chunk0")] = _WriterState(
        writer=mock_writer,
        key="chunk0",
        index=0,
        start=0,
        end=8,
    )
    conn._active_writers[("req-1", "chunk1")] = _WriterState(
        writer=mock_writer,
        key="chunk1",
        index=1,
        start=8,
        end=16,
    )
    conn._active_writers[("req-2", "chunk0")] = _WriterState(
        writer=mock_writer,
        key="chunk0",
        index=0,
        start=0,
        end=8,
    )

    request = SimpleNamespace(request_id="req-1")
    conn.request_finished(request, None)

    assert ("req-1", "chunk0") not in conn._active_writers
    assert ("req-1", "chunk1") not in conn._active_writers
    assert ("req-2", "chunk0") in conn._active_writers  # other request kept


# ═══════════════════════════════════════════════════════════════════
# 10. Lifecycle: get_finished accepts set or list
# ═══════════════════════════════════════════════════════════════════


def test_get_finished_with_set(tmp_path):
    """get_finished accepts set[str] without error."""
    conn = DummyConnector(tmp_path)
    result = conn.get_finished({"req-1", "req-2"})
    assert result == (None, None)


def test_get_finished_with_list(tmp_path):
    """get_finished accepts list[str] without error."""
    conn = DummyConnector(tmp_path)
    result = conn.get_finished(["req-1"])
    assert result == (None, None)


# ═══════════════════════════════════════════════════════════════════
# 11. Lifecycle: build_connector_meta reads finished_req_ids from
#     scheduler_output
# ═══════════════════════════════════════════════════════════════════


def test_build_connector_meta_reads_finished_from_scheduler(tmp_path):
    """Read finished_req_ids from scheduler_output when param is None."""
    conn = DummyConnector(tmp_path)
    conn._request_trackers["done-req"] = _RequestTracker(
        token_ids=list(range(5)), block_ids=[0]
    )

    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=None,
        num_scheduled_tokens={},
        finished_req_ids={"done-req"},
    )

    conn.build_connector_meta(scheduler_output)

    assert "done-req" not in conn._request_trackers
