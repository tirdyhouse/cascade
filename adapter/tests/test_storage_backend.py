# SPDX-License-Identifier: Apache-2.0
"""Tests for storage backends (GDS + POSIX).

Level 1 (pure mock / CPU)
==========================
These tests run on any machine — no GPU, no GDS driver required::

    pytest adapter/tests/test_storage_backend.py -v

Level 2 (CPU fallback with GPU)
===============================
These tests need ``torch.cuda.is_available()`` but no GDS driver::

    pytest adapter/tests/test_storage_backend.py -v -k "gpu"

Level 3 (real GDS)
===================
These tests need ``cufile`` or ``hipfile`` installed.  Skipped by default.
"""

from __future__ import annotations

import contextlib
import json
import os
import struct
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from adapter.storage.backend import (
    StorageBackend,
    _HEADER_SIZE,
    _pack_header,
    _unpack_header,
    create_storage_backend,
)
from adapter.storage.posix_backend import PosixBackend


# ═══════════════════════════════════════════════════════════════════
# Level 1 — pure CPU, no GPU, no GDS driver
# ═══════════════════════════════════════════════════════════════════

class TestPosixBackend:
    """PosixBackend should work with CPU tensors (no GPU needed)."""

    @pytest.fixture
    def backend(self):
        return PosixBackend()

    @pytest.fixture
    def tmp_path(self):
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def test_save_and_load_cpu_tensor(self, backend, tmp_path):
        """Round-trip a CPU tensor through save/load."""
        path = tmp_path / "test.safetensors"
        tensor = torch.randn(2, 256, 8, 128, dtype=torch.bfloat16)
        backend.save(path, tensor)
        assert path.exists()

        loaded = backend.load(path, device="cpu")
        assert torch.equal(loaded, tensor)

    def test_save_and_load_gpu_fallback(self, backend, tmp_path):
        """When GPU is not available, save/load works the same."""
        path = tmp_path / "test.safetensors"
        tensor = torch.randn(2, 256, 8, 128, dtype=torch.bfloat16)
        backend.save(path, tensor)
        loaded = backend.load(path, device="cpu")
        assert torch.equal(loaded, tensor)

    def test_is_available(self, backend):
        assert backend.is_available() is True


class TestHeaderFormat:
    """Metadata header packing/unpacking (used by GDS backend)."""

    def test_pack_unpack_roundtrip(self):
        tensor = torch.zeros(4, 8, dtype=torch.float32)
        packed = _pack_header(tensor)
        assert len(packed) == _HEADER_SIZE

        meta = _unpack_header(packed)
        assert meta["dtype"] == "torch.float32"
        assert meta["shape"] == [4, 8]
        assert meta["nbytes"] == 4 * 8 * 4  # float32 = 4 bytes
        assert meta["version"] == 1

    def test_pack_unpack_bfloat16(self):
        tensor = torch.zeros(2, 256, 8, 128, dtype=torch.bfloat16)
        packed = _pack_header(tensor)
        meta = _unpack_header(packed)
        assert meta["dtype"] == "torch.bfloat16"
        assert meta["shape"] == [2, 256, 8, 128]

    def test_pack_header_fixed_size(self):
        """Header is always exactly _HEADER_SIZE bytes."""
        for shape in [(1,), (2, 256, 8, 128), (4, 8, 16, 32, 64)]:
            tensor = torch.zeros(shape, dtype=torch.float16)
            packed = _pack_header(tensor)
            assert len(packed) == _HEADER_SIZE


class TestFactory:
    """Factory function auto-selection."""

    def test_prefer_posix(self):
        backend = create_storage_backend(prefer="posix")
        assert isinstance(backend, PosixBackend)

    def test_prefer_posix_case_insensitive(self):
        backend = create_storage_backend(prefer="POSIX")
        assert isinstance(backend, PosixBackend)

    def test_gds_unavailable_falls_back_to_posix(self):
        """When GDS is forced unavailable (mock), returns PosixBackend."""
        with mock.patch("adapter.storage.backend._try_gds", return_value=None):
            backend = create_storage_backend()
            assert isinstance(backend, PosixBackend)


class TestCreateStorageBackend:
    """End-to-end tests for the factory."""

    def test_default_is_posix_on_non_gds_system(self):
        """On systems without GDS, create_storage_backend() returns PosixBackend."""
        backend = create_storage_backend()
        name = type(backend).__name__
        assert name in ("PosixBackend", "NvFileBackend")
        if name == "NvFileBackend":
            # GDS is actually available on this system — that's fine too
            assert backend.is_available()


# ═══════════════════════════════════════════════════════════════════
# Level 2 — GPU tests (need torch.cuda)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
class TestPosixBackendWithGPU:
    """PosixBackend with real GPU tensor path (cudaMemcpy fallback)."""

    @pytest.fixture
    def backend(self):
        return PosixBackend()

    @pytest.fixture
    def tmp_path(self):
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def test_save_and_load_gpu_tensor(self, backend, tmp_path):
        """GPU tensor → save → load → compare on GPU."""
        path = tmp_path / "test.safetensors"
        tensor = torch.randn(2, 256, 8, 128, dtype=torch.bfloat16, device="cuda")
        backend.save(path, tensor)

        loaded = backend.load(path, device="cuda")
        assert loaded.is_cuda
        assert loaded.dtype == tensor.dtype
        assert loaded.shape == tensor.shape
        assert torch.equal(loaded.cpu(), tensor.cpu())

    def test_save_keeps_gpu_tensor_unchanged(self, backend, tmp_path):
        """Saving should not modify the original GPU tensor."""
        path = tmp_path / "test.safetensors"
        tensor = torch.randn(4, 128, dtype=torch.float16, device="cuda")
        original = tensor.clone()
        backend.save(path, tensor)
        assert torch.equal(tensor, original)



@contextlib.contextmanager
def _mock_nvfile_backend():
    class _MockCuFile:
        def __init__(self, path, mode, use_direct_io=False):
            self.path = path
            self.mode = mode

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def close(self):
            pass

        @staticmethod
        def _ptr_addr(ptr):
            if isinstance(ptr, int):
                return ptr
            return ptr.value

        def write(self, ptr, nbytes, file_offset=0, dev_offset=0):
            import ctypes
            buf = (ctypes.c_byte * nbytes).from_address(self._ptr_addr(ptr))
            with open(self.path, "r+b") as f:
                f.seek(file_offset)
                f.write(bytes(buf))
            return nbytes

        def read(self, ptr, nbytes, file_offset=0, dev_offset=0):
            import ctypes
            with open(self.path, "rb") as f:
                f.seek(file_offset)
                data = f.read(nbytes)
            buf = (ctypes.c_byte * nbytes).from_address(self._ptr_addr(ptr))
            buf[:] = data
            return nbytes

    class _MockBinding:
        name = "cufile (mock)"
        _mod = mock.MagicMock()
        _mod.CuFile = _MockCuFile

        def driver_open(self):
            pass

        def driver_close(self):
            pass

    from adapter.storage.nvfile_backend import NvFileBackend

    with mock.patch("adapter.storage.nvfile_backend._detect_binding",
                    return_value=_MockBinding()):
        yield NvFileBackend()

# ═══════════════════════════════════════════════════════════════════
# Level 3 — GDS mock tests (no real driver needed)
# ═══════════════════════════════════════════════════════════════════

class TestNvFileBackendMocked:
    """NvFileBackend with mocked cufile — runs on any machine."""

    @pytest.fixture
    def tmp_path(self):
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    @pytest.fixture
    def mock_cufile(self):
        """Create a mock 'cufile' module that simulates GDS I/O using POSIX.

        The mock 'CuFile.write()' writes raw bytes at the given file offset;
        'CuFile.read()' reads them back.  This lets us verify the GDS
        integration path without real hardware.
        """
        class MockCuFile:
            def __init__(self, path, mode, use_direct_io=False):
                self.path = path
                self.mode = mode

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

            def close(self):
                pass

            def write(self, gpu_addr_ptr, nbytes, file_offset=0, dev_offset=0):
                # gpu_addr_ptr is already a ctypes.c_void_p from _CuFileHandle
                import ctypes
                buf = (ctypes.c_byte * nbytes).from_address(
                    gpu_addr_ptr.value
                )
                with open(self.path, "r+b") as f:
                    f.seek(file_offset)
                    f.write(bytes(buf))
                return nbytes

            def read(self, gpu_addr_ptr, nbytes, file_offset=0, dev_offset=0):
                import ctypes
                with open(self.path, "rb") as f:
                    f.seek(file_offset)
                    data = f.read(nbytes)
                buf = (ctypes.c_byte * nbytes).from_address(
                    gpu_addr_ptr.value
                )
                buf[:] = data
                return nbytes

        class MockCuFileDriver:
            def __init__(self):
                pass

        mock_module = mock.MagicMock()
        mock_module.CuFile = MockCuFile
        mock_module.CuFileDriver = MockCuFileDriver
        return mock_module

    @pytest.fixture
    def backend(self):
        with _mock_nvfile_backend() as backend:
            yield backend

    def test_save_and_load_cpu_as_gpu(self, backend, tmp_path):
        """Simulate save/load with CPU tensors."""
        path = tmp_path / "test.kvcache"
        tensor = torch.randn(2, 256, 8, 128, dtype=torch.bfloat16)
        backend.save(path, tensor)
        assert path.exists()
        assert path.stat().st_size > _HEADER_SIZE
        loaded = backend.load(path, device="cpu")
        assert torch.equal(loaded, tensor)

    def test_save_file_format(self, backend, tmp_path):
        """Verify the on-disk format: JSON header + raw tensor data."""
        path = tmp_path / "test.kvcache"
        tensor = torch.arange(16, dtype=torch.int32).reshape(4, 4)
        backend.save(path, tensor)
        with open(path, "rb") as f:
            header_blob = f.read(_HEADER_SIZE)
        meta = _unpack_header(header_blob)
        assert meta["shape"] == [4, 4]
        assert meta["dtype"] == "torch.int32"
        with open(path, "rb") as f:
            f.seek(_HEADER_SIZE)
            raw = f.read(meta["nbytes"])
        expected = tensor.numpy().tobytes()
        assert raw == expected

    def test_is_available_true(self, backend):
        assert backend.is_available() is True

    def test_save_load_small_tensor(self, backend, tmp_path):
        """Small tensors round-trip correctly."""
        path = tmp_path / "small.kvcache"
        tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
        backend.save(path, tensor)
        loaded = backend.load(path, device="cpu")
        assert torch.equal(loaded, tensor)

    def test_save_failure_cleans_up_temp(self, backend, tmp_path):
        """If GDS write fails, the temp file should be cleaned up."""
        from adapter.storage.nvfile_backend import _CuFileHandle
        original_write = _CuFileHandle.write
        def failing_write(self, *args, **kwargs):
            raise RuntimeError("GDS write failed")
        _CuFileHandle.write = failing_write
        try:
            path = tmp_path / "fail.kvcache"
            tensor = torch.randn(2, 8, dtype=torch.float32)
            with pytest.raises(RuntimeError, match="GDS write failed"):
                backend.save(path, tensor)
            tmp_files = list(tmp_path.glob("*.tmp*"))
            assert len(tmp_files) == 0
            assert not path.exists()
        finally:
            _CuFileHandle.write = original_write


# ═══════════════════════════════════════════════════════════════════
# Level 3 — real GDS (requires hardware, skipped by default)
# ═══════════════════════════════════════════════════════════════════

def _has_gds() -> bool:
    """Check if a GDS library is installed."""
    for lib in ("cufile", "nvfile", "hipfile"):
        try:
            __import__(lib)
            return True
        except ImportError:
            continue
    return False


@pytest.mark.skipif(not _has_gds(), reason="Requires GDS library (cufile/nvfile/hipfile)")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
class TestRealGdsBackend:
    """Smoke tests against real GDS hardware.

    These are the Level 3 validation tests — they verify that real
    cuFile/nvfile calls work end-to-end with GPU tensors.
    """

    @pytest.fixture
    def tmp_path(self):
        # Use a real ext4/XFS path (tmpfs/overlayfs may not support GDS)
        import os
        path = Path(os.environ.get("LMCACHE_TEST_TMPDIR", "/tmp")) / "gds-test"
        path.mkdir(parents=True, exist_ok=True)
        yield path
        import shutil
        shutil.rmtree(path, ignore_errors=True)

    def test_gds_backend_init(self):
        """Creating the backend should succeed when GDS library is present."""
        from adapter.storage import create_storage_backend
        backend = create_storage_backend(prefer="gds")
        assert "NvFile" in type(backend).__name__ or "File" in type(backend).__name__

    def test_gds_save_and_load(self, tmp_path):
        """Full GDS write+read round-trip with a real GPU tensor."""
        from adapter.storage import create_storage_backend
        backend = create_storage_backend(prefer="gds")

        path = tmp_path / "smoke.kvcache"
        tensor = torch.randn(2, 256, 8, 128, dtype=torch.bfloat16, device="cuda")

        backend.save(path, tensor)
        assert path.exists()

        loaded = backend.load(path, device="cuda")
        assert loaded.is_cuda
        assert loaded.dtype == tensor.dtype
        assert loaded.shape == tensor.shape
        assert torch.equal(loaded.cpu(), tensor.cpu())


class TestNvFileBackendInternals:
    """Focused unit tests for GDS binding/handle edge cases."""

    def test_aligned_device_buffer_preserves_aligned_view(self):
        from adapter.storage.nvfile_backend import _aligned_device_buffer

        backing, aligned, offset = _aligned_device_buffer(8192, "cpu")

        assert aligned.data_ptr() % 4096 == 0
        assert aligned.data_ptr() == backing.data_ptr() + offset
        assert aligned.numel() == 8192

    def test_cufile_handle_low_level_registers_and_deregisters_fd(self, tmp_path):
        from adapter.storage.nvfile_backend import _CuFileHandle

        path = tmp_path / "raw.bin"
        path.write_bytes(b"\x00" * 32)

        class LowLevelBinding:
            name = "low-level-mock"

            def __init__(self):
                self.registered_fd = None
                self.deregistered = []
                self.writes = []
                self.reads = []

            def handle_register(self, fd):
                self.registered_fd = fd
                return "handle-1"

            def handle_deregister(self, handle):
                self.deregistered.append(handle)

            def write(self, handle, gpu_ptr, size, file_offset, dev_offset):
                self.writes.append((handle, gpu_ptr, size, file_offset, dev_offset))
                return size

            def read(self, handle, gpu_ptr, size, file_offset, dev_offset):
                self.reads.append((handle, gpu_ptr, size, file_offset, dev_offset))
                return size

        binding = LowLevelBinding()
        with _CuFileHandle(binding, str(path), "r+") as handle:
            assert binding.registered_fd is not None
            assert binding.registered_fd >= 0
            assert handle.write(12345, 7, file_offset=4, dev_offset=2) == 7
            assert handle.read(67890, 5, file_offset=8, dev_offset=3) == 5

        assert binding.deregistered == ["handle-1"]
        assert binding.writes == [("handle-1", 12345, 7, 4, 2)]
        assert binding.reads == [("handle-1", 67890, 5, 8, 3)]
        assert handle._fd is None
        assert handle._handle is None

    def test_cufile_handle_high_level_passes_c_void_p_and_offsets(self, tmp_path):
        from adapter.storage.nvfile_backend import _CuFileHandle

        path = tmp_path / "raw.bin"
        path.write_bytes(b"\x00" * 32)
        calls = []

        class MockCuFile:
            def __init__(self, path_arg, mode_arg):
                self.path = path_arg
                self.mode = mode_arg
                self.closed = False

            def write(self, ptr, nbytes, file_offset=0, dev_offset=0):
                calls.append(("write", ptr.value, nbytes, file_offset, dev_offset))
                return nbytes

            def read(self, ptr, nbytes, file_offset=0, dev_offset=0):
                calls.append(("read", ptr.value, nbytes, file_offset, dev_offset))
                return nbytes

            def close(self):
                self.closed = True

        class HighLevelBinding:
            name = "high-level-mock"
            _mod = mock.Mock(CuFile=MockCuFile)

        with _CuFileHandle(HighLevelBinding(), str(path), "r+") as handle:
            assert handle.write(0xABC, 11, file_offset=6, dev_offset=1) == 11
            assert handle.read(0xDEF, 13, file_offset=9, dev_offset=2) == 13

        assert calls == [
            ("write", 0xABC, 11, 6, 1),
            ("read", 0xDEF, 13, 9, 2),
        ]
        assert handle._handle is None

    def test_detect_binding_prefers_cuda_bindings_when_available(self, monkeypatch):
        from adapter.storage import nvfile_backend

        fake_cufile = mock.Mock()
        fake_cufile.driver_open.return_value = None
        fake_cufile.driver_close.return_value = None
        fake_cufile.handle_register.return_value = "handle"
        fake_cufile.handle_deregister.return_value = None
        fake_cufile.read.return_value = 1
        fake_cufile.write.return_value = 1

        cuda_module = SimpleNamespace()
        bindings_module = SimpleNamespace(cufile=fake_cufile)
        monkeypatch.setitem(sys.modules, "cuda", cuda_module)
        monkeypatch.setitem(sys.modules, "cuda.bindings", bindings_module)
        monkeypatch.setitem(sys.modules, "cuda.bindings.cufile", fake_cufile)

        binding = nvfile_backend._detect_binding()

        assert binding is not None
        assert binding.name == "cuda.bindings.cufile"
        fake_cufile.driver_open.assert_called_once()

    def test_nvfile_backend_raises_clear_error_when_no_binding(self):
        from adapter.storage.nvfile_backend import NvFileBackend

        with mock.patch("adapter.storage.nvfile_backend._detect_binding", return_value=None):
            with pytest.raises(RuntimeError, match="No GDS library found"):
                NvFileBackend()

    def test_save_raises_on_short_gds_write_and_cleans_tmp(self, tmp_path):
        from adapter.storage.nvfile_backend import _CuFileHandle

        with _mock_nvfile_backend() as backend:
            original_write = _CuFileHandle.write

            def short_write(self, gpu_addr, nbytes, **kwargs):
                return nbytes - 1

            _CuFileHandle.write = short_write
            try:
                path = tmp_path / "short-write.kvcache"
                tensor = torch.arange(8, dtype=torch.float32)
                with pytest.raises(RuntimeError, match="GDS write: expected"):
                    backend.save(path, tensor)
                assert not path.exists()
                assert list(tmp_path.glob("*.tmp*")) == []
            finally:
                _CuFileHandle.write = original_write

    def test_load_raises_on_short_gds_read(self, tmp_path):
        from adapter.storage.nvfile_backend import _CuFileHandle

        with _mock_nvfile_backend() as backend:
            path = tmp_path / "short-read.kvcache"
            tensor = torch.arange(8, dtype=torch.float32)
            backend.save(path, tensor)

            original_read = _CuFileHandle.read

            def short_read(self, gpu_addr, nbytes, **kwargs):
                return nbytes - 1

            _CuFileHandle.read = short_read
            try:
                with pytest.raises(RuntimeError, match="GDS read: expected"):
                    backend.load(path, device="cpu")
            finally:
                _CuFileHandle.read = original_read

    def test_forced_gds_unavailable_falls_back_to_posix(self):
        from adapter.storage import backend as storage_backend

        with mock.patch("adapter.storage.backend._try_gds", return_value=None):
            result = storage_backend.create_storage_backend(prefer="gds")

        assert isinstance(result, PosixBackend)


# ═══════════════════════════════════════════════════════════════════
# POSIX positional I/O (write_tensor_at / load_tensor_slices)
# ═══════════════════════════════════════════════════════════════════


class TestPosixPositionalIO:
    """Positional I/O through PosixBackend."""

    @pytest.fixture
    def backend(self):
        return PosixBackend()

    @pytest.fixture
    def tmp_path(self):
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def _prepare_file(self, tmp_path, size=65536):
        p = tmp_path / "chunk.cobj"
        p.write_bytes(b"\x00" * size)
        return p

    def test_write_tensor_at(self, backend, tmp_path):
        p = self._prepare_file(tmp_path)
        tensor = torch.randn(2, 64, dtype=torch.bfloat16)
        nbytes = tensor.nbytes
        stored = (nbytes + 4095) // 4096 * 4096
        backend.write_tensor_at(p, tensor, 65536, stored)
        from adapter.storage.backend import TensorSliceSpec
        specs = [TensorSliceSpec(65536, nbytes, stored, (2, 64), torch.bfloat16)]
        loaded = backend.load_tensor_slices(p, specs, "cpu")
        assert len(loaded) == 1
        assert torch.equal(loaded[0], tensor)

    def test_load_tensor_slices_multiple(self, backend, tmp_path):
        p = self._prepare_file(tmp_path, size=262144)
        t1 = torch.randn(4, 8, dtype=torch.float32)
        t2 = torch.randn(2, 4, dtype=torch.bfloat16)
        stored1 = (t1.nbytes + 4095) // 4096 * 4096
        stored2 = (t2.nbytes + 4095) // 4096 * 4096
        off1, off2 = 65536, 65536 + stored1
        backend.write_tensor_at(p, t1, off1, stored1)
        backend.write_tensor_at(p, t2, off2, stored2)
        from adapter.storage.backend import TensorSliceSpec
        specs = [
            TensorSliceSpec(off1, t1.nbytes, stored1, (4, 8), torch.float32),
            TensorSliceSpec(off2, t2.nbytes, stored2, (2, 4), torch.bfloat16),
        ]
        loaded = backend.load_tensor_slices(p, specs, "cpu")
        assert len(loaded) == 2
        assert torch.equal(loaded[0], t1)
        assert loaded[0].dtype == torch.float32
        assert torch.equal(loaded[1], t2)
        assert loaded[1].dtype == torch.bfloat16

    def test_load_tensor_slices_bfloat16(self, backend, tmp_path):
        """bfloat16 round-trip via uint8 view — no numpy bf16 needed."""
        p = self._prepare_file(tmp_path)
        tensor = torch.randn(3, 16, dtype=torch.bfloat16)
        nbytes = tensor.nbytes
        stored = (nbytes + 4095) // 4096 * 4096
        backend.write_tensor_at(p, tensor, 65536, stored)
        from adapter.storage.backend import TensorSliceSpec
        specs = [TensorSliceSpec(65536, nbytes, stored, (3, 16), torch.bfloat16)]
        loaded = backend.load_tensor_slices(p, specs, "cpu")
        assert loaded[0].dtype == torch.bfloat16
        assert torch.equal(loaded[0], tensor)

    def test_load_rejects_truncated_alignment_padding(self, backend, tmp_path):
        from adapter.storage.backend import TensorSliceSpec

        p = self._prepare_file(tmp_path)
        tensor = torch.arange(8, dtype=torch.float32)
        stored = 4096
        backend.write_tensor_at(p, tensor, 65536, stored)
        with open(p, "r+b") as f:
            f.truncate(65536 + tensor.nbytes)

        spec = TensorSliceSpec(
            65536, tensor.nbytes, stored, tuple(tensor.shape), tensor.dtype
        )
        with pytest.raises(RuntimeError, match="expected 4096 bytes"):
            backend.load_tensor_slices(p, [spec], "cpu")

    def test_write_tensor_at_short_write_raises(self, backend, tmp_path):
        """stored_nbytes less than nbytes should raise."""
        p = self._prepare_file(tmp_path)
        tensor = torch.arange(8, dtype=torch.float32)
        with pytest.raises(ValueError, match="stored_nbytes"):
            backend.write_tensor_at(p, tensor, 65536, 4)


# ═══════════════════════════════════════════════════════════════════
# ChunkObject tests (pure CPU, no GPU needed)
# ═══════════════════════════════════════════════════════════════════


class TestChunkObject:
    """ChunkObjectWriter and load_chunk_object round-trip."""

    @pytest.fixture
    def tmp_path(self):
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def _writer(self, tmp_path, namespace="test-ns", key="k" * 64,
                shard=0, index=0, start=0, end=10,
                expected_layers=None, backend=None):
        from adapter.storage.chunk_object import ChunkObjectWriter
        if expected_layers is None:
            expected_layers = {"k", "v"}
        return ChunkObjectWriter(
            work_dir=tmp_path,
            namespace=namespace,
            key=key,
            shard=shard,
            index=index,
            start=start,
            end=end,
            expected_layers=expected_layers,
            backend=backend or PosixBackend(),
        )

    def test_roundtrip(self, tmp_path):
        from adapter.storage.chunk_object import load_chunk_object
        k_tensor = torch.randn(4, 8, dtype=torch.bfloat16)
        v_tensor = torch.randn(4, 8, dtype=torch.bfloat16)
        with self._writer(tmp_path) as w:
            w.add_layer("k", k_tensor)
            w.add_layer("v", v_tensor)
            manifest = w.finalize()
        assert manifest.complete
        assert manifest.namespace == "test-ns"
        assert manifest.start == 0
        assert manifest.end == 10
        assert len(manifest.layers) == 2
        assert w.final_path.exists()
        header, tensors = load_chunk_object(
            w.final_path, ["k", "v"], device="cpu", backend=PosixBackend(),
        )
        assert header.complete
        assert header.namespace == "test-ns"
        assert header.start == 0
        assert header.end == 10
        assert torch.equal(tensors["k"], k_tensor)
        assert tensors["k"].dtype == torch.bfloat16
        assert torch.equal(tensors["v"], v_tensor)

    def test_float8_dtype_roundtrip(self, tmp_path):
        if not hasattr(torch, "float8_e4m3fn"):
            pytest.skip("PyTorch build has no float8_e4m3fn")

        raw = torch.arange(32, dtype=torch.uint8)
        tensor = raw.view(torch.float8_e4m3fn).reshape(4, 8)
        with self._writer(tmp_path, expected_layers={"fp8"}) as w:
            w.add_layer("fp8", tensor)
            w.finalize()

        from adapter.storage.chunk_object import load_chunk_object
        _, tensors = load_chunk_object(
            w.final_path, ["fp8"], device="cpu", backend=PosixBackend()
        )
        assert tensors["fp8"].dtype == torch.float8_e4m3fn
        assert torch.equal(
            tensors["fp8"].view(torch.uint8), raw.reshape(4, 8)
        )

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"expected_layers": set()}, "expected_layers must be non-empty"),
            ({"index": -1}, "index must be >= 0"),
            ({"start": 4, "end": 4}, "Invalid token range"),
        ],
    )
    def test_writer_rejects_invalid_header_fields(
        self, tmp_path, kwargs, message
    ):
        with pytest.raises(ValueError, match=message):
            self._writer(tmp_path, **kwargs)

    def test_writer_rejects_too_many_shape_dimensions(self, tmp_path):
        with self._writer(tmp_path, expected_layers={"k"}) as w:
            with pytest.raises(ValueError, match="max is 8"):
                w.add_layer("k", torch.ones((1,) * 9))

    def test_missing_layer_rejected(self, tmp_path):
        with self._writer(tmp_path) as w:
            w.add_layer("k", torch.randn(4, 8, dtype=torch.bfloat16))
            with pytest.raises(ValueError, match="do not match expected"):
                w.finalize()

    def test_unexpected_layer_rejected(self, tmp_path):
        with self._writer(tmp_path, expected_layers={"k"}) as w:
            w.add_layer("k", torch.randn(4, 8, dtype=torch.bfloat16))
            with pytest.raises(ValueError, match="not in expected_layers"):
                w.add_layer("extra", torch.randn(4, 8, dtype=torch.bfloat16))

    def test_abort_cleans_temp(self, tmp_path):
        w = self._writer(tmp_path)
        w.add_layer("k", torch.randn(4, 8, dtype=torch.bfloat16))
        w.abort()
        assert not w.final_path.exists()

    def test_double_finalize_raises(self, tmp_path):
        with self._writer(tmp_path) as w:
            w.add_layer("k", torch.randn(4, 8, dtype=torch.bfloat16))
            w.add_layer("v", torch.randn(4, 8, dtype=torch.bfloat16))
            w.finalize()
            with pytest.raises(RuntimeError, match="already finalized"):
                w.finalize()

    def test_load_corrupted_magic(self, tmp_path):
        from adapter.storage.chunk_object import load_chunk_object
        with self._writer(tmp_path) as w:
            w.add_layer("k", torch.randn(4, 8, dtype=torch.bfloat16))
            w.add_layer("v", torch.randn(4, 8, dtype=torch.bfloat16))
            w.finalize()
        data = w.final_path.read_bytes()
        w.final_path.write_bytes(b"\x00" * 8 + data[8:])
        with pytest.raises(ValueError, match="Bad magic"):
            load_chunk_object(w.final_path, ["k", "v"], device="cpu")

    def test_load_incomplete_rejected(self, tmp_path):
        """Load of an incomplete (partial) chunk object should raise."""
        from adapter.storage.chunk_object import (_HEADER_SIZE, _HEADER_MAGIC,
                                                   _HEADER_VERSION)
        buf = bytearray(_HEADER_SIZE)
        struct.pack_into("<8s", buf, 0, _HEADER_MAGIC)
        struct.pack_into("<H", buf, 8, _HEADER_VERSION)
        struct.pack_into("<B", buf, 10, 0)  # complete = 0
        p = tmp_path / "incomplete.cobj"
        p.write_bytes(bytes(buf))
        with pytest.raises(ValueError, match="incomplete"):
            from adapter.storage.chunk_object import load_chunk_object
            load_chunk_object(p, ["k"], device="cpu")

    def test_add_layer_idempotent(self, tmp_path):
        """Adding the same layer name twice should be a no-op."""
        with self._writer(tmp_path) as w:
            t1 = torch.randn(4, 8, dtype=torch.bfloat16)
            w.add_layer("k", t1)
            w.add_layer("k", torch.randn(4, 8, dtype=torch.bfloat16))
            w.add_layer("v", torch.randn(4, 8, dtype=torch.bfloat16))
            w.finalize()
        from adapter.storage.chunk_object import load_chunk_object
        _, tensors = load_chunk_object(
            w.final_path, ["k"], device="cpu", backend=PosixBackend(),
        )
        assert torch.equal(tensors["k"], t1)


    def _complete(self, writer):
        writer.add_layer("k", torch.randn(4, 8, dtype=torch.bfloat16))
        writer.add_layer("v", torch.randn(4, 8, dtype=torch.bfloat16))
        return writer.finalize()

    def test_abort_removes_temp_file_and_is_idempotent(self, tmp_path):
        w = self._writer(tmp_path)
        w.add_layer("k", torch.randn(4, 8, dtype=torch.bfloat16))
        temp_path = w._tmp_path
        assert temp_path is not None and temp_path.exists()

        w.abort()
        w.abort()

        assert not temp_path.exists()
        assert list(tmp_path.rglob("*.cobj.tmp")) == []
        assert not w.final_path.exists()

    def test_constructor_failure_cleans_temp_file(self, tmp_path, monkeypatch):
        import builtins

        real_open = builtins.open
        opened = []

        def failing_open(*args, **kwargs):
            opened.append(args[0])
            if str(args[0]).endswith(".cobj.tmp"):
                raise OSError("open failed")
            return real_open(*args, **kwargs)

        monkeypatch.setattr(builtins, "open", failing_open)
        with pytest.raises(OSError, match="open failed"):
            self._writer(tmp_path)

        assert opened
        assert list(tmp_path.rglob("*.cobj.tmp")) == []

    def test_same_identity_reuses_existing_object_without_overwrite(self, tmp_path):
        first = self._writer(tmp_path, namespace="ns", key="key", shard="s")
        first_header = self._complete(first)
        final_path = first.final_path
        original = final_path.read_bytes()

        second = self._writer(tmp_path, namespace="ns", key="key", shard="s")
        second_header = self._complete(second)

        assert second_header == first_header
        assert final_path.read_bytes() == original
        assert not second._tmp_path.exists()

    def test_conflicting_identity_does_not_overwrite_existing_object(self, tmp_path):
        first = self._writer(tmp_path, namespace="ns", key="key", shard="s")
        self._complete(first)
        final_path = first.final_path
        original = final_path.read_bytes()

        second = self._writer(
            tmp_path,
            namespace="ns",
            key="key",
            shard="s",
            end=11,
        )
        second.add_layer("k", torch.randn(4, 8, dtype=torch.bfloat16))
        second.add_layer("v", torch.randn(4, 8, dtype=torch.bfloat16))
        with pytest.raises(ValueError, match="invalid or conflicts"):
            second.finalize()

        assert final_path.read_bytes() == original
        assert not second._tmp_path.exists()

    def test_existing_corrupt_object_is_not_overwritten(self, tmp_path):
        first = self._writer(tmp_path, namespace="ns", key="key", shard="s")
        self._complete(first)
        final_path = first.final_path
        final_path.write_bytes(b"corrupt")

        second = self._writer(tmp_path, namespace="ns", key="key", shard="s")
        second.add_layer("k", torch.randn(4, 8, dtype=torch.bfloat16))
        second.add_layer("v", torch.randn(4, 8, dtype=torch.bfloat16))
        with pytest.raises(ValueError, match="invalid or conflicts"):
            second.finalize()

        assert final_path.read_bytes() == b"corrupt"
        assert not second._tmp_path.exists()

    def test_file_exists_race_validates_competing_object(self, tmp_path, monkeypatch):
        import adapter.storage.chunk_object as chunk_object

        first = self._writer(tmp_path, namespace="ns", key="key", shard="s")
        expected = self._complete(first)
        second = self._writer(tmp_path, namespace="ns", key="key", shard="s")
        second.add_layer("k", torch.randn(4, 8, dtype=torch.bfloat16))
        second.add_layer("v", torch.randn(4, 8, dtype=torch.bfloat16))
        real_link = chunk_object.os.link

        def race_link(source, destination):
            if not Path(destination).exists():
                Path(destination).parent.mkdir(parents=True, exist_ok=True)
                real_link(first.final_path, destination)
            raise FileExistsError(destination)

        monkeypatch.setattr(chunk_object.os, "link", race_link)
        result = second.finalize()

        assert result == expected
        assert first.final_path.exists()
        assert not second._tmp_path.exists()

    def test_finalize_fsyncs_parent_directory(self, tmp_path, monkeypatch):
        import adapter.storage.chunk_object as chunk_object

        fsynced = []
        monkeypatch.setattr(
            chunk_object,
            "_fsync_directory",
            lambda path: fsynced.append(Path(path)),
        )
        writer = self._writer(tmp_path, namespace="ns", key="key", shard="s")
        self._complete(writer)

        assert fsynced == [writer.final_path.parent]

# ═══════════════════════════════════════════════════════════════════
# GDS positional I/O mock tests (no real driver needed)
# ═══════════════════════════════════════════════════════════════════


class TestNvFileBackendPositionalIOMocked:
    """NvFileBackend write_tensor_at / load_tensor_slices with mocked cufile."""

    @pytest.fixture
    def tmp_path(self):
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    @pytest.fixture
    def backend(self):
        with _mock_nvfile_backend() as backend:
            yield backend

    def _prepare_file(self, tmp_path, size=262144):
        p = tmp_path / "chunk.cobj"
        p.write_bytes(b"\x00" * size)
        return p

    def test_gds_write_tensor_at(self, backend, tmp_path):
        p = self._prepare_file(tmp_path)
        tensor = torch.randn(4, 16, dtype=torch.bfloat16)
        nbytes = tensor.nbytes
        stored = (nbytes + 4095) // 4096 * 4096
        backend.write_tensor_at(p, tensor, 65536, stored)
        from adapter.storage.backend import TensorSliceSpec
        specs = [TensorSliceSpec(65536, nbytes, stored, (4, 16), torch.bfloat16)]
        loaded = backend.load_tensor_slices(p, specs, "cpu")
        assert torch.equal(loaded[0], tensor)

    def test_gds_load_tensor_slices_multiple(self, backend, tmp_path):
        p = self._prepare_file(tmp_path, size=262144)
        t1 = torch.randn(2, 8, dtype=torch.float32)
        t2 = torch.randn(3, 12, dtype=torch.bfloat16)
        stored1 = (t1.nbytes + 4095) // 4096 * 4096
        stored2 = (t2.nbytes + 4095) // 4096 * 4096
        off1, off2 = 65536, 65536 + stored1
        backend.write_tensor_at(p, t1, off1, stored1)
        backend.write_tensor_at(p, t2, off2, stored2)
        from adapter.storage.backend import TensorSliceSpec
        specs = [
            TensorSliceSpec(off1, t1.nbytes, stored1, (2, 8), torch.float32),
            TensorSliceSpec(off2, t2.nbytes, stored2, (3, 12), torch.bfloat16),
        ]
        loaded = backend.load_tensor_slices(p, specs, "cpu")
        assert len(loaded) == 2
        assert torch.equal(loaded[0], t1)
        assert torch.equal(loaded[1], t2)

    def test_gds_aligned_offset_check(self, backend, tmp_path):
        """Non-aligned offset should raise."""
        p = self._prepare_file(tmp_path)
        tensor = torch.arange(4, dtype=torch.float32)
        with pytest.raises(ValueError, match="aligned"):
            backend.write_tensor_at(p, tensor, 100, 4096)

    def test_gds_aligned_stored_nbytes_check(self, backend, tmp_path):
        """Non-aligned stored_nbytes should raise."""
        p = self._prepare_file(tmp_path)
        tensor = torch.arange(4, dtype=torch.float32)
        with pytest.raises(ValueError, match="aligned"):
            backend.write_tensor_at(p, tensor, 65536, 100)
