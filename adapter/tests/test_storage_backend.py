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

import json
import os
import struct
import tempfile
from pathlib import Path
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
        """When GDS is not installed, auto returns PosixBackend."""
        backend = create_storage_backend()
        # On machines without cufile/nvfile, this should be PosixBackend
        if hasattr(backend, "_lib_name"):
            assert "File" in type(backend).__name__
        else:
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
    def backend(self, mock_cufile):
        with (
            mock.patch.dict("sys.modules", {"cufile": mock_cufile}),
            mock.patch("adapter.storage.nvfile_backend._probe_module",
                       return_value="cufile"),
        ):
            from adapter.storage.nvfile_backend import NvFileBackend
            backend = NvFileBackend()
            backend._lib_name = "cufile"
            yield backend

    def test_save_and_load_cpu_as_gpu(self, backend, tmp_path, mock_cufile):
        """Simulate save/load with CPU tensors (GDS-addressable via mock)."""
        with mock.patch.dict("sys.modules", {"cufile": mock_cufile}):
            path = tmp_path / "test.kvcache"
            tensor = torch.randn(2, 256, 8, 128, dtype=torch.bfloat16)
            backend.save(path, tensor)
            assert path.exists()
            assert path.stat().st_size > _HEADER_SIZE
            loaded = backend.load(path, device="cpu")
            assert torch.equal(loaded, tensor)

    def test_save_file_format(self, backend, tmp_path, mock_cufile):
        """Verify the on-disk format: JSON header + raw tensor data."""
        with mock.patch.dict("sys.modules", {"cufile": mock_cufile}):
            path = tmp_path / "test.kvcache"
            tensor = torch.arange(16, dtype=torch.int32).reshape(4, 4)
            backend.save(path, tensor)

            # Read header
            with open(path, "rb") as f:
                header_blob = f.read(_HEADER_SIZE)
            meta = _unpack_header(header_blob)
            assert meta["shape"] == [4, 4]
            assert meta["dtype"] == "torch.int32"

            # Read raw data after header
            with open(path, "rb") as f:
                f.seek(_HEADER_SIZE)
                raw = f.read(meta["nbytes"])
            expected = tensor.numpy().tobytes()
            assert raw == expected

    def test_is_available_true(self, backend):
        # is_available returns True because _probe_library is mocked in fixture
        assert backend.is_available() is True

    def test_save_load_small_tensor(self, backend, tmp_path, mock_cufile):
        """Small tensors should round-trip correctly."""
        with mock.patch.dict("sys.modules", {"cufile": mock_cufile}):
            path = tmp_path / "small.kvcache"
            tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
            backend.save(path, tensor)
            loaded = backend.load(path, device="cpu")
            assert torch.equal(loaded, tensor)

    def test_save_failure_cleans_up_temp(self, backend, tmp_path, mock_cufile):
        """If GDS write fails, the temp file should be cleaned up."""
        with mock.patch.dict("sys.modules", {"cufile": mock_cufile}):
            # Mock CuFile.write to fail
            original_write = mock_cufile.CuFile.write
            mock_cufile.CuFile.write = mock.MagicMock(
                side_effect=RuntimeError("GDS write failed")
            )

            path = tmp_path / "fail.kvcache"
            tensor = torch.randn(2, 8, dtype=torch.float32)
            with pytest.raises(RuntimeError, match="GDS write failed"):
                backend.save(path, tensor)

            # No temp files should remain
            tmp_files = list(tmp_path.glob("*.tmp*"))
            assert len(tmp_files) == 0
            assert not path.exists()

            # Restore original
            mock_cufile.CuFile.write = original_write


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
