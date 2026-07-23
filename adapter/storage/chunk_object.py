# SPDX-License-Identifier: Apache-2.0
"""Chunk-object format: a single logical chunk aggregating all local layers.

File layout
-----------
::

    [0 .. 65535]        64 KiB binary header  (see below)
    [65536 .. ]         Layer payload data, each 4 KiB aligned

Header structure (fixed 384 bytes + per-layer 192-byte entries)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
+---------------+--------+-----------------------------------------+
| Field         | Bytes  | Description                             |
+---------------+--------+-----------------------------------------+
| magic         |      8 | Magic bytes ``PREDCOBJ``                |
| version       |      2 | uint16, currently 2                     |
| complete      |      1 | uint8, 1 when all expected layers set   |
| namespace     |     96 | Namespace string (null-padded)          |
| key           |     64 | Chunk key string (null-padded)          |
| shard         |     64 | Shard string (null-padded)              |
| index         |      4 | uint32 chunk index                      |
| start         |      4 | uint32 start token position (inclusive) |
| end           |      4 | uint32 end token position (exclusive)   |
| num_layers    |      1 | uint8 number of layer entries           |
| reserved      |    136 | Padding to 384 bytes                    |
+---------------+--------+-----------------------------------------+
Then *num_layers* × **layer entries** of 192 bytes each:
+---------------+--------+-----------------------------------------+
| Field         | Bytes  | Description                             |
+---------------+--------+-----------------------------------------+
| name          |     80 | Layer name (null-padded)                |
| dtype         |     32 | Data type string (null-padded)          |
| shape_ndim    |      1 | uint8 number of shape dimensions        |
| shape_bytes   |     32 | Up to 8 × uint32 shape dims            |
| offset        |      8 | uint64 byte offset from end of header   |
| nbytes        |      8 | uint64 logical tensor size in bytes     |
| stored_nbytes |      8 | uint64 stored (padded) size in bytes   |
| reserved      |     23 | Padding to 192 bytes                    |
+---------------+--------+-----------------------------------------+
Remaining header bytes (up to 65536) are zero-filled.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
from urllib.parse import quote

import torch

from adapter.storage.backend import (
    StorageBackend,
    TensorSliceSpec,
    create_storage_backend,
)

# ── Constants ───────────────────────────────────────────────────────────

_HEADER_SIZE = 64 * 1024  # 64 KiB
_PAYLOAD_ALIGN = 4 * 1024  # 4 KiB

_HEADER_MAGIC = b"PREDCOBJ"
_HEADER_VERSION = 2
_MAX_LAYERS = 64
_MAX_SHAPE_DIMS = 8

# Fixed header: 384 bytes.
# V2 stores the full string shard identifier used by Go metadata.
#   magic       8s
#   version     H  (uint16)
#   complete    B  (uint8)
#   namespace   96s
#   key         64s
#   shard       64s
#   index       I  (uint32)
#   start       I  (uint32)
#   end         I  (uint32)
#   num_layers  B  (uint8)
#   reserved    136x
_FIXED_HEADER_FMT = struct.Struct("<8s H B 96s 64s 64s I I I B 136x")
_FIXED_HEADER_SIZE = 384

# Per-layer entry: 192 bytes, enough for vLLM names and FP8 dtype strings.
#   name          80s
#   dtype         32s
#   shape_ndim    B  (uint8)
#   shape_bytes   32s  (packed as up to 8 × uint32)
#   offset        Q  (uint64)
#   nbytes        Q  (uint64)
#   stored_nbytes Q  (uint64)
#   reserved      23x
_LAYER_ENTRY_FMT = struct.Struct("<80s 32s B 32s Q Q Q 23x")
_LAYER_ENTRY_SIZE = 192


def _fsync_directory(path: Path) -> None:
    """Fsync a directory after publishing or removing a cache object."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# ── Data types ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LayerSpec:
    """Metadata for a single layer stored in a chunk object."""
    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    offset: int
    nbytes: int
    stored_nbytes: int


@dataclass(frozen=True)
class ChunkObjectHeader:
    """Parsed header of a chunk-object file."""
    version: int
    complete: bool
    namespace: str
    key: str
    shard: str
    index: int
    start: int
    end: int
    layers: tuple[LayerSpec, ...]


# ── Public helpers ──────────────────────────────────────────────────────


def _align_up(size: int, alignment: int) -> int:
    """Return *size* rounded up to the next multiple of *alignment*."""
    return (size + alignment - 1) // alignment * alignment


def _encode_fixed_string(s: str, length: int) -> bytes:
    """Encode a fixed string, rejecting ambiguous truncation."""
    if "\x00" in s:
        raise ValueError("Fixed-width strings cannot contain NUL bytes")
    encoded = s.encode("utf-8")
    if len(encoded) > length:
        raise ValueError(
            f"String is {len(encoded)} bytes, exceeds fixed field {length}: {s!r}"
        )
    return encoded.ljust(length, b"\x00")


def _decode_fixed_string(b: bytes) -> str:
    """Decode a null-padded (or space-padded) fixed-length string."""
    return b.rstrip(b"\x00").decode("utf-8")


def _dtype_to_str(dt: torch.dtype) -> str:
    return str(dt)


def _str_to_dtype(s: str) -> torch.dtype:
    names = (
        "float16", "bfloat16", "float32", "float64",
        "float8_e4m3fn", "float8_e5m2", "uint8", "int8",
        "int16", "int32", "int64",
    )
    dtype_map = {
        f"torch.{name}": dtype
        for name in names
        if (dtype := getattr(torch, name, None)) is not None
    }
    dtype = dtype_map.get(s)
    if dtype is None:
        raise ValueError(f"Unknown or unsupported dtype: {s}")
    return dtype


def _pack_shape(shape: tuple[int, ...]) -> bytes:
    """Pack shape dims as up to 8 × uint32 into a 32-byte block."""
    if len(shape) > _MAX_SHAPE_DIMS:
        raise ValueError(f"Shape has {len(shape)} dims, max is {_MAX_SHAPE_DIMS}")
    buf = bytearray(32)
    for i, d in enumerate(shape):
        struct.pack_into("<I", buf, i * 4, d)
    return bytes(buf)


def _unpack_shape(data: bytes, ndim: int) -> tuple[int, ...]:
    """Unpack up to *ndim* uint32 dims from a 32-byte block."""
    shape = []
    for i in range(ndim):
        shape.append(struct.unpack_from("<I", data, i * 4)[0])
    return tuple(shape)


# ── Writer ──────────────────────────────────────────────────────────────


class ChunkObjectWriter:
    """Write a chunk-object file incrementally without caching GPU tensors.

    Usage::

        writer = ChunkObjectWriter(tmp_path, namespace, key, shard,
                                   index, start, end,
                                   expected_layers={"k_cache", "v_cache"})
        writer.add_layer("k_cache", k_tensor)
        writer.add_layer("v_cache", v_tensor)
        manifest = writer.finalize()
    """

    def __init__(
        self,
        work_dir: str | Path,
        namespace: str,
        key: str,
        shard: str,
        index: int,
        start: int,
        end: int,
        expected_layers: set[str],
        backend: Optional[StorageBackend] = None,
    ):
        shard = str(shard)
        expected = frozenset(expected_layers)
        if not namespace or not key or not shard:
            raise ValueError("namespace, key, and shard must be non-empty")
        if index < 0:
            raise ValueError(f"index must be >= 0, got {index}")
        if start < 0 or end <= start:
            raise ValueError(f"Invalid token range: [{start}, {end})")
        if not expected:
            raise ValueError("expected_layers must be non-empty")
        if len(expected) > _MAX_LAYERS:
            raise ValueError(
                f"expected_layers has {len(expected)} entries, max is {_MAX_LAYERS}"
            )

        # Validate fixed-width fields before creating a temporary file.  This
        # keeps every object writer capable of producing a loadable header.
        _encode_fixed_string(namespace, 96)
        _encode_fixed_string(key, 64)
        _encode_fixed_string(shard, 64)
        for name in expected:
            if not name:
                raise ValueError("expected layer names must be non-empty")
            _encode_fixed_string(name, 80)

        self._work_dir = Path(work_dir)
        self._work_dir.mkdir(parents=True, exist_ok=True)

        self._namespace = namespace
        self._key = key
        self._shard = shard
        self._index = index
        self._start = start
        self._end = end
        self._expected_layers = expected
        self._backend = backend or create_storage_backend()

        # Temporary file — written synchronously per layer.  Keep the path as
        # a Path and clean it up if opening/reserving the file fails: __init__
        # failures do not run a context manager's __exit__ method.
        import tempfile

        self._tmp_path: Optional[Path] = None
        self._file = None
        fd = -1
        try:
            fd, tmp_path = tempfile.mkstemp(
                suffix=".cobj.tmp",
                dir=str(self._work_dir),
            )
            self._tmp_path = Path(tmp_path)
            os.close(fd)
            fd = -1

            # Reserve header space.
            self._file = open(self._tmp_path, "wb")
            self._file.write(b"\x00" * _HEADER_SIZE)
            self._file.flush()
        except BaseException:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if self._file is not None:
                try:
                    self._file.close()
                except Exception:
                    pass
                self._file = None
            if self._tmp_path is not None:
                try:
                    self._tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
            raise

        # Track added layers
        self._layers: dict[str, LayerSpec] = {}
        self._finalized = False
        self._next_offset = _HEADER_SIZE  # first payload byte
        self._aborted = False


    @property
    def final_path(self) -> Path:
        """The final on-disk path (available before and after finalize)."""
        return self._resolve_final_path()

    # ── public API ─────────────────────────────────────────────────────

    def add_layer(self, name: str, tensor: torch.Tensor) -> None:
        """Write one layer at its aligned offset without retaining the tensor."""
        if self._finalized:
            raise RuntimeError("ChunkObjectWriter already finalized")
        if self._aborted:
            raise RuntimeError("ChunkObjectWriter already aborted")
        if name in self._layers:
            return
        if name not in self._expected_layers:
            raise ValueError(
                f"Layer {name!r} is not in expected_layers: "
                f"{sorted(self._expected_layers)}"
            )

        tensor = tensor.contiguous()
        nbytes = tensor.nbytes
        if nbytes <= 0:
            raise ValueError(f"Layer {name!r} tensor must be non-empty")
        shape = tuple(tensor.shape)
        _pack_shape(shape)
        _encode_fixed_string(_dtype_to_str(tensor.dtype), 32)
        stored_nbytes = _align_up(nbytes, _PAYLOAD_ALIGN)
        offset = _align_up(self._next_offset, _PAYLOAD_ALIGN)
        self._file.truncate(offset + stored_nbytes)
        self._file.flush()
        self._backend.write_tensor_at(
            Path(self._tmp_path), tensor, offset, stored_nbytes
        )
        self._file.seek(offset + stored_nbytes)
        self._next_offset = offset + stored_nbytes
        self._layers[name] = LayerSpec(
            name=name,
            dtype=tensor.dtype,
            shape=shape,
            offset=offset,
            nbytes=nbytes,
            stored_nbytes=stored_nbytes,
        )

    def finalize(self) -> ChunkObjectHeader:
        """Finalise the file and atomically publish an immutable object.

        A same-filesystem hard link gives create-if-absent semantics.  If a
        competing writer already published this deterministic object, its full
        structural contract must match before it is reused.
        """
        if self._finalized:
            raise RuntimeError("ChunkObjectWriter already finalized")

        added = frozenset(self._layers.keys())
        if added != self._expected_layers:
            raise ValueError(
                f"Added layers {sorted(added)} do not match expected "
                f"{sorted(self._expected_layers)}"
            )

        header = self._build_header(complete=True)
        assert self._file is not None
        assert self._tmp_path is not None
        self._file.seek(0)
        self._file.write(self._serialize_header(header))
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        self._file = None

        final_path = self._resolve_final_path()
        final_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(self._tmp_path, final_path)
        except FileExistsError:
            try:
                existing = self._validate_existing_object(final_path, header)
            except Exception:
                self._cleanup_temp()
                self._aborted = True
                raise
            self._cleanup_temp()
            self._finalized = True
            return existing
        except Exception:
            self._cleanup_temp()
            self._aborted = True
            raise

        try:
            self._tmp_path.unlink(missing_ok=True)
            _fsync_directory(final_path.parent)
        except Exception:
            # The final link is intentionally retained. Removing it here could
            # destroy a publication that another process already sees.
            self._finalized = True
            raise

        self._finalized = True
        return header

    def abort(self) -> None:
        """Abort writing and remove the temporary file."""
        if self._finalized:
            return
        self._aborted = True
        self._cleanup_temp()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._finalized:
            self.abort()

    # ── internal helpers ──────────────────────────────────────────────

    def _build_header(self, complete: bool) -> ChunkObjectHeader:
        layers = tuple(
            LayerSpec(
                name=spec.name,
                dtype=spec.dtype,
                shape=spec.shape,
                offset=spec.offset,
                nbytes=spec.nbytes,
                stored_nbytes=spec.stored_nbytes,
            )
            for spec in self._layers.values()
        )
        return ChunkObjectHeader(
            version=_HEADER_VERSION,
            complete=complete,
            namespace=self._namespace,
            key=self._key,
            shard=self._shard,
            index=self._index,
            start=self._start,
            end=self._end,
            layers=layers,
        )

    def _serialize_header(self, header: ChunkObjectHeader) -> bytes:
        buf = bytearray(_HEADER_SIZE)

        # Fixed portion
        _FIXED_HEADER_FMT.pack_into(
            buf, 0,
            _HEADER_MAGIC,
            _HEADER_VERSION,
            1 if header.complete else 0,
            _encode_fixed_string(header.namespace, 96),
            _encode_fixed_string(header.key, 64),
            _encode_fixed_string(header.shard, 64),
            header.index,
            header.start,
            header.end,
            len(header.layers),
        )

        # Layer entries
        base = _FIXED_HEADER_SIZE
        for i, layer in enumerate(header.layers):
            off = base + i * _LAYER_ENTRY_SIZE
            _LAYER_ENTRY_FMT.pack_into(
                buf, off,
                _encode_fixed_string(layer.name, 80),
                _encode_fixed_string(_dtype_to_str(layer.dtype), 32),
                len(layer.shape),
                _pack_shape(layer.shape),
                layer.offset,
                layer.nbytes,
                layer.stored_nbytes,
            )

        return bytes(buf)

    def _resolve_final_path(self) -> Path:
        """Return the final on-disk path for this chunk object."""
        return (
            self._work_dir
            / "v2"
            / "chain"
            / quote(self._namespace, safe="")
            / quote(self._key, safe="")
            / f"{quote(self._shard, safe='')}.cobj"
        )

    def _cleanup_temp(self) -> None:
        """Close and remove the temporary object, if it still exists."""
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
        if self._tmp_path is not None:
            try:
                self._tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _validate_existing_object(
        self,
        path: Path,
        expected: ChunkObjectHeader,
    ) -> ChunkObjectHeader:
        """Validate an object published by a competing writer."""
        try:
            with open(path, "rb") as f:
                blob = f.read(_HEADER_SIZE)
            if len(blob) != _HEADER_SIZE:
                raise ValueError(
                    f"existing chunk object header is truncated: {path}"
                )
            existing = _parse_header(blob)
            if existing.version != _HEADER_VERSION or not existing.complete:
                raise ValueError(
                    f"existing chunk object is not a complete v{_HEADER_VERSION} "
                    f"object: {path}"
                )

            scalar_fields = (
                ("namespace", existing.namespace, expected.namespace),
                ("key", existing.key, expected.key),
                ("shard", existing.shard, expected.shard),
                ("index", existing.index, expected.index),
                ("start", existing.start, expected.start),
                ("end", existing.end, expected.end),
            )
            for name, actual, wanted in scalar_fields:
                if actual != wanted:
                    raise ValueError(
                        f"existing chunk object {name} mismatch: "
                        f"{actual!r} != {wanted!r}"
                    )

            expected_layers = {layer.name: layer for layer in expected.layers}
            existing_layers = {layer.name: layer for layer in existing.layers}
            if set(existing_layers) != set(expected_layers):
                raise ValueError(
                    "existing chunk object layer set mismatch: "
                    f"{sorted(existing_layers)} != {sorted(expected_layers)}"
                )
            for name, wanted in expected_layers.items():
                actual = existing_layers[name]
                if (
                    actual.dtype != wanted.dtype
                    or actual.shape != wanted.shape
                    or actual.nbytes != wanted.nbytes
                    or actual.stored_nbytes != wanted.stored_nbytes
                ):
                    raise ValueError(
                        f"existing chunk object layer contract mismatch: {name}"
                    )

            file_size = path.stat().st_size
            if file_size < _HEADER_SIZE:
                raise ValueError(
                    f"existing chunk object is smaller than its header: {path}"
                )
            if existing.layers and max(
                layer.offset + layer.stored_nbytes
                for layer in existing.layers
            ) > file_size:
                raise ValueError(
                    f"existing chunk object payload exceeds file size: {path}"
                )
            return existing
        except Exception as exc:
            raise ValueError(
                f"existing chunk object is invalid or conflicts: {exc}"
            ) from exc


# ── Loader ──────────────────────────────────────────────────────────────


def resolve_chunk_object_path(
    cache_root: str | Path,
    namespace: str,
    key: str,
    shard: str,
) -> Path:
    """Resolve a chunk-object path from its logical components.

    Format: ``{root}/v2/chain/{namespace}/{key}/{shard}.cobj``
    """
    root = Path(cache_root).resolve()
    return (
        root
        / "v2"
        / "chain"
        / quote(namespace, safe="")
        / quote(key, safe="")
        / f"{quote(shard, safe='')}.cobj"
    )


def load_chunk_object(
    path: str | Path,
    layer_names: Sequence[str],
    device: str = "cuda",
    backend: Optional[StorageBackend] = None,
) -> tuple[ChunkObjectHeader, dict[str, torch.Tensor]]:
    """Load specific layers from a chunk-object file.

    Parameters
    ----------
    path:
        Path to the ``.cobj`` file.
    layer_names:
        Names of layers to load.  Must be a subset of the layers stored
        in the file.
    device:
        Target device for the loaded tensors.
    backend:
        Storage backend to use.  Defaults to auto-detected.

    Returns
    -------
    tuple[ChunkObjectHeader, dict[str, torch.Tensor]]
        The parsed header and a mapping from layer name to tensor.

    Raises
    ------
    ValueError
        If the magic, version, complete flag, namespace, key, shard,
        token range, or requested layer set is invalid.
    """
    path = Path(path)
    if not layer_names or len(set(layer_names)) != len(layer_names):
        raise ValueError("layer_names must be non-empty and unique")
    backend = backend or create_storage_backend()

    # Read header
    with open(path, "rb") as f:
        header_blob = f.read(_HEADER_SIZE)

    if len(header_blob) < _HEADER_SIZE:
        raise ValueError(
            f"Chunk object file too small: {len(header_blob)} bytes "
            f"(expected at least {_HEADER_SIZE})"
        )

    header = _parse_header(header_blob)

    # Validate header fields
    if header.version != _HEADER_VERSION:
        raise ValueError(
            f"Unsupported chunk-object version {header.version} "
            f"(expected {_HEADER_VERSION})"
        )
    if not header.complete:
        raise ValueError(
            f"Chunk object is incomplete (complete flag is false): {path}"
        )

    # Validate requested layers exist
    layer_map = {l.name: l for l in header.layers}
    requested_set = set(layer_names)
    available_set = set(layer_map.keys())
    missing = requested_set - available_set
    if missing:
        raise ValueError(
            f"Requested layers {sorted(missing)} not found in chunk object. "
            f"Available: {sorted(available_set)}"
    )

    # Build TensorSliceSpec list in the order requested
    specs: list[TensorSliceSpec] = []
    for name in layer_names:
        ls = layer_map[name]
        specs.append(TensorSliceSpec(
            offset=ls.offset,
            nbytes=ls.nbytes,
            stored_nbytes=ls.stored_nbytes,
            shape=ls.shape,
            dtype=ls.dtype,
        ))

    file_size = path.stat().st_size
    if header.layers and max(
        layer.offset + layer.stored_nbytes for layer in header.layers
    ) > file_size:
        raise ValueError(
            f"Chunk object payload exceeds file size {file_size}: {path}"
        )
    tensors = backend.load_tensor_slices(path, specs, device)
    if len(tensors) != len(specs):
        raise RuntimeError(
            f"Storage backend returned {len(tensors)} tensors for {len(specs)} specs"
        )
    result = dict(zip(layer_names, tensors))
    return header, result


def _parse_header(blob: bytes) -> ChunkObjectHeader:
    """Parse a 64 KiB header blob and return a :class:`ChunkObjectHeader`."""
    # Fixed portion
    magic, version, complete, ns_bytes, key_bytes, shard_bytes, idx, start, end, num_layers = (
        _FIXED_HEADER_FMT.unpack_from(blob, 0)
    )

    if magic != _HEADER_MAGIC:
        raise ValueError(
            f"Bad magic: {magic!r} (expected {_HEADER_MAGIC!r})"
        )

    if not complete:
        return ChunkObjectHeader(
            version=version,
            complete=False,
            namespace=_decode_fixed_string(ns_bytes),
            key=_decode_fixed_string(key_bytes),
            shard=_decode_fixed_string(shard_bytes),
            index=idx,
            start=start,
            end=end,
            layers=(),
        )
    if num_layers == 0:
        raise ValueError("Complete chunk object declares no layers")
    if num_layers > _MAX_LAYERS:
        raise ValueError(f"Header declares too many layers: {num_layers}")
    if end <= start:
        raise ValueError(f"Invalid token range in header: [{start}, {end})")

    layers: list[LayerSpec] = []
    layer_names: set[str] = set()
    base = _FIXED_HEADER_SIZE
    previous_end = _HEADER_SIZE
    for i in range(num_layers):
        off = base + i * _LAYER_ENTRY_SIZE
        name_b, dtype_b, ndim, shape_bytes, offset, nbytes, stored_nbytes = (
            _LAYER_ENTRY_FMT.unpack_from(blob, off)
        )
        if ndim > _MAX_SHAPE_DIMS:
            raise ValueError(f"Layer {i} has too many dimensions: {ndim}")
        if not name_b.rstrip(b"\x00"):
            raise ValueError(f"Layer {i} has an empty name")
        if offset < _HEADER_SIZE or offset % _PAYLOAD_ALIGN != 0:
            raise ValueError(f"Layer {i} has invalid offset {offset}")
        if nbytes <= 0 or stored_nbytes < nbytes or stored_nbytes % _PAYLOAD_ALIGN != 0:
            raise ValueError(f"Layer {i} has invalid byte sizes")
        if offset < previous_end:
            raise ValueError(f"Layer {i} payload overlaps a previous layer")
        previous_end = offset + stored_nbytes
        name = _decode_fixed_string(name_b)
        if name in layer_names:
            raise ValueError(f"Header declares duplicate layer name: {name!r}")
        layer_names.add(name)
        dtype = _str_to_dtype(_decode_fixed_string(dtype_b))
        shape = _unpack_shape(shape_bytes, ndim)
        expected_nbytes = torch.empty((), dtype=dtype).element_size()
        for dim in shape:
            expected_nbytes *= dim
        if expected_nbytes != nbytes:
            raise ValueError(
                f"Layer {i} shape/dtype requires {expected_nbytes} bytes, "
                f"header declares {nbytes}"
            )
        layers.append(LayerSpec(
            name=name,
            dtype=dtype,
            shape=shape,
            offset=offset,
            nbytes=nbytes,
            stored_nbytes=stored_nbytes,
        ))

    return ChunkObjectHeader(
        version=version,
        complete=bool(complete),
        namespace=_decode_fixed_string(ns_bytes),
        key=_decode_fixed_string(key_bytes),
        shard=_decode_fixed_string(shard_bytes),
        index=idx,
        start=start,
        end=end,
        layers=tuple(layers),
    )
