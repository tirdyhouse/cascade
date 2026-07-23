# SPDX-License-Identifier: Apache-2.0
"""
Storage backends for disk cache data I/O.

Provides GPU Direct Storage (GDS) via nvfile/cufile and POSIX fallback.
Usage::

    from adapter.storage import create_storage_backend

    backend = create_storage_backend()
    backend.save(path, gpu_tensor)       # GPU↔disk, either GDS or POSIX
    tensor  = backend.load(path, device) # disk↔GPU, either GDS or POSIX

Chunk-object format::

    from adapter.storage.chunk_object import ChunkObjectWriter, load_chunk_object

    writer = ChunkObjectWriter(...)
    writer.add_layer("k", k_tensor)
    writer.add_layer("v", v_tensor)
    manifest = writer.finalize()
    header, tensors = load_chunk_object(path, ["k", "v"])
"""

from adapter.storage.backend import (
    StorageBackend,
    TensorSliceSpec,
    create_storage_backend,
)

__all__ = [
    "StorageBackend",
    "TensorSliceSpec",
    "create_storage_backend",
]
