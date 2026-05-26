# SPDX-License-Identifier: Apache-2.0
"""
Storage backends for disk cache data I/O.

Provides GPU Direct Storage (GDS) via nvfile/cufile and POSIX fallback.
Usage::

    from adapter.storage import create_storage_backend

    backend = create_storage_backend()
    backend.save(path, gpu_tensor)       # GPU↔disk, either GDS or POSIX
    tensor  = backend.load(path, device) # disk↔GPU, either GDS or POSIX
"""

from adapter.storage.backend import StorageBackend, create_storage_backend

__all__ = [
    "StorageBackend",
    "create_storage_backend",
]
