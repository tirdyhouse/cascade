"""LMCache GDS compatibility shim for small-BAR1 GPUs.

LMCache 0.4.3 registers its entire GDS staging pool with cuFile.  A Tesla T4
has a 256 MiB BAR1 aperture and cannot register the roughly 224 MiB required
to retrieve this benchmark's full 4K-token request after driver reservations.
cuFile explicitly supports unregistered GPU destinations through its internal
bounce buffers, so this opt-in test shim swaps only the staging allocator.
"""

from __future__ import annotations

import os


if os.environ.get("LMCACHE_UNREGISTERED_CUFILE", "0") == "1":
    from lmcache.v1.memory_management import GPUMemoryAllocator
    from lmcache.v1.storage_backend.gds_backend import GdsBackend

    def _initialize_unregistered_allocator(self, config, metadata):
        del metadata
        if config.cufile_buffer_size is None:
            raise ValueError("cufile_buffer_size is required")
        return GPUMemoryAllocator(
            config.cufile_buffer_size * 1024**2,
            device=self.dst_device,
            align_bytes=4096,
        )

    GdsBackend.initialize_allocator = _initialize_unregistered_allocator
