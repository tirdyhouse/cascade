#!/usr/bin/env python3
"""Benchmark actual .cobj POSIX read-ahead plus compiled KV injection."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import statistics
import time

import torch

from adapter.storage.chunk_object import (
    _HEADER_SIZE,
    _parse_header,
    load_chunk_object,
    read_chunk_object_layout,
)
from adapter.storage.posix_backend import PosixBackend
from adapter.vllm.compiled_transfer import create_compiled_multi_layer_transfer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-root", type=Path, required=True)
    parser.add_argument("--object-count", type=int, default=16)
    parser.add_argument("--workers", default="4,8,16")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()

    paths = sorted(args.object_root.rglob("*.cobj"))[: args.object_count]
    if len(paths) != args.object_count:
        raise RuntimeError(
            f"expected {args.object_count} objects under {args.object_root}, "
            f"found {len(paths)}"
        )

    with paths[0].open("rb") as file:
        first_header = _parse_header(file.read(_HEADER_SIZE))
    layer_names = [layer.name for layer in first_header.layers]
    if not layer_names:
        raise RuntimeError("first object has no layers")

    backend = PosixBackend()
    transfer = create_compiled_multi_layer_transfer()
    device = torch.device("cuda:0")
    total_tokens = 0
    object_tokens = []
    for path in paths:
        with path.open("rb") as file:
            header = _parse_header(file.read(_HEADER_SIZE))
        tokens = header.end - header.start
        object_tokens.append(tokens)
        total_tokens += tokens
    cached_layouts = {
        path: read_chunk_object_layout(path, layer_names) for path in paths
    }
    persistent_buffers = {}
    persistent_fds = {}
    for path, (_, specs) in cached_layouts.items():
        read_start = min(spec.offset for spec in specs)
        read_end = max(spec.offset + spec.stored_nbytes for spec in specs)
        persistent_buffers[path] = (
            read_start,
            torch.empty(read_end - read_start, dtype=torch.uint8, pin_memory=True),
        )
        persistent_fds[path] = os.open(path, os.O_RDONLY)

    sample_header, sample_tensors = load_chunk_object(
        paths[0], layer_names, "cpu", backend, pin_memory=True
    )
    sample_device = backend.move_pinned_tensor_slices_to_device(
        [sample_tensors[name] for name in layer_names], str(device)
    )
    hidden_shape = sample_device[0].shape[2:]
    dtype = sample_device[0].dtype
    num_blocks = (total_tokens + args.block_size - 1) // args.block_size
    destinations = [
        torch.zeros(
            num_blocks,
            2,
            args.block_size,
            *hidden_shape,
            dtype=dtype,
            device=device,
        )
        for _ in layer_names
    ]
    del sample_header, sample_tensors, sample_device
    torch.cuda.synchronize()

    def run_once(
        worker_count: int,
        cache_layout: bool,
        persistent_pinned: bool = False,
        persistent_fd: bool = False,
    ) -> float:
        started = time.perf_counter()
        slot_start = 0
        path_iter = iter(zip(paths, object_tokens))

        def read_object(path: Path):
            if cache_layout:
                header, specs = cached_layouts[path]
                if persistent_pinned:
                    read_start, buffer = persistent_buffers[path]
                    if persistent_fd:
                        bytes_read = os.preadv(
                            persistent_fds[path],
                            [buffer.numpy()],
                            read_start,
                        )
                    else:
                        with path.open("rb") as file:
                            file.seek(read_start)
                            bytes_read = file.readinto(buffer.numpy())
                    if bytes_read != buffer.numel():
                        raise RuntimeError(
                            f"short persistent read: {bytes_read} != {buffer.numel()}"
                        )
                    tensors = [
                        buffer.narrow(
                            0,
                            spec.offset - read_start,
                            spec.nbytes,
                        )
                        .view(dtype=spec.dtype)
                        .reshape(spec.shape)
                        for spec in specs
                    ]
                else:
                    tensors = backend.load_tensor_slices_to_pinned_cpu(path, specs)
                return header, dict(zip(layer_names, tensors))
            return load_chunk_object(path, layer_names, "cpu", backend, pin_memory=True)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            pending = deque()
            for _ in range(worker_count):
                item = next(path_iter, None)
                if item is None:
                    break
                path, tokens = item
                pending.append((tokens, executor.submit(read_object, path)))

            while pending:
                tokens, future = pending.popleft()
                _, host_tensors = future.result()
                item = next(path_iter, None)
                if item is not None:
                    next_path, next_tokens = item
                    pending.append(
                        (next_tokens, executor.submit(read_object, next_path))
                    )

                sources = backend.move_pinned_tensor_slices_to_device(
                    [host_tensors[name] for name in layer_names], str(device)
                )
                slots = torch.arange(
                    slot_start,
                    slot_start + tokens,
                    dtype=torch.int64,
                    device=device,
                )
                transfer.inject(
                    sources,
                    destinations,
                    slots,
                    args.block_size,
                )
                slot_start += tokens

        torch.cuda.synchronize()
        return (time.perf_counter() - started) * 1000

    worker_counts = [int(value) for value in args.workers.split(",")]
    results = {}
    for workers in worker_counts:
        run_once(workers, False)
        samples = [run_once(workers, False) for _ in range(args.iterations)]
        run_once(workers, True)
        cached_samples = [run_once(workers, True) for _ in range(args.iterations)]
        run_once(workers, True, True)
        persistent_samples = [
            run_once(workers, True, True) for _ in range(args.iterations)
        ]
        run_once(workers, True, True, True)
        persistent_fd_samples = [
            run_once(workers, True, True, True) for _ in range(args.iterations)
        ]
        results[str(workers)] = {
            "full_layout_validation": {
                "mean_ms": statistics.mean(samples),
                "median_ms": statistics.median(samples),
                "min_ms": min(samples),
                "max_ms": max(samples),
            },
            "cached_layout": {
                "mean_ms": statistics.mean(cached_samples),
                "median_ms": statistics.median(cached_samples),
                "min_ms": min(cached_samples),
                "max_ms": max(cached_samples),
            },
            "cached_layout_persistent_pinned": {
                "mean_ms": statistics.mean(persistent_samples),
                "median_ms": statistics.median(persistent_samples),
                "min_ms": min(persistent_samples),
                "max_ms": max(persistent_samples),
            },
            "cached_layout_persistent_pinned_fd": {
                "mean_ms": statistics.mean(persistent_fd_samples),
                "median_ms": statistics.median(persistent_fd_samples),
                "min_ms": min(persistent_fd_samples),
                "max_ms": max(persistent_fd_samples),
            },
        }

    for fd in persistent_fds.values():
        os.close(fd)

    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(device),
                "objects": len(paths),
                "tokens": total_tokens,
                "layers": len(layer_names),
                "results": results,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
