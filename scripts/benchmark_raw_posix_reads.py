#!/usr/bin/env python3
"""Compare warm POSIX payload reads from two cache layouts."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import statistics
import time

import torch


def collect(root: Path, pattern: str, count: int) -> list[Path]:
    paths = sorted(root.rglob(pattern))[:count]
    if len(paths) != count:
        raise RuntimeError(f"expected {count} {pattern} files under {root}")
    return paths


def benchmark(
    paths: list[Path],
    *,
    payload_offset: int,
    workers: int,
    iterations: int,
) -> dict[str, float]:
    buffers = [
        torch.empty(
            path.stat().st_size - payload_offset,
            dtype=torch.uint8,
            pin_memory=True,
        )
        for path in paths
    ]
    fds = [os.open(path, os.O_RDONLY) for path in paths]

    def run_once() -> float:
        started = time.perf_counter()

        def read_one(index: int) -> None:
            buffer = buffers[index]
            read = os.preadv(fds[index], [buffer.numpy()], payload_offset)
            if read != buffer.numel():
                raise RuntimeError(f"short read {read} != {buffer.numel()}")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(read_one, range(len(paths))))
        return (time.perf_counter() - started) * 1000

    try:
        run_once()
        samples = [run_once() for _ in range(iterations)]
    finally:
        for fd in fds:
            os.close(fd)

    return {
        "bytes": float(sum(buffer.numel() for buffer in buffers)),
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cascade-root", type=Path, required=True)
    parser.add_argument("--lmcache-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--workers", default="1,4")
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    cascade = collect(args.cascade_root, "*.cobj", args.count)
    lmcache = collect(args.lmcache_root, "*.pt", args.count)
    result = {}
    for workers_text in args.workers.split(","):
        workers = int(workers_text)
        result[str(workers)] = {
            "cascade": benchmark(
                cascade,
                payload_offset=64 * 1024,
                workers=workers,
                iterations=args.iterations,
            ),
            "lmcache": benchmark(
                lmcache,
                payload_offset=0,
                workers=workers,
                iterations=args.iterations,
            ),
        }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
