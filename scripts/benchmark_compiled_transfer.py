#!/usr/bin/env python3
"""Validate and benchmark Cascade's optional compiled KV scatter."""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Callable

import torch

from adapter.vllm.compiled_transfer import create_compiled_multi_layer_transfer


def measure_ms(fn: Callable[[], None], warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def summarize(samples: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--tokens", type=int, default=4087)
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--hidden-elements", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    dtype = torch.float16
    num_blocks = (args.tokens + args.block_size - 1) // args.block_size
    slot_mapping = torch.arange(args.tokens, dtype=torch.int64, device=device)

    full_sources = [
        torch.randn(
            args.chunk_tokens,
            2,
            args.hidden_elements,
            dtype=dtype,
            device=device,
        )
        for _ in range(args.layers)
    ]
    destinations = [
        torch.zeros(
            num_blocks,
            2,
            args.block_size,
            args.hidden_elements,
            dtype=dtype,
            device=device,
        )
        for _ in range(args.layers)
    ]
    reference = [torch.zeros_like(tensor) for tensor in destinations]

    chunks: list[tuple[int, int, list[torch.Tensor]]] = []
    for start in range(0, args.tokens, args.chunk_tokens):
        end = min(start + args.chunk_tokens, args.tokens)
        length = end - start
        sources = [source[:length] for source in full_sources]
        chunks.append((start, end, sources))

    transfer = create_compiled_multi_layer_transfer()

    # Exactness check over the full multi-chunk restore.
    for start, end, sources in chunks:
        transfer.inject(
            sources,
            destinations,
            slot_mapping[start:end],
            args.block_size,
        )
        blocks = slot_mapping[start:end] // args.block_size
        offsets = slot_mapping[start:end] % args.block_size
        for layer, source in enumerate(sources):
            reference[layer][blocks, :, offsets] = source
    torch.cuda.synchronize()
    if not all(
        torch.equal(actual, expected)
        for actual, expected in zip(destinations, reference)
    ):
        raise RuntimeError("compiled transfer does not match PyTorch reference")

    def compiled_restore() -> None:
        for start, end, sources in chunks:
            transfer.inject(
                sources,
                destinations,
                slot_mapping[start:end],
                args.block_size,
            )

    def pytorch_microbatch_restore() -> None:
        for group_start in range(0, len(chunks), 4):
            group = chunks[group_start : group_start + 4]
            group_slots = torch.cat(
                [slot_mapping[start:end] for start, end, _ in group]
            )
            blocks = group_slots // args.block_size
            offsets = group_slots % args.block_size
            for layer in range(args.layers):
                source = torch.cat([item[2][layer] for item in group], dim=0)
                destinations[layer][blocks, :, offsets] = source

    compiled_samples = measure_ms(compiled_restore, args.warmup, args.iterations)
    pytorch_samples = measure_ms(
        pytorch_microbatch_restore, args.warmup, args.iterations
    )
    result = {
        "correct": True,
        "gpu": torch.cuda.get_device_name(device),
        "layers": args.layers,
        "tokens": args.tokens,
        "chunks": len(chunks),
        "block_size": args.block_size,
        "hidden_elements": args.hidden_elements,
        "compiled": summarize(compiled_samples),
        "pytorch_4_chunk_microbatch": summarize(pytorch_samples),
        "speedup": statistics.mean(pytorch_samples) / statistics.mean(compiled_samples),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
