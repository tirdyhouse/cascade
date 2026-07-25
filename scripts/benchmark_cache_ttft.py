#!/usr/bin/env python3
"""Collect per-request TTFT samples for an already running vLLM server.

The script intentionally keeps warmup and query phases separate.  A caller can
therefore collect cache-engine statistics between them and calculate a query
hit rate without mixing in the initial population requests.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record["success"]]
    ttfts = [record["ttft_seconds"] for record in successful]
    totals = [record["total_seconds"] for record in successful]
    prompt_tokens = [
        record["prompt_tokens"]
        for record in successful
        if record["prompt_tokens"] is not None
    ]
    output_matches = [
        record["output_matches_expected"]
        for record in records
        if record.get("output_matches_expected") is not None
    ]
    return {
        "success": len(successful),
        "total": len(records),
        "mean_ttft_seconds": sum(ttfts) / len(ttfts) if ttfts else None,
        "p50_ttft_seconds": percentile(ttfts, 0.50),
        "p95_ttft_seconds": percentile(ttfts, 0.95),
        "min_ttft_seconds": min(ttfts) if ttfts else None,
        "max_ttft_seconds": max(ttfts) if ttfts else None,
        "mean_total_seconds": sum(totals) / len(totals) if totals else None,
        "observed_prompt_tokens": prompt_tokens,
        "mean_prompt_tokens": (
            sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else None
        ),
        "matching_outputs": sum(output_matches) if output_matches else None,
        "checked_outputs": len(output_matches),
    }


def make_documents(count: int, target_tokens: int) -> list[str]:
    """Make deterministic, distinct long prompts.

    The character multiplier deliberately over-allocates a little for the
    Qwen tokenizer.  The server-reported prompt token counts are stored in the
    result so the benchmark does not rely on this approximation.
    """
    # This multiplier was calibrated against the Qwen2.5 chat endpoint on the
    # benchmark host. The API-reported count remains the source of truth.
    target_chars = round(target_tokens * 8.4)
    filler = "cache retrieval benchmark content "
    repeat_count = max(1, math.ceil(target_chars / len(filler)))
    body = filler * repeat_count
    return [
        f"Document {index}. Unique cache namespace {index}. {body}"
        for index in range(count)
    ]


def usage_value(usage: Any, field: str) -> int | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        value = usage.get(field)
    else:
        value = getattr(usage, field, None)
    return int(value) if value is not None else None


async def send_request(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    document_index: int,
    max_tokens: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    first_token_time: float | None = None
    usage: Any = None
    output_parts: list[str] = []
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
            for choice in chunk.choices or []:
                delta = choice.delta
                content = getattr(delta, "content", None) if delta else None
                reasoning = getattr(delta, "reasoning_content", None) if delta else None
                if content:
                    output_parts.append(content)
                if reasoning:
                    output_parts.append(reasoning)
                if first_token_time is None and (
                    content is not None or reasoning is not None
                ):
                    first_token_time = time.perf_counter()
        finish = time.perf_counter()
        output_text = "".join(output_parts)
        return {
            "document_index": document_index,
            "success": first_token_time is not None,
            "ttft_seconds": (first_token_time - start) if first_token_time else None,
            "total_seconds": finish - start,
            "prompt_tokens": usage_value(usage, "prompt_tokens"),
            "completion_tokens": usage_value(usage, "completion_tokens"),
            "output_text": output_text,
            "output_sha256": hashlib.sha256(output_text.encode()).hexdigest(),
            "output_matches_expected": None,
            "error": (
                None if first_token_time is not None else "stream ended without a token"
            ),
        }
    except Exception as exc:
        finish = time.perf_counter()
        return {
            "document_index": document_index,
            "success": False,
            "ttft_seconds": None,
            "total_seconds": finish - start,
            "prompt_tokens": None,
            "completion_tokens": None,
            "output_text": "",
            "output_sha256": None,
            "output_matches_expected": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    client = AsyncOpenAI(
        base_url=f"http://{args.host}:{args.port}/v1",
        api_key="benchmark-key",
        timeout=args.request_timeout,
    )
    try:
        await client.models.list()
        documents = make_documents(args.num_documents, args.document_tokens)
        records: list[dict[str, Any]] = []
        for document_index, prompt in enumerate(documents):
            record = await send_request(
                client,
                args.model,
                prompt,
                document_index,
                args.max_tokens,
            )
            records.append(record)
            status = "ok" if record["success"] else "failed"
            ttft = record["ttft_seconds"]
            print(
                (
                    f"{args.phase} document={document_index} status={status} "
                    f"ttft_ms={ttft * 1000:.3f}"
                    if ttft is not None
                    else f"{args.phase} document={document_index} status={status} error={record['error']}"
                ),
                flush=True,
            )

        if args.expected_output is not None:
            expected_data = json.loads(args.expected_output.read_text())
            expected_by_index = {
                int(record["document_index"]): record.get("output_sha256")
                for record in expected_data.get("records", [])
            }
            for record in records:
                expected = expected_by_index.get(record["document_index"])
                matches = expected is not None and record["output_sha256"] == expected
                record["output_matches_expected"] = matches
                if args.strict_output and record["success"] and not matches:
                    record["success"] = False
                    record["error"] = "deterministic output differs from warmup"
    finally:
        await client.close()

    return {
        "phase": args.phase,
        "model": args.model,
        "host": args.host,
        "port": args.port,
        "num_documents": args.num_documents,
        "target_document_tokens": args.document_tokens,
        "max_tokens": args.max_tokens,
        "records": records,
        "summary": summarize(records),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("warmup", "query"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-documents", type=int, default=10)
    parser.add_argument("--document-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=10)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-output", type=Path)
    parser.add_argument(
        "--strict-output",
        action="store_true",
        help=(
            "treat an output hash mismatch as a failed latency sample; by "
            "default transport-success TTFT remains valid and hash agreement "
            "is reported separately"
        ),
    )
    args = parser.parse_args()

    result = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2), flush=True)
    return 0 if result["summary"]["success"] == args.num_documents else 1


if __name__ == "__main__":
    raise SystemExit(main())
