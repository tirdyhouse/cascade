#!/usr/bin/env python3
"""Build a comparable cache benchmark summary from one remote test run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


MODES = (
    ("cascade-host", "Cascade pinned host tier (4 GiB)", "cascade"),
    ("cascade-posix", "Cascade POSIX (4-way read-ahead)", "cascade"),
    ("cascade-gds", "Cascade cuFile/GDS compatibility", "cascade"),
    ("lmcache-local-cpu", "LMCache LocalCPU", "lmcache"),
    ("lmcache-local-disk", "LMCache LocalDisk (POSIX file)", "lmcache"),
    ("lmcache-gds", "LMCache GDS compatibility", "lmcache"),
)

LMCACHE_HIT_RE = re.compile(
    r"Total tokens:?\s*(?P<total>\d+),\s*Inference Engine computed tokens: "
    r"(?P<computed>\d+),\s*LMCache hit tokens:?\s*(?P<hit>\d+),\s*"
    r"need to load:\s*(?P<load>\d+)"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def display_ms(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 1000:.3f} ms"


def display_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.2f}%"


def delta(before: dict[str, Any], after: dict[str, Any], key: str) -> int:
    return int(after.get(key, 0)) - int(before.get(key, 0))


def query_prompt_tokens(result: dict[str, Any]) -> int:
    return sum(
        int(record["prompt_tokens"])
        for record in result.get("records", [])
        if record.get("success") and record.get("prompt_tokens") is not None
    )


def query_required_forward_tokens(result: dict[str, Any]) -> int:
    """Count vLLM's mandatory final-token forwards for first-token logits."""
    return sum(
        1
        for record in result.get("records", [])
        if record.get("success")
        and record.get("prompt_tokens") is not None
        and int(record["prompt_tokens"]) > 0
    )


def physical_kv_target_tokens(result: dict[str, Any]) -> int:
    """Maximum external-KV restore target for this query phase.

    KV state alone cannot produce the first generated-token logits. vLLM
    executes one final prompt token for every successful generation, even for
    a full cache lookup. That common forward work is not a cache miss.
    """
    return max(0, query_prompt_tokens(result) - query_required_forward_tokens(result))


def parse_lmcache_hits(
    mode_dir: Path,
    expected_count: int,
    *,
    log_name: str = "vllm.log",
) -> dict[str, Any]:
    log = mode_dir / log_name
    if not log.exists():
        return {"observed": 0, "hits": []}
    hits = [
        {key: int(value) for key, value in match.groupdict().items()}
        for match in LMCACHE_HIT_RE.finditer(log.read_text(errors="replace"))
    ]
    # A log can include warmup traffic and CUDA progress updates with carriage
    # returns. The final benchmark query is the last expected request records.
    return {"observed": len(hits), "hits": hits[-expected_count:]}


def cascade_hit_metrics(mode_dir: Path, query: dict[str, Any]) -> dict[str, Any]:
    before = read_json(mode_dir / "stats-before-query.json")
    after = read_json(mode_dir / "stats-after-query.json")
    requests = delta(before, after, "MatchRequests")
    hits = delta(before, after, "MatchHits")
    matched_tokens = delta(before, after, "MatchedTokens")
    prompt_tokens = query_prompt_tokens(query)
    required_forward_tokens = query_required_forward_tokens(query)
    kv_target_tokens = physical_kv_target_tokens(query)
    # Since Cascade now looks up the full prompt just like LMCache, engine
    # ``MatchedTokens`` is the logical coverage. A full prompt hit still has
    # one token per request forwarded by vLLM to produce first-token logits.
    external_kv_tokens = min(matched_tokens, kv_target_tokens)
    return {
        "request_hits": hits,
        "request_total": requests,
        "request_hit_rate": hits / requests if requests else None,
        "logical_lookup_tokens": matched_tokens,
        "logical_lookup_hit_rate": (
            matched_tokens / prompt_tokens if prompt_tokens else None
        ),
        "external_kv_tokens": external_kv_tokens,
        "prompt_tokens": prompt_tokens,
        "required_forward_tokens": required_forward_tokens,
        "kv_target_tokens": kv_target_tokens,
        "kv_restore_hit_rate": (
            external_kv_tokens / kv_target_tokens if kv_target_tokens else None
        ),
    }


def lmcache_hit_metrics(
    mode_dir: Path,
    query: dict[str, Any],
    *,
    restarted: bool = False,
) -> dict[str, Any]:
    expected = int(query["summary"]["success"])
    parsed = parse_lmcache_hits(
        mode_dir,
        expected,
        log_name="restart-vllm.log" if restarted else "vllm.log",
    )
    hits = parsed["hits"]
    total_tokens = sum(item["total"] for item in hits)
    logical_hit_tokens = sum(item["hit"] for item in hits)
    external_kv_tokens = sum(item["load"] for item in hits)
    hit_requests = sum(item["hit"] > 0 for item in hits)
    prompt_tokens = query_prompt_tokens(query)
    required_forward_tokens = query_required_forward_tokens(query)
    kv_target_tokens = physical_kv_target_tokens(query)
    return {
        "request_hits": hit_requests,
        "request_total": expected,
        "request_hit_rate": hit_requests / expected if expected else None,
        "logical_lookup_tokens": logical_hit_tokens,
        "logical_lookup_hit_rate": (
            logical_hit_tokens / prompt_tokens if prompt_tokens else None
        ),
        "external_kv_tokens": external_kv_tokens,
        "prompt_tokens": prompt_tokens or total_tokens,
        "required_forward_tokens": required_forward_tokens,
        "kv_target_tokens": kv_target_tokens,
        "kv_restore_hit_rate": (
            external_kv_tokens / kv_target_tokens if kv_target_tokens else None
        ),
        "lmcache_log_records": parsed["observed"],
    }


def build_record(mode: str, label: str, kind: str, run_dir: Path) -> dict[str, Any]:
    mode_dir = run_dir / mode
    warmup = read_json(mode_dir / "warmup.json")
    restarted = (mode_dir / "restart-query.json").exists()
    query = read_json(mode_dir / ("restart-query.json" if restarted else "query.json"))
    if mode == "lmcache-gds":
        label = (
            "LMCache GDS (restart, persistent)"
            if restarted
            else "LMCache GDS (pure GDS, in process)"
        )
    elif mode == "lmcache-local-disk" and restarted:
        label = "LMCache LocalDisk (POSIX, restart)"
    metrics = (
        cascade_hit_metrics(mode_dir, query)
        if kind == "cascade"
        else lmcache_hit_metrics(mode_dir, query, restarted=restarted)
    )
    evidence_path = mode_dir / "backend-evidence.txt"
    evidence = (
        evidence_path.read_text(errors="replace").strip()
        if evidence_path.exists()
        else ""
    )
    return {
        "mode": mode,
        "label": label,
        "kind": kind,
        "query_source": "restart" if restarted else "in_process",
        "query_concurrency": int(query.get("concurrency", 1)),
        "warmup": warmup["summary"],
        "query": query["summary"],
        "hit_metrics": metrics,
        "backend_evidence": evidence,
    }


def markdown(records: list[dict[str, Any]], run_dir: Path) -> str:
    query_concurrencies = sorted({record["query_concurrency"] for record in records})
    concurrency_label = ", ".join(str(value) for value in query_concurrencies)
    lines = [
        "# Cache GDS Retest",
        "",
        f"Raw artifacts: `{run_dir}`",
        "",
        f"Query concurrency: {concurrency_label}",
        "",
        "Logical lookup coverage uses every prompt token for both systems. "
        "Physical KV restore coverage uses prompt tokens minus one vLLM "
        "first-token-forward token per successful request; that forward "
        "produces logits and is not a cache miss.",
        "",
        "| Backend | Cache scope | Query mean TTFT | p50 | p95 | Throughput | Request hit rate | Logical lookup coverage | Physical KV restore coverage | Query success |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        query = record["query"]
        hits = record["hit_metrics"]
        lines.append(
            "| {label} | {query_source} | {mean} | {p50} | {p95} | {throughput} | {request_rate} "
            "({request_hits}/{request_total}) | {logical_rate} "
            "({logical_tokens}/{prompt_tokens}) | {kv_rate} "
            "({external_kv}/{kv_target}) | {success}/{total} |".format(
                label=record["label"],
                query_source=record["query_source"],
                mean=display_ms(query.get("mean_ttft_seconds")),
                p50=display_ms(query.get("p50_ttft_seconds")),
                p95=display_ms(query.get("p95_ttft_seconds")),
                throughput=(
                    f"{query['request_throughput_rps']:.2f} req/s"
                    if query.get("request_throughput_rps") is not None
                    else "N/A"
                ),
                request_rate=display_percent(hits.get("request_hit_rate")),
                request_hits=hits.get("request_hits", 0),
                request_total=hits.get("request_total", 0),
                logical_rate=display_percent(hits.get("logical_lookup_hit_rate")),
                logical_tokens=hits.get("logical_lookup_tokens", 0),
                prompt_tokens=hits.get("prompt_tokens", 0),
                kv_rate=display_percent(hits.get("kv_restore_hit_rate")),
                external_kv=hits.get("external_kv_tokens", 0),
                kv_target=hits.get("kv_target_tokens", 0),
                success=query.get("success", 0),
                total=query.get("total", 0),
            )
        )
    lines.extend(["", "## Backend Evidence", ""])
    for record in records:
        lines.append(f"### {record['label']}")
        lines.append("")
        lines.append("```")
        lines.append(record["backend_evidence"] or "No backend evidence captured")
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    records = [
        build_record(mode, label, kind, args.run_dir) for mode, label, kind in MODES
    ]
    summary = {"run_dir": str(args.run_dir), "records": records}
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.run_dir / "summary.md").write_text(markdown(records, args.run_dir) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
