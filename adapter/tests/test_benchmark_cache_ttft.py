import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "benchmark_cache_ttft.py"
SPEC = importlib.util.spec_from_file_location("benchmark_cache_ttft", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
OPENAI_STUB = ModuleType("openai")
OPENAI_STUB.AsyncOpenAI = object
PREVIOUS_OPENAI = sys.modules.get("openai")
sys.modules["openai"] = OPENAI_STUB
try:
    SPEC.loader.exec_module(BENCHMARK)
finally:
    if PREVIOUS_OPENAI is None:
        del sys.modules["openai"]
    else:
        sys.modules["openai"] = PREVIOUS_OPENAI


class _FakeModels:
    async def list(self):
        return []


class _FakeClient:
    def __init__(self, **kwargs):
        del kwargs
        self.models = _FakeModels()

    async def close(self):
        return None


@pytest.mark.parametrize("concurrency", [1, 4])
def test_run_limits_in_flight_requests(monkeypatch, concurrency):
    active = 0
    max_active = 0

    async def fake_send_request(client, model, prompt, document_index, max_tokens):
        nonlocal active, max_active
        del client, model, prompt, max_tokens
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return {
            "document_index": document_index,
            "success": True,
            "ttft_seconds": 0.005,
            "total_seconds": 0.01,
            "prompt_tokens": 4088,
            "completion_tokens": 1,
            "output_text": "ok",
            "output_sha256": "hash",
            "output_matches_expected": None,
            "error": None,
        }

    monkeypatch.setattr(BENCHMARK, "AsyncOpenAI", _FakeClient)
    monkeypatch.setattr(BENCHMARK, "send_request", fake_send_request)
    args = SimpleNamespace(
        host="127.0.0.1",
        port=8000,
        request_timeout=30.0,
        num_documents=10,
        document_tokens=4096,
        concurrency=concurrency,
        model="model",
        max_tokens=1,
        phase="query",
        expected_output=None,
        strict_output=False,
    )

    result = asyncio.run(BENCHMARK.run(args))

    assert max_active == concurrency
    assert result["concurrency"] == concurrency
    assert [record["document_index"] for record in result["records"]] == list(range(10))
    assert result["summary"]["success"] == 10
    assert result["summary"]["request_throughput_rps"] > 0
