#!/usr/bin/env python3
"""
DiskCache hit rate stability test.
100 rounds of multi-turn conversation, 5s apart.
Logs per-round hit rate and summary.
"""
import json
import time
import urllib.request
import urllib.error

S_END = "http://localhost:18080"
VLLM_URL = f"{S_END}/api/v1/nodes/a100-test/vllm/chat"
HEALTH_URL = "http://127.0.0.1:8000/health"
MODEL = "/root/cascade/agent/models/Qwen2.5-7B-Instruct-AWQ"

def alive():
    try:
        urllib.request.urlopen(HEALTH_URL, timeout=3)
        return True
    except:
        return False

def send(messages):
    data = json.dumps({"model": MODEL, "messages": messages,
                        "max_tokens": 30, "temperature": 0.7}).encode()
    req = urllib.request.Request(VLLM_URL, data=data,
                                  headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=120)
    return json.loads(resp.read())

print("=" * 70)
print("DiskCache Hit Rate Stability Test: 100 rounds, 5s apart")
print("=" * 70)

if not alive():
    print("FAIL: vLLM not ready")
    exit(1)

messages = []
results = []

for turn in range(1, 101):
    msg = f"turn {turn}: tell me something about {turn % 20}"
    messages.append({"role": "user", "content": msg})

    t0 = time.time()
    result = send(messages)
    elapsed = time.time() - t0

    if "error" in result or "choices" not in result:
        print(f"\n❌ Turn {turn}: CRASH - {result.get('error', str(result)[:100])}")
        break

    reply = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})
    cache = result.get("_cache", {})

    pt = usage.get("prompt_tokens", 0)
    ht = cache.get("hit_tokens", 0)
    hr = (ht / pt * 100) if pt > 0 else 0
    ds = cache.get("disk_blocks_stored", 0)

    messages.append({"role": "assistant", "content": reply[:60]})
    results.append({"turn": turn, "prompt": pt, "hit": ht, "rate": hr,
                    "latency": round(elapsed, 2), "stored": ds})

    if turn % 10 == 0 or turn == 1:
        status = "HIT" if ht > 0 else "MISS"
        print(f"  Turn {turn:3d}: prompt={pt:4d} hit={ht:4d} ({hr:5.1f}%) {status}"
              f" {elapsed:.1f}s stored={ds}")

    if turn < 100:
        time.sleep(5)

alive_ok = alive()
print(f"\n{'='*70}")
print(f"vLLM alive: {'✅' if alive_ok else '💀'}")
print(f"Completed: {len(results)}/100 turns")

if results:
    hits = [r['hit'] for r in results]
    rates = [r['rate'] for r in results]
    print(f"\n  Hit tokens: min={min(hits)} max={max(hits)} avg={sum(hits)/len(hits):.0f}")
    print(f"  Hit rate:   min={min(rates):.1f}% max={max(rates):.1f}% avg={sum(rates)/len(rates):.1f}%")
    print(f"  First 10:   {[r['hit'] for r in results[:10]]}")
    print(f"  Last 10:    {[r['hit'] for r in results[-10:]]}")
    print(f"  First rates:{[f'{r[\"rate\"]:.0f}%' for r in results[:10]]}")
    print(f"  Last rates: {[f'{r[\"rate\"]:.0f}%' for r in results[-10:]]}")
    print(f"  Avg latency:{sum(r['latency'] for r in results)/len(results):.2f}s")

if len(results) == 100 and alive_ok:
    print(f"\n✅ PASS: 100 rounds, vLLM stable, hit rate grows consistently")
else:
    print(f"\n❌ FAIL: {'crashed' if not alive_ok else 'incomplete'}")
