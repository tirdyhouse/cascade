#!/usr/bin/env python3
"""
DiskCache 对比测试脚本
自动启动/停止 vLLM，跑测试用例，监控 GPU 显存，记录结果。
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

MODEL_PATH = "/data/models/Qwen/Qwen3___6-27B-FP8"
API_URL = "http://localhost:8000/v1"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def gpu_memory_mb() -> tuple:
    """返回 (used_mb, total_mb)"""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        parts = r.stdout.strip().split(",")
        return int(parts[0]), int(parts[1])
    except:
        return 0, 0


def wait_for_vllm(timeout=120):
    """等待 vLLM API 就绪"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            req = urllib.request.Request(f"{API_URL}/models")
            urllib.request.urlopen(req, timeout=5)
            return True
        except:
            time.sleep(2)
    return False


def run_test_case(prompt: str, max_tokens: int, runs: int = 3) -> list:
    """运行单个测试用例多次，返回结果列表"""
    results = []
    model = MODEL_PATH

    for i in range(runs):
        mem_before = gpu_memory_mb()
        data = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": False,
        }).encode()

        t0 = time.perf_counter()
        req = urllib.request.Request(f"{API_URL}/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=300)
        result = json.loads(resp.read())
        elapsed = time.perf_counter() - t0
        mem_after = gpu_memory_mb()

        usage = result.get("usage", {})
        prompt_tok = usage.get("prompt_tokens", 0)
        gen_tok = usage.get("completion_tokens", 0)

        results.append({
            "run": i + 1,
            "prompt_tokens": prompt_tok,
            "gen_tokens": gen_tok,
            "time_s": round(elapsed, 3),
            "tps": round(gen_tok / elapsed, 1) if elapsed > 0 else 0,
            "gpu_mem_before_mb": mem_before[0],
            "gpu_mem_after_mb": mem_after[0],
            "gpu_mem_total_mb": mem_before[1],
        })
        time.sleep(1)  # cool down

    return results


def run_prefix_test(prefix_prompt: str, suffix: str, max_tokens: int, runs: int = 3) -> dict:
    """测试前缀缓存：先发一次（冷启动），再发一次（命中）"""
    full_prompt = prefix_prompt + suffix

    print("  首次（冷启动）...")
    cold = run_test_case(full_prompt, max_tokens, runs=1)[0]

    print("  第二次（预期命中）...")
    hot = run_test_case(full_prompt, max_tokens, runs=1)[0]

    return {"cold": cold, "hot": hot}


def start_vllm(mode: str, extra_args: list = None) -> subprocess.Popen:
    """启动 vLLM"""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH,
        "--tensor-parallel-size", "1",
        "--max-model-len", "16384",
        "--gpu-memory-utilization", "0.95",
        "--dtype", "auto",
        "--trust-remote-code",
        "--port", "8000",
    ]
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    # lmcache 需要 PYTHONPATH
    if mode == "diskcache":
        env["PYTHONPATH"] = "/data/disk-cache:" + env.get("PYTHONPATH", "")

    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["native", "lmcache", "diskcache"], required=True)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    # 停掉可能正在运行的 vLLM
    os.system("pkill -f vllm.entrypoints.openai 2>/dev/null; sleep 2")

    # 根据 mode 设置额外参数
    extra = []
    if args.mode == "lmcache":
        extra = ["--kv-transfer-config",
                 '{"kv_connector":"lmcache","kv_connector_extra_config":{"local_disk":"/data/kv-cache"}}']
    elif args.mode == "diskcache":
        extra = ["--kv-transfer-config",
                 '{"kv_connector":"disk-cache","kv_connector_extra_config":{"disk_cache_path":"/data/kv-cache","disk_cache_engine_addr":"http://localhost:9100"}}']

    # 对于 diskcache，先启动 Go 引擎
    go_proc = None
    if args.mode == "diskcache":
        go_proc = subprocess.Popen(
            ["/data/disk-cache/bin/disk-cache",
             "--cache-path", "/data/kv-cache",
             "--max-size", "50GB"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(1)

    # 启动 vLLM
    print(f"\n🚀 启动 vLLM (mode={args.mode})...")
    proc = start_vllm(args.mode, extra)

    if not wait_for_vllm():
        print("❌ vLLM 启动超时")
        proc.kill()
        if go_proc:
            go_proc.kill()
        sys.exit(1)

    print("✅ vLLM 就绪")

    # 测试用例
    test_cases = [
        ("short_prompt_32gen",  "What is deep learning?", 32),
        ("short_prompt_256gen", "Tell me a story about a dragon.", 256),
        ("long_prompt_128gen",  "Write about " + "machine learning " * 2000, 128),
    ]

    results = {
        "mode": args.mode,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cases": {}
    }

    for name, prompt, max_tok in test_cases:
        print(f"\n📊 [{name}]")
        res = run_test_case(prompt, max_tok, runs=args.runs)
        results["cases"][name] = res
        # 打印摘要
        times = [r["time_s"] for r in res]
        tps = [r["tps"] for r in res]
        mem = [r["gpu_mem_after_mb"] for r in res]
        print(f"  时间: {sum(times)/len(times):.2f}s avg | TPS: {sum(tps)/len(tps):.1f} avg | 显存: {sum(mem)/len(mem):.0f}MB")

    # 前缀命中测试
    print("\n📊 [prefix_cache]")
    prefix = "Write a detailed explanation of " + "neural networks " * 1000
    suffix = " Include mathematics and code examples."
    pr = run_prefix_test(prefix, suffix, 128, runs=1)
    results["cases"]["prefix_cache"] = pr
    print(f"  冷启动: {pr['cold']['tps']} TPS ({pr['cold']['time_s']:.2f}s)")
    print(f"  命中:   {pr['hot']['tps']} TPS ({pr['hot']['time_s']:.2f}s)")

    # 保存结果
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_file = os.path.join(RESULTS_DIR, f"{args.mode}.json")
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ 结果保存到 {result_file}")

    # 停止
    proc.kill()
    if go_proc:
        go_proc.kill()


if __name__ == "__main__":
    main()
