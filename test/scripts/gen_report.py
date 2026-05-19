#!/usr/bin/env python3
"""
测试报告生成器 — 读取 test/results/*.json 生成对比报告。
"""

import json
import os
from pathlib import Path

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_results():
    results = {}
    for f in sorted(Path(RESULTS_DIR).glob("*.json")):
        with open(f) as fp:
            data = json.load(fp)
            results[data["mode"]] = data
    return results


def avg(lst, key):
    vals = [x[key] for x in lst]
    return sum(vals) / len(vals)


def main():
    results = load_results()

    if not results:
        print("❌ 没有找到测试结果，请先运行 run_bench.py")
        return

    print("=" * 80)
    print("  DiskCache 对比测试报告")
    print("=" * 80)
    print(f"\n  测试模式: {', '.join(results.keys())}")
    print(f"  运行时间: {min(r['timestamp'] for r in results.values())}")
    print()

    # ── 硬件信息 ──
    print("  ── 环境 ──")
    print(f"  GPU: {results[list(results.keys())[0]]['cases']['short_prompt_32gen'][0]['gpu_mem_total_mb']} MB")
    print()

    # ── 性能对比表 ──
    cases = ["short_prompt_32gen", "short_prompt_256gen", "long_prompt_128gen"]

    print("  ── 性能对比 ──")
    header = f"  {'场景':<25} {'指标':<10}"
    for m in results:
        header += f" {m:<15}"
    print(header)
    print("  " + "-" * len(header))

    for case in cases:
        # TPS
        line = f"  {case:<25} {'TPS':<10}"
        for m in results:
            if case in results[m]["cases"]:
                val = avg(results[m]["cases"][case], "tps")
                line += f" {val:<15.1f}"
            else:
                line += f" {'N/A':<15}"
        print(line)

        # 延迟
        line = f"  {'':<25} {'time(s)':<10}"
        for m in results:
            if case in results[m]["cases"]:
                val = avg(results[m]["cases"][case], "time_s")
                line += f" {val:<15.3f}"
            else:
                line += f" {'N/A':<15}"
        print(line)

        # 显存
        line = f"  {'':<25} {'GPU(MB)':<10}"
        for m in results:
            if case in results[m]["cases"]:
                val = avg(results[m]["cases"][case], "gpu_mem_after_mb")
                line += f" {val:<15.0f}"
            else:
                line += f" {'N/A':<15}"
        print(line)
        print()

    # ── 前缀命中 ──
    print("  ── 前缀缓存命中对比 ──")
    header = f"  {'模式':<20} {'冷启动 TPS':<15} {'命中 TPS':<15} {'加速比':<10}"
    print(header)
    print("  " + "-" * len(header))
    for m in results:
        if "prefix_cache" in results[m]["cases"]:
            p = results[m]["cases"]["prefix_cache"]
            cold_tps = p["cold"]["tps"]
            hot_tps = p["hot"]["tps"]
            ratio = hot_tps / cold_tps if cold_tps > 0 else 0
            print(f"  {m:<20} {cold_tps:<15.1f} {hot_tps:<15.1f} {ratio:<10.2f}x")
    print()

    # ── 总结 ──
    print("  ── 总结 ──")
    # 取第一个测试项的平均值
    for case in cases:
        vals = {}
        for m in results:
            if case in results[m]["cases"]:
                vals[m] = avg(results[m]["cases"][case], "tps")
        if len(vals) >= 2:
            best = max(vals, key=vals.get)
            others = [k for k in vals if k != best]
            for o in others:
                diff = (vals[best] - vals[o]) / vals[o] * 100
                print(f"  {case}: {best} 比 {o} 快 {diff:+.0f}%")
        print()

    print("=" * 80)


if __name__ == "__main__":
    main()
