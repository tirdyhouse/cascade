#!/usr/bin/env python3
"""
DiskCache 引擎专项 I/O 测试 — 测试 Go 引擎的 put/get 延迟。

模拟实际 KV cache block 的大小和行为。

用法:
    # 先启动 Go 引擎
    ./bin/disk-cache --cache-path /tmp/bench-cache --max-size 10GB
    
    # 再跑测试
    python3 scripts/disk-bench-cache.py http://localhost:9100
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error


def fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def go_put(addr: str, hash_val: int, file_path: str, size: int) -> float:
    """调用 Go 引擎的 /put，返回耗时(ms)"""
    data = json.dumps({"hash": hash_val, "file_path": file_path, "size": size}).encode()
    start = time.perf_counter()
    req = urllib.request.Request(f"{addr}/put", data=data,
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=10)
    return (time.perf_counter() - start) * 1000


def go_get(addr: str, hash_val: int) -> float:
    """调用 Go 引擎的 /get，返回耗时(ms)"""
    start = time.perf_counter()
    urllib.request.urlopen(f"{addr}/get?hash={hash_val:016x}", timeout=10)
    return (time.perf_counter() - start) * 1000


def file_write(path: str, size: int) -> tuple[float, str]:
    """写文件到磁盘，返回 (耗时ms, 文件路径)"""
    data = os.urandom(size)
    start = time.perf_counter()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    elapsed = (time.perf_counter() - start) * 1000
    return elapsed, path


def file_read(path: str) -> float:
    """读文件，返回耗时(ms)"""
    start = time.perf_counter()
    with open(path, "rb") as f:
        f.read()
    return (time.perf_counter() - start) * 1000


def main():
    parser = argparse.ArgumentParser(description="DiskCache 引擎 I/O 测试")
    parser.add_argument("engine_addr", help="Go 引擎地址 (http://localhost:9100)")
    parser.add_argument("--cache-root", default="/tmp/bench-cache", help="缓存根目录")
    parser.add_argument("--blocks", default=100, type=int, help="测试 block 数量")
    parser.add_argument("--min-size", default="1M", help="最小 block 大小")
    parser.add_argument("--max-size", default="64M", help="最大 block 大小")
    args = parser.parse_args()

    def _sz(s):
        if s.endswith("M"):
            return int(float(s[:-1]) * 1024**2)
        if s.endswith("K"):
            return int(float(s[:-1]) * 1024)
        return int(s)

    min_sz = _sz(args.min_size)
    max_sz = _sz(args.max_size)
    n = args.blocks

    # 检查引擎连接
    try:
        urllib.request.urlopen(f"{args.engine_addr}/stats", timeout=5)
    except Exception as e:
        print(f"❌ 无法连接 Go 引擎: {e}")
        print(f"请先启动: ./bin/disk-cache --cache-path {args.cache_root} --max-size 10GB")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  DiskCache 引擎 I/O 测试")
    print(f"{'='*60}")
    print(f"  引擎地址:    {args.engine_addr}")
    print(f"  缓存根目录:  {args.cache_root}")
    print(f"  Block 数量:  {n}")
    print(f"  Block 大小:  {fmt_size(min_sz)} ~ {fmt_size(max_sz)}")
    print(f"{'='*60}\n")

    # 清理之前的测试数据
    os.system(f"rm -rf {args.cache_root}/*")

    # 测试 1: 连续写（模拟推理中的 save_kv_layer）
    print("[1/4] 连续写...")
    write_latencies = []
    import random
    for i in range(n):
        size = random.randint(min_sz, max_sz)
        hash_val = i + 1
        h = format(hash_val, "016x")
        rel_path = f"{h[:2]}/{h[2:4]}/{h}.bin"
        
        # 写文件
        ms, full_path = file_write(f"{args.cache_root}/{rel_path}", size)
        file_path = f"{h[:2]}/{h[2:4]}/{h}.bin"
        
        # Go Put
        ms2 = go_put(args.engine_addr, hash_val, file_path, size)
        write_latencies.append(ms)
        
        if (i + 1) % 20 == 0:
            print(f"   ... {i+1}/{n}")

    avg_write = sum(write_latencies) / len(write_latencies)
    total_data = n * (min_sz + max_sz) / 2
    print(f"  ✅ 平均写延迟: {avg_write:.2f} ms")
    print(f"  ✅ 等效带宽:   {fmt_size(total_data / (sum(write_latencies) / 1000))}/s")

    # 测试 2: 连续读
    print("\n[2/4] 连续读（缓存命中）...")
    read_latencies = []
    for i in range(n):
        hash_val = i + 1
        
        # Go Get
        ms = go_get(args.engine_addr, hash_val)
        read_latencies.append(ms)
        
    avg_read = sum(read_latencies) / len(read_latencies)
    print(f"  ✅ 平均读延迟: {avg_read:.2f} ms")

    # 测试 3: 缓存未命中
    print("\n[3/4] 缓存未命中...")
    miss_latencies = []
    for i in range(n):
        hash_val = 999999 + i
        try:
            ms = go_get(args.engine_addr, hash_val)
            miss_latencies.append(ms)
        except urllib.error.HTTPError:
            miss_latencies.append(0.5)  # 404 很快
    print(f"  ✅ 未命中延迟: {sum(miss_latencies)/len(miss_latencies):.2f} ms")

    # 测试 4: LRU 淘汰
    print("\n[4/4] LRU 淘汰...")
    stats = json.loads(urllib.request.urlopen(f"{args.engine_addr}/stats").read())
    print(f"  ✅ 当前缓存: {stats['BlocksStored']} blocks, {fmt_size(stats['DiskUsedBytes'])}")

    print(f"\n{'='*60}")
    print(f"  测试完成")
    print(f"{'='*60}")
    print(f"  总结:")
    print(f"    写延迟 (p50):    {sorted(write_latencies)[len(write_latencies)//2]:.2f} ms")
    print(f"    读延迟 (p50):    {sorted(read_latencies)[len(read_latencies)//2]:.2f} ms")
    print(f"    总写入数据:      {fmt_size(total_data)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
