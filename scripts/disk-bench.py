#!/usr/bin/env python3
"""
磁盘 I/O 基准测试 — 在跑 vLLM 之前先摸清磁盘能力。

用法:
    # 测试 /mnt/nvme 目录下的磁盘性能
    python3 scripts/disk-bench.py /mnt/nvme
    
    # 指定测试文件大小和块大小
    python3 scripts/disk-bench.py /mnt/nvme --file-size 4G --block-size 1M

输出:
    Sequential Write:   xxx MB/s
    Sequential Read:    xxx MB/s
    Random Write IOPS:  xxx
    Random Read IOPS:   xxx
    File Create:        xxx files/s
"""

import argparse
import os
import sys
import tempfile
import time
import threading


def parse_size(s: str) -> int:
    s = s.strip().upper()
    if s.endswith("G"):
        return int(float(s[:-1]) * 1024**3)
    if s.endswith("M"):
        return int(float(s[:-1]) * 1024**2)
    if s.endswith("K"):
        return int(float(s[:-1]) * 1024)
    return int(s)


def fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def fmt_iops(n: float) -> str:
    return f"{n:,.0f}"


def fmt_bw(n: float) -> str:
    return f"{n:.2f}"


def sequential_write(path: str, file_size: int, block_size: int) -> float:
    """顺序写测试，返回 MB/s"""
    data = os.urandom(block_size)
    file_path = os.path.join(path, ".disk-bench-write.tmp")
    written = 0
    start = time.perf_counter()
    try:
        with open(file_path, "wb", buffering=0) as f:
            while written < file_size:
                n = min(block_size, file_size - written)
                f.write(data[:n])
                written += n
    finally:
        elapsed = time.perf_counter() - start
        os.remove(file_path)
    return (file_size / elapsed) / 1024**2


def sequential_read(path: str, file_size: int, block_size: int) -> float:
    """顺序读测试，返回 MB/s"""
    file_path = os.path.join(path, ".disk-bench-read.tmp")
    # 先写一个测试文件
    data = os.urandom(block_size)
    with open(file_path, "wb", buffering=0) as f:
        written = 0
        while written < file_size:
            n = min(block_size, file_size - written)
            f.write(data[:n])
            written += n

    # 顺序读
    total = 0
    start = time.perf_counter()
    try:
        with open(file_path, "rb", buffering=0) as f:
            while True:
                buf = f.read(block_size)
                if not buf:
                    break
                total += len(buf)
    finally:
        elapsed = time.perf_counter() - start
        os.remove(file_path)
    return (total / elapsed) / 1024**2


def random_read_iops(path: str, file_size: int, block_size: int, num_ops: int) -> float:
    """随机读 IOPS"""
    file_path = os.path.join(path, ".disk-bench-randread.tmp")
    # 写测试文件
    data = os.urandom(block_size)
    with open(file_path, "wb", buffering=0) as f:
        written = 0
        while written < file_size:
            n = min(block_size, file_size - written)
            f.write(data[:n])
            written += n

    # 随机读
    import random
    f = open(file_path, "rb", buffering=0)
    max_offset = file_size - block_size
    start = time.perf_counter()
    for _ in range(num_ops):
        offset = random.randint(0, max_offset)
        f.seek(offset)
        f.read(block_size)
    elapsed = time.perf_counter() - start
    f.close()
    os.remove(file_path)
    return num_ops / elapsed


def random_write_iops(path: str, block_size: int, num_ops: int) -> float:
    """随机写 IOPS（写入不同文件）"""
    data = os.urandom(block_size)
    start = time.perf_counter()
    for i in range(num_ops):
        file_path = os.path.join(path, f".disk-bench-randwrite-{i}.tmp")
        with open(file_path, "wb", buffering=0) as f:
            f.write(data)
        os.remove(file_path)
    elapsed = time.perf_counter() - start
    return num_ops / elapsed


def file_create_delete(path: str, num_files: int, file_size: int) -> tuple[float, float]:
    """文件创建/删除速度，返回 (create files/s, delete files/s)"""
    data = os.urandom(file_size)

    # 创建
    start = time.perf_counter()
    for i in range(num_files):
        file_path = os.path.join(path, f".disk-bench-file-{i}.tmp")
        with open(file_path, "wb") as f:
            f.write(data)
    create_elapsed = time.perf_counter() - start

    # 删除
    start = time.perf_counter()
    for i in range(num_files):
        file_path = os.path.join(path, f".disk-bench-file-{i}.tmp")
        os.remove(file_path)
    delete_elapsed = time.perf_counter() - start

    return num_files / create_elapsed, num_files / delete_elapsed


def main():
    parser = argparse.ArgumentParser(description="磁盘 I/O 基准测试")
    parser.add_argument("path", nargs="?", default=".", help="测试目录（默认为当前目录）")
    parser.add_argument("--file-size", default="2G", help="测试文件大小（默认 2G）")
    parser.add_argument("--block-size", default="1M", help="I/O 块大小（默认 1M）")
    parser.add_argument("--rand-ops", default=10000, type=int, help="随机 I/O 次数（默认 10000）")
    parser.add_argument("--num-files", default=5000, type=int, help="文件创建测试数量（默认 5000）")
    args = parser.parse_args()

    path = args.path
    file_size = parse_size(args.file_size)
    block_size = parse_size(args.block_size)
    rand_ops = args.rand_ops
    num_files = args.num_files

    # 检查目录
    if not os.path.isdir(path):
        print(f"❌ 目录不存在: {path}")
        sys.exit(1)

    # 检查剩余空间（需要至少 2 倍 file_size）
    stat = os.statvfs(path)
    free = stat.f_frsize * stat.f_bavail
    if free < file_size * 2:
        print(f"⚠️  磁盘剩余空间 {fmt_size(free)}，可能需要清理")
        cont = input("继续？(y/n): ")
        if cont.lower() != "y":
            return

    print(f"\n{'='*60}")
    print(f"  磁盘 I/O 基准测试")
    print(f"{'='*60}")
    print(f"  测试目录:    {os.path.abspath(path)}")
    print(f"  文件系统:    {stat.f_fstypestr if hasattr(stat, 'f_fstypestr') else 'N/A'}")
    print(f"  块大小:      {fmt_size(block_size)}")
    print(f"  测试文件大小: {fmt_size(file_size)}")
    print(f"{'='*60}\n")

    # 1. 顺序写
    print("[1/5] 顺序写...", end=" ", flush=True)
    bw = sequential_write(path, file_size, block_size)
    print(f"✅  {fmt_bw(bw)} MB/s")

    # 2. 顺序读
    print("[2/5] 顺序读...", end=" ", flush=True)
    bw = sequential_read(path, file_size, block_size)
    print(f"✅  {fmt_bw(bw)} MB/s")

    # 3. 随机读 IOPS
    print(f"[3/5] 随机读 ({rand_ops} ops)...", end=" ", flush=True)
    iops = random_read_iops(path, file_size, block_size, rand_ops)
    print(f"✅  {fmt_iops(iops)} IOPS ({fmt_bw(iops * block_size / 1024**2)} MB/s)")

    # 4. 随机写 IOPS
    print(f"[4/5] 随机写 ({rand_ops} ops)...", end=" ", flush=True)
    iops = random_write_iops(path, block_size, min(rand_ops, 5000))
    print(f"✅  {fmt_iops(iops)} IOPS ({fmt_bw(iops * block_size / 1024**2)} MB/s)")

    # 5. 文件创建/删除
    print(f"[5/5] 文件创建/删除 ({num_files} files)...", end=" ", flush=True)
    cps, dps = file_create_delete(path, min(num_files, 2000), 4096)
    print(f"✅  创建 {fmt_iops(cps)} files/s, 删除 {fmt_iops(dps)} files/s")

    print(f"\n{'='*60}")
    print(f"  测试完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
