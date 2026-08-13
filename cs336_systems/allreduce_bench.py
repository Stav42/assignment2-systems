"""Benchmarks all-reduce latency on a single node.

Sweeps float32 payload sizes (1MB / 10MB / 100MB / 1GB) against world sizes
(2 / 4 / 6). Defaults to NCCL on GPU; pass --backend gloo to smoke-test the
harness on a CPU-only machine.

    uv run python cs336_systems/allreduce_bench.py
    uv run python cs336_systems/allreduce_bench.py --backend gloo --world-sizes 2
"""

import argparse
import csv
import os
import timeit

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SIZES_MB = [1, 10, 100, 1024]
WORLD_SIZES = [2, 4, 6]
WARMUP = 5
ITERS = 10
OUT_CSV = "allreduce_bench.csv"


def setup(rank, world_size, backend):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank)
        return f"cuda:{rank}"
    return "cpu"


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def worker(rank, world_size, backend, results_path):
    device = setup(rank, world_size, backend)
    rows = []

    for size_mb in SIZES_MB:
        numel = size_mb * 1024 * 1024 // 4          # float32
        data = torch.rand(numel, dtype=torch.float32, device=device)

        for _ in range(WARMUP):
            dist.all_reduce(data, op=dist.ReduceOp.SUM)
        sync(device)
        dist.barrier()                               # line every rank up first

        start = timeit.default_timer()
        for _ in range(ITERS):
            dist.all_reduce(data, op=dist.ReduceOp.SUM)
        sync(device)
        elapsed_ms = (timeit.default_timer() - start) / ITERS * 1000

        # A collective is only as fast as its slowest participant.
        t = torch.tensor([elapsed_ms], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        worst_ms = t.item()

        if rank == 0:
            # Ring all-reduce moves 2*(N-1)/N * D bytes per rank.
            moved_mb = 2 * (world_size - 1) / world_size * size_mb
            bw = moved_mb / 1024 / (worst_ms / 1000)          # GB/s
            rows.append({
                "backend": backend,
                "world_size": world_size,
                "size_mb": size_mb,
                "latency_ms": f"{worst_ms:.3f}",
                "eff_bandwidth_gb_s": f"{bw:.2f}",
            })
            print(f"{backend:>5} ws={world_size} {size_mb:>5}MB | "
                  f"{worst_ms:9.3f} ms | {bw:7.2f} GB/s", flush=True)

        del data
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if rank == 0:
        write_header = not os.path.exists(results_path)
        with open(results_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="nccl", choices=["nccl", "gloo"])
    ap.add_argument("--world-sizes", type=int, nargs="+", default=WORLD_SIZES)
    args = ap.parse_args()

    if os.path.exists(OUT_CSV):
        os.remove(OUT_CSV)

    available = torch.cuda.device_count() if args.backend == "nccl" else os.cpu_count()

    print(f"{'be':>5} {'world':>8} {'size':>7} | {'latency':>12} | {'bandwidth':>11}")
    print("-" * 52)

    for world_size in args.world_sizes:
        if args.backend == "nccl" and world_size > available:
            print(f"skipping world_size={world_size}: only {available} GPUs visible")
            continue
        mp.spawn(worker, args=(world_size, args.backend, OUT_CSV), nprocs=world_size, join=True)

    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
