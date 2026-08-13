import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse, csv, os, timeit


CONSTANTS = {
    "SIZES_MB": [1, 10, 100, 1024],
    "WORLD_SIZES": [2, 4, 6],
    "WARMUP": 5,
    "ITERS": 10,
}

RESULTS_CSV = "allreduce_results.csv"


def setup(rank, world_size, backend):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank)
        return f"cuda:{rank}"
    return "cpu"


def worker(rank, world_size, args):
    device = setup(rank, world_size, args.backend)
    rows = []

    for size in args.sizes:
        numel = size * 1024 * 1024 // 4  # float32 is 4 bytes per element
        tensor = torch.randn(numel, dtype=torch.float32, device=device)

        for _ in range(CONSTANTS["WARMUP"]):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        if device.startswith("cuda"):
            torch.cuda.synchronize()  # Ensure all processes have completed the warmup
        dist.barrier()  # Synchronize all processes before timing

        start_time = timeit.default_timer()
        for _ in range(CONSTANTS["ITERS"]):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        if device.startswith("cuda"):
            torch.cuda.synchronize()  # Ensure all processes have completed the operation
        elapsed_time = (timeit.default_timer() - start_time) / CONSTANTS["ITERS"] * 1000  # ms

        t = torch.tensor([elapsed_time], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)  # Get the maximum
        worst_time = t.item()

        if rank == 0:
            # Ring all-reduce moves 2*(N-1)/N times the payload per rank.
            moved_gb = 2 * (world_size - 1) / world_size * size / 1024
            bandwidth = moved_gb / (worst_time / 1000)
            rows.append({
                "backend": args.backend,
                "world_size": world_size,
                "size_mb": size,
                "latency_ms": round(worst_time, 4),
                "bandwidth_gb_s": round(bandwidth, 3),
            })
            print(
                f"  {args.backend:<5} world_size={world_size:<2} "
                f"size={size:>5} MB | latency {worst_time:9.3f} ms | "
                f"bandwidth {bandwidth:7.2f} GB/s",
                flush=True,
            )

        del tensor
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if rank == 0:
        write_header = not os.path.exists(RESULTS_CSV)
        with open(RESULTS_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    dist.destroy_process_group()  # Free the port for the next spawn


def print_summary(sizes, world_sizes):
    """Pivot the CSV into one row per payload size, one column per world size."""
    if not os.path.exists(RESULTS_CSV):
        return
    with open(RESULTS_CSV) as f:
        data = {(int(r["world_size"]), int(r["size_mb"])): r for r in csv.DictReader(f)}

    done = [w for w in world_sizes if any((w, s) in data for s in sizes)]
    if not done:
        return

    for metric, label, unit in [
        ("latency_ms", "Latency", "ms"),
        ("bandwidth_gb_s", "Effective bandwidth", "GB/s"),
    ]:
        print(f"\n{label} ({unit})")
        print(f"{'size':>8} " + "".join(f"{'ws=' + str(w):>12}" for w in done))
        print("-" * (8 + 12 * len(done)))
        for s in sizes:
            cells = "".join(
                f"{data[(w, s)][metric]:>12}" if (w, s) in data else f"{'-':>12}"
                for w in done
            )
            print(f"{str(s) + 'MB':>8} {cells}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Allreduce Benchmark")
    parser.add_argument("--backend", type=str, default="nccl", choices=["gloo", "nccl"], help="Distributed backend to use")
    parser.add_argument("--world-sizes", type=int, nargs="+", default=CONSTANTS["WORLD_SIZES"], help="List of world sizes to test")
    parser.add_argument("--sizes", type=int, nargs="+", default=CONSTANTS["SIZES_MB"], help="List of tensor sizes in MB to test")

    args = parser.parse_args()

    if os.path.exists(RESULTS_CSV):
        os.remove(RESULTS_CSV)

    available = torch.cuda.device_count() if args.backend == "nccl" else os.cpu_count()

    for world_size in args.world_sizes:
        if args.backend == "nccl" and world_size > available:
            print(f"skipping world_size={world_size}: only {available} GPU(s) visible")
            continue
        print(f"\n=== world_size={world_size} ({args.backend}) ===")
        mp.spawn(worker, args=(world_size, args), nprocs=world_size, join=True)

    print_summary(args.sizes, args.world_sizes)
    print(f"\nwrote {RESULTS_CSV}")
