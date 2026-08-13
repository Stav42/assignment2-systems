"""Benchmarks naive DDP training on a single node.

Reports total time per training step and how much of that is spent
all-reducing gradients. The assignment specifies the xl model on 2 GPUs;
defaults here are the medium config, which fits 24GB cards.

    uv run python cs336_systems/ddp_bench.py
    uv run python cs336_systems/ddp_bench.py --backend gloo --world-size 2   # local check
"""

import argparse
import os
import timeit

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.ddp import DDP

# Table 1 sizes. medium fits comfortably on a 24GB card; xl needs ~55GB/rank.
MODEL_CONFIGS = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


def setup(rank, world_size, backend):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29511")
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank)
        return f"cuda:{rank}"
    return "cpu"


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def worker(rank, world_size, args):
    device = setup(rank, world_size, args.backend)
    cfg = MODEL_CONFIGS[args.model_size]

    # Ranks deliberately start from different weights; DDP broadcasts rank 0's.
    torch.manual_seed(rank)
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        rope_theta=10000.0,
        **cfg,
    ).to(device)
    ddp_model = DDP(model)
    optimizer = AdamW(ddp_model.parameters(), lr=1e-4)

    n_params = sum(p.numel() for p in ddp_model.parameters())
    grad_mb = n_params * 4 / 1024**2

    # Each rank owns a disjoint shard of the global batch.
    assert args.batch_size % world_size == 0, "batch size must divide across ranks"
    local_bs = args.batch_size // world_size
    x = torch.randint(0, args.vocab_size, (local_bs, args.context_length), device=device)
    y = torch.randint(0, args.vocab_size, (local_bs, args.context_length), device=device)

    def train_step():
        """Returns (total_s, comm_s) for one step."""
        sync(device)
        t0 = timeit.default_timer()

        optimizer.zero_grad()
        loss = cross_entropy(ddp_model(x), y)
        loss.backward()

        sync(device)
        t1 = timeit.default_timer()

        ddp_model.finish_gradient_synchronization()

        sync(device)
        t2 = timeit.default_timer()

        optimizer.step()

        sync(device)
        t3 = timeit.default_timer()
        return t3 - t0, t2 - t1

    for _ in range(args.warmup_steps):
        train_step()

    totals, comms = [], []
    for _ in range(args.steps):
        total, comm = train_step()
        totals.append(total)
        comms.append(comm)

    mean_total = sum(totals) / len(totals)
    mean_comm = sum(comms) / len(comms)

    # A step is only done when the slowest rank is done.
    stats = torch.tensor([mean_total, mean_comm], device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    total_s, comm_s = stats.tolist()

    if rank == 0:
        print(f"\nmodel={args.model_size} ({n_params / 1e6:.0f}M params)  "
              f"world_size={world_size}  backend={args.backend}")
        print(f"global batch={args.batch_size} (local {local_bs})  "
              f"context={args.context_length}")
        print(f"gradients all-reduced per step: {grad_mb:.1f} MB")
        print()
        print(f"  total time per step      : {total_s * 1000:9.2f} ms")
        print(f"  gradient communication   : {comm_s * 1000:9.2f} ms")
        print(f"  proportion in comms      : {comm_s / total_s * 100:9.1f} %")
        if device.startswith("cuda"):
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  peak memory per rank     : {peak:9.2f} GiB")

    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="nccl", choices=["nccl", "gloo"])
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--model-size", default="medium", choices=list(MODEL_CONFIGS))
    ap.add_argument("--batch-size", type=int, default=8, help="global batch, split across ranks")
    ap.add_argument("--context-length", type=int, default=512)
    ap.add_argument("--vocab-size", type=int, default=10000)
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()

    if args.backend == "nccl" and torch.cuda.device_count() < args.world_size:
        raise SystemExit(f"need {args.world_size} GPUs, found {torch.cuda.device_count()}")

    mp.spawn(worker, args=(args.world_size, args), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
