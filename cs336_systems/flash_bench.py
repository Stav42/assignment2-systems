"""Benchmarks Triton FlashAttention-2 against plain PyTorch attention.

Batch size 1 and causal masking throughout, per the assignment. Sweeps
sequence length x head dim x precision, reporting forward, backward and
end-to-end latency for both implementations. Configurations that run out of
memory are recorded as OOM and the sweep continues.
"""

import csv
import math

import torch
import triton.testing

import cs336_systems.FlashAttentionTriton as flash_mod
from cs336_systems.FlashAttentionTriton import FlashAttentionTriton

SEQ_LENS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
D_MODELS = [16, 32, 64, 128]
DTYPES = [("bf16", torch.bfloat16), ("fp32", torch.float32)]
BATCH = 1
OUT_CSV = "flash_bench.csv"


def pytorch_attention(q, k, v, is_causal=True):
    d = q.shape[-1]
    s = torch.einsum("bqd,bkd->bqk", q, k) / math.sqrt(d)
    if is_causal:
        q_idx = torch.arange(q.shape[1], device=q.device)
        k_idx = torch.arange(k.shape[1], device=q.device)
        s = s.masked_fill(q_idx[:, None] < k_idx[None, :], -1e6)
    p = torch.softmax(s, dim=-1)
    return torch.einsum("bqk,bkd->bqd", p, v)


def tile_size_for(seq_len, d_model):
    # Larger tiles mean fewer loop iterations, but Q/K/V/O and the score tile
    # all have to fit in SRAM, so back off as the head dim grows.
    cap = 32 if d_model >= 128 else 64
    return max(16, min(cap, seq_len))


def bench_one(impl, q, k, v, dO):
    """Returns (fwd_ms, bwd_ms, e2e_ms). Backward comes from the difference so
    we never need retain_graph, which conflicts with compiled backward passes."""
    fwd = triton.testing.do_bench(lambda: impl(q, k, v, True))
    e2e = triton.testing.do_bench(lambda: impl(q, k, v, True).backward(dO))
    return fwd, e2e - fwd, e2e


def run():
    rows = []
    for dtype_name, dtype in DTYPES:
        for d_model in D_MODELS:
            for seq_len in SEQ_LENS:
                tile = tile_size_for(seq_len, d_model)
                flash_mod.DEFAULT_Q_TILE = tile
                flash_mod.DEFAULT_K_TILE = tile

                row = {
                    "dtype": dtype_name,
                    "d_model": d_model,
                    "seq_len": seq_len,
                    "tile": tile,
                }

                q = k = v = dO = None
                try:
                    q, k, v = (
                        torch.randn(BATCH, seq_len, d_model, device="cuda",
                                    dtype=dtype, requires_grad=True)
                        for _ in range(3)
                    )
                    dO = torch.randn(BATCH, seq_len, d_model, device="cuda", dtype=dtype)

                    for label, impl in [("triton", FlashAttentionTriton.apply),
                                        ("pytorch", pytorch_attention)]:
                        try:
                            fwd, bwd, e2e = bench_one(impl, q, k, v, dO)
                            row[f"{label}_fwd"] = f"{fwd:.3f}"
                            row[f"{label}_bwd"] = f"{bwd:.3f}"
                            row[f"{label}_e2e"] = f"{e2e:.3f}"
                        except torch.OutOfMemoryError:
                            row[f"{label}_fwd"] = row[f"{label}_bwd"] = row[f"{label}_e2e"] = "OOM"
                            torch.cuda.empty_cache()
                        finally:
                            for t in (q, k, v):
                                t.grad = None
                except torch.OutOfMemoryError:
                    for label in ("triton", "pytorch"):
                        row[f"{label}_fwd"] = row[f"{label}_bwd"] = row[f"{label}_e2e"] = "OOM"
                finally:
                    del q, k, v, dO
                    torch.cuda.empty_cache()

                rows.append(row)
                print(
                    f"{dtype_name:>5} d={d_model:<4} s={seq_len:<6} tile={tile:<4} | "
                    f"triton {row['triton_fwd']:>9} {row['triton_bwd']:>9} {row['triton_e2e']:>9} | "
                    f"pytorch {row['pytorch_fwd']:>9} {row['pytorch_bwd']:>9} {row['pytorch_e2e']:>9}",
                    flush=True,
                )

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    header = f"{'dtype':>5} {'d':<6} {'seq':<8} {'tile':<9} | " \
             f"{'triton: fwd':>9} {'bwd':>9} {'e2e':>9} | {'pytorch: fwd':>9} {'bwd':>9} {'e2e':>9}"
    print(header)
    print("-" * len(header))
    run()
