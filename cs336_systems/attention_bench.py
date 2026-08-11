import torch
import timeit

from cs336_basics.model import scaled_dot_product_attention
import torch._dynamo

d_models = [16,32,64,128]
seq_lens = [256,1024,4096,8192,16384]
BATCH = 8
N_ITERS = 100

torch._dynamo.config.cache_size_limit = 64
compiled_attn = torch.compile(scaled_dot_product_attention)


def bench(fn, q, k, v):
    # warm up BOTH graphs (forward and backward) so nothing compiles inside a timer
    for _ in range(10):
        fn(q, k, v).sum().backward()
    torch.cuda.synchronize()

    q.grad = k.grad = v.grad = None      # so the memory reading matches theory

    # --- forward only ---
    start = timeit.default_timer()
    for _ in range(N_ITERS):
        out = fn(q, k, v)
    torch.cuda.synchronize()
    fwd = (timeit.default_timer() - start) / N_ITERS * 1000

    mem = torch.cuda.memory_allocated() / 1024**2
    del out

    # --- forward + backward ---
    start = timeit.default_timer()
    for _ in range(N_ITERS):
        fn(q, k, v).sum().backward()
    torch.cuda.synchronize()
    total = (timeit.default_timer() - start) / N_ITERS * 1000

    return fwd, total - fwd, mem

if __name__ == "__main__":

    for d_model in d_models:
        for seq_len in seq_lens:
            q = torch.randn(BATCH, seq_len, d_model, device='cuda', requires_grad=True)
            k = torch.randn(BATCH, seq_len, d_model, device='cuda', requires_grad=True)
            v = torch.randn(BATCH, seq_len, d_model, device='cuda', requires_grad=True)

            for label, fn in [("eager", scaled_dot_product_attention), ("compiled", compiled_attn)]:
                fwd, bwd, mem = bench(fn, q, k, v)
                print(f"{label} | d={d_model} s={seq_len} | fwd {fwd:.4f} ms | bwd {bwd:.4f} ms | mem {mem:.2f} MB")

            del q, k, v
            torch.cuda.empty_cache()