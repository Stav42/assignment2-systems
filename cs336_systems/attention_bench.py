import torch
import timeit

from cs336_basics.model import scaled_dot_product_attention

d_models = [16,32,64,128]
seq_lens = [256,1024,4096,8192,16384]
BATCH = 8
N_ITERS = 100


if __name__ == "__main__":

    for d_model in d_models:
        for seq_len in seq_lens:
            q = torch.randn(BATCH, seq_len, d_model, device='cuda', requires_grad=True)
            k = torch.randn(BATCH, seq_len, d_model, device='cuda', requires_grad=True)
            v = torch.randn(BATCH, seq_len, d_model, device='cuda', requires_grad=True)

            # for warmup
            for _ in range(10):
                scaled_dot_product_attention(q, k, v)

            torch.cuda.synchronize()

            # note time right now
            start = timeit.default_timer()

            out = None

            for _ in range(N_ITERS):
                out = scaled_dot_product_attention(q, k, v)

            torch.cuda.synchronize()
            elapsed = timeit.default_timer() - start
            mem = torch.cuda.memory_allocated()
            print(f"seq_len: {seq_len}, d_model: {d_model}, time per iteration: {elapsed/N_ITERS*1000:.4f} ms, out shape: {out.shape}, memory allocated: {mem/1024**2:.2f} MB")

            loss = out.sum()
            torch.cuda.synchronize()
            start = timeit.default_timer()

            for _ in range(N_ITERS):
                loss.backward(retain_graph=True)

            torch.cuda.synchronize()
            elapsed = timeit.default_timer() - start
            print(f"seq_len: {seq_len}, d_model: {d_model}, time per backward iteration: {elapsed/N_ITERS*1000:.4f} ms, memory allocated: {mem/1024**2:.2f} MB")

            del q, k, v, out, loss
            torch.cuda.empty_cache()