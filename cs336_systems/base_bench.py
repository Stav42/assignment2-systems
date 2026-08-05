import argparse
import math
from contextlib import nullcontext
import cs336_basics.model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import softmax
from einops import einsum
import torch
from cs336_basics.data import get_batch
import timeit
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
import torch.cuda.nvtx as nvtx


def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))
    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)
    with nvtx.range("attention final matmul"):
        output = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    return output


cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

# import modal

# image = modal.Image.from_registry("nvidia/cuda:13.3.1-cudnn-devel-ubuntu24.04", add_python="3.12").uv_sync(extra_options="--no-install-package cs336-basics")
# # image = modal.Image.debian_slim(python_version="3.12").uv_sync(extra_options="--no-install-package cs336-basics")

# image = image.add_local_python_source("cs336_basics", "cs336_systems")
# app = modal.App("cs336_systems", image=image)


parser = argparse.ArgumentParser(description="Base benchmarking script for LM.")

# parser.add_argument("hyperparams", help="Hyperparameter set for LM and optimizer")
parser.add_argument("--device", default="cpu", help="Device to run the benchmark on (default: cuda)")
parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size for the model")
parser.add_argument("--context_length", type=int, default=128, help="Context length for the model")
parser.add_argument("--d_model", type=int, default=512, help="Dimension of the model")
parser.add_argument("--num_layers", type=int, default=6, help="Number of layers in the model")
parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads in the model")
parser.add_argument("--d_ff", type=int, default=2048, help="Dimension of the feedforward network")
parser.add_argument("--rope_theta", type=float, default=10000.0, help="Theta value for RoPE")
parser.add_argument("--evaluation_steps", type=int, default=10, help="Number of evaluation steps for benchmarking")
parser.add_argument("--warmup_steps", type=int, default=5, help="Number of warm-up steps before benchmarking")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--dataset_path", type=str, default=None, help="Path to the dataset file")    
parser.add_argument("--forward_only", action="store_true", default=False, help="If set, only perform forward pass without backward pass and optimizer step")
parser.add_argument("--forward_backward", action="store_true", default=False, help="If set, perform both forward and backward pass with optimizer step")
parser.add_argument("--optimizer", action="store_true", default=False, help="Optimizer to use (default: adamw)")
parser.add_argument("--local", action="store_true", default=False, help="If set, run the benchmark locally without using Modal")
parser.add_argument("--mixed_precision", action="store_true", default=False, help="If set, run forward/loss under BF16 autocast mixed precision")
parser.add_argument("--memory_profiling", action="store_true", default=False, help="If set, set memory profiling and dump memory snapshot to file")
args = parser.parse_args()

'''
To finish the task, I must:
1. Write a script that takes in hyperparameters
2. Find a way to load the model with those hyperparameters
3. Load a dataset
4. Get a random batch of data from the dataset
5. Run warm up steps before time benchmarking


Now the next thing for the modal integration:
1. Choose the right image for the modal app
2. Choose the right GPU

How do I choose the right image? It must consist of torch and everything else that is there in the uv thing. 
I must understand how this integrates with the uv system that we have setup right now. What exactly is uv in any case
How will the image in the modal app work with the uv system? The code will obviously run on their computer
uv system is there on my local PC. What's up with all that?
'''

# @app.function(gpu="A100", image=image)
def benchmark(params: dict):

    vocab_size = params["vocab_size"]
    context_length = params["context_length"]
    d_model = params["d_model"]
    num_layers = params["num_layers"]
    num_heads = params["num_heads"]
    d_ff = params["d_ff"]
    rope_theta = params["rope_theta"]
    batch_size = params["batch_size"]
    dataset_path = params["dataset_path"]
    forward_only = params["forward_only"]
    forward_backward = params["forward_backward"]
    optimizer = params["optimizer"]
    device = params["device"]
    evaluation_steps = params["evaluation_steps"]
    warmup_steps = params["warmup_steps"]
    mixed_precision = params["mixed_precision"]

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if mixed_precision and "cuda" in device
        else nullcontext()
    )
    print(f"Mixed precision (BF16 autocast): {mixed_precision and 'cuda' in device}")

    if forward_only:
        benchmark_type = "forward_only"
        print("Benchmarking forward pass only...")
    elif forward_backward:
        benchmark_type = "forward_backward"
        print("Benchmarking forward and backward pass...")
    else:
        benchmark_type = "optimizer"
        print("Benchmarking optimizer step...")
        
    lm = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta
    )

    lm.to(device)

    optimizer = AdamW(lm.parameters(), lr=1e-3, weight_decay=0.01)

    if dataset_path is not None:
        dataset = torch.load(dataset_path)
        data = get_batch(dataset, batch_size=batch_size, context_length=context_length, device=device)
        x = data[0]  # Use only the input tensor for benchmarking
        y = data[1]
    else:
        # If no dataset path is provided, create a random dataset for benchmarking
        dataset = torch.randint(0, vocab_size, (55000,), dtype=torch.int64)
        data = get_batch(dataset.numpy(), batch_size=batch_size, context_length=context_length, device=device)
        x = data[0]  # Use only the input tensor for benchmarking
        y = data[1]
     
    # Warm up steps
    for _ in range(warmup_steps):
        lm(x)

    if args.memory_profiling:
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    torch.cuda.synchronize() if "cuda" in device else None
    start_time = timeit.default_timer()
    print(f"Starting benchmark for {benchmark_type} with {evaluation_steps} steps...")

    # Evaluation steps
    for _ in range(evaluation_steps):
        if benchmark_type == "forward_only":
            with torch.no_grad(), nvtx.range("forward"), autocast_ctx:
                logits = lm(x)
        if benchmark_type == "forward_backward":
            with nvtx.range("forward"), autocast_ctx:
                logits = lm(x)
                loss = cross_entropy(logits, y)  # Compute loss for benchmarking
            optimizer.zero_grad()
            with nvtx.range("backward"):
                loss.backward()
            # optimizer.step()
        if benchmark_type == "optimizer":
            with nvtx.range("forward"), autocast_ctx:
                logits = lm(x)
                loss = cross_entropy(logits, y)  # Compute loss for benchmarking
            optimizer.zero_grad()
            with nvtx.range("backward"):
                loss.backward()
            with nvtx.range("optimizer_step"):
                optimizer.step()

        torch.cuda.synchronize() if "cuda" in device else None
        output_time = timeit.default_timer() - start_time
        print(f"Step {_ + 1}/{evaluation_steps}, Time taken: {output_time:.2f} seconds")

    end_time = timeit.default_timer()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Average time per step: {(end_time - start_time) / evaluation_steps:.2f} seconds")
    print(f"Total iterations: {evaluation_steps}")

    if args.memory_profiling:
        torch.cuda.memory._dump_snapshot("./memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

    # return lm
    return


if __name__ == "__main__":

    hyperparams = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "rope_theta": args.rope_theta,
        "batch_size": args.batch_size,
        "dataset_path": args.dataset_path,
        "forward_only": args.forward_only,
        "forward_backward": args.forward_backward,
        "optimizer": args.optimizer,
        "device": args.device,
        "evaluation_steps": args.evaluation_steps,
        "warmup_steps": args.warmup_steps,
        "mixed_precision": args.mixed_precision
    }
    benchmark(hyperparams)
#     else:
#         with modal.enable_output():
#             with app.run():

#                 vocab_size = 10000
#                 context_length = 128
#                 d_model = 512
#                 num_layers = 6
#                 num_heads = 8
#                 d_ff = 2048
#                 rope_theta = 10000.0
#                 batch_size = 32
#                 dataset_path = None  # Set to the path of your dataset if available
#                 forward_only = True
#                 forward_backward = False
#                 optimizer = False
#                 device = "cuda"
#                 evaluation_steps = 10
#                 warmup_steps = 5


#                 hyperparams = {
#                     "vocab_size": vocab_size,
#                     "context_length": context_length,
#                     "d_model": d_model,
#                     "num_layers": num_layers,
#                     "num_heads": num_heads,
#                     "d_ff": d_ff,
#                     "rope_theta": rope_theta,
#                     "batch_size": batch_size,
#                     "dataset_path": dataset_path,
#                     "forward_only": forward_only,
#                     "forward_backward": forward_backward,
#                     "optimizer": optimizer, 
#                     "device": device,
#                     "evaluation_steps": evaluation_steps,
#                     "warmup_steps": warmup_steps
#                 }

#                 benchmark.remote(hyperparams)
#     # print(lm)

