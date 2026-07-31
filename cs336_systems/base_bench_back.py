import argparse
from cs336_basics import BasicsTransformerLM
import torch
from cs336_basics.data import get_batch
import timeit
from cs336_basics.nn_utils import cross_entropy

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
args = parser.parse_args()

'''
To finish the task, I must:
1. Write a script that takes in hyperparameters
2. FInd a way to load the model with those hyperparameters
3. Load a dataset
4. Get a random batch of data from the dataset
5. Run w warm up steps before time benchmarking
6. 
'''

def benchmark(params: dict):

    vocab_size = params["vocab_size"]
    context_length = params["context_length"]
    d_model = params["d_model"]
    num_layers = params["num_layers"]
    num_heads = params["num_heads"]
    d_ff = params["d_ff"]
    rope_theta = params["rope_theta"]
    batch_size = params["batch_size"]

    lm = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta
    )

    dataset = torch.load("cs336_systems/dataset.pt")
    batch = get_batch(dataset, batch_size=batch_size, context_length=context_length, device=args.device)

    # Warm up steps
    for _ in range(args.warmup_steps):
        lm(*batch)

    torch.cuda.synchronize() if "cuda" in args.device else None
    start_time = timeit.default_timer()

    # Evaluation steps
    for _ in range(args.evaluation_steps):
        lm(*batch)
        torch.cuda.synchronize() if "cuda" in args.device else None

    end_time = timeit.default_timer()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    return lm


if __name__ == "__main__":
    import json

    hyperparams = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "rope_theta": args.rope_theta,
        "batch_size": args.batch_size
    }
    lm = benchmark(hyperparams)
    print(lm)

