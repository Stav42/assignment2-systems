#!/usr/bin/env bash
# Usage: ./run_profile.sh <size:small|large> <context_length> <mode:forward_only|forward_backward|optimizer>
# Runs both an nsys-wrapped profile and a plain (non-nsys) timeit baseline for the same config,
# then prints the forward-pass NVTX stats from the profile.
set -euo pipefail

SIZE=$1
CTX=$2
MODE=$3

case $SIZE in
  small) D_MODEL=768;  D_FF=3072;  LAYERS=12; HEADS=12 ;;
  large) D_MODEL=1280; D_FF=5120;  LAYERS=36; HEADS=20 ;;
  *) echo "unknown size $SIZE (expected small|large)"; exit 1 ;;
esac

case $MODE in
  forward_only)     FLAG="--forward_only" ;;
  forward_backward) FLAG="--forward_backward" ;;
  optimizer)        FLAG="" ;;
  *) echo "unknown mode $MODE (expected forward_only|forward_backward|optimizer)"; exit 1 ;;
esac

TAG="${SIZE}_${CTX}_${MODE}"
COMMON_ARGS="--vocab_size 10000 --batch_size 4 --context_length $CTX --d_model $D_MODEL --num_layers $LAYERS --num_heads $HEADS --d_ff $D_FF"

echo "=== nsys profile: $TAG ==="
uv run nsys profile -o "$TAG" --trace=cuda,cudnn,cublas,osrt,nvtx \
  --pytorch=functions-trace,autograd-shapes-nvtx -- \
  python cs336_systems/base_bench.py --device cuda $FLAG $COMMON_ARGS \
  2>&1 | tee "${TAG}_nsys.log"

echo "=== plain timeit baseline: $TAG ==="
uv run python cs336_systems/base_bench.py --device cuda $FLAG $COMMON_ARGS \
  2>&1 | tee "${TAG}_timeit.log"

echo "=== forward-pass NVTX stats (GPU-projected) ==="
nsys stats --report nvtx_gpu_proj_sum "${TAG}.nsys-rep" 2>/dev/null \
  || nsys stats --report nvtx_sum "${TAG}.nsys-rep"
