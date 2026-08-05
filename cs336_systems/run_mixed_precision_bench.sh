#!/usr/bin/env bash
# Times forward+backward with vs without BF16 mixed precision, across Table 1 model sizes.
# Continues past OOM/failures on individual sizes instead of aborting the whole run.
set -uo pipefail

CONTEXT_LENGTH=512

declare -A SIZES=(
  [small]="768 3072 12 12"
  [medium]="1024 4096 24 16"
  [large]="1280 5120 36 20"
  [xl]="2560 10240 32 32"
)

extract_avg() {
  # Pull the numeric seconds out of "Average time per step: X.XX seconds"
  grep "Average time per step" | grep -oE '[0-9]+\.[0-9]+'
}

run_one() {
  local d_model=$1 d_ff=$2 layers=$3 heads=$4 mp_flag=$5
  uv run python cs336_systems/base_bench.py --device cuda --forward_backward \
    --vocab_size 10000 --batch_size 4 --context_length "$CONTEXT_LENGTH" \
    --d_model "$d_model" --num_layers "$layers" --num_heads "$heads" --d_ff "$d_ff" \
    $mp_flag
}

printf "%-8s %-14s %-14s %-10s\n" "size" "full_fp32(s)" "bf16_mixed(s)" "speedup"

for size in small medium large xl; do
  read -r d_model d_ff layers heads <<< "${SIZES[$size]}"

  fp32_out=$(run_one "$d_model" "$d_ff" "$layers" "$heads" "" 2>&1)
  fp32_time=$(echo "$fp32_out" | extract_avg)

  bf16_out=$(run_one "$d_model" "$d_ff" "$layers" "$heads" "--mixed_precision" 2>&1)
  bf16_time=$(echo "$bf16_out" | extract_avg)

  if [ -z "$fp32_time" ]; then
    fp32_display="OOM/FAILED"
  else
    fp32_display="$fp32_time"
  fi

  if [ -z "$bf16_time" ]; then
    bf16_display="OOM/FAILED"
  else
    bf16_display="$bf16_time"
  fi

  if [ -n "$fp32_time" ] && [ -n "$bf16_time" ]; then
    speedup=$(awk -v a="$fp32_time" -v b="$bf16_time" 'BEGIN { printf "%.2fx", a/b }')
  else
    speedup="n/a"
  fi

  printf "%-8s %-14s %-14s %-10s\n" "$size" "$fp32_display" "$bf16_display" "$speedup"

  # Save full logs for later inspection
  echo "$fp32_out" > "mp_${size}_fp32.log"
  echo "$bf16_out" > "mp_${size}_bf16.log"
done
