#!/usr/bin/env bash
# Generates 8 memory snapshots for the `large` model:
#   {ctx 128, 2048} x {forward, full step} x {fp32, bf16 mixed}
# Run from the repo root. Continues past OOM/failures instead of aborting.
set -uo pipefail

D_MODEL=1280; D_FF=5120; LAYERS=36; HEADS=20   # large, Table 1
BATCH_SIZE=4
WARMUP=3
STEPS=3          # keep snapshots small; the timeline only needs a few steps

OUTDIR=memory_snapshots
DUMP=./memory_snapshot.pickle   # hardcoded path inside base_bench.py
mkdir -p "$OUTDIR"

run_one() {
  local ctx=$1 mode=$2 prec=$3
  local tag="large_${ctx}_${mode}_${prec}"

  local mode_flag=""
  [ "$mode" = "forward" ] && mode_flag="--forward_only"   # no flag = full training step

  local prec_flag=""
  [ "$prec" = "bf16" ] && prec_flag="--mixed_precision"

  echo "=== $tag ==="
  rm -f "$DUMP"   # so a failed run can't be mislabeled with the previous run's file

  uv run python cs336_systems/base_bench.py \
    --device cuda --memory_profiling \
    --vocab_size 10000 --batch_size "$BATCH_SIZE" --context_length "$ctx" \
    --d_model "$D_MODEL" --num_layers "$LAYERS" --num_heads "$HEADS" --d_ff "$D_FF" \
    --warmup_steps "$WARMUP" --evaluation_steps "$STEPS" \
    $mode_flag $prec_flag 2>&1 | tee "$OUTDIR/${tag}.log"

  if [ -f "$DUMP" ]; then
    mv "$DUMP" "$OUTDIR/${tag}.pickle"
    echo "--> $OUTDIR/${tag}.pickle ($(du -h "$OUTDIR/${tag}.pickle" | cut -f1))"
  else
    echo "--> FAILED (no snapshot; check $OUTDIR/${tag}.log)"
  fi
}

for ctx in 128 2048; do
  for mode in forward full_step; do
    for prec in fp32 bf16; do
      run_one "$ctx" "$mode" "$prec"
    done
  done
done

echo
echo "=== results ==="
ls -lh "$OUTDIR"/*.pickle 2>/dev/null || echo "no snapshots produced"

tar czf memory_snapshots.tar.gz "$OUTDIR"/*.pickle 2>/dev/null \
  && echo "packaged: memory_snapshots.tar.gz ($(du -h memory_snapshots.tar.gz | cut -f1))"
