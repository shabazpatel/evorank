#!/bin/bash
# Phase 4 ablation: {memory_on, memory_off} x seeds {0,1,2}, 40 iterations each.
# Resume-aware: picks up each run from its latest checkpoint and runs only the
# remaining iterations, so the script is safe to re-launch after interruption.
# Runs in 3 waves of 2 concurrent. Aggregates at the end.
set -u
REPO=/Users/shabazpatel/Desktop/Project/skydiscover
TOTAL_ITERS=40
cd "$REPO"
set -a; source .env; set +a
export EVORANK_EVAL_TIMEOUT_S=0 OMP_NUM_THREADS=3

latest_checkpoint() {
  ls -d "evorank/runs/$1/checkpoints/checkpoint_"* 2>/dev/null \
    | sed 's/.*checkpoint_//' | sort -n | tail -1
}

run_one() {
  local cond=$1 seed=$2
  local name="mem_${cond}_s${seed}"
  local feedback="minimal"
  [ "$cond" = "on" ] && feedback="rich"
  local done_iters resume_args=""
  done_iters=$(latest_checkpoint "$name")
  done_iters=${done_iters:-0}
  local remaining=$((TOTAL_ITERS - done_iters))
  if [ "$remaining" -le 0 ]; then
    echo "[$(date +%H:%M:%S)] ${name} already at ${done_iters}/${TOTAL_ITERS}, skipping"
    return 0
  fi
  [ "$done_iters" -gt 0 ] && resume_args="--checkpoint evorank/runs/${name}/checkpoints/checkpoint_${done_iters}"
  echo "[$(date +%H:%M:%S)] launching ${name} (done ${done_iters}, running ${remaining} more)"
  EVORANK_FEEDBACK=$feedback uv run python -m skydiscover.cli \
    evorank/seed/initial_program.py evorank/eval/evaluator.py \
    -c "evorank/runs/_configs/${name}.yaml" -s adaevolve -i "$remaining" \
    $resume_args \
    -o "evorank/runs/${name}" >> "evorank/runs/_configs/${name}.stdout" 2>&1
  echo "[$(date +%H:%M:%S)] ${name} exited with $?"
}

for seed in 0 1 2; do
  echo "=== wave seed ${seed} ==="
  run_one on "$seed" &
  PID_ON=$!
  run_one off "$seed" &
  PID_OFF=$!
  wait "$PID_ON" "$PID_OFF"
done

echo "=== ablation matrix finished; aggregating ==="
cd "$REPO/evorank" && uv run python analyze/aggregate.py
echo "=== DONE $(date +%H:%M:%S) ==="
