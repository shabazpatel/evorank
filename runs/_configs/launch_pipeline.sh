#!/bin/bash
# EvoRank-Pipeline overnight campaign: 3 seeds x 50 iterations over the
# features + model + loss + ensemble search space. Resume-aware from
# checkpoints; waves of 2 concurrent runs. Aggregates at the end.
set -u
REPO=/Users/shabazpatel/Desktop/Project/skydiscover
TOTAL=50
cd "$REPO"
set -a; source .env; set +a
export EVORANK_EVAL_TIMEOUT_S=0 OMP_NUM_THREADS=3

latest_checkpoint() {
  ls -d "evorank/runs/$1/checkpoints/checkpoint_"* 2>/dev/null \
    | sed 's/.*checkpoint_//' | sort -n | tail -1
}

run_one() {
  local seed=$1
  local name="pipeline_s${seed}"
  local done_iters resume_args=""
  done_iters=$(latest_checkpoint "$name"); done_iters=${done_iters:-0}
  local remaining=$((TOTAL - done_iters))
  if [ "$remaining" -le 0 ]; then
    echo "[$(date +%H:%M:%S)] ${name} complete (${done_iters}/${TOTAL}), skipping"
    return 0
  fi
  [ "$done_iters" -gt 0 ] && resume_args="--checkpoint evorank/runs/${name}/checkpoints/checkpoint_${done_iters}"
  sed "s/^random_seed: 0$/random_seed: ${seed}/" \
      evorank/configs/evorank_pipeline.yaml > "evorank/runs/_configs/${name}.yaml"
  echo "[$(date +%H:%M:%S)] launching ${name} (done ${done_iters}, running ${remaining})"
  uv run python -m skydiscover.cli \
    evorank/seed/initial_pipeline.py evorank/eval/evaluator_pipeline.py \
    -c "evorank/runs/_configs/${name}.yaml" -s adaevolve -i "$remaining" \
    $resume_args \
    -o "evorank/runs/${name}" > "evorank/runs/_configs/${name}.stdout" 2>&1
  echo "[$(date +%H:%M:%S)] ${name} exited with $?"
}

run_one 0 &
P0=$!
run_one 1 &
P1=$!
wait "$P0" "$P1"
run_one 2

echo "=== pipeline campaign finished; aggregating ==="
cd "$REPO/evorank" && uv run python analyze/aggregate.py
echo "=== DONE $(date +%H:%M:%S) ==="
