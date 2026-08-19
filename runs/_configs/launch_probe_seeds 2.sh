#!/bin/bash
# Feedback ablation, seeds 1 and 2 (both arms), extending the seed-0 probe to a
# 3-seeded comparison on the regenerated fast subset. Two waves of two runs.
set -u
REPO=/Users/shabazpatel/Desktop/Project/skydiscover
cd "$REPO"
set -a; source .env; set +a
export EVORANK_EVAL_TIMEOUT_S=0 OMP_NUM_THREADS=3

run_one() {
  local feedback=$1 seed=$2
  local name="probe_${feedback/minimal/scores}_s${seed}"
  name=${name/diagnostic/diag}
  echo "[$(date +%H:%M:%S)] launching ${name}"
  sed "s/^random_seed: 0$/random_seed: ${seed}/" \
      evorank/configs/evorank_memory_off.yaml > "evorank/runs/_configs/${name}.yaml"
  EVORANK_FEEDBACK=$feedback uv run python -m skydiscover.cli \
    evorank/seed/initial_program.py evorank/eval/evaluator.py \
    -c "evorank/runs/_configs/${name}.yaml" -s adaevolve -i 40 \
    -o "evorank/runs/${name}" > "evorank/runs/_configs/${name}.stdout" 2>&1
  echo "[$(date +%H:%M:%S)] ${name} exited with $?"
}

for seed in 1 2; do
  echo "=== wave seed ${seed} ==="
  run_one minimal "$seed" &
  P1=$!
  run_one diagnostic "$seed" &
  P2=$!
  wait "$P1" "$P2"
done
echo "=== probe seeds finished $(date +%H:%M:%S) ==="
cd "$REPO/evorank" && uv run python analyze/aggregate.py
echo "=== DONE ==="
