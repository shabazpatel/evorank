#!/bin/bash
# Feedback-ablation probe: scores-only vs diagnostic feedback, priors OFF,
# seed 0, 40 iterations each, on the regenerated fast subset (6k-query val).
# Decision gate for the full 2x2 (see paper/approach.md section 3).
#
# NOTE: runs on the regenerated fast subset, so results are comparable between
# these two arms (and future 2x2 arms), NOT with the July 3 morning runs.
set -u
REPO=/Users/shabazpatel/Desktop/Project/skydiscover
cd "$REPO"
set -a; source .env; set +a
export EVORANK_EVAL_TIMEOUT_S=0 OMP_NUM_THREADS=3

run_one() {
  local feedback=$1 name=$2
  echo "[$(date +%H:%M:%S)] launching ${name} (EVORANK_FEEDBACK=${feedback})"
  EVORANK_FEEDBACK=$feedback uv run python -m skydiscover.cli \
    evorank/seed/initial_program.py evorank/eval/evaluator.py \
    -c evorank/configs/evorank_memory_off.yaml -s adaevolve -i 40 \
    -o "evorank/runs/${name}" > "evorank/runs/_configs/${name}.stdout" 2>&1
  echo "[$(date +%H:%M:%S)] ${name} exited with $?"
}

run_one minimal    probe_scores_s0 &
P1=$!
run_one diagnostic probe_diag_s0 &
P2=$!
wait "$P1" "$P2"

echo "=== probe finished; aggregating ==="
cd "$REPO/evorank" && uv run python analyze/aggregate.py
echo "=== DONE $(date +%H:%M:%S) ==="
