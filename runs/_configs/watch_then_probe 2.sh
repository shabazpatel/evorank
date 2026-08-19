#!/bin/bash
# Wait for the CPU-heavy detached jobs (full eval + simplified retune) to exit,
# then launch the feedback-ablation probe. Waits on pids passed as arguments.
set -u
LOG=/Users/shabazpatel/Desktop/Project/skydiscover/evorank/runs/_configs
echo "[$(date +%H:%M:%S)] watcher started, waiting on pids: $*"
for pid in "$@"; do
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
  done
  echo "[$(date +%H:%M:%S)] pid ${pid} exited"
done
echo "[$(date +%H:%M:%S)] CPU free, launching probe"
bash "$LOG/launch_probe.sh"
