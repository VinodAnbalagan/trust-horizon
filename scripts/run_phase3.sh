#!/usr/bin/env bash
# Phase 3 GPU stages: bank (encode training clips), then dreams from new start states.
#   mkdir -p runs && nohup bash scripts/run_phase3.sh <price> > runs/phase3.log 2>&1 &
#   tail -n 30 runs/phase3.log        # check progress and ETA (prints and returns)
# Resumable: re-running skips finished items.
set -uo pipefail
PRICE=${1:-0.50}
export HF_HOME=${HF_HOME:-/workspace/hf_cache}
for s in bank dreams; do
  echo "=================== phase3 $s  $(date) ==================="
  python phase3.py "$s" --usd-per-hour "$PRICE"
done
echo "=================== phase3 finished  $(date) ==================="
