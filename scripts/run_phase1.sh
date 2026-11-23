#!/usr/bin/env bash
# Run Phase 1 stages in order, in the background, surviving a closed browser tab.
#   nohup bash scripts/run_phase1.sh 0.50 repro > runs/phase1.log 2>&1 &
#   nohup bash scripts/run_phase1.sh 0.50 fit test unseen > runs/phase1.log 2>&1 &
#   tail -f runs/phase1.log            # watch progress (Ctrl+C stops watching, not the run)
# Every stage resumes where it stopped, so re-running after an interruption is safe.
set -uo pipefail
PRICE=${1:-0.50}; shift || true
STAGES=${@:-repro fit test unseen}
export HF_HOME=${HF_HOME:-/workspace/hf_cache}
for s in $STAGES; do
  echo "=================== stage $s  $(date) ==================="
  python phase1.py "$s" --usd-per-hour "$PRICE"
done
echo "=================== all requested stages finished  $(date) ==================="
