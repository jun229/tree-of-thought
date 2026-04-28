#!/usr/bin/env bash
# Extension 2: ToT with deterministic reach-24 heuristic value (no LLM eval calls).
# Compare against game24_tot_b5.sh for the same backend/slice.
set -euo pipefail
cd "$(dirname "$0")/.."

BACKEND="${BACKEND:-claude_cli:sonnet}"
START="${START:-900}"
END="${END:-1000}"

python run.py \
  --task game24 \
  --backend "$BACKEND" \
  --method_generate propose \
  --method_evaluate heuristic \
  --method_select greedy \
  --n_generate_sample 1 \
  --n_evaluate_sample 1 \
  --n_select_sample 5 \
  --task_start_index "$START" \
  --task_end_index "$END"
