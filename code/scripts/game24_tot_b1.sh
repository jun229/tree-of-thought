#!/usr/bin/env bash
# ToT BFS, beam width = 1.
set -euo pipefail
cd "$(dirname "$0")/.."

BACKEND="${BACKEND:-claude_cli:sonnet}"
START="${START:-900}"
END="${END:-1000}"

python run.py \
  --task game24 \
  --backend "$BACKEND" \
  --method_generate propose \
  --method_evaluate value \
  --method_select greedy \
  --n_generate_sample 1 \
  --n_evaluate_sample 3 \
  --n_select_sample 1 \
  --task_start_index "$START" \
  --task_end_index "$END"
