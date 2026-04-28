#!/usr/bin/env bash
# Extension 1: sweep beam width b in {1, 3, 5, 7}. Paper only reports b=1, b=5.
set -euo pipefail
cd "$(dirname "$0")/.."

BACKEND="${BACKEND:-gemini:gemini-2.0-flash}"
START="${START:-900}"
END="${END:-1000}"

for B in 1 3 5 7; do
  echo "=== ablate_beam: b=$B ==="
  python run.py \
    --task game24 \
    --backend "$BACKEND" \
    --method_generate propose \
    --method_evaluate value \
    --method_select greedy \
    --n_generate_sample 1 \
    --n_evaluate_sample 3 \
    --n_select_sample "$B" \
    --task_start_index "$START" \
    --task_end_index "$END"
done
