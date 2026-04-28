#!/usr/bin/env bash
# Tiny end-to-end smoke test (5 puzzles). Default backend is groq because it's
# free and fast. Override with BACKEND=claude_cli:sonnet for the headline tier.
set -euo pipefail
cd "$(dirname "$0")/.."

BACKEND="${BACKEND:-groq:llama-3.3-70b-versatile}"
START="${START:-900}"
END="${END:-905}"

python run.py \
  --task game24 \
  --backend "$BACKEND" \
  --temperature 0.7 \
  --method_generate propose \
  --method_evaluate value \
  --method_select greedy \
  --n_generate_sample 1 \
  --n_evaluate_sample 3 \
  --n_select_sample 5 \
  --task_start_index "$START" \
  --task_end_index "$END" \
  --verbose
