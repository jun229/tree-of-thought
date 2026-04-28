#!/usr/bin/env bash
# CoT baseline: 1 chain-of-thought sample, no search.
set -euo pipefail
cd "$(dirname "$0")/.."

BACKEND="${BACKEND:-claude_cli:sonnet}"
START="${START:-900}"
END="${END:-1000}"

python run.py \
  --task game24 \
  --backend "$BACKEND" \
  --naive_run \
  --prompt_sample cot \
  --n_generate_sample 1 \
  --task_start_index "$START" \
  --task_end_index "$END"
