#!/usr/bin/env bash
# IO baseline: 1 sample, no chain-of-thought, no search.
set -euo pipefail
cd "$(dirname "$0")/.."

BACKEND="${BACKEND:-claude_cli:sonnet}"
START="${START:-900}"
END="${END:-1000}"

python run.py \
  --task game24 \
  --backend "$BACKEND" \
  --naive_run \
  --prompt_sample standard \
  --n_generate_sample "${N:-1}" \
  --task_start_index "$START" \
  --task_end_index "$END"
