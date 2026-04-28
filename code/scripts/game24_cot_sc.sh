#!/usr/bin/env bash
# CoT-SC baseline: 100 chain-of-thought samples, majority vote on test_output.
# (Selection is downstream of generation; we just rely on cnt_any over n=100.)
set -euo pipefail
cd "$(dirname "$0")/.."

BACKEND="${BACKEND:-claude_cli:sonnet}"
START="${START:-900}"
END="${END:-1000}"
N="${N:-100}"

python run.py \
  --task game24 \
  --backend "$BACKEND" \
  --naive_run \
  --prompt_sample cot \
  --n_generate_sample "$N" \
  --task_start_index "$START" \
  --task_end_index "$END"
