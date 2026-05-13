#!/usr/bin/env bash
# ToT Monte Carlo rollouts: repeat stochastic BFS rollouts and pool finals.
set -euo pipefail
cd "$(dirname "$0")/.."

BACKEND="${BACKEND:-openai:gpt-4o-mini}"
START="${START:-900}"
END="${END:-1000}"
MC="${MC:-5}"

python run.py \
  --task game24 \
  --backend "$BACKEND" \
  --method_generate propose \
  --method_evaluate value \
  --method_select sample \
  --n_generate_sample 1 \
  --n_evaluate_sample 3 \
  --n_select_sample 5 \
  --n_monte_carlo "$MC" \
  --task_start_index "$START" \
  --task_end_index "$END"
