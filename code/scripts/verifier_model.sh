#!/usr/bin/env bash
# Extension 4: strong proposer + cheap verifier.
# Tests the hypothesis that evaluator quality matters less than proposer
# quality for ToT on Game-24. Compares against game24_tot_b5.sh (symmetric).
#
# Defaults: propose with claude_cli:sonnet (strong), evaluate with
# groq:llama-3.3-70b-versatile (free + fast). Override either via env vars.
set -euo pipefail
cd "$(dirname "$0")/.."

PROPOSE_BACKEND="${PROPOSE_BACKEND:-claude_cli:sonnet}"
EVAL_BACKEND="${EVAL_BACKEND:-groq:llama-3.3-70b-versatile}"
START="${START:-900}"
END="${END:-1000}"

python run.py \
  --task game24 \
  --backend "$PROPOSE_BACKEND" \
  --backend_evaluate "$EVAL_BACKEND" \
  --method_generate propose \
  --method_evaluate value \
  --method_select greedy \
  --n_generate_sample 1 \
  --n_evaluate_sample 3 \
  --n_select_sample 5 \
  --task_start_index "$START" \
  --task_end_index "$END"
