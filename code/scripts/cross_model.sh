#!/usr/bin/env bash
# Extension 3: same E5 (ToT b=5) condition across providers/models.
# Plot accuracy vs tokens-per-puzzle in analysis.
set -euo pipefail
cd "$(dirname "$0")/.."

START="${START:-900}"
END="${END:-1000}"

BACKENDS=(
  "claude_cli:sonnet"
  "claude_cli:haiku"
  "gemini:gemini-2.0-flash"
  "groq:llama-3.3-70b-versatile"
)

for B in "${BACKENDS[@]}"; do
  echo "=== cross_model: $B ==="
  python run.py \
    --task game24 \
    --backend "$B" \
    --method_generate propose \
    --method_evaluate value \
    --method_select greedy \
    --n_generate_sample 1 \
    --n_evaluate_sample 3 \
    --n_select_sample 5 \
    --task_start_index "$START" \
    --task_end_index "$END"
done
