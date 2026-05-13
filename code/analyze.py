"""Aggregate run summaries into a single CSV for plotting.

Walks `results/<task>/<backend>/*.summary.json` and emits a CSV row per run with
the args, accuracy, and token/cost usage. Use to populate figures comparing
methods / beam widths / backends.

Each row also carries a `_lenient` accuracy pair computed from the run's
JSONL by `tot.scoring.lenient` — this catches answers that are factually
correct but get scored r=0 by the strict upstream `test_output` because the
model's output has a trailing markdown block (see `tot/scoring/lenient.py`).
The strict columns are unchanged so headline numbers stay traceable to the
verbatim Princeton scorer.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

# Make `tot` importable when run from the project root.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from tot.scoring.lenient import lenient_score  # noqa: E402


def _lenient_accuracies(summary_path: Path) -> tuple[float, float] | tuple[None, None]:
    """Re-score the JSONL beside `summary_path` with the lenient parser.
    Returns (acc_avg_lenient, acc_any_lenient), or (None, None) if the
    JSONL is missing (older summaries from upstream-only runs)."""
    jsonl_path = summary_path.with_suffix("").with_suffix(".jsonl")
    if not jsonl_path.exists():
        return (None, None)
    total = 0
    sum_avg = 0.0
    sum_any = 0
    with jsonl_path.open() as f:
        for line in f:
            rec = json.loads(line)
            ys = rec.get("ys") or []
            problem_input = rec.get("input", "")
            if not ys:
                total += 1
                continue
            scores = [lenient_score(problem_input, y) for y in ys]
            sum_avg += sum(scores) / len(scores)
            sum_any += int(any(scores))
            total += 1
    if total == 0:
        return (0.0, 0.0)
    return (sum_avg / total, sum_any / total)


def aggregate(results_dir: Path, out_csv: Path) -> None:
    rows = []
    for summary in sorted(results_dir.rglob("*.summary.json")):
        # Skip archived runs (e.g. _archive_thinking_on/ holds the original
        # thinking-enabled CoT/CoT-SC numbers, kept for diff'ing only).
        if any(part.startswith("_archive") for part in summary.parts):
            continue
        with summary.open() as f:
            d = json.load(f)
        args = d.get("args", {})
        usage = d.get("usage", {})
        # Naive runs aren't search runs, so report algo="" rather than the
        # default "bfs" — keeps the CSV honest about what was actually run.
        algo = "" if args.get("naive_run") else (args.get("algo") or "")
        acc_avg_lenient, acc_any_lenient = _lenient_accuracies(summary)
        rows.append({
            "run_id": d.get("run_id", summary.stem),
            "task": args.get("task", ""),
            "backend": args.get("backend", ""),
            "algo": algo,
            "method_generate": args.get("method_generate") or "",
            "method_evaluate": args.get("method_evaluate") or "",
            "method_select": args.get("method_select") or "",
            "naive_run": args.get("naive_run", False),
            "prompt_sample": args.get("prompt_sample") or "",
            "n_generate_sample": args.get("n_generate_sample", 0),
            "n_evaluate_sample": args.get("n_evaluate_sample", 0),
            "n_select_sample": args.get("n_select_sample", 0),
            "task_start_index": args.get("task_start_index", 0),
            "task_end_index": args.get("task_end_index", 0),
            "n_puzzles": d.get("n_puzzles", 0),
            "accuracy_avg": d.get("accuracy_avg", 0.0),
            "accuracy_any": d.get("accuracy_any", 0.0),
            "accuracy_avg_lenient": acc_avg_lenient if acc_avg_lenient is not None else "",
            "accuracy_any_lenient": acc_any_lenient if acc_any_lenient is not None else "",
            "completion_tokens": usage.get("completion_tokens", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "cost_usd": usage.get("cost", 0.0),
            "elapsed_s": d.get("elapsed_s", 0.0),
        })
    if not rows:
        print(f"No *.summary.json files found under {results_dir}", file=sys.stderr)
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    aggregate(HERE / "results", HERE / "results" / "summary.csv")
