"""Aggregate run summaries into a single CSV for plotting.

Walks `results/<task>/<backend>/*.summary.json` and emits a CSV row per run with
the args, accuracy, and token/cost usage. Use to populate figures comparing
methods / beam widths / backends.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


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
    here = Path(__file__).resolve().parent
    aggregate(here / "results", here / "results" / "summary.csv")
