"""Merge sharded ToT runs into a single canonical run.

When a long ToT run is split across N parallel processes (each handling a
disjoint puzzle range), we get N JSONL files and N summary.json files. This
script concatenates them into a single canonical pair using the full puzzle
range, e.g., shards covering 900-905, 905-910, ..., 920-925 produce
`<run_id without range>_900-925.{jsonl,summary.json}`.

Usage:
    python merge_shards.py results/game24/claude_cli-haiku \\
        --pattern "claude_cli-haiku_T0.7_bfs_propose1_value3_greedy1" \\
        --start 900 --end 925
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def merge(results_dir: Path, pattern: str, start: int, end: int) -> None:
    # Find all summary.json files matching the pattern (e.g.,
    # ".../<pattern>_<start>-<end>.summary.json"). The shard ranges differ.
    rng_re = re.compile(rf"{re.escape(pattern)}_(\d+)-(\d+)\.summary\.json$")
    shards = []
    for p in results_dir.glob(f"{pattern}_*.summary.json"):
        m = rng_re.search(p.name)
        if not m:
            continue
        s, e = int(m.group(1)), int(m.group(2))
        if s >= start and e <= end:
            shards.append((s, e, p))
    shards.sort()
    if not shards:
        raise SystemExit(f"No shards found matching {pattern} under {results_dir}")

    # Sanity check coverage.
    covered = sum(e - s for s, e, _ in shards)
    expected = end - start
    if covered != expected:
        ranges = ", ".join(f"{s}-{e}" for s, e, _ in shards)
        raise SystemExit(
            f"Shard coverage mismatch: shards cover {covered} puzzles "
            f"({ranges}), but range {start}-{end} expects {expected}."
        )

    # Concatenate JSONLs in puzzle-index order.
    out_jsonl = results_dir / f"{pattern}_{start}-{end}.jsonl"
    with out_jsonl.open("w") as out:
        for s, e, summary_path in shards:
            shard_jsonl = summary_path.with_suffix("").with_suffix(".jsonl")
            with shard_jsonl.open() as f:
                for line in f:
                    out.write(line)

    # Aggregate summary.
    total_n = 0
    total_correct_avg = 0.0  # weighted by puzzles per shard
    total_any = 0
    total_completion = 0
    total_prompt = 0
    total_cost = 0.0
    max_elapsed = 0.0  # parallel runs => take the max wall-clock as effective elapsed
    args_template = None
    for _, _, summary_path in shards:
        with summary_path.open() as f:
            d = json.load(f)
        n = d.get("n_puzzles", 0)
        total_n += n
        total_correct_avg += d.get("accuracy_avg", 0.0) * n
        total_any += int(d.get("accuracy_any", 0.0) * n)
        usage = d.get("usage", {}) or {}
        total_completion += usage.get("completion_tokens", 0)
        total_prompt += usage.get("prompt_tokens", 0)
        total_cost += usage.get("cost", 0.0)
        max_elapsed = max(max_elapsed, d.get("elapsed_s", 0.0))
        if args_template is None:
            args_template = dict(d.get("args", {}))

    if args_template is not None:
        args_template["task_start_index"] = start
        args_template["task_end_index"] = end

    merged = {
        "args": args_template or {},
        "run_id": f"{pattern}_{start}-{end}",
        "n_puzzles": total_n,
        "accuracy_avg": total_correct_avg / total_n if total_n else 0.0,
        "accuracy_any": total_any / total_n if total_n else 0.0,
        "usage": {
            "completion_tokens": total_completion,
            "prompt_tokens": total_prompt,
            "cached_tokens": 0,
            "cost": total_cost,
        },
        "elapsed_s": max_elapsed,
        "merged_from": [f.name for _, _, f in shards],
    }
    out_summary = results_dir / f"{pattern}_{start}-{end}.summary.json"
    with out_summary.open("w") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged {len(shards)} shards covering {start}-{end} ({total_n} puzzles)")
    print(f"  accuracy_avg = {merged['accuracy_avg']:.2%}")
    print(f"  accuracy_any = {merged['accuracy_any']:.2%}")
    print(f"  cost = ${merged['usage']['cost']:.2f}")
    print(f"  -> {out_jsonl}")
    print(f"  -> {out_summary}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("results_dir", type=Path,
                   help="e.g. results/game24/claude_cli-haiku")
    p.add_argument("--pattern", required=True,
                   help="run_id minus the trailing _<start>-<end> range. "
                        "e.g. claude_cli-haiku_T0.7_bfs_propose1_value3_greedy1")
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    args = p.parse_args()
    merge(args.results_dir, args.pattern, args.start, args.end)
