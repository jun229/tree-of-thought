"""CLI entry point. Mirrors upstream run.py flags so existing prompt-tape
scripts port over.

Usage examples:
    python run.py --task game24 --backend groq:llama-3.3-70b-versatile \
        --naive_run --prompt_sample cot --n_generate_sample 1 \
        --task_start_index 900 --task_end_index 905

    python run.py --task game24 --backend claude_cli:sonnet \
        --method_generate propose --method_evaluate value --method_select greedy \
        --n_generate_sample 1 --n_evaluate_sample 3 --n_select_sample 5 \
        --task_start_index 900 --task_end_index 1000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure imports work when run from project root or from code/.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from tot.tasks import get_task  # noqa: E402
from tot.search.bfs import solve, naive_solve  # noqa: E402
from tot.models import gpt_usage, reset_usage  # noqa: E402


def _slug(s: str) -> str:
    return s.replace(":", "-").replace("/", "_")


def _run_id(args) -> str:
    backend = _slug(args.backend)
    if args.backend_evaluate and args.backend_evaluate != args.backend:
        backend = f"{backend}_eval-{_slug(args.backend_evaluate)}"
    if args.naive_run:
        return (
            f"{backend}_T{args.temperature}_naive_{args.prompt_sample}"
            f"_n{args.n_generate_sample}"
            f"_{args.task_start_index}-{args.task_end_index}"
        )
    return (
        f"{backend}_T{args.temperature}_{args.algo}"
        f"_{args.method_generate}{args.n_generate_sample}"
        f"_{args.method_evaluate}{args.n_evaluate_sample}"
        f"_{args.method_select}{args.n_select_sample}"
    )
    if args.n_monte_carlo > 1:
        run_id += f"_mc{args.n_monte_carlo}"
    return f"{run_id}_{args.task_start_index}-{args.task_end_index}"


def _solve_with_monte_carlo(args, task, idx):
    if args.naive_run or args.n_monte_carlo <= 1:
        return solve(args, task, idx, to_print=args.verbose)

    all_ys = []
    rollouts = []
    for rollout in range(args.n_monte_carlo):
        if args.verbose:
            print(f"== monte carlo rollout {rollout + 1}/{args.n_monte_carlo} ==")
        ys, info = solve(args, task, idx, to_print=args.verbose)
        all_ys.extend(ys)
        rollouts.append({
            "rollout": rollout,
            "ys": ys,
            "info": info,
        })
    return all_ys, {"monte_carlo_rollouts": rollouts}


def run(args):
    if not args.naive_run and args.algo != "bfs":
        raise NotImplementedError(
            f"--algo {args.algo!r} not implemented yet; only 'bfs' has a "
            f"search module right now (tot/search/bfs.py)."
        )
    reset_usage()
    task = get_task(args.task)
    run_id = _run_id(args)

    log_dir = Path(args.log_dir) / args.task / _slug(args.backend)
    log_dir.mkdir(parents=True, exist_ok=True)
    jsonl_file = log_dir / f"{run_id}.jsonl"
    summary_file = log_dir / f"{run_id}.summary.json"

    cnt_avg = 0.0
    cnt_any = 0
    n = args.task_end_index - args.task_start_index
    t0 = time.time()

    with jsonl_file.open("w") as f:
        for i in range(args.task_start_index, args.task_end_index):
            try:
                if args.naive_run:
                    ys, info = naive_solve(args, task, i, to_print=args.verbose)
                else:
                    ys, info = _solve_with_monte_carlo(args, task, i)
            except Exception as e:
                print(f"[idx {i}] ERROR: {e!r}", file=sys.stderr)
                ys, info = [], {"error": repr(e)}

            results = [task.test_output(i, y) for y in ys] if ys else [{"r": 0}]
            accs = [r["r"] for r in results]
            cnt_avg += (sum(accs) / len(accs)) if accs else 0.0
            cnt_any += int(any(accs))

            record = {
                "idx": i,
                "input": task.get_input(i),
                "ys": ys,
                "results": results,
                "usage_so_far": gpt_usage(),
                "info": info,
            }
            f.write(json.dumps(record, default=str) + "\n")
            f.flush()

            elapsed = time.time() - t0
            print(
                f"[{i - args.task_start_index + 1}/{n}] idx={i} "
                f"acc_any={any(accs)} cnt_avg={cnt_avg:.2f} cnt_any={cnt_any} "
                f"usage={gpt_usage()} elapsed={elapsed:.1f}s"
            )

    summary = {
        "args": vars(args),
        "run_id": run_id,
        "n_puzzles": n,
        "accuracy_avg": (cnt_avg / n) if n else 0.0,  # avg over per-puzzle pass-rate
        "accuracy_any": (cnt_any / n) if n else 0.0,  # at least one ys correct
        "usage": gpt_usage(),
        "elapsed_s": time.time() - t0,
    }
    with summary_file.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))


def parse_args():
    p = argparse.ArgumentParser(description="Tree of Thoughts re-implementation")
    p.add_argument(
        "--backend", type=str, default="claude_cli:sonnet",
        help="<provider>:<model>. E.g. claude_cli:sonnet, claude_cli:haiku, "
             "gemini:gemini-2.0-flash, groq:llama-3.3-70b-versatile, "
             "openrouter:meta-llama/llama-3.1-70b-instruct:free, "
             "gemma3:google/gemma-3-4b-it, gemma3:google/gemma-3-12b-it:4bit",
    )
    p.add_argument(
        "--backend_evaluate", type=str, default=None,
        help="Optional asymmetric backend for the value/vote evaluator. "
             "If unset, uses --backend for both proposer and evaluator. "
             "Extension 4: cheap verifier (e.g. groq:llama-3.3-70b-versatile) "
             "with strong proposer (e.g. claude_cli:sonnet).",
    )
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature (ignored by claude_cli).")
    p.add_argument("--task", type=str, required=True, choices=["game24"])
    p.add_argument("--task_start_index", type=int, default=900)
    p.add_argument("--task_end_index", type=int, default=1000)

    p.add_argument("--naive_run", action="store_true",
                   help="Run IO/CoT/CoT-SC baselines (no tree search).")
    p.add_argument("--prompt_sample", type=str, choices=["standard", "cot"],
                   help="For --naive_run or --method_generate sample.")

    p.add_argument("--method_generate", type=str, choices=["sample", "propose"])
    p.add_argument("--method_evaluate", type=str,
                   choices=["value", "vote", "heuristic"],
                   help="`heuristic` uses task.heuristic_value (deterministic, "
                        "no LLM calls). Extension 2 in the project plan.")
    p.add_argument("--method_select", type=str, choices=["sample", "greedy"], default="greedy")
    p.add_argument(
        "--algo", type=str, default="bfs", choices=["bfs", "dfs", "mcts"],
        help="Tree-search algorithm. Currently only 'bfs' is implemented; the "
             "flag exists so future MCTS/DFS runs get distinct run_ids without "
             "colliding with these BFS results.",
    )
    p.add_argument("--n_generate_sample", type=int, default=1)
    p.add_argument("--n_evaluate_sample", type=int, default=1)
    p.add_argument("--n_select_sample", type=int, default=1)
    p.add_argument(
        "--n_monte_carlo", type=int, default=1,
        help="For ToT only: repeat solve() this many stochastic rollouts per puzzle "
             "and score over the pooled final candidates. Pair with "
             "--method_select sample for Monte Carlo branch selection.",
    )

    p.add_argument("--log_dir", type=str, default="results")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("ARGS:", vars(args))
    run(args)
