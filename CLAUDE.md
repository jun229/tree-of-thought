# CLAUDE.md

Context for future Claude Code sessions working in this repo.

## What this is

CS 4782 final project: re-implementation of *Tree of Thoughts: Deliberate Problem Solving with Large Language Models* (Yao et al., NeurIPS 2023). Scope is **Game of 24 only** (paper Table 2 / §4.1). Upstream reference: https://github.com/princeton-nlp/tree-of-thought-llm.

Detailed plan lives at `~/.claude/plans/jaunty-roaming-umbrella.md`.

## Repo layout

```
code/
  run.py                 # CLI entry point (mirrors upstream flag set)
  analyze.py             # aggregate *.summary.json -> results/summary.csv
  tot/
    models.py            # backend dispatcher + on-disk cache (the unusual file)
    search/bfs.py        # BFS solve() / naive_solve()
    tasks/{base,game24}.py
    prompts/game24.py    # VERBATIM from upstream — DO NOT MODIFY
    heuristics/reach24.py
    utils/cache.py
  tests/                 # pytest, run with: python -m pytest tests/
  scripts/*.sh           # one per experiment condition
data/24/24.csv           # 1,362 puzzles, paper evaluates indices 900-1000
results/<task>/<backend>/<run_id>.{jsonl,summary.json}
.ref/                    # gitignored clone of upstream (parity tests + reference)
.cache/llm/              # gitignored on-disk LLM response cache
```

## Critical conventions

- **Prompts are verbatim from upstream.** `tot/prompts/game24.py` is a 1:1 copy. Do not edit. They ARE the method.
- **`test_output` is the success metric.** Drift here silently changes every reported number. The parity test (`tests/test_game24_checker.py::test_parity_with_upstream`) runs upstream's `test_output` in a subprocess (with `PYTHONPATH=.ref/src`) and requires byte-identical labels on a fixed set of cases. If you change anything in `tasks/game24.py::test_output`, this test must still pass.
- **Backend strings are `<provider>:<model>`.** Examples: `claude_cli:sonnet`, `claude_cli:haiku`, `gemini:gemini-2.0-flash`, `groq:llama-3.3-70b-versatile`, `openrouter:<slug>`. A bare model name (no colon) defaults to `claude_cli`.
- **Default backend is `claude_cli:sonnet`** — shells out to `claude -p` against the user's Pro/Max OAuth. No API key required. Other backends require `GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY` respectively.
- **Cache is correctness, not optimization.** Every `gpt()` call is sha256-keyed by `(backend, prompt, temperature, n, stop)` in `.cache/llm/`. Cached calls do NOT count toward `gpt_usage()`. Identical re-runs are byte-identical and zero-call. Don't disable.

## Cost & quota gotchas

- One `claude -p --model haiku` call ≈ ~$0.16 of subscription quota (mostly cache_creation overhead from Claude Code internals — even with `--system-prompt ""` and `--tools ""`).
- ToT b=5 makes ~16 calls per Game-24 puzzle → ~$2.50 quota per puzzle on haiku, ~3–5× that on sonnet.
- For 100-puzzle slices on sonnet, expect to consume meaningful Pro/Max session quota. Recommend testing on a 25-puzzle slice first.
- The cache means re-running the same config is free. Use this — don't ablate by re-running, run new conditions and let analyze.py compare.

## Common tasks

```bash
# tests
cd code && python -m pytest tests/ -v

# smoke (5 puzzles, free Groq backend)
cd code && BACKEND=groq:llama-3.3-70b-versatile bash scripts/dev_smoke.sh

# headline ToT b=5 on 100 puzzles, claude_cli:sonnet
cd code && bash scripts/game24_tot_b5.sh

# aggregate after runs
python code/analyze.py     # writes results/summary.csv
```

## Extensions tracked (rubric J — Independent Exploration)

1. **Beam-width sweep b∈{1,3,5,7}** — `scripts/ablate_beam.sh`
2. **Reach-24 heuristic value** — `tot/heuristics/reach24.py` + `scripts/heuristic_value.sh` (deterministic, zero LLM eval calls)
3. **Cross-model frontier** — `scripts/cross_model.sh` (claude_cli/gemini/groq, same matrix)
4. **Verifier-model evaluator** — `scripts/verifier_model.sh` (`--backend_evaluate` flag splits proposer/evaluator backends; `bfs.solve` threads two `gpt_fn`s)

## Deadlines

- **Poster** due April 30 OR May 5 (depends on assigned session). PDF goes in `poster/`.
- **2-page report + GitHub repo** due **May 12**. PDF goes in `report/group_treeofthought_2page_report.pdf`.

## Things NOT to do

- Don't modify `tot/prompts/game24.py`. It's verbatim from upstream by design.
- Don't bypass the cache for "freshness" — the seed is captured and re-runs are supposed to be deterministic w.r.t. backend output.
- Don't run experiments inside `.ref/` — that's read-only reference, regenerated via `git clone`.
- Don't commit `.cache/`, `.ref/`, or `.env`. They're in `.gitignore`.
- Don't add new tasks (text, crosswords) before the headline Game-24 numbers are solid. Project plan explicitly scopes to Game-24.
