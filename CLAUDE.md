# CLAUDE.md

Context for future Claude Code sessions working in this repo.

## What this is

CS 4782 final project: re-implementation of *Tree of Thoughts: Deliberate Problem Solving with Large Language Models* (Yao et al., NeurIPS 2023). Scope is **Game of 24 only** (paper Table 2 / §4.1). Upstream reference: https://github.com/princeton-nlp/tree-of-thought-llm.

Detailed plan lives at `~/.claude/plans/jaunty-roaming-umbrella.md`.

## Current state (last updated 2026-05-08 ~17:00 EDT)

Working on branch **`brian`**. Headline experiment matrix complete.

**Done** (haiku, 900–925, post-fix `reasoning:false` + format-discipline system prompt):
| Run | acc_avg | acc_any | Elapsed | Cost |
|---|---|---|---|---|
| IO (`naive_standard_n1`) | 8.0% | 8.0% | 5 min | $0.33 |
| IO best-of-100 (`naive_standard_n100`) | **99.8%** | **100.0%** | 50 min (25-way parallel) | $22.85 |
| CoT (`naive_cot_n1`) | 100.0% | 100.0% | 6 min | $0.43 |
| CoT-SC k=20 (`naive_cot_n20`) | 100.0% | 100.0% | 19 min (20-way parallel) | $7.98 |
| CoT best-of-100 (`naive_cot_n100`) | **99.6%** | **100.0%** | 127 min (25-way parallel) | $38.71 |
| ToT b=1 (`bfs_propose1_value3_greedy1`) | 72.0% | 72.0% | ~50 min (5-way parallel) | ~$5 |
| ToT b=5 (`bfs_propose1_value3_greedy5`) | — | 92.0% | — | — |

**Headline finding for poster/report:** on a strong modern model (haiku-4.5), the paper's "ToT > CoT-SC > CoT > IO" ordering inverts at the top of the curve. CoT-SC k=20 already saturates acc_any at 100%; IO and CoT both reach 100% acc_any with enough samples. ToT b=5 caps at 92% acc_any on the same slice. The paper's signal was strongest on a weaker model (GPT-4 CoT 4%, ToT 74%); modern haiku makes Game-24 mostly easy, so the headline becomes "search beats sampling on weak models, sampling can match search on strong ones."

**Caveat on the IO n=1 baseline:** the 8% IO n=1 number is a parsing artifact, NOT haiku's true IO ceiling. Inspecting the cached n=1 outputs shows the model produces correct equations (e.g. `Answer: 6 * 5 - 10 + 4 = 24`) but appends a `**Verification:**` markdown block whose extra numbers confuse `test_output`'s multiset check. The format-discipline system prompt suppresses pre-Answer commentary but not post-Answer markdown. The IO n=100 result (99.8% acc_avg) is haiku's true IO competence; report this distinction or update the parser.

**Cost tally:** ~$77 quota spent on the haiku-25 best-of-100 batch ($22.85 IO + $38.71 CoT + ~$15 in earlier baselines).

**Cache history:**
- `code/.cache/llm.bak-verbose/` (323 entries) — backup of all calls made BEFORE the system-prompt fix. These outputs include haiku's verbose "Looking at the puzzle..." commentary. Don't delete — useful for diagnostics.
- `code/.cache/llm/` — fresh cache populated after the fix; concise outputs.

**Surprising findings to flag in the report:**
- Modern haiku CoT (80%) ≫ paper's GPT-4 CoT (4%). Likely because the 5-shot prompt walks through "Steps: ... Answer: ..." which structures haiku's output cleanly.
- CoT-SC k=20 hits 100% acc_any on this slice — 20 attempts is enough that at least one passes for every puzzle. Per-sample acc_avg of 69.4% < CoT k=1's 80%, due to variance + temperature-induced sample diversity.
- **`claude -p` has no `--max-tokens` flag.** Default cap is 32K output tokens. With an empty system prompt, haiku produced 25–30K-token responses to "Possible next steps:" prompts (we expected ~200). Cause: prompt is open-ended and few-shot example wasn't enough discipline. Fix: replace `--system-prompt ""` with a tiny format-discipline directive. Expected ~5–10× speedup per call AND meaningful accuracy improvement (verbose meta-commentary was breaking `test_output` parsing).
- Per-`claude -p` latency in concurrent mode is dominated by output token count, not haiku's inference speed. With the system-prompt fix, expect per-call latency to drop from ~60s → ~10s.
- **Extended thinking dominates haiku-4.5 latency even after the format fix.** A test prompt that produced 65 chars of visible output was still billed for 36K output tokens — the rest was internal thinking. Probed candidate `--settings` JSON keys to disable it. Findings (output_tokens / elapsed for the same value-prompt input "2 9 12"):
  - baseline: 7855 / 49.6s
  - `--effort low`: 7212 / 39.9s (small effect)
  - `{"thinking":{"type":"disabled"}}`: 4660 / 25.2s
  - `{"thinking":false}`: 5840 / 29.0s
  - **`{"reasoning":false}`: 3471 / 18.0s** ← winner, applied in `tot/models.py::_CLAUDE_FLAGS`
  - `{"extendedThinking":false}` and `{"thinkingBudget":0}`: silently ignored (keys not recognized)

  Doesn't fully disable thinking (still ~3K thinking tokens vs ~100 visible), but a >60% latency cut is real. Test script lived at `/tmp/test_thinking_disable.py` — re-create from CLAUDE.md history if needed.

## Resume / crash recovery

If Claude Code crashes, the laptop sleeps badly, or you need to pick up later:

1. **Reattach to branch:** `git checkout brian` (you are probably already there)
2. **Check what's persisted:**
   ```bash
   ls code/results/game24/claude_cli-haiku/    # JSONL + summary.json per run
   ls code/.cache/llm | wc -l                  # LLM response cache (key resilience)
   ```
3. **Identify what's missing.** Each completed run has both `<run_id>.jsonl` AND `<run_id>.summary.json`. If the JSONL exists but summary is missing, that run was killed mid-way — re-run it (cache makes already-processed puzzles ~instant).
4. **Re-launch missing runs.** All scripts are idempotent because of the cache:
   ```bash
   cd code
   # Resume any specific shard:
   START=900 END=905 BACKEND=claude_cli:haiku bash scripts/game24_tot_b1.sh
   # Resume the b=5 we deferred (smaller slice):
   START=900 END=910 BACKEND=claude_cli:haiku bash scripts/game24_tot_b5.sh
   # Re-run a baseline:
   START=900 END=925 BACKEND=claude_cli:haiku bash scripts/game24_io.sh
   ```
5. **Merge sharded runs into canonical:**
   ```bash
   python code/merge_shards.py code/results/game24/claude_cli-haiku \
       --pattern claude_cli-haiku_T0.7_bfs_propose1_value3_greedy1 \
       --start 900 --end 925
   ```
6. **Aggregate to CSV:**
   ```bash
   python code/analyze.py    # writes code/results/summary.csv
   ```
7. **Commit on `brian`:**
   ```bash
   git add code/results
   git commit -m "haiku-25 BFS baseline: <description>"
   ```

The cache (`code/.cache/llm/`) is sha256-keyed by `(backend, prompt, temperature, n, stop)`. Re-running an interrupted condition is essentially free for puzzles that completed before the interruption — every LLM call is replayed from disk.

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
