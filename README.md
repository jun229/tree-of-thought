# Tree of Thoughts — Re-implementation

CS 4782 Final Project (Cornell, Spring 2026). Re-implementation of *Tree of Thoughts: Deliberate Problem Solving with Large Language Models* (Yao et al., NeurIPS 2023).

## 1. Introduction

ToT generalizes Chain-of-Thought prompting by exploring a tree of intermediate "thoughts" with explicit search (BFS/DFS) and LLM-based self-evaluation, enabling planning, look-ahead, and backtracking on tasks that resist single-pass generation. We re-implement the method and reproduce the **Game of 24** experiment (paper Table 2 / §4.1), the cleanest demonstration that BFS+value-prompting beats IO/CoT/CoT-SC baselines.

## 2. Chosen Result

**Game of 24, indices 900–1000 of `data/24/24.csv`** (Table 2). Paper reports with GPT-4:
- IO ~7.3% · CoT ~4.0% · CoT-SC (k=100) ~9.0%
- ToT b=1 ~45% · **ToT b=5 ~74%**

We target reproducing the IO/CoT/CoT-SC vs ToT b∈{1,5} ordering, accepting absolute-number drift since we run on different (newer) models.

## 3. GitHub Contents

- `code/tot/` — re-implementation package: `models.py` (backend dispatcher + on-disk cache), `search/bfs.py` (BFS + naive_solve), `tasks/game24.py` (state, value, `test_output` checker), `prompts/game24.py` (verbatim from upstream), `heuristics/reach24.py` (extension 2: deterministic value).
- `code/run.py` — CLI mirroring upstream's flag set.
- `code/scripts/*.sh` — one script per experiment condition (IO, CoT, CoT-SC, ToT b=1/b=5, beam ablation, heuristic value, cross-model).
- `code/tests/` — unit tests, including parity test against upstream's `test_output`.
- `code/analyze.py` — aggregates run summaries into `results/summary.csv` for plotting.
- `data/24/24.csv` — Game-24 dataset (1,362 puzzles).
- `results/` — JSONL traces + per-run summary JSON, organized by `<task>/<backend>/<run_id>`.
- `poster/`, `report/` — submission PDFs.
 <!-- - `.ref/` — gitignored clone of upstream for reference and parity tests. -->

## 4. Re-implementation Details

- **Algorithm** — BFS over thought-trees: at each depth, expand each surviving candidate via *propose* (one prompt, structured numeric continuations) or *sample* (CoT prompt, n samples), then score every candidate via *value* (LLM rates "sure/likely/impossible") or *vote* or `heuristic` (our reach-24 deterministic check), then prune to top-`b` greedily or by score-weighted sampling.
- **Backends** (provider-agnostic dispatcher; select via `--backend <provider>:<model>`):
  - `openai:gpt-4o-mini` — OpenAI API.
  - `claude_cli:sonnet` / `claude_cli:haiku` — shells out to `claude -p` against the user's Pro/Max OAuth (no API key).
  - `gemini:gemini-2.0-flash` — Google AI Gemini API.
  - `groq:llama-3.3-70b-versatile` — Groq free tier.
  - `openrouter:<slug>` — any OpenRouter model.
- **Caching** — every `(backend, prompt, temperature, n, stop)` request is sha256-keyed in `.cache/llm/`. Re-runs are byte-identical and zero-call. Cached calls do not count toward `gpt_usage()`.
- **Prompts** — `code/tot/prompts/game24.py` is adapted from upstream `src/tot/prompts/game24.py` (cited at top of file). The few-shot examples are unchanged; format-discipline instructions were added to `propose_prompt`, `cot_prompt`, `value_prompt`, and `value_last_step_prompt` to suppress verbose model output that broke downstream parsing.
- **Checker parity** — `tests/test_game24_checker.py::test_parity_with_upstream` runs both our `test_output` and upstream's on identical (puzzle, candidate) pairs and requires identical pass/fail labels on every case.
- **Extensions**: (1) beam-width sweep b∈{1,3,5,7}; (2) deterministic reach-24 heuristic value (zero LLM eval calls); (3) cross-model frontier (claude_cli vs gemini vs groq); (4) verifier-model evaluator (strong proposer + cheap evaluator).

## 5. Reproduction Steps

To reproduce the results for Claude Haiku, go to the "brian" branch. To reproduce the results for GPT-4o-mini, go to the "ethan" branch. To reproduce the results for qwen-3.5-2B, go to the "gemma3" branch

### 5.1: for GPT-4o-mini or Claude Haiku
```bash
# 1. Install
pip install -r code/requirements.txt

# 2. Pick a backend. For Pro/Max users, no API key needed:
export BACKEND=claude_cli:sonnet
# Or, free options (require API keys):
#   export OPENAI_API_KEY=...    BACKEND=openai:gpt-4o-mini
#   export GROQ_API_KEY=...      BACKEND=groq:llama-3.3-70b-versatile
#   export GEMINI_API_KEY=...    BACKEND=gemini:gemini-2.0-flash
#   export OPENROUTER_API_KEY=.. BACKEND=openrouter:meta-llama/llama-3.1-70b-instruct:free

# 3. Smoke test (5 puzzles, ~5 min depending on backend)
cd code && bash scripts/dev_smoke.sh

# 4. Run the matrix (each ~30–90 min depending on backend)
bash scripts/game24_io.sh        # IO baseline
bash scripts/game24_cot.sh       # CoT baseline
bash scripts/game24_cot_sc.sh    # CoT-SC k=100
bash scripts/game24_tot_b1.sh    # ToT b=1
bash scripts/game24_tot_b5.sh    # ToT b=5

# 5. Extensions
bash scripts/ablate_beam.sh      # beam width sweep
bash scripts/heuristic_value.sh  # extension 2
bash scripts/cross_model.sh      # extension 3

# 6. Aggregate
python code/analyze.py
```

**Compute**: no GPU required; all compute is API-bound. CPU-only laptop is sufficient. Subprocess overhead for `claude -p` is ~5–10s per call; budget ~3–5 minutes per ToT b=5 puzzle. Cache makes re-runs free.

### 5.2 For Qwen3.5-2B (Google Colab)

**Compute:** requires a GPU runtime — select Runtime → Change runtime type → **A100 GPU**.

1. Upload this repo to Google Drive.
2. Open `colab_gemma3.ipynb` in Colab and run all cells in order.
   - Cell 2 installs dependencies (`transformers`, `accelerate`, `sympy`, `pandas`).
   - Cell 5 prompts a HuggingFace login (`notebook_login()`) — a free HF account is required.
   - Adjust `START`/`END` in the Experiments cell (default: 900–1000 as in the paper) to control how many of the 100 paper puzzles to evaluate.



## 6. Results / Insights

`acc_any` across models on Game of 24 (indices 900–1000; haiku evaluated on 900–925). Run `python code/analyze.py` to regenerate from `results/`.

| Condition | Paper (GPT-4) | Haiku 4.5 | GPT-4o-mini | Qwen 3.5-2B |
|---|---|---|---|---|
| IO n=1 | 7.3% | 8.0%† | 4.0% | 2.0% |
| IO best-of-100 | 33% | 100.0% | — | 58.0% |
| CoT n=1 | 4.0% | 100.0% | 3.0% | 3.0% |
| CoT best-of-100 | 49% | 100.0% | 35.0% | 69.0% |
| ToT b=1 | 45% | 72.0% | 20.0% | 0.0% |
| ToT b=5 | **74%** | 92.0% | **41.0%** | 0.0%‡ |

†Haiku IO n=1 is a parsing artifact — the model produces correct equations but appends a `**Verification:**` block that confuses the checker; IO best-of-100 (100%) reflects true competence. ‡Qwen ToT b=5 evaluated on 30 puzzles.

**Headline findings:** ToT's advantage is highly model-dependent. On strong models (haiku), CoT alone saturates the task (100% acc\_any) and ToT caps at 92% — the paper's "ToT > CoT" ordering inverts. On mid-tier models (GPT-4o-mini), the paper's ordering holds: ToT b=5 (41%) substantially outperforms CoT (3%). On the weak 2B model (Qwen), ToT collapses entirely (0%) — the model cannot reliably evaluate intermediate states — while best-of-100 sampling still reaches 69% via sheer volume.

## 7. Conclusion

We reproduce the core finding of Yao et al. — ToT outperforms IO and CoT baselines — while surfacing an important caveat: the magnitude of the gap is highly model-dependent. On the weaker GPT-4 baseline in the paper (CoT 4%, ToT 74%), structured search provides a dramatic lift; on modern Claude Haiku 4.5, CoT alone saturates the task. The reach-24 heuristic (extension 2) replaces LLM value calls with a deterministic arithmetic reachability check at zero additional cost, offering a practical path to cheaper ToT on tasks with verifiable intermediate states.

## 8. References

- Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., & Narasimhan, K. (2023). *Tree of Thoughts: Deliberate Problem Solving with Large Language Models.* NeurIPS 2023. https://arxiv.org/abs/2305.10601
- Upstream code: https://github.com/princeton-nlp/tree-of-thought-llm (MIT) — prompts and dataset reproduced here under the same license.

## 9. Acknowledgements

Completed as the final project for CS 4782 (Cornell University, Spring 2026). Prompt strings and the Game-24 dataset are reproduced verbatim from the original Princeton NLP repository under MIT.
