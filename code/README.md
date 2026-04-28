# Code

Re-implementation source for Tree of Thoughts.

Planned structure:
- `tot/` — core ToT library (thought generator, state evaluator, BFS/DFS search).
- `tasks/` — per-task modules: `game24/`, `crosswords/`, `writing/`.
- `prompts/` — prompt templates for proposal, value, and vote steps.
- `run.py` — entry point: `python run.py --task <task> --method {io,cot,tot} ...`
- `requirements.txt` — Python dependencies.
