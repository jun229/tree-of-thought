"""BFS smoke test with a deterministic mocked LLM.

Verifies tree-shape invariants without spending any real LLM quota:
- propose+greedy at b=5 expands to ~exactly 5 candidates per step (depending
  on how many lines the mocked propose returns)
- naive_solve produces n_generate_sample outputs for IO/CoT
- heuristic evaluator path exercises task.heuristic_value (no LLM at all)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tot.tasks.game24 import Game24Task
from tot.search import bfs as bfs_mod


class Args:
    pass


def _make_args(**overrides):
    a = Args()
    a.backend = "mock:dummy"
    a.temperature = 0.7
    a.task = "game24"
    a.task_start_index = 0
    a.task_end_index = 1
    a.naive_run = False
    a.prompt_sample = "cot"
    a.method_generate = "propose"
    a.method_evaluate = "heuristic"  # avoids LLM call entirely
    a.method_select = "greedy"
    a.algo = "bfs"
    a.n_generate_sample = 1
    a.n_evaluate_sample = 1
    a.n_select_sample = 5
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


def _mock_gpt_factory(propose_lines: int = 8):
    """Returns a callable matching gpt(prompt, n=..., stop=...)."""
    def _mock(prompt, n=1, stop=None, **kwargs):
        # For propose prompts, return `propose_lines` plausible "X op Y = Z (left: ...)" lines.
        if "Possible next steps" in prompt:
            current = prompt.strip().split("Input: ")[-1].split("\n", 1)[0]
            lines_by_input = {
                "1 1 4 6": [
                    "1 + 1 = 2 (left: 2 4 6)",
                    "1 + 4 = 5 (left: 1 5 6)",
                    "1 + 6 = 7 (left: 1 4 7)",
                    "4 + 6 = 10 (left: 1 1 10)",
                    "4 - 1 = 3 (left: 1 3 6)",
                    "6 - 1 = 5 (left: 1 4 5)",
                    "6 - 4 = 2 (left: 1 1 2)",
                    "4 * 6 = 24 (left: 1 1 24)",
                ],
                "2 4 6": [
                    "2 + 4 = 6 (left: 6 6)",
                    "2 * 6 = 12 (left: 4 12)",
                ],
                "1 5 6": [
                    "1 + 5 = 6 (left: 6 6)",
                    "5 + 6 = 11 (left: 1 11)",
                ],
                "1 4 7": [
                    "1 + 7 = 8 (left: 4 8)",
                    "4 * 7 = 28 (left: 1 28)",
                ],
                "1 1 10": [
                    "1 + 1 = 2 (left: 2 10)",
                    "10 - 1 = 9 (left: 1 9)",
                ],
                "1 3 6": [
                    "1 + 3 = 4 (left: 4 6)",
                    "3 * 6 = 18 (left: 1 18)",
                ],
            }
            lines = lines_by_input.get(current, [])[:propose_lines]
            return ["\n".join(lines)]
        # For sample prompts, return a single fake answer trajectory.
        return [f"\nAnswer: {i+1} = 24" for i in range(n)]
    return _mock


def test_bfs_propose_greedy_b5_expands_to_b_per_step():
    task = Game24Task()
    task.data[0] = "1 1 4 6"
    args = _make_args(n_select_sample=5)
    gpt_fn = _mock_gpt_factory(propose_lines=8)
    ys, info = bfs_mod.solve(args, task, 0, to_print=False, gpt_fn=gpt_fn)
    # After step 0: ys had 1 entry, propose returned 8 -> new_ys=8 -> select 5
    step0 = info["steps"][0]
    assert len(step0["new_ys"]) == 8
    assert len(step0["select_new_ys"]) == 5


def test_bfs_propose_greedy_b1_keeps_single_path():
    task = Game24Task()
    task.data[0] = "1 1 4 6"
    args = _make_args(n_select_sample=1)
    gpt_fn = _mock_gpt_factory(propose_lines=4)
    ys, info = bfs_mod.solve(args, task, 0, to_print=False, gpt_fn=gpt_fn)
    for step in info["steps"]:
        assert len(step["select_new_ys"]) == 1


def test_sample_selection_without_replacement():
    ids = bfs_mod._select([1, 1, 1], n_select_sample=5, method_select="sample")
    assert sorted(ids) == [0, 1, 2]


def test_naive_solve_produces_n_samples():
    task = Game24Task()
    task.data[0] = "1 1 4 6"
    args = _make_args(naive_run=True, n_generate_sample=7, prompt_sample="cot")
    gpt_fn = _mock_gpt_factory()
    ys, _ = bfs_mod.naive_solve(args, task, 0, to_print=False, gpt_fn=gpt_fn)
    assert len(ys) == 7


if __name__ == "__main__":
    test_bfs_propose_greedy_b5_expands_to_b_per_step()
    test_bfs_propose_greedy_b1_keeps_single_path()
    test_naive_solve_produces_n_samples()
    print("OK")
