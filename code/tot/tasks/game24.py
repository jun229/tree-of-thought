"""Game-of-24 task adapter.

The wrapper methods (`*_prompt_wrap`, `value_outputs_unwrap`, `test_output`) are
intentionally close to upstream Princeton ToT (`.ref/src/tot/tasks/game24.py`):

  * The prompts they wrap are verbatim from upstream by design (see
    `tot/prompts/game24.py` and `CLAUDE.md`).
  * `test_output` is the success metric; drift here silently changes every
    reported number. `tests/test_game24_checker.py::test_parity_with_upstream`
    runs upstream's `test_output` in a clean subprocess and requires
    byte-identical labels on a fixed set of cases.
  * The wrappers themselves are 1-4 lines of glue around the verbatim prompts.

The novel surface area in this file is `heuristic_value()`, which is a
deterministic drop-in replacement for the LLM value prompt (extension 2).
"""
import os
import re
from collections import Counter

import sympy
import pandas as pd

from tot.tasks.base import Task, DATA_PATH
from tot.prompts.game24 import (
    standard_prompt,
    cot_prompt,
    propose_prompt,
    value_prompt,
    value_last_step_prompt,
)


def get_current_numbers(y: str) -> str:
    last_line = y.strip().split("\n")[-1]
    return last_line.split("left: ")[-1].split(")")[0]


class Game24Task(Task):
    """
    Input  (x): 4 space-separated integers
    Output (y): trajectory of intermediate steps + final answer
    Reward (r): 1 if final expression uses each input exactly once and equals 24, else 0
    """

    def __init__(self, file: str = "24.csv"):
        super().__init__()
        path = os.path.join(DATA_PATH, "24", file)
        self.data = list(pd.read_csv(path)["Puzzles"])
        self.value_cache: dict[str, float] = {}
        self.steps = 4
        self.stops = ["\n"] * 4

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        return self.data[idx]

    def test_output(self, idx: int, output: str) -> dict:
        expression = (
            output.strip()
            .split("\n")[-1]
            .lower()
            .replace("answer: ", "")
            .split("=")[0]
        )
        numbers = re.findall(r"\d+", expression)
        problem_numbers = re.findall(r"\d+", self.data[idx])
        if sorted(numbers) != sorted(problem_numbers):
            return {"r": 0}
        try:
            return {"r": int(sympy.simplify(expression) == 24)}
        except Exception:
            return {"r": 0}

    @staticmethod
    def standard_prompt_wrap(x: str, y: str = "") -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y: str = "") -> str:
        return cot_prompt.format(input=x) + y

    @staticmethod
    def propose_prompt_wrap(x: str, y: str = "") -> str:
        current_numbers = get_current_numbers(y if y else x)
        if current_numbers == "24":
            return cot_prompt.format(input=x) + "Steps:" + y
        return propose_prompt.format(input=current_numbers)

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        last_line = y.strip().split("\n")[-1]
        if "left: " not in last_line:
            ans = last_line.lower().replace("answer: ", "")
            return value_last_step_prompt.format(input=x, answer=ans)
        current_numbers = get_current_numbers(y)
        return value_prompt.format(input=current_numbers)

    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        # Trajectory has 4 lines but no answer line -> malformed, score 0.
        if len(y.strip().split("\n")) == 4 and "answer" not in y.lower():
            return 0.0
        weights = {"impossible": 0.001, "likely": 1.0, "sure": 20.0}
        last_token_counts = Counter(out.split("\n")[-1] for out in value_outputs)
        return sum(w * last_token_counts.get(label, 0) for label, w in weights.items())

    # --- Extension: deterministic heuristic value (drop-in for LLM evaluator) ---

    def heuristic_value(self, x: str, y: str) -> float:
        from tot.heuristics.reach24 import is_reachable_24

        last_line = y.strip().split("\n")[-1]
        if "left: " in last_line:
            try:
                nums = [float(n) for n in get_current_numbers(y).split()]
            except ValueError:
                return 0.001
            return 20.0 if is_reachable_24(nums) else 0.001
        # final-answer line: deterministic checker
        expr = last_line.lower().replace("answer: ", "").split("=")[0]
        numbers = re.findall(r"\d+", expr)
        problem_numbers = re.findall(r"\d+", x)
        if sorted(numbers) != sorted(problem_numbers):
            return 0.001
        try:
            return 20.0 if sympy.simplify(expr) == 24 else 0.001
        except Exception:
            return 0.001
