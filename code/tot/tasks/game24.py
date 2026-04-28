import os
import re
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
        if len(y.strip().split("\n")) == 4 and "answer" not in y.lower():
            return 0
        value_names = [_.split("\n")[-1] for _ in value_outputs]
        value_map = {"impossible": 0.001, "likely": 1, "sure": 20}
        return sum(v * value_names.count(name) for name, v in value_map.items())

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
