"""Deterministic reach-24 reachability check.

Drop-in replacement for the LLM `value_prompt` in Game-24 ToT (extension 2).
Uses exact rational arithmetic via fractions.Fraction so floating-point noise
never causes a false negative on integer-reachable targets like 24.

Time complexity: at most O((n choose 2) * 6 * recurse(n-1)) where 6 is the
fanout of {a+b, a-b, b-a, a*b, a/b, b/a} per pair. For n=4 the search is
trivially fast (sub-millisecond per puzzle in practice).
"""
from fractions import Fraction
from itertools import combinations
from typing import Iterable


def is_reachable_24(nums: Iterable[float], target: int = 24) -> bool:
    """Return True iff `nums` can be combined with +, -, *, / to reach `target`.

    Each number must be used exactly once. Division by zero is skipped (not an
    error). Integer/fractional reachability are both detected since we operate
    in Fraction.
    """
    nums = [Fraction(n).limit_denominator(10**12) for n in nums]
    if not nums:
        return False
    return _reach(nums, Fraction(target))


def _reach(nums: list[Fraction], target: Fraction) -> bool:
    if len(nums) == 1:
        return nums[0] == target
    for i, j in combinations(range(len(nums)), 2):
        a, b = nums[i], nums[j]
        rest = [n for k, n in enumerate(nums) if k != i and k != j]
        candidates = [a + b, a - b, b - a, a * b]
        if b != 0:
            candidates.append(a / b)
        if a != 0:
            candidates.append(b / a)
        for c in candidates:
            if _reach(rest + [c], target):
                return True
    return False
