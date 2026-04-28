"""Unit tests for the deterministic reach-24 heuristic."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tot.heuristics.reach24 import is_reachable_24


def test_classic_reachable_puzzles():
    # All from the upstream 24.csv: known to be solvable.
    assert is_reachable_24([4, 4, 6, 8])  # (4 + 8) * (6 - 4) = 24
    assert is_reachable_24([2, 9, 10, 12])  # 2 * 12 * (10 - 9)
    assert is_reachable_24([4, 9, 10, 13])  # (13 - 9) * (10 - 4)
    assert is_reachable_24([1, 4, 8, 8])    # (8 / 4 + 1) * 8
    assert is_reachable_24([5, 5, 5, 9])    # 5 + 5 + 5 + 9
    assert is_reachable_24([1, 1, 4, 6])    # 6 * (4 - (1 - 1))


def test_unreachable():
    assert not is_reachable_24([1, 1, 1, 1])
    assert not is_reachable_24([1, 1, 1, 2])
    assert not is_reachable_24([])


def test_intermediate_states():
    # Two-number subgoals from a partial trajectory.
    assert is_reachable_24([10, 14])      # 10 + 14
    assert not is_reachable_24([11, 12])  # 11+12=23, 11*12=132, etc.
    assert is_reachable_24([3, 8])        # 3 * 8
    assert is_reachable_24([24])          # singleton match
    assert not is_reachable_24([23])


def test_division_safe():
    # Zero divisors are skipped silently (not raised). And [0, 0, 24, 1] IS
    # reachable: 24 + 0 + 0 + 1 - 1 = 24, more directly 24 * 1 + 0 + 0 = 24.
    assert is_reachable_24([0, 0, 24, 1]) is True


if __name__ == "__main__":
    test_classic_reachable_puzzles()
    test_unreachable()
    test_intermediate_states()
    test_division_safe()
    print("OK")
