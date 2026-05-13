"""Tests for tot/scoring/lenient.py.

These pin the lenient parser against the real failure modes observed in the
haiku IO n=1 cached outputs. If any of these regresses, the IO n=1 corrected
accuracy number on the poster will move silently — that is the whole reason
the strict parser is staying as-is.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tot.scoring.lenient import lenient_score


def test_answer_line_with_trailing_verification():
    """The canonical haiku IO failure mode."""
    out = (
        "Answer: 6 * 5 - 10 + 4 = 24\n"
        "\n"
        "**Verification:**\n"
        "- 6 × 5 = 30\n"
        "- 30 - 10 = 20\n"
        "- 20 + 4 = 24 ✓"
    )
    assert lenient_score("4 5 6 10", out) == 1


def test_bold_wrapped_equation_no_answer_prefix():
    """Puzzle 919 in the cached IO n=1 batch — no `Answer:` keyword at all."""
    out = (
        "Looking at the numbers 3, 3, 6, 7, I need a combination that equals 24.\n"
        "\n"
        "**3 * 7 + 6 - 3 = 24**\n"
        "\n"
        "Verification:\n"
        "- 3 * 7 = 21\n"
        "- 21 + 6 = 27\n"
        "- 27 - 3 = 24 ✓"
    )
    assert lenient_score("3 3 6 7", out) == 1


def test_concise_single_line_answer():
    """The format-discipline happy path; strict and lenient must agree."""
    assert lenient_score("4 5 6 10", "Answer: 4 * 5 + 10 - 6 = 24") == 1


def test_wrong_value_rejected():
    """Lenient must NOT accept an equation that doesn't equal 24."""
    assert lenient_score("4 5 6 10", "Answer: 4 + 5 + 6 + 10 = 25") == 0


def test_wrong_multiset_rejected():
    """Lenient must reject equations that don't use each input number once."""
    # uses 5, 6, 10 but missing 4 and re-uses 6
    assert lenient_score("4 5 6 10", "Answer: 6 * 6 - 10 - 2 = 24") == 0


def test_unicode_operators():
    """Haiku occasionally uses × and ÷; the parser should normalize them."""
    assert lenient_score("4 5 6 10", "**6 × 5 - 10 + 4 = 24**") == 1


def test_empty_output():
    assert lenient_score("4 5 6 10", "") == 0


def test_multiple_candidates_first_correct_wins():
    """If the output offers two equations and one is correct, accept it."""
    out = "Answer: 4 + 5 + 6 + 10 = 25\nOr: 4 * 5 + 10 - 6 = 24"
    assert lenient_score("4 5 6 10", out) == 1
