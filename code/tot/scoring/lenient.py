"""Lenient re-scorer for Game-of-24 model outputs.

Strict `Game24Task.test_output` (verbatim from upstream Princeton ToT) only
inspects the LAST line of the model output. On chatty modern models — e.g.
Claude haiku-4.5 — the correct equation is on an `Answer:` line that is
followed by a `**Verification:**` markdown block. The strict parser then
sees the last verification step (e.g. `- 20 + 4 = 24 ✓`) and rejects it
because the multiset of numbers doesn't match the puzzle input. The
upstream parser is preserved as-is (so `test_parity_with_upstream` stays
passing); this module provides a post-hoc, more permissive re-scorer that
analyze.py applies on top of the cached JSONL.

The lenient scorer accepts an output as correct iff ANY substring of the
form `<expr> = <value>` in the output satisfies:
  - sorted re.findall(r"\\d+", expr) == sorted(puzzle's input numbers)
  - sympy.simplify(expr) == 24

That single rule covers the three failure modes seen in the haiku IO n=1
data: `Answer:` line followed by verification, `**bold**`-wrapped answers
without an `Answer:` prefix, and "Or alternatively: ..." second answers.
It never accepts an answer that doesn't actually evaluate to 24, so the
lenient number is a valid alternative-scoring of the same data, not a
relaxation of the success criterion.
"""
from __future__ import annotations

import re
from typing import Iterable

import sympy


# Equation form within a single line: "<expr> = 24" (or "= 24.0", etc).
# Captures the LHS. LHS char class includes markdown bold markers (*) and
# unicode operators (× ÷); _normalize strips the markdown noise before sympy
# sees it. The class deliberately excludes newlines so the LHS can't span
# multiple lines (otherwise a stray "24." in surrounding prose pulls the
# capture earlier than the actual answer).
_EQ_RE = re.compile(r"([0-9 \t\+\-\*\/\(\)\.×÷]+?)\s*=\s*(-?\d+(?:\.\d+)?)")
_MARKDOWN_BOLD_RE = re.compile(r"\*{2,}")


def _normalize(expr: str) -> str:
    expr = _MARKDOWN_BOLD_RE.sub("", expr)
    return expr.replace("×", "*").replace("÷", "/").strip(" \t.-")


def lenient_score(problem_input: str, output: str) -> int:
    """Return 1 if `output` contains an equation on any single line that
    uses each problem number exactly once and evaluates to 24; else 0."""
    if not output:
        return 0
    problem_numbers = sorted(re.findall(r"\d+", problem_input))
    for line in output.split("\n"):
        for match in _EQ_RE.finditer(line):
            lhs = _normalize(match.group(1))
            if not lhs:
                continue
            nums = sorted(re.findall(r"\d+", lhs))
            if nums != problem_numbers:
                continue
            try:
                if sympy.simplify(lhs) == 24:
                    return 1
            except Exception:
                continue
    return 0


def lenient_score_many(problem_input: str, outputs: Iterable[str]) -> list[int]:
    return [lenient_score(problem_input, y) for y in outputs]
