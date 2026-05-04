"""Unit tests for Game24Task.test_output (the success metric).

Cross-checks our re-implementation against upstream's `test_output` on
sampled (puzzle, candidate-output) pairs to guarantee we haven't drifted
on the metric that gates every reported number.
"""
import os
import sys
from pathlib import Path

# Make `tot.*` importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Upstream clone, used by the parity test only (via subprocess).
ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_SRC = ROOT / ".ref" / "src"


from tot.tasks.game24 import Game24Task


def _ours():
    return Game24Task()


def test_accepts_known_solution():
    t = _ours()
    # idx 900 from upstream — we don't need to know the puzzle; we use
    # one of the prompt examples whose puzzle is implicitly position 0.
    # Construct a synthetic puzzle that we control.
    t.data[0] = "4 4 6 8"
    out = "Steps:\n4 + 8 = 12 (left: 4 6 12)\n6 - 4 = 2 (left: 2 12)\n2 * 12 = 24 (left: 24)\nAnswer: (4 + 8) * (6 - 4) = 24"
    assert t.test_output(0, out) == {"r": 1}


def test_rejects_wrong_result():
    t = _ours()
    t.data[0] = "4 4 6 8"
    out = "Answer: (4 + 8) * (6 - 4) + 1 = 25"
    assert t.test_output(0, out) == {"r": 0}


def test_rejects_reused_number():
    t = _ours()
    t.data[0] = "4 4 6 8"
    out = "Answer: 4 * 4 + 8 = 24"  # reuses 4, drops 6
    assert t.test_output(0, out) == {"r": 0}


def test_rejects_extra_number():
    t = _ours()
    t.data[0] = "4 4 6 8"
    out = "Answer: (4 + 8) * (6 - 4) * 1 = 24"  # extra 1
    assert t.test_output(0, out) == {"r": 0}


def test_handles_division():
    t = _ours()
    t.data[0] = "1 4 8 8"
    out = "Answer: (8 / 4 + 1) * 8 = 24"
    assert t.test_output(0, out) == {"r": 1}


def test_accepts_valid_trajectory_without_answer():
    t = _ours()
    t.data[0] = "3 4 4 13"
    out = (
        "3 + 4 = 7 (left: 4 7 13)\n"
        "13 - 7 = 6 (left: 4 6)\n"
        "4 * 6 = 24 (left: 24)\n"
    )
    assert t.test_output(0, out) == {"r": 1}


def test_accepts_valid_trajectory_before_bad_final_line():
    t = _ours()
    t.data[0] = "1 2 4 7"
    out = (
        "7 + 1 = 8 (left: 2 4 8)\n"
        "8 - 2 = 6 (left: 4 6)\n"
        "4 * 6 = 24 (left: 24)\n"
        "Answer: 3 * 9 - 3 = 24\n"
    )
    assert t.test_output(0, out) == {"r": 1}


def test_rejects_invalid_trajectory_that_claims_left_24():
    t = _ours()
    t.data[0] = "1 2 4 7"
    out = (
        "7 + 1 = 8 (left: 2 4 8)\n"
        "8 - 2 = 6 (left: 4 6)\n"
        "4 * 6 = 25 (left: 24)\n"
    )
    assert t.test_output(0, out) == {"r": 0}


def test_handles_malformed_output():
    t = _ours()
    t.data[0] = "4 4 6 8"
    assert t.test_output(0, "garbage no expression") == {"r": 0}
    assert t.test_output(0, "") == {"r": 0}


def test_parity_with_upstream():
    """If the upstream repo is available at .ref/, run identical inputs
    through *upstream's* checker and require agreement on every case.

    Naive `importlib.import_module("tot.tasks.game24")` returns the version
    already cached in sys.modules (ours, since this test file imported it at
    the top). To actually load upstream we run its checker in a clean Python
    subprocess that has only `.ref/src` on its path.
    """
    if not UPSTREAM_SRC.exists():
        return

    cases = [
        ("4 4 6 8", "Answer: (4 + 8) * (6 - 4) = 24", 1),
        ("4 4 6 8", "Answer: (4 + 8) * (6 - 4) + 1 = 25", 0),
        ("4 4 6 8", "Answer: 4 * 4 + 8 = 24", 0),
        ("4 4 6 8", "Answer: (4 + 8) * (6 - 4) * 1 = 24", 0),
        ("1 4 8 8", "Answer: (8 / 4 + 1) * 8 = 24", 1),
        ("2 9 10 12", "Answer: 2 * 12 * (10 - 9) = 24", 1),
        ("5 5 5 9", "Answer: 5 + 5 + 5 + 9 = 24", 1),
        ("4 9 10 13", "Answer: (13 - 9) * (10 - 4) = 24", 1),
        ("1 1 4 6", "Answer: 6 * (4 - (1 - 1)) = 24", 1),
    ]

    import json
    import subprocess

    # 1) Run our checker in-process.
    ours_task = _ours()
    ours_labels = []
    for puzzle, output, _ in cases:
        ours_task.data[0] = puzzle
        ours_labels.append(ours_task.test_output(0, output)["r"])

    # 2) Run upstream's checker in a clean subprocess with PYTHONPATH set to
    #    upstream's src. This guarantees `tot.tasks.game24` resolves to upstream.
    driver = (
        "import sys, json, os\n"
        "from tot.tasks.game24 import Game24Task\n"
        "cases = json.loads(sys.stdin.read())\n"
        "t = Game24Task.__new__(Game24Task)\n"
        "t.data = ['']\n"
        "out = []\n"
        "for puzzle, output in cases:\n"
        "    t.data[0] = puzzle\n"
        "    out.append(t.test_output(0, output)['r'])\n"
        "json.dump(out, sys.stdout)\n"
    )
    env = dict(**os.environ)
    env["PYTHONPATH"] = str(UPSTREAM_SRC)
    proc = subprocess.run(
        [sys.executable, "-c", driver],
        input=json.dumps([(p, o) for p, o, _ in cases]),
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert proc.returncode == 0, f"upstream subprocess failed: {proc.stderr}"
    upstream_labels = json.loads(proc.stdout)

    # 3) Both must equal the expected oracle, and each other.
    expected = [r for _, _, r in cases]
    assert ours_labels == expected, f"ours diverges from oracle: {ours_labels} vs {expected}"
    assert upstream_labels == expected, f"upstream diverges from oracle: {upstream_labels} vs {expected}"
    assert ours_labels == upstream_labels  # belt and suspenders


if __name__ == "__main__":
    test_accepts_known_solution()
    test_rejects_wrong_result()
    test_rejects_reused_number()
    test_rejects_extra_number()
    test_handles_division()
    test_handles_malformed_output()
    test_parity_with_upstream()
    print("OK")
