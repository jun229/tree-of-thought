"""BFS search for Tree of Thoughts.

Mirrors upstream `src/tot/methods/bfs.py` (~150 lines) but threads `gpt`
explicitly instead of mutating a module-level global, and adds support for
`method_evaluate="heuristic"` which calls `task.heuristic_value(x, y)` instead
of issuing an LLM value-prompt call (extension 2 in the project plan).
"""
from __future__ import annotations

import itertools
import re
from functools import partial

import numpy as np

from tot.models import gpt as _gpt_default


_PROPOSAL_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\d+[.)]\s*)?\**\s*"
    r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)"
    r"\s*\**\s*\(left:\s*([^)]+)\)\s*(?:\[[^\]]*\])?\s*$"
)
_ANSWER_RE = re.compile(r"^\s*(?:answer:\s*)?(.+=\s*24)\s*$", re.IGNORECASE)


def get_value(task, x, y, n_evaluate_sample, gpt_fn, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = gpt_fn(value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value


def get_values(task, x, ys, n_evaluate_sample, gpt_fn, cache_value=True):
    values = []
    local_value_cache: dict[str, float] = {}
    for y in ys:
        if y in local_value_cache:
            values.append(0)
        else:
            v = get_value(task, x, y, n_evaluate_sample, gpt_fn, cache_value=cache_value)
            local_value_cache[y] = v
            values.append(v)
    return values


def get_votes(task, x, ys, n_evaluate_sample, gpt_fn):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt_fn(vote_prompt, n=n_evaluate_sample, stop=None)
    return task.vote_outputs_unwrap(vote_outputs, len(ys))


def get_heuristic_values(task, x, ys):
    """Deterministic per-state value via task.heuristic_value (extension 2)."""
    return [task.heuristic_value(x, y) for y in ys]


def _numbers_close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


def _current_numbers(x, y):
    if not y.strip():
        return [float(n) for n in x.split()]
    last_line = y.strip().split("\n")[-1]
    if "left: " not in last_line:
        return []
    raw = last_line.split("left: ")[-1].split(")")[0]
    return [float(n) for n in raw.replace(",", " ").split()]


def _same_multiset(xs, ys):
    if len(xs) != len(ys):
        return False
    remaining = list(xs)
    for y in ys:
        for i, x in enumerate(remaining):
            if _numbers_close(x, y):
                remaining.pop(i)
                break
        else:
            return False
    return True


def _format_number(n):
    n = float(n)
    return str(int(n)) if n.is_integer() else f"{n:g}"


def _canonical_proposal_line(line, current_numbers):
    match = _PROPOSAL_RE.match(line)
    if not match:
        if len(current_numbers) == 1 and _numbers_close(current_numbers[0], 24):
            answer = _ANSWER_RE.match(line)
            if answer:
                text = answer.group(1)
                return text if text.lower().startswith("answer:") else f"Answer: {text}"
        return None

    a, op, b, result, left_raw = match.groups()
    a_f, b_f, result_f = float(a), float(b), float(result)
    if op == "+":
        expected = a_f + b_f
    elif op == "-":
        expected = a_f - b_f
    elif op == "*":
        expected = a_f * b_f
    elif op == "/" and not _numbers_close(b_f, 0):
        expected = a_f / b_f
    else:
        return None
    if not _numbers_close(expected, result_f):
        return None

    left = [float(n) for n in left_raw.replace(",", " ").split()]
    expected_left = list(current_numbers)
    for used in (a_f, b_f):
        for i, n in enumerate(expected_left):
            if _numbers_close(n, used):
                expected_left.pop(i)
                break
        else:
            return None
    expected_left.append(result_f)
    if not _same_multiset(expected_left, left):
        return None

    return (
        f"{_format_number(a_f)} {op} {_format_number(b_f)} = {_format_number(result_f)} "
        f"(left: {' '.join(_format_number(n) for n in left)})"
    )


def get_proposals(task, x, y, gpt_fn):
    propose_prompt = task.propose_prompt_wrap(x, y)
    raw = gpt_fn(propose_prompt, n=1, stop=None)[0]
    current_numbers = _current_numbers(x, y)
    lines = []
    seen = set()
    for line in raw.split("\n"):
        canonical = _canonical_proposal_line(line, current_numbers)
        if canonical and canonical not in seen:
            lines.append(y + canonical + "\n")
            seen.add(canonical)
    return lines


def get_samples(task, x, y, n_generate_sample, prompt_sample, stop, gpt_fn):
    if prompt_sample == "standard":
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == "cot":
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f"prompt_sample {prompt_sample!r} not recognized")
    samples = gpt_fn(prompt, n=n_generate_sample, stop=stop)
    return [y + s for s in samples]


def _select(values, n_select_sample, method_select):
    ids = list(range(len(values)))
    if method_select == "sample":
        total = float(sum(values))
        if total <= 0:
            ps = np.array([1.0 / len(values)] * len(values))
        else:
            ps = np.array(values) / total
        size = min(n_select_sample, len(ids))
        return np.random.choice(ids, size=size, replace=False, p=ps).tolist()
    if method_select == "greedy":
        return sorted(ids, key=lambda i: values[i], reverse=True)[:n_select_sample]
    raise ValueError(f"method_select {method_select!r} not recognized")


def _bind_gpt(args, gpt_fn=None, gpt_propose=None, gpt_evaluate=None):
    """Resolve up to two LLM callables: one for generation, one for evaluation.

    Precedence: explicit kwargs > args.backend_evaluate > args.backend.
    `gpt_fn` is a back-compat alias that sets both to the same callable.
    """
    if gpt_fn is not None:
        return gpt_fn, gpt_fn
    if gpt_propose is None:
        gpt_propose = partial(_gpt_default, model=args.backend, temperature=args.temperature)
    if gpt_evaluate is None:
        eval_backend = getattr(args, "backend_evaluate", None) or args.backend
        if eval_backend == args.backend:
            gpt_evaluate = gpt_propose
        else:
            gpt_evaluate = partial(_gpt_default, model=eval_backend, temperature=args.temperature)
    return gpt_propose, gpt_evaluate


def solve(args, task, idx, to_print=True, gpt_fn=None, gpt_propose=None, gpt_evaluate=None):
    gpt_propose, gpt_evaluate = _bind_gpt(args, gpt_fn, gpt_propose, gpt_evaluate)
    x = task.get_input(idx)
    ys = [""]
    infos = []
    for step in range(task.steps):
        # --- generation ---
        if args.method_generate == "sample":
            new_ys = [
                get_samples(
                    task, x, y,
                    args.n_generate_sample,
                    prompt_sample=args.prompt_sample,
                    stop=task.stops[step],
                    gpt_fn=gpt_propose,
                )
                for y in ys
            ]
        elif args.method_generate == "propose":
            new_ys = [get_proposals(task, x, y, gpt_propose) for y in ys]
        else:
            raise ValueError(f"method_generate {args.method_generate!r} not recognized")
        new_ys = list(itertools.chain.from_iterable(new_ys))
        if not new_ys:
            break

        # --- evaluation ---
        if args.method_evaluate == "vote":
            values = get_votes(task, x, new_ys, args.n_evaluate_sample, gpt_evaluate)
        elif args.method_evaluate == "value":
            values = get_values(task, x, new_ys, args.n_evaluate_sample, gpt_evaluate)
        elif args.method_evaluate == "heuristic":
            values = get_heuristic_values(task, x, new_ys)
        else:
            raise ValueError(f"method_evaluate {args.method_evaluate!r} not recognized")

        # --- selection ---
        select_ids = _select(values, args.n_select_sample, args.method_select)
        select_new_ys = [new_ys[i] for i in select_ids]

        if to_print:
            top = sorted(zip(new_ys, values), key=lambda p: p[1], reverse=True)[:5]
            print(f"-- step {step} (top {len(top)}) --")
            for y, v in top:
                print(f"  v={v}: {y!r}")
            print(f"-- selected --: {select_new_ys}")

        infos.append({
            "step": step,
            "x": x,
            "ys": ys,
            "new_ys": new_ys,
            "values": values,
            "select_new_ys": select_new_ys,
        })
        ys = select_new_ys

    if to_print:
        print("FINAL:", ys)
    return ys, {"steps": infos}


def naive_solve(args, task, idx, to_print=True, gpt_fn=None):
    gpt_propose, _ = _bind_gpt(args, gpt_fn)
    x = task.get_input(idx)
    ys = get_samples(
        task, x, "",
        args.n_generate_sample,
        prompt_sample=args.prompt_sample,
        stop=None,
        gpt_fn=gpt_propose,
    )
    if to_print:
        print("FINAL:", ys)
    return ys, {}
