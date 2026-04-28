"""Backend-dispatching LLM client for Tree of Thoughts.

Mirrors upstream `src/tot/models.py` in signature so search code (`bfs.py`)
ports unchanged: `gpt(prompt, model, temperature, max_tokens, n, stop) -> list[str]`.

Backends are selected via `model="<provider>:<name>"`:
    claude_cli:sonnet              -> shells out to `claude -p` (Pro/Max OAuth)
    claude_cli:haiku
    gemini:gemini-2.0-flash        -> Google AI Gemini API ($GEMINI_API_KEY)
    groq:llama-3.3-70b-versatile   -> Groq free tier ($GROQ_API_KEY)
    openrouter:<slug>              -> OpenRouter ($OPENROUTER_API_KEY)

A bare `model="sonnet"` defaults to claude_cli.

All calls go through an on-disk cache keyed by full request. Cached results
do NOT count toward `gpt_usage()` so re-runs are free.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any

from tot.utils import cache as _cache

# --- module-level usage counters (mirrors upstream API) ---
completion_tokens: int = 0
prompt_tokens: int = 0
cached_tokens: int = 0
total_cost_usd: float = 0.0


def reset_usage() -> None:
    global completion_tokens, prompt_tokens, cached_tokens, total_cost_usd
    completion_tokens = 0
    prompt_tokens = 0
    cached_tokens = 0
    total_cost_usd = 0.0


def gpt_usage(backend: str | None = None) -> dict[str, Any]:
    return {
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "cost": total_cost_usd,
    }


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def gpt(
    prompt: str,
    model: str = "claude_cli:sonnet",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    n: int = 1,
    stop: Any = None,
) -> list[str]:
    """Return n string completions. Cached by request signature."""
    global completion_tokens, prompt_tokens, total_cost_usd

    provider, model_name = _split_model(model)
    key = _cache.make_key(model, prompt, temperature, n, stop)
    cached = _cache.get(key)
    if cached is not None:
        return cached["outputs"]

    if provider == "claude_cli":
        outputs, usage = _call_claude_cli(prompt, model_name, n, max_tokens)
    elif provider == "gemini":
        outputs, usage = _call_gemini(prompt, model_name, temperature, max_tokens, n, stop)
    elif provider == "groq":
        outputs, usage = _call_groq(prompt, model_name, temperature, max_tokens, n, stop)
    elif provider == "openrouter":
        outputs, usage = _call_openrouter(prompt, model_name, temperature, max_tokens, n, stop)
    else:
        raise ValueError(f"Unknown provider {provider!r} (model={model!r})")

    completion_tokens += usage.get("output_tokens", 0)
    prompt_tokens += usage.get("input_tokens", 0)
    total_cost_usd += usage.get("cost_usd", 0.0)

    _cache.put(key, {"outputs": outputs, "usage": usage})
    return outputs


def _split_model(model: str) -> tuple[str, str]:
    if ":" in model:
        provider, name = model.split(":", 1)
        return provider, name
    # bare alias -> claude_cli (since that's the default backend in this project)
    return "claude_cli", model


# ---------------------------------------------------------------------------
# Backend: claude -p subprocess (Pro/Max OAuth, no API key)
# ---------------------------------------------------------------------------

# Strip per-machine context from system prompt for cache reuse + lower overhead.
# We can't pass --bare (it disables OAuth), so we do the next best thing:
# replace the default system prompt with empty and disable tools/slash-commands.
_CLAUDE_FLAGS = [
    "-p",
    "--output-format", "json",
    "--system-prompt", "",
    "--tools", "",
    "--no-session-persistence",
    "--disable-slash-commands",
    "--exclude-dynamic-system-prompt-sections",
]


def _call_claude_cli(
    prompt: str, model: str, n: int, max_tokens: int, timeout: int = 240, max_retries: int = 3
) -> tuple[list[str], dict]:
    out: list[str] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    for _ in range(n):
        for attempt in range(max_retries):
            try:
                proc = subprocess.run(
                    ["claude", *_CLAUDE_FLAGS, "--model", model],
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"claude -p exit {proc.returncode}: {proc.stderr[:300]}"
                    )
                data = json.loads(proc.stdout)
                if data.get("is_error"):
                    raise RuntimeError(f"claude -p reported error: {data.get('result', '')[:300]}")
                out.append(data.get("result", ""))
                u = data.get("usage", {}) or {}
                usage["input_tokens"] += int(u.get("input_tokens", 0)) + int(u.get("cache_creation_input_tokens", 0))
                usage["output_tokens"] += int(u.get("output_tokens", 0))
                usage["cost_usd"] += float(data.get("total_cost_usd", 0.0))
                break
            except (subprocess.TimeoutExpired, json.JSONDecodeError, RuntimeError) as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
    return out, usage


# ---------------------------------------------------------------------------
# Backend: Gemini (google-genai)
# ---------------------------------------------------------------------------

def _call_gemini(prompt, model, temperature, max_tokens, n, stop):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    out: list[str] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    stop_seqs = [stop] if isinstance(stop, str) else (list(stop) if stop else None)
    cfg = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        stop_sequences=stop_seqs,
    )
    # google-genai returns one candidate per call; loop n times for sample diversity.
    for _ in range(n):
        resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
        out.append(getattr(resp, "text", "") or "")
        meta = getattr(resp, "usage_metadata", None)
        if meta is not None:
            usage["input_tokens"] += int(getattr(meta, "prompt_token_count", 0) or 0)
            usage["output_tokens"] += int(getattr(meta, "candidates_token_count", 0) or 0)
    return out, usage


# ---------------------------------------------------------------------------
# Backend: Groq (free tier)
# ---------------------------------------------------------------------------

def _call_groq(prompt, model, temperature, max_tokens, n, stop):
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    # Groq supports n natively; but free-tier RPM is tight, so we still batch
    # only when n>1 in a single request.
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
        stop=stop if not isinstance(stop, list) or stop else (stop or None),
    )
    out = [c.message.content or "" for c in resp.choices]
    usage = {
        "input_tokens": getattr(resp.usage, "prompt_tokens", 0) if resp.usage else 0,
        "output_tokens": getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0,
        "cost_usd": 0.0,
    }
    return out, usage


# ---------------------------------------------------------------------------
# Backend: OpenRouter (HTTP)
# ---------------------------------------------------------------------------

def _call_openrouter(prompt, model, temperature, max_tokens, n, stop):
    import requests

    api_key = os.environ["OPENROUTER_API_KEY"]
    url = "https://openrouter.ai/api/v1/chat/completions"
    out: list[str] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/cs4782",
        "X-Title": "tree-of-thought-cs4782",
    }
    for _ in range(n):
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            body["stop"] = stop
        for attempt in range(3):
            r = requests.post(url, json=body, headers=headers, timeout=120)
            if r.status_code in (429, 502, 503):
                time.sleep(2 ** attempt + 1)
                continue
            r.raise_for_status()
            break
        else:
            r.raise_for_status()
        data = r.json()
        out.append(data["choices"][0]["message"]["content"])
        u = data.get("usage", {}) or {}
        usage["input_tokens"] += int(u.get("prompt_tokens", 0))
        usage["output_tokens"] += int(u.get("completion_tokens", 0))
    return out, usage
