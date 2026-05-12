"""Backend-dispatching LLM client for Tree of Thoughts.

Mirrors upstream `src/tot/models.py` in signature so search code (`bfs.py`)
ports unchanged: `gpt(prompt, model, temperature, max_tokens, n, stop) -> list[str]`.

Backends are selected via `model="<provider>:<name>"`:
    openai:gpt-4o-mini            -> OpenAI API ($OPENAI_API_KEY)
    claude_cli:sonnet              -> shells out to `claude -p` (Pro/Max OAuth)
    claude_cli:haiku
    gemini:gemini-2.0-flash        -> Google AI Gemini API ($GEMINI_API_KEY)
    groq:llama-3.3-70b-versatile   -> Groq free tier ($GROQ_API_KEY)
    openrouter:<slug>              -> OpenRouter ($OPENROUTER_API_KEY)
    hf:<model-id>                  -> HuggingFace Transformers local inference (Colab GPU)
    hf:Qwen/Qwen3.5-2B-Instruct
    hf:google/gemma-3-4b-it:4bit  -> append :4bit for 4-bit quantization (needs bitsandbytes)

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
    max_tokens: int = 200,
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
    elif provider == "openai":
        outputs, usage = _call_openai(prompt, model_name, temperature, max_tokens, n, stop)
    elif provider == "gemini":
        outputs, usage = _call_gemini(prompt, model_name, temperature, max_tokens, n, stop)
    elif provider == "groq":
        outputs, usage = _call_groq(prompt, model_name, temperature, max_tokens, n, stop)
    elif provider == "openrouter":
        outputs, usage = _call_openrouter(prompt, model_name, temperature, max_tokens, n, stop)
    elif provider in ("hf", "gemma3"):
        outputs, usage = _call_gemma3(prompt, model_name, temperature, max_tokens, n, stop)
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

# Strip per-machine context from the system prompt for cache reuse + lower
# overhead. We can't pass --bare (it disables OAuth), so we replace the
# default system prompt with a tiny format-discipline instruction.
#
# WHY THE NON-EMPTY SYSTEM PROMPT: claude -p has no --max-tokens flag, and
# its default max output is 32K tokens. With an empty system prompt, haiku
# was generating 25-30K-token responses to "Possible next steps:" prompts
# (hundreds of items instead of ~8). The few-shot examples in the prompt
# specify the format, but haiku ignored their length without explicit
# instruction. This cuts per-call latency by ~5-10x without affecting
# correctness (downstream code already truncates at \n boundaries).
#
# Bumping this string changes output behavior — but the on-disk cache key
# does not include the system prompt, so old (verbose) and new (concise)
# entries can coexist for the same prompt. Treat verbose-era entries as
# stale; new conditions automatically use the concise prompt going forward.
_CLAUDE_SYSTEM_PROMPT = (
    "Match the format and length of the few-shot examples exactly. "
    "Do not add commentary, explanations, or items beyond what the examples show."
)
# `--settings '{"reasoning":false}'` partially disables extended thinking
# on reasoning-capable models (haiku-4.5, sonnet-4.6). Empirically (probe at
# /tmp/test_thinking_disable.py — see CLAUDE.md "Findings" section): drops
# output_tokens 56% and per-call latency 64% vs baseline on haiku. Other
# candidates we tried (`thinkingBudget:0`, `extendedThinking:false`) were
# silently ignored. `thinking:{type:"disabled"}` also works but `reasoning`
# is the cleanest result.
_CLAUDE_FLAGS = [
    "-p",
    "--output-format", "json",
    "--system-prompt", _CLAUDE_SYSTEM_PROMPT,
    "--settings", '{"reasoning":false}',
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
# Backend: OpenAI (official Python SDK)
# ---------------------------------------------------------------------------

def _call_openai(prompt, model, temperature, max_tokens, n, stop):
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
        stop=stop or None,
    )
    out = [c.message.content or "" for c in resp.choices]
    usage = {
        "input_tokens": getattr(resp.usage, "prompt_tokens", 0) if resp.usage else 0,
        "output_tokens": getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0,
        "cost_usd": 0.0,
    }
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


# ---------------------------------------------------------------------------
# Backend: Gemma 3 via HuggingFace Transformers (Colab / local GPU)
# ---------------------------------------------------------------------------
# Model name format: "google/gemma-3-4b-it" or "google/gemma-3-12b-it:4bit"
# The ":4bit" suffix enables 4-bit quantization (requires bitsandbytes).
# Recommended Colab pairings:
#   T4 (16 GB)  -> google/gemma-3-4b-it (bfloat16, ~8 GB)
#   T4 (16 GB)  -> google/gemma-3-12b-it:4bit (~6 GB)
#   A100 (40 GB)-> google/gemma-3-12b-it or google/gemma-3-27b-it:4bit

_hf_model = None
_hf_tokenizer = None
_hf_model_id: str | None = None


def _get_hf_model(model_id: str, load_in_4bit: bool = False):
    global _hf_model, _hf_tokenizer, _hf_model_id
    tag = model_id + (":4bit" if load_in_4bit else "")
    if _hf_model_id != tag:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        load_kwargs: dict = {"device_map": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        model.eval()

        _hf_tokenizer = tokenizer
        _hf_model = model
        _hf_model_id = tag

    return _hf_model, _hf_tokenizer


def _call_gemma3(prompt: str, model: str, temperature: float, max_tokens: int, n: int, stop):
    import torch

    load_in_4bit = model.endswith(":4bit")
    model_id = model.removesuffix(":4bit")

    hf_model, tokenizer = _get_hf_model(model_id, load_in_4bit=load_in_4bit)

    messages = [
        {"role": "system", "content": "Follow the exact output format shown in the examples. Output only the required numbers and equations, no other text. If there is only one number left, write the Answer line and stop."},
        {"role": "user", "content": prompt},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(hf_model.device)
    input_len = inputs["input_ids"].shape[1]

    stop_seqs = ([stop] if isinstance(stop, str) else list(stop)) if stop else []

    gen_kwargs: dict = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
        "temperature": max(temperature, 1e-4),
        "pad_token_id": tokenizer.eos_token_id,
    }

    use_batch = "qwen" in model_id.lower()
    if use_batch and n > 1:
        gen_kwargs["num_return_sequences"] = n

    out: list[str] = []
    total_output_tokens = 0
    with torch.no_grad():
        if use_batch:
            output_ids = hf_model.generate(**inputs, **gen_kwargs)
            for seq in output_ids:
                new_ids = seq[input_len:]
                text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                for s in stop_seqs:
                    if s in text:
                        text = text[:text.index(s)]
                out.append(text)
                total_output_tokens += len(new_ids)
        else:
            # Generate one at a time to avoid OOM on models with large VRAM footprint
            for _ in range(n):
                output_ids = hf_model.generate(**inputs, **gen_kwargs)
                new_ids = output_ids[0][input_len:]
                text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                for s in stop_seqs:
                    if s in text:
                        text = text[:text.index(s)]
                out.append(text)
                total_output_tokens += len(new_ids)

    usage = {
        "input_tokens": input_len * n,
        "output_tokens": total_output_tokens,
        "cost_usd": 0.0,
    }
    return out, usage
