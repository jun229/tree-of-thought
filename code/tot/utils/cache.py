"""Simple on-disk cache for LLM prompt -> response.

Keyed by sha256 of (backend, prompt, temperature, n, stop). Each entry is a
JSON file containing {"outputs": [...], "usage": {...}}. The cache is a pure
correctness improvement: identical re-runs return identical bytes and make
zero backend calls. This is critical because Game-24 ToT runs are long and
re-running for analysis would otherwise re-bill subscription quota.
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Any

CACHE_DIR = Path(os.environ.get("TOT_CACHE_DIR", ".cache/llm"))


def _ensure_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def make_key(backend: str, prompt: str, temperature: float, n: int, stop) -> str:
    payload = json.dumps(
        {"backend": backend, "prompt": prompt, "temperature": temperature, "n": n, "stop": stop},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get(key: str) -> dict | None:
    p = _ensure_dir() / f"{key}.json"
    if not p.exists():
        return None
    try:
        with p.open() as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def put(key: str, value: dict) -> None:
    p = _ensure_dir() / f"{key}.json"
    tmp = p.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(value, f)
    os.replace(tmp, p)


def stats() -> dict[str, Any]:
    if not CACHE_DIR.exists():
        return {"entries": 0, "bytes": 0}
    entries = list(CACHE_DIR.glob("*.json"))
    return {"entries": len(entries), "bytes": sum(p.stat().st_size for p in entries)}
