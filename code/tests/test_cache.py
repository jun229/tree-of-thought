"""Cache correctness: identical re-runs are byte-identical and zero-call."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_cache_roundtrip(monkeypatch=None):
    with tempfile.TemporaryDirectory() as td:
        os.environ["TOT_CACHE_DIR"] = td

        # Re-import after env change so cache picks up new dir.
        import importlib
        from tot.utils import cache as cache_mod
        importlib.reload(cache_mod)

        key = cache_mod.make_key("test:model", "hello", 0.7, 1, None)
        assert cache_mod.get(key) is None
        cache_mod.put(key, {"outputs": ["hi"], "usage": {}})
        got = cache_mod.get(key)
        assert got["outputs"] == ["hi"]
        # Different params -> different key.
        key2 = cache_mod.make_key("test:model", "hello", 0.8, 1, None)
        assert key != key2
        assert cache_mod.get(key2) is None


def test_cache_skips_real_call():
    """Hot path: when cache has an entry, gpt() must not invoke the backend."""
    with tempfile.TemporaryDirectory() as td:
        os.environ["TOT_CACHE_DIR"] = td

        import importlib
        from tot.utils import cache as cache_mod
        importlib.reload(cache_mod)
        from tot import models as models_mod
        importlib.reload(models_mod)

        # Pre-populate cache with the exact key gpt() will compute.
        key = cache_mod.make_key("groq:fake", "ping", 0.0, 1, None)
        cache_mod.put(key, {"outputs": ["pong"], "usage": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}})

        # If gpt() ignored the cache it would try to import groq + read GROQ_API_KEY.
        out = models_mod.gpt("ping", model="groq:fake", temperature=0.0, n=1, stop=None)
        assert out == ["pong"]


if __name__ == "__main__":
    test_cache_roundtrip()
    test_cache_skips_real_call()
    print("OK")
