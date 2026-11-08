"""Minimal OpenAI-compatible chat client (stdlib only) for a local server.

Every call is returned together with a trace record so that callers can
write it to JSONL. No cloud endpoint is used; the base URL defaults to a
local llama.cpp `llama-server`.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request

DEFAULT_BASE = os.environ.get("SBS_LLM_BASE", "http://127.0.0.1:18471/v1")
DEFAULT_MODEL = os.environ.get("SBS_LLM_MODEL", "qwen3.6-35b-a3b-ud-q3_k_m")

_inflight = threading.Semaphore(int(os.environ.get("SBS_MAX_INFLIGHT", "8")))


def chat(messages: list[dict], *, seed: int, temperature: float = 0.7,
         max_tokens: int = 160, base: str = DEFAULT_BASE, model: str = DEFAULT_MODEL,
         json_mode: bool = True, retries: int = 3, timeout: float = 300.0) -> dict:
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    data = json.dumps(body).encode("utf-8")
    last_err = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            with _inflight:
                req = urllib.request.Request(base + "/chat/completions", data=data,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    out = json.loads(resp.read().decode("utf-8"))
            text = out["choices"][0]["message"].get("content") or ""
            usage = out.get("usage", {})
            return {
                "ok": True,
                "text": text,
                "usage": {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens")},
                "latency_s": round(time.time() - t0, 3),
                "attempt": attempt,
                "seed": seed,
                "temperature": temperature,
                "model": model,
            }
        except Exception as e:  # network / server error: retry with backoff
            last_err = repr(e)
            time.sleep(2 * (attempt + 1))
    return {"ok": False, "text": "", "error": last_err, "usage": {}, "latency_s": None,
            "attempt": retries, "seed": seed, "temperature": temperature, "model": model}


def parse_json_obj(text: str) -> dict | None:
    """Return the first JSON object in `text`, or None."""
    text = text.strip()
    try:
        v = json.loads(text)
        return v if isinstance(v, dict) else None
    except Exception:
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for j in range(start, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        v = json.loads(text[start:j + 1])
                        if isinstance(v, dict):
                            return v
                    except Exception:
                        break
        start = text.find("{", start + 1)
    return None
