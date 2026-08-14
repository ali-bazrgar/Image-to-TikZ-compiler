from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class DownstreamLLMError(RuntimeError):
    pass


def chat_completion(
    endpoint: str,
    model: str,
    prompt: str,
    *,
    api_key: str = "",
    temperature: float = 0.0,
    max_tokens: int = 8192,
    timeout: float = 180.0,
) -> str:
    """Call an OpenAI-compatible downstream text/code LLM."""
    url = endpoint.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise DownstreamLLMError(f"LLM HTTP {exc.code}: {detail[:4000]}") from exc
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise DownstreamLLMError(f"Could not call downstream LLM: {exc}") from exc
    try:
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise DownstreamLLMError(f"Unexpected LLM response schema: {body!r}") from exc
    if not isinstance(text, str) or not text.strip():
        raise DownstreamLLMError("Downstream LLM returned empty content")
    return text.strip()
