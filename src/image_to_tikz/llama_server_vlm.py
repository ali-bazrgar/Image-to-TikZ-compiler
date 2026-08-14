from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_ALLOWED_MODEL_BYTES = 3_000_000_000
RECOMMENDED_MAX_MODEL_BYTES = 2_500_000_000


class LlamaServerVLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class GGUFModelInfo:
    model_path: str
    mmproj_path: str
    total_bytes: int


def validate_gguf_pair(
    model_path: str | Path,
    mmproj_path: str | Path,
    *,
    max_bytes: int = MAX_ALLOWED_MODEL_BYTES,
) -> GGUFModelInfo:
    model = Path(model_path)
    mmproj = Path(mmproj_path)
    for path in (model, mmproj):
        if not path.is_file():
            raise LlamaServerVLMError(f"GGUF file does not exist: {path}")
        if path.suffix.lower() != ".gguf":
            raise LlamaServerVLMError(f"Expected a .gguf file: {path}")
    total = model.stat().st_size + mmproj.stat().st_size
    if total > min(max_bytes, MAX_ALLOWED_MODEL_BYTES):
        raise LlamaServerVLMError(
            f"GGUF model + mmproj total {total} bytes, exceeding the configured ceiling {max_bytes} bytes"
        )
    return GGUFModelInfo(str(model), str(mmproj), total)


def _data_uri(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/webp"
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


class LlamaServerVisionObserver:
    """Semantic observer for a running llama.cpp multimodal OpenAI-compatible server."""

    def __init__(
        self,
        model_path: str | Path,
        mmproj_path: str | Path,
        *,
        base_url: str = "http://127.0.0.1:8080/v1",
        model_name: str = "SmolVLM2-2.2B-Instruct",
        timeout: float = 120.0,
        max_new_tokens: int = 180,
        max_model_bytes: int = RECOMMENDED_MAX_MODEL_BYTES,
    ) -> None:
        self.info = validate_gguf_pair(model_path, mmproj_path, max_bytes=max_model_bytes)
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.max_new_tokens = max_new_tokens

    def describe(
        self,
        image_path: str | Path,
        *,
        visual_record: str = "",
        crop_reason: str = "whole_image",
    ) -> dict[str, Any]:
        prompt = (
            "Analyze this scientific or technical diagram region. Give a concise semantic hypothesis "
            "about what it shows, visible symbols, labels, local relationships, and likely role in the "
            "whole figure. Do not invent exact coordinates. This is a semantic hint only. "
            f"Crop reason: {crop_reason}.\n\n"
            "Deterministic visual evidence:\n"
            + visual_record[:12000]
        )
        body = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": _data_uri(image_path)}},
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": self.max_new_tokens,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise LlamaServerVLMError(
                f"Could not reach llama.cpp vision server at {self.base_url}: {exc}"
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise LlamaServerVLMError(f"Invalid response from llama.cpp vision server: {exc}") from exc

        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LlamaServerVLMError(f"Unexpected llama.cpp response schema: {payload!r}") from exc
        if not isinstance(text, str) or not text.strip():
            raise LlamaServerVLMError("llama.cpp vision server returned an empty semantic response")
        return {"text": text.strip(), "model": self.model_name, "weight_bytes": self.info.total_bytes}


def build_llama_server_command(
    server_executable: str | Path,
    model_path: str | Path,
    mmproj_path: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    context: int = 4096,
    gpu_layers: int = 99,
) -> list[str]:
    """Build a conservative llama.cpp server command for SmolVLM2 GGUF."""
    return [
        str(server_executable),
        "-m",
        str(model_path),
        "--mmproj",
        str(mmproj_path),
        "--host",
        host,
        "--port",
        str(port),
        "-c",
        str(context),
        "-ngl",
        str(gpu_layers),
    ]
