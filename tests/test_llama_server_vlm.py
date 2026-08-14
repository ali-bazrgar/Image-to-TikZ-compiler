from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import image_to_tikz.llama_server_vlm as adapter
from image_to_tikz.llama_server_vlm import (
    LlamaServerVLMError,
    LlamaServerVisionObserver,
    build_llama_server_command,
    validate_gguf_pair,
)


def test_validate_gguf_pair_enforces_configured_and_hard_limits(tmp_path):
    model = tmp_path / "model.gguf"
    mmproj = tmp_path / "mmproj.gguf"
    model.write_bytes(b"x" * 100)
    mmproj.write_bytes(b"y" * 200)
    info = validate_gguf_pair(model, mmproj, max_bytes=500)
    assert info.total_bytes == 300

    with pytest.raises(LlamaServerVLMError, match="exceeding"):
        validate_gguf_pair(model, mmproj, max_bytes=250)


def test_build_llama_server_command_contains_mmproj_and_gpu_layers(tmp_path):
    command = build_llama_server_command(
        tmp_path / "llama-server.exe",
        tmp_path / "model.gguf",
        tmp_path / "mmproj.gguf",
    )
    assert "--mmproj" in command
    assert "-ngl" in command
    assert "99" in command


def test_llama_server_observer_posts_openai_compatible_image_message(tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    mmproj = tmp_path / "mmproj.gguf"
    image = tmp_path / "crop.png"
    model.write_bytes(b"m" * 10)
    mmproj.write_bytes(b"p" * 20)
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "Likely a connector."}}]}).encode()

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(adapter.urllib.request, "urlopen", fake_urlopen)
    observer = LlamaServerVisionObserver(model, mmproj, base_url="http://127.0.0.1:8080/v1")
    result = observer.describe(image, visual_record="line_1 is horizontal", crop_reason="arrow")

    payload = json.loads(captured["request"].data.decode())
    parts = payload["messages"][0]["content"]
    image_part = next(part for part in parts if part["type"] == "image_url")
    assert result["text"] == "Likely a connector."
    assert payload["model"] == "SmolVLM2-2.2B-Instruct"
    assert "line_1 is horizontal" in next(part["text"] for part in parts if part["type"] == "text")
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert base64.b64decode(image_part["image_url"]["url"].split(",", 1)[1]).startswith(b"\x89PNG")
