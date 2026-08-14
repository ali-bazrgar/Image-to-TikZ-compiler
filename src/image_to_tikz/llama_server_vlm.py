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


def validate_gguf_pair(model_path: str | Path, mmproj_path: str | Path, *, max_bytes: int = MAX_ALLOWED_MODEL_BYTES) -> GGUFModelInfo:
    model = Path(model_path)
    mmproj = Path(mmproj_path)
    for path in (model, mmproj):
        if not path.is_file():
            raise LlamaServerVLMError(f"GGUF file does not exist: {path}")
        if path.suffix.lower() != ".gguf":
            raise LlamaServerVLMError(f"Expected a .gguf file: {path}")
    total = model.stat().st_size + mmproj.stat().st_size
    ceiling = min(int(max_bytes), MAX_ALLOWED_MODEL_BYTES)
    if total > ceiling:
        raise LlamaServerVLMError(f"GGUF model + mmproj total {total} bytes, exceeding the configured ceiling {ceiling} bytes")
    return GGUFModelInfo(str(model), str(mmproj), total)


def _data_uri(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/webp"
    return f"data:{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode('ascii')}"


def _get_json(url: str, *, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise LlamaServerVLMError(f"llama.cpp returned HTTP {exc.code} for {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LlamaServerVLMError(f"Could not reach llama.cpp server at {url}: {exc}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise LlamaServerVLMError(f"Invalid response from llama.cpp server at {url}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LlamaServerVLMError(f"Unexpected JSON response from {url}: {payload!r}")
    return payload


class LlamaServerVisionObserver:
    def __init__(self, model_path: str | Path, mmproj_path: str | Path, *, base_url: str = "http://127.0.0.1:8080/v1", model_name: str = "SmolVLM2-2.2B-Instruct", timeout: float = 120.0, max_new_tokens: int = 96, max_model_bytes: int = RECOMMENDED_MAX_MODEL_BYTES) -> None:
        self.info = validate_gguf_pair(model_path, mmproj_path, max_bytes=max_model_bytes)
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.max_new_tokens = max_new_tokens

    def check_server(self) -> dict[str, Any]:
        root = self.base_url.removesuffix("/v1")
        props = _get_json(f"{root}/props", timeout=min(self.timeout, 10.0))
        modalities = props.get("modalities") or {}
        if not bool(modalities.get("vision")):
            raise LlamaServerVLMError(
                "llama.cpp is reachable but vision is disabled. "
                f"/props reports modalities.vision=false; model={props.get('model_path', 'unknown')}; "
                f"build={props.get('build_info', 'unknown')}. Start a current multimodal llama-server with --mmproj <matching-mmproj.gguf>."
            )
        return props

    def describe(self, image_path: str | Path, *, visual_record: str = "", crop_reason: str = "whole_image") -> dict[str, Any]:
        prompt = (
            "Analyze this scientific or technical diagram region. Give a concise semantic hypothesis about what it shows, "
            "visible symbols, labels, local relationships, and likely role in the whole figure. Do not invent exact coordinates. "
            f"This is a semantic hint only. Crop reason: {crop_reason}.\n\nDeterministic visual evidence:\n"
            + visual_record[:12000]
        )
        body = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _data_uri(image_path)}},
            ]}],
            "temperature": 0.1,
            "max_tokens": self.max_new_tokens,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LlamaServerVLMError(f"llama.cpp vision request failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LlamaServerVLMError(f"Could not reach llama.cpp vision server at {self.base_url}: {exc}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise LlamaServerVLMError(f"Invalid response from llama.cpp vision server: {exc}") from exc
        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LlamaServerVLMError(f"Unexpected llama.cpp response schema: {payload!r}") from exc
        if not isinstance(text, str) or not text.strip():
            raise LlamaServerVLMError("llama.cpp vision server returned an empty semantic response")
        return {"text": text.strip(), "model": self.model_name, "weight_bytes": self.info.total_bytes}


def enrich_scene_with_llama_server_vlm(scene: Any, image_path: str | Path, model_path: str | Path, mmproj_path: str | Path, *, base_url: str = "http://127.0.0.1:8080/v1", model_name: str = "SmolVLM2-2.2B-Instruct", max_crops: int = 8, max_model_bytes: int = RECOMMENDED_MAX_MODEL_BYTES) -> Any:
    from .semantic_crops import crop_image, select_semantic_crops
    from .serialize import to_llm_context
    observer = LlamaServerVisionObserver(model_path, mmproj_path, base_url=base_url, model_name=model_name, max_model_bytes=max_model_bytes)
    observer.check_server()
    crops = select_semantic_crops(scene, image_path, max_crops=max_crops)
    if not crops:
        scene.warnings.append("LLAMA_SERVER_VLM skipped: no high-value semantic crops were selected.")
        return scene
    temp_dir = Path(image_path).parent / ".semantic_crops_llama"
    hypotheses: list[dict[str, Any]] = []
    try:
        for crop in crops:
            crop_path = crop_image(image_path, crop, temp_dir)
            result = observer.describe(crop_path, visual_record=to_llm_context(scene), crop_reason=", ".join(crop.reasons))
            hypotheses.append({"crop_id": crop.id, "bbox_px": [crop.bbox.x, crop.bbox.y, crop.bbox.width, crop.bbox.height], "reasons": list(crop.reasons), "priority": crop.priority, "text": result["text"]})
    finally:
        for path in temp_dir.glob("*.png"):
            path.unlink(missing_ok=True)
        try:
            temp_dir.rmdir()
        except OSError:
            pass
    scene.image["micro_vlm"] = {"enabled": True, "backend": "llama.cpp-server", "model": model_name, "model_path": str(model_path), "mmproj_path": str(mmproj_path), "endpoint": base_url, "weight_bytes_including_mmproj": observer.info.total_bytes, "hard_model_limit_bytes": MAX_ALLOWED_MODEL_BYTES, "recommended_model_limit_bytes": RECOMMENDED_MAX_MODEL_BYTES, "crops_analyzed": len(hypotheses), "policy": "semantic hypotheses only; authoritative geometry remains deterministic"}
    scene.image["micro_vlm_hypotheses"] = hypotheses
    scene.warnings.append("LLAMA_SERVER_VLM output is a semantic hypothesis and must not override measured geometry or topology.")
    return scene


def build_llama_server_command(
    server_executable: str | Path,
    model_path: str | Path,
    mmproj_path: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    context: int = 4096,
    gpu_layers: int | None = None,
    parallel: int = 1,
    fit_target_mib: int = 384,
    no_mmproj_offload: bool = False,
) -> list[str]:
    """Return QProcess arguments; leave GPU layers unset so llama.cpp can fit VRAM automatically."""
    command = [
        "-m", str(model_path),
        "--mmproj", str(mmproj_path),
        "--host", host,
        "--port", str(port),
        "-c", str(context),
        "-np", str(max(1, int(parallel))),
        "--fit", "on",
        "--fit-target", str(max(128, int(fit_target_mib))),
    ]
    if gpu_layers is not None:
        command.extend(["-ngl", str(int(gpu_layers))])
    if no_mmproj_offload:
        command.append("--no-mmproj-offload")
    return command
