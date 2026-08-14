from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_ALLOWED_MODEL_BYTES = 3_000_000_000
RECOMMENDED_MAX_MODEL_BYTES = 2_500_000_000
DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


class MicroVLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class SemanticHypothesis:
    model_id: str
    model_size_bytes: int
    text: str
    confidence: float | None = None


def validate_model_directory(model_dir: str | Path, *, max_bytes: int = MAX_ALLOWED_MODEL_BYTES) -> int:
    """Validate local model weights against the 3 GB hard ceiling."""
    root = Path(model_dir)
    if not root.exists() or not root.is_dir():
        raise MicroVLMError(f"Model directory does not exist: {root}")
    weight_suffixes = {".safetensors", ".bin", ".onnx", ".pt", ".pth"}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in weight_suffixes]
    if not files:
        raise MicroVLMError("No supported model weight file found in model directory")
    total = sum(p.stat().st_size for p in files)
    if total > max_bytes:
        raise MicroVLMError(f"Model weights total {total} bytes, exceeding the {max_bytes}-byte hard ceiling")
    return total


class SmolVLMSemanticObserver:
    """Optional VLM observer; it supplies hypotheses and never authoritative geometry."""

    def __init__(self, model_dir: str | Path, *, device: str = "auto", max_new_tokens: int = 180, max_model_bytes: int = RECOMMENDED_MAX_MODEL_BYTES) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_model_bytes = max_model_bytes
        self._model: Any = None
        self._processor: Any = None
        self.model_size_bytes = validate_model_directory(self.model_dir, max_bytes=min(max_model_bytes, MAX_ALLOWED_MODEL_BYTES))

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as exc:
            raise MicroVLMError("Lightweight VLM support requires the optional 'micro-vlm' dependencies.") from exc

        if self.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device
        dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self._processor = AutoProcessor.from_pretrained(self.model_dir)
        kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
        if device == "cuda":
            kwargs["device_map"] = {"": 0}
        self._model = AutoModelForVision2Seq.from_pretrained(self.model_dir, **kwargs)
        if device != "cuda":
            self._model = self._model.to(device)
        self._device = device

    def describe(self, image_path: str | Path, *, visual_record: str = "", crop_reason: str = "whole_image") -> SemanticHypothesis:
        self._load()
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": (
                "Analyze this scientific or technical diagram region. Give a concise semantic hypothesis "
                "about what it shows, its symbols, labels, local relationships, and likely role in the whole "
                "figure. Do not invent exact coordinates. This is a semantic hint only. Crop reason: "
                + crop_reason + "\nDeterministic visual evidence:\n" + visual_record[:10000]
            )},
        ]}]
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=prompt, images=[image], return_tensors="pt").to(self._device)
        generated = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        input_len = inputs["input_ids"].shape[-1]
        text = self._processor.batch_decode(generated[:, input_len:], skip_special_tokens=True)[0].strip()
        return SemanticHypothesis(str(self.model_dir), self.model_size_bytes, text)


def enrich_scene_with_micro_vlm(scene: Any, image_path: str | Path, model_dir: str | Path, *, device: str = "auto", max_crops: int = 8, max_model_bytes: int = RECOMMENDED_MAX_MODEL_BYTES, crop_dir: str | Path | None = None) -> Any:
    """Inspect only high-value crops and attach semantic hypotheses to the VIR."""
    from .semantic_crops import crop_image, select_semantic_crops
    from .serialize import to_llm_context

    observer = SmolVLMSemanticObserver(model_dir, device=device, max_crops=max_crops if False else 180, max_model_bytes=max_model_bytes)
    crops = select_semantic_crops(scene, image_path, max_crops=max_crops)
    if not crops:
        scene.warnings.append("MICRO_VLM skipped: no high-value semantic crops were selected.")
        return scene

    temp_dir = Path(crop_dir) if crop_dir is not None else Path(image_path).parent / ".semantic_crops"
    hypotheses = []
    try:
        for crop in crops:
            crop_path = crop_image(image_path, crop, temp_dir)
            hypothesis = observer.describe(crop_path, visual_record=to_llm_context(scene), crop_reason=", ".join(crop.reasons))
            hypotheses.append({
                "crop_id": crop.id,
                "bbox_px": [crop.bbox.x, crop.bbox.y, crop.bbox.width, crop.bbox.height],
                "reasons": list(crop.reasons),
                "priority": crop.priority,
                "text": hypothesis.text,
            })
    finally:
        if crop_dir is None:
            for path in temp_dir.glob("*.png"):
                path.unlink(missing_ok=True)
            try:
                temp_dir.rmdir()
            except OSError:
                pass

    scene.image["micro_vlm"] = {
        "enabled": True,
        "model": DEFAULT_MODEL_ID,
        "local_model_dir": str(model_dir),
        "weight_bytes": observer.model_size_bytes,
        "hard_model_limit_bytes": MAX_ALLOWED_MODEL_BYTES,
        "recommended_model_limit_bytes": RECOMMENDED_MAX_MODEL_BYTES,
        "crops_analyzed": len(hypotheses),
        "policy": "semantic hypotheses only; authoritative geometry remains deterministic",
    }
    scene.image["micro_vlm_hypotheses"] = hypotheses
    scene.warnings.append("MICRO_VLM output is a semantic hypothesis and must not override measured geometry or topology.")
    return scene
