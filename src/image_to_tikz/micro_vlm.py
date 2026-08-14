from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_ALLOWED_MODEL_BYTES = 1_000_000_000
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
    """Validate that all model weight files in a local directory fit the 1GB policy."""
    root = Path(model_dir)
    if not root.exists() or not root.is_dir():
        raise MicroVLMError(f"Model directory does not exist: {root}")
    weight_suffixes = {".safetensors", ".bin", ".onnx", ".pt", ".pth"}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in weight_suffixes]
    if not files:
        raise MicroVLMError("No supported model weight file found in model directory")
    total = sum(p.stat().st_size for p in files)
    if total > max_bytes:
        raise MicroVLMError(
            f"Model weights total {total} bytes, exceeding the {max_bytes}-byte policy"
        )
    return total


class SmolVLMSemanticObserver:
    """Optional sub-1GB vision-language observer; never supplies authoritative geometry."""

    def __init__(self, model_dir: str | Path, *, device: str = "auto", max_new_tokens: int = 180) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model: Any = None
        self._processor: Any = None
        self.model_size_bytes = validate_model_directory(self.model_dir)

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as exc:
            raise MicroVLMError(
                "Lightweight VLM support requires the optional 'micro-vlm' dependencies."
            ) from exc

        if self.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self._processor = AutoProcessor.from_pretrained(self.model_dir)
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_dir,
            torch_dtype=dtype,
            _attn_implementation="flash_attention_2" if device == "cuda" else "eager",
        ).to(device)
        self._device = device

    def describe(self, image_path: str | Path, *, visual_record: str = "") -> SemanticHypothesis:
        self._load()
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": (
                            "Analyze this scientific/technical diagram. Give a concise semantic hypothesis "
                            "about what the whole figure represents, its major components, their relationships, "
                            "and any visible labels or symbols. Do not invent exact coordinates. "
                            "The following deterministic computer-vision record is evidence; reconcile with the image:\n"
                            + visual_record[:12000]
                        ),
                    },
                ],
            }
        ]
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=prompt, images=[image], return_tensors="pt").to(self._device)
        generated = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        input_len = inputs["input_ids"].shape[-1]
        text = self._processor.batch_decode(generated[:, input_len:], skip_special_tokens=True)[0].strip()
        return SemanticHypothesis(
            model_id=str(self.model_dir),
            model_size_bytes=self.model_size_bytes,
            text=text,
            confidence=None,
        )
