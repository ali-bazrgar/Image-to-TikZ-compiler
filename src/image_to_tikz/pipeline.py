from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .analyzer_api import ImageAnalyzer
from .curves import enrich_curves
from .micro_vlm import MicroVLMError, enrich_scene_with_micro_vlm
from .multiscale import MultiscaleAnalyzer
from .ocr import LightweightOCRError, enrich_scene_with_ocr
from .scene_grammar import enrich_scene_grammar
from .serialize import to_llm_context
from .structure import enrich_structure
from .text_structure import enrich_text_structure


def analyze_image(
    image_path: str | Path,
    *,
    multiscale: bool = True,
    ocr: str = "auto",
    ocr_score_threshold: float = 0.35,
    micro_vlm_dir: str | Path | None = None,
    micro_vlm_device: str = "auto",
    micro_vlm_max_crops: int = 8,
    micro_vlm_max_model_bytes: int = 2_500_000_000,
) -> tuple[Any, str]:
    """Run deterministic image analysis with optional lightweight semantic inspection."""
    if ocr not in {"auto", "on", "off"}:
        raise ValueError("ocr must be one of: auto, on, off")

    path = str(image_path)
    scene = MultiscaleAnalyzer().analyze(path) if multiscale else ImageAnalyzer().analyze(path)
    enrich_curves(scene, path)
    enrich_structure(scene)

    if ocr != "off":
        try:
            enrich_scene_with_ocr(scene, path, score_threshold=ocr_score_threshold)
        except LightweightOCRError as exc:
            if ocr == "on":
                raise
            scene.image["ocr"] = {"enabled": False, "engine": None, "reason": str(exc)}
            scene.warnings.append(str(exc))

    enrich_text_structure(scene)
    enrich_scene_grammar(scene)

    if micro_vlm_dir is not None:
        enrich_scene_with_micro_vlm(
            scene,
            path,
            micro_vlm_dir,
            device=micro_vlm_device,
            max_crops=micro_vlm_max_crops,
            max_model_bytes=micro_vlm_max_model_bytes,
        )

    return scene, to_llm_context(scene)


def save_artifacts(scene: Any, context: str, output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "scene.json"
    text_path = out / "llm_context.txt"
    json_path.write_text(json.dumps(scene.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    text_path.write_text(context, encoding="utf-8")
    return {"json": str(json_path), "context": str(text_path)}
