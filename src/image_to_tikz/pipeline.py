from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .analyzer_api import ImageAnalyzer
from .serialize import to_llm_context
from .structure import enrich_structure


def analyze_image(image_path: str | Path) -> tuple[Any, str]:
    """Run the complete deterministic, model-free image-analysis pipeline."""
    scene = ImageAnalyzer().analyze(str(image_path))
    enrich_structure(scene)
    return scene, to_llm_context(scene)


def save_artifacts(scene: Any, context: str, output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "scene.json"
    text_path = out / "llm_context.txt"
    json_path.write_text(json.dumps(scene.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    text_path.write_text(context, encoding="utf-8")
    return {"json": str(json_path), "context": str(text_path)}
