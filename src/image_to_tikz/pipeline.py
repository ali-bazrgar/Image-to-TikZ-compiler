from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .analyzer import ImageAnalyzer
from .graph import enrich_scene
from .semantic import render_llm_context


def analyze_image(image_path: str | Path) -> tuple[Any, str]:
    """Run the complete deterministic image-analysis pipeline."""
    scene = ImageAnalyzer().analyze(str(image_path))
    scene = enrich_scene(scene, str(image_path))
    return scene, render_llm_context(scene)


def save_artifacts(scene: Any, context: str, output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "scene.json"
    text_path = out / "llm_context.txt"
    json_path.write_text(json.dumps(scene.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    text_path.write_text(context, encoding="utf-8")
    return {"json": str(json_path), "context": str(text_path)}
