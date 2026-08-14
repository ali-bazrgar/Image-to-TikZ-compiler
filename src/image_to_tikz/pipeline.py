from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .analyzer_api import ImageAnalyzer
from .canonical_graph import enrich_canonical_graph
from .curves import enrich_curves
from .domain_router import enrich_domain_routing
from .llama_server_vlm import LlamaServerVLMError, enrich_scene_with_llama_server_vlm
from .micro_vlm import enrich_scene_with_micro_vlm
from .multiscale import MultiscaleAnalyzer
from .ocr import LightweightOCRError, enrich_scene_with_ocr
from .provenance import enrich_provenance
from .scene_grammar import enrich_scene_grammar
from .serialize import to_llm_context
from .specialized_detectors import enrich_specialized_detectors
from .structure import enrich_structure
from .text_structure import enrich_text_structure


def analyze_image(
    image_path: str | Path,
    *,
    multiscale: bool = True,
    ocr: str = "auto",
    ocr_score_threshold: float = 0.35,
    micro_vlm_backend: str = "none",
    micro_vlm_dir: str | Path | None = None,
    micro_vlm_device: str = "auto",
    micro_vlm_model_path: str | Path | None = None,
    micro_vlm_mmproj_path: str | Path | None = None,
    micro_vlm_base_url: str = "http://127.0.0.1:8080/v1",
    micro_vlm_model_name: str = "SmolVLM2-2.2B-Instruct",
    micro_vlm_max_crops: int = 8,
    micro_vlm_max_model_bytes: int = 2_500_000_000,
) -> tuple[Any, str]:
    """Run deterministic image analysis with optional lightweight semantic inspection."""
    if ocr not in {"auto", "on", "off"}:
        raise ValueError("ocr must be one of: auto, on, off")
    if micro_vlm_backend not in {"none", "transformers", "llama-server"}:
        raise ValueError("micro_vlm_backend must be one of: none, transformers, llama-server")

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
    enrich_domain_routing(scene)
    enrich_specialized_detectors(scene, path)
    enrich_scene_grammar(scene)
    enrich_canonical_graph(scene)
    enrich_provenance(scene)

    if micro_vlm_backend == "transformers":
        if micro_vlm_dir is None:
            raise ValueError("micro_vlm_dir is required when micro_vlm_backend='transformers'")
        enrich_scene_with_micro_vlm(
            scene,
            path,
            micro_vlm_dir,
            device=micro_vlm_device,
            max_crops=micro_vlm_max_crops,
            max_model_bytes=micro_vlm_max_model_bytes,
        )
        enrich_provenance(scene)
    elif micro_vlm_backend == "llama-server":
        if micro_vlm_model_path is None or micro_vlm_mmproj_path is None:
            raise ValueError("micro_vlm_model_path and micro_vlm_mmproj_path are required for llama-server backend")
        try:
            enrich_scene_with_llama_server_vlm(
                scene,
                path,
                micro_vlm_model_path,
                micro_vlm_mmproj_path,
                base_url=micro_vlm_base_url,
                model_name=micro_vlm_model_name,
                max_crops=micro_vlm_max_crops,
                max_model_bytes=micro_vlm_max_model_bytes,
            )
        except LlamaServerVLMError:
            raise
        enrich_provenance(scene)

    return scene, to_llm_context(scene)


def save_artifacts(scene: Any, context: str, output_dir: str | Path) -> dict[str, str]:
    """Save lossless machine JSON without pretty-printing huge coordinate arrays."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "scene.json"
    text_path = out / "llm_context.txt"

    payload = scene.to_dict()
    # Compact separators preserve every value while avoiding one JSON array item per line.
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    text_path.write_text(context, encoding="utf-8")
    return {"json": str(json_path), "context": str(text_path)}
