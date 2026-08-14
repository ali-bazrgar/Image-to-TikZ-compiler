from __future__ import annotations

from pathlib import Path
from typing import Any

from .llama_server_vlm import LlamaServerVisionObserver
from .semantic_crops import crop_image, select_semantic_crops
from .serialize import to_llm_context


def enrich_scene_with_llama_server_vlm(
    scene: Any,
    image_path: str | Path,
    model_path: str | Path,
    mmproj_path: str | Path,
    *,
    base_url: str = "http://127.0.0.1:8080/v1",
    model_name: str = "SmolVLM2-2.2B-Instruct",
    max_crops: int = 8,
    max_model_bytes: int = 2_500_000_000,
    crop_dir: str | Path | None = None,
) -> Any:
    """Use a running llama.cpp multimodal server only on selected high-value crops."""
    observer = LlamaServerVisionObserver(
        model_path,
        mmproj_path,
        base_url=base_url,
        model_name=model_name,
        max_model_bytes=max_model_bytes,
    )
    crops = select_semantic_crops(scene, image_path, max_crops=max_crops)
    if not crops:
        scene.warnings.append("LLAMA_SERVER_VLM skipped: no high-value semantic crops were selected.")
        return scene

    root = Path(crop_dir) if crop_dir is not None else Path(image_path).parent / ".semantic_crops"
    hypotheses: list[dict[str, Any]] = []
    try:
        for crop in crops:
            crop_path = crop_image(image_path, crop, root)
            result = observer.describe(
                crop_path,
                visual_record=to_llm_context(scene),
                crop_reason=", ".join(crop.reasons),
            )
            hypotheses.append({
                "crop_id": crop.id,
                "bbox_px": [crop.bbox.x, crop.bbox.y, crop.bbox.width, crop.bbox.height],
                "reasons": list(crop.reasons),
                "priority": crop.priority,
                "text": result["text"],
            })
    finally:
        if crop_dir is None:
            for p in root.glob("*.png"):
                p.unlink(missing_ok=True)
            try:
                root.rmdir()
            except OSError:
                pass

    scene.image["micro_vlm"] = {
        "enabled": True,
        "backend": "llama.cpp-openai-compatible",
        "model": model_name,
        "local_model": str(model_path),
        "mmproj": str(mmproj_path),
        "weight_bytes": observer.info.total_bytes,
        "crops_analyzed": len(hypotheses),
        "policy": "semantic hypotheses only; deterministic geometry/topology remain authoritative",
    }
    scene.image["micro_vlm_hypotheses"] = hypotheses
    scene.warnings.append("SmolVLM2 output is semantic evidence only; never override measured geometry/topology.")
    return scene
