from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from .vir import BoundingBox, VisualElement, VisualScene


@dataclass(frozen=True)
class SemanticCrop:
    id: str
    bbox: BoundingBox
    reasons: tuple[str, ...]
    priority: float


def select_semantic_crops(
    scene: VisualScene,
    image_path: str | Path,
    *,
    max_crops: int = 8,
    min_size: int = 96,
    padding: int = 36,
) -> list[SemanticCrop]:
    """Select deterministic, high-value image regions for optional VLM inspection."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return []
    h, w = image.shape[:2]
    candidates: list[SemanticCrop] = []

    # Text-heavy regions are the most useful semantic targets.
    for text in scene.texts:
        b = _expand(text.bbox, padding, w, h)
        size_score = min(1.0, (b.width * b.height) / max(1, w * h) * 18.0)
        role = text.role or "text_region"
        candidates.append(SemanticCrop(f"text:{text.id}", b, ("text", role), 0.75 + size_score * 0.15))

    # Ambiguous primitives benefit from a local visual interpretation.
    for element in scene.elements:
        reasons: list[str] = []
        if element.confidence < 0.72:
            reasons.append("low_confidence")
        if element.kind in {"polyline_or_arc", "curve_path"}:
            reasons.append("curve_or_path")
        if element.kind == "line_segment" and element.geometry.get("endpoint_evidence", 0):
            reasons.append("junction_or_arrow_candidate")
        if element.geometry.get("possible_role"):
            reasons.append("role_ambiguous")
        if not reasons:
            continue
        b = _expand(element.bbox, padding, w, h)
        priority = 0.55 + 0.08 * len(reasons)
        candidates.append(SemanticCrop(f"element:{element.id}", b, tuple(sorted(set(reasons))), priority))

    # Prefer larger windows around dense connected regions.
    for index, group in enumerate(_group_boxes(scene.elements), 1):
        b = _expand(group, padding, w, h)
        if b.width >= min_size and b.height >= min_size:
            candidates.append(SemanticCrop(f"group:{index}", b, ("dense_connected_region",), 0.68))

    selected: list[SemanticCrop] = []
    for candidate in sorted(candidates, key=lambda c: (-c.priority, c.bbox.y, c.bbox.x)):
        if candidate.bbox.width < min_size or candidate.bbox.height < min_size:
            continue
        if any(_iou(candidate.bbox, other.bbox) > 0.72 for other in selected):
            continue
        selected.append(candidate)
        if len(selected) >= max_crops:
            break
    return selected


def crop_image(image_path: str | Path, crop: SemanticCrop, output_dir: str | Path) -> Path:
    """Materialize a selected crop for a VLM adapter."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {image_path}")
    x, y = int(crop.bbox.x), int(crop.bbox.y)
    x1 = min(image.shape[1], int(crop.bbox.x + crop.bbox.width))
    y1 = min(image.shape[0], int(crop.bbox.y + crop.bbox.height))
    crop_array = image[y:y1, x:x1]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{_safe(crop.id)}.png"
    if not cv2.imwrite(str(path), crop_array):
        raise ValueError(f"Could not write semantic crop: {path}")
    return path


def _group_boxes(elements: list[VisualElement]) -> list[BoundingBox]:
    boxes: list[BoundingBox] = []
    for element in elements:
        if not element.geometry.get("structure_group"):
            continue
        group = element.geometry["structure_group"]
        matched = next((i for i, b in enumerate(boxes) if getattr(b, "_group", None) == group), None)
        if matched is None:
            b = BoundingBox(element.bbox.x, element.bbox.y, element.bbox.width, element.bbox.height)
            setattr(b, "_group", group)
            boxes.append(b)
        else:
            boxes[matched] = _union(boxes[matched], element.bbox)
            setattr(boxes[matched], "_group", group)
    return boxes


def _union(a: BoundingBox, b: BoundingBox) -> BoundingBox:
    x0 = min(a.x, b.x)
    y0 = min(a.y, b.y)
    x1 = max(a.x + a.width, b.x + b.width)
    y1 = max(a.y + a.height, b.y + b.height)
    return BoundingBox(x0, y0, x1 - x0, y1 - y0)


def _expand(box: BoundingBox, padding: int, w: int, h: int) -> BoundingBox:
    x0 = max(0, box.x - padding)
    y0 = max(0, box.y - padding)
    x1 = min(w, box.x + box.width + padding)
    y1 = min(h, box.y + box.height + padding)
    return BoundingBox(float(x0), float(y0), float(x1 - x0), float(y1 - y0))


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    x0 = max(a.x, b.x)
    y0 = max(a.y, b.y)
    x1 = min(a.x + a.width, b.x + b.width)
    y1 = min(a.y + a.height, b.y + b.height)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union else 0.0


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)[:80]
