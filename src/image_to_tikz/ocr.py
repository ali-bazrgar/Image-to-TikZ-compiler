from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .vir import BoundingBox, TextBlock, VisualScene

MAX_ALLOWED_MODEL_BYTES = 1_000_000_000


class LightweightOCRError(RuntimeError):
    pass


def _load_engine():
    """Prefer the current RapidOCR API, then the legacy package, and enforce the model budget."""
    candidates = ("rapidocr", "rapidocr_onnxruntime")
    last_error: Exception | None = None
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
            engine = module.RapidOCR()
            _enforce_model_budget(engine, module)
            return engine
        except ImportError as exc:
            last_error = exc
    raise LightweightOCRError(
        "Lightweight OCR is not installed. Install the optional 'ocr' extra."
    ) from last_error


def _enforce_model_budget(engine: Any, module: Any) -> None:
    roots: list[Path] = []
    cfg = getattr(engine, "cfg", None)
    global_cfg = getattr(cfg, "Global", None)
    model_root = getattr(global_cfg, "model_root_dir", None)
    if model_root:
        roots.append(Path(model_root))
    module_paths = getattr(module, "__path__", None) or []
    roots.extend(Path(p) for p in module_paths)

    offenders: list[tuple[Path, int]] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists() or root in seen:
            continue
        seen.add(root)
        try:
            for path in root.rglob("*.onnx"):
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                if size > MAX_ALLOWED_MODEL_BYTES:
                    offenders.append((path, size))
        except OSError:
            continue
    if offenders:
        details = ", ".join(f"{p} ({s / 1_000_000:.1f} MB)" for p, s in offenders)
        raise LightweightOCRError(
            f"OCR model budget exceeded: {details}. This compiler allows models below 1 GB only."
        )


def _quad_to_bbox(box: Any) -> BoundingBox:
    points = np.asarray(box, dtype=float).reshape(-1, 2)
    if points.size == 0:
        return BoundingBox(0, 0, 0, 0)
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    return BoundingBox(float(x0), float(y0), float(x1 - x0), float(y1 - y0))


def _normalize_result(result: Any) -> list[tuple[Any, str, float]]:
    """Normalize RapidOCR current and legacy outputs to (box, text, score)."""
    if result is None:
        return []

    boxes = getattr(result, "boxes", None)
    txts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    if boxes is not None and txts is not None and scores is not None:
        return [
            (box, str(text), float(score))
            for box, text, score in zip(boxes, txts, scores)
            if str(text).strip()
        ]

    payload = result[0] if isinstance(result, tuple) and len(result) == 2 else result
    if payload is None:
        return []
    if isinstance(payload, np.ndarray):
        payload = payload.tolist()

    normalized: list[tuple[Any, str, float]] = []
    for item in payload:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        box, text, score = item[0], item[1], item[2]
        try:
            normalized.append((box, str(text), float(score)))
        except (TypeError, ValueError):
            continue
    return [x for x in normalized if x[1].strip()]


def extract_ocr(image: np.ndarray, *, score_threshold: float = 0.35) -> list[TextBlock]:
    """Run optional lightweight OCR on an image array."""
    if image is None or image.size == 0:
        return []
    engine = _load_engine()
    result = engine(image)
    rows = _normalize_result(result)
    texts: list[TextBlock] = []
    for index, (box, text, score) in enumerate(rows, 1):
        if score < score_threshold:
            continue
        bbox = _quad_to_bbox(box)
        if bbox.width < 2 or bbox.height < 2:
            continue
        texts.append(
            TextBlock(
                id=f"ocr_text_{index}",
                text=text.strip(),
                bbox=bbox,
                confidence=round(score, 4),
                language="rapidocr",
            )
        )
    return texts


def enrich_scene_with_ocr(
    scene: VisualScene, image_path: str | Path, *, score_threshold: float = 0.35
) -> VisualScene:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise LightweightOCRError(f"Could not decode image for OCR: {image_path}")
    recognized = extract_ocr(image, score_threshold=score_threshold)

    for text in recognized:
        match = _best_text_region(text, scene.texts)
        if match is not None and _iou(text.bbox, match.bbox) >= 0.10:
            match.text = text.text
            match.confidence = text.confidence
            match.language = text.language
        else:
            scene.texts.append(text)

    for text in recognized:
        for element in scene.elements:
            if _distance_to_box(text.bbox.center.x, text.bbox.center.y, element.bbox) <= max(
                24.0, element.bbox.width * 0.12, element.bbox.height * 0.12
            ):
                if text.id not in element.text_refs:
                    element.text_refs.append(text.id)
                break

    scene.image["ocr"] = {
        "enabled": True,
        "engine": "rapidocr",
        "recognized_regions": len(recognized),
        "model_policy": "only lightweight OCR models under 1GB are supported",
    }
    scene.warnings = [w for w in scene.warnings if "Text glyphs are not decoded" not in w]
    return scene


def _best_text_region(text: TextBlock, regions: list[TextBlock]) -> TextBlock | None:
    if not regions:
        return None
    return max(regions, key=lambda r: _iou(text.bbox, r.bbox))


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    x0 = max(a.x, b.x)
    y0 = max(a.y, b.y)
    x1 = min(a.x + a.width, b.x + b.width)
    y1 = min(a.y + a.height, b.y + b.height)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def _distance_to_box(x: float, y: float, box: BoundingBox) -> float:
    dx = max(box.x - x, 0.0, x - (box.x + box.width))
    dy = max(box.y - y, 0.0, y - (box.y + box.height))
    return float(np.hypot(dx, dy))
