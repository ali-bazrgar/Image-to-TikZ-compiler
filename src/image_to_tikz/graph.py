from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot
from typing import Any

import cv2
import numpy as np

from .vir import Point, Relation, VisualElement, VisualScene


@dataclass
class Primitive:
    id: str
    kind: str
    geometry: dict[str, Any]
    confidence: float


def enrich_scene(scene: VisualScene, image_path: str) -> VisualScene:
    """Add line/arrow semantics and a higher-level connection graph to an existing VIR scene."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        scene.warnings.append("graph: image could not be decoded")
        return scene

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(20, min(image.shape[:2]) // 25),
                            minLineLength=max(12, min(image.shape[:2]) // 40), maxLineGap=8)
    if lines is None:
        return scene

    next_id = 1 + len(scene.elements)
    for raw in lines[:, 0, :]:
        x1, y1, x2, y2 = map(int, raw)
        length = hypot(x2 - x1, y2 - y1)
        if length < max(image.shape[:2]) * 0.025:
            continue
        angle = degrees(atan2(y2 - y1, x2 - x1))
        if angle < 0:
            angle += 180
        # Avoid flooding the scene with near-duplicate Hough detections.
        if _near_duplicate_line(scene.elements, x1, y1, x2, y2, image.shape[1], image.shape[0]):
            continue
        bbox = {
            "x": min(x1, x2) / image.shape[1],
            "y": min(y1, y2) / image.shape[0],
            "width": abs(x2 - x1) / image.shape[1],
            "height": abs(y2 - y1) / image.shape[0],
        }
        from .vir import BoundingBox
        scene.elements.append(VisualElement(
            id=f"line_{next_id}", kind="line", bbox=BoundingBox(**bbox),
            center=Point((x1 + x2) / (2 * image.shape[1]), (y1 + y2) / (2 * image.shape[0])),
            confidence=0.72,
            geometry={
                "start": [x1 / image.shape[1], y1 / image.shape[0]],
                "end": [x2 / image.shape[1], y2 / image.shape[0]],
                "length_px": round(length, 2),
                "angle_deg": round(angle, 2),
            },
        ))
        next_id += 1

    _add_geometric_relations(scene)
    _infer_arrowheads(scene, gray)
    return scene


def _near_duplicate_line(elements: list[VisualElement], x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> bool:
    a = np.array([x1 / w, y1 / h, x2 / w, y2 / h])
    for e in elements:
        if e.kind != "line" or "start" not in e.geometry:
            continue
        b = np.array([*e.geometry["start"], *e.geometry["end"]])
        if min(np.linalg.norm(a[:2] - b[:2]) + np.linalg.norm(a[2:] - b[2:]),
               np.linalg.norm(a[:2] - b[2:]) + np.linalg.norm(a[2:] - b[:2])) < 0.035:
            return True
    return False


def _add_geometric_relations(scene: VisualScene) -> None:
    for a in scene.elements:
        for b in scene.elements:
            if a.id >= b.id:
                continue
            dx = abs(a.center.x - b.center.x)
            dy = abs(a.center.y - b.center.y)
            if dy < 0.018:
                scene.relations.append(Relation(a.id, "horizontally_aligned_with", b.id, 0.84))
            if dx < 0.018:
                scene.relations.append(Relation(a.id, "vertically_aligned_with", b.id, 0.84))
            gap_x = max(b.bbox.x - (a.bbox.x + a.bbox.width), a.bbox.x - (b.bbox.x + b.bbox.width), 0)
            gap_y = max(b.bbox.y - (a.bbox.y + a.bbox.height), a.bbox.y - (b.bbox.y + b.bbox.height), 0)
            if max(gap_x, gap_y) < 0.035:
                scene.relations.append(Relation(a.id, "near", b.id, 0.68,
                                                {"gap_normalized": round(max(gap_x, gap_y), 4)}))


def _infer_arrowheads(scene: VisualScene, gray: np.ndarray) -> None:
    h, w = gray.shape[:2]
    # A conservative heuristic: inspect small endpoint neighborhoods for a compact dark triangular cluster.
    for e in scene.elements:
        if e.kind != "line":
            continue
        pts = [e.geometry.get("start"), e.geometry.get("end")]
        for p in pts:
            if not p:
                continue
            x, y = int(p[0] * w), int(p[1] * h)
            r = max(4, min(h, w) // 80)
            crop = gray[max(0, y-r):min(h, y+r+1), max(0, x-r):min(w, x+r+1)]
            if crop.size == 0:
                continue
            dark = float((crop < 100).mean())
            if dark > 0.33:
                e.geometry.setdefault("endpoint_candidates", []).append({"x": p[0], "y": p[1], "possible_arrowhead": True})
                e.confidence = min(0.9, e.confidence + 0.05)
