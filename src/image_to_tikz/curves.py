from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .vir import BoundingBox, Point, VisualElement, VisualScene


def enrich_curves(scene: VisualScene, image_path: str) -> VisualScene:
    """Detect deterministic curve/path candidates and attach their geometry."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return scene
    h, w = image.shape[:2]
    edges = cv2.Canny(cv2.GaussianBlur(image, (3, 3), 0), 35, 125, L2gradient=True)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    next_id = 1
    existing = {(e.kind, round(e.center.x, 1), round(e.center.y, 1)) for e in scene.elements}
    added: list[VisualElement] = []
    for contour in contours:
        if len(contour) < 18:
            continue
        perimeter = float(cv2.arcLength(contour, False))
        if perimeter < max(30.0, min(w, h) * 0.06):
            continue
        approx = cv2.approxPolyDP(contour, 0.012 * perimeter, False)
        if len(approx) < 4 or len(approx) > 80:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 12 or bh < 8:
            continue
        ratio = bw / max(1.0, bh)
        if ratio > 18 or ratio < 1 / 18:
            continue

        pts = contour[:, 0, :].astype(float)
        # Reject contours that are already well represented by a straight segment.
        if len(pts) >= 2:
            p0, p1 = pts[0], pts[-1]
            chord = float(np.hypot(*(p1 - p0)))
            straightness = chord / max(perimeter, 1e-6)
        else:
            straightness = 1.0
        if straightness > 0.94:
            continue

        # Curvature sign changes are a useful model-free hint for polyline-like paths.
        delta = np.diff(pts, axis=0)
        angles = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
        turning = np.diff(angles)
        turning = turning[np.isfinite(turning)]
        turning_abs = float(np.mean(np.abs(turning))) if turning.size else 0.0

        if len(approx) <= 10 and turning_abs > 0.10:
            kind = "polyline_or_arc"
            confidence = 0.67
        else:
            kind = "curve_path"
            confidence = 0.61

        cx = x + bw / 2.0
        cy = y + bh / 2.0
        key = (kind, round(cx, 1), round(cy, 1))
        if key in existing:
            continue

        points = [[round(float(px), 2), round(float(py), 2)] for px, py in pts[::max(1, len(pts) // 32)]]
        element = VisualElement(
            id=f"curve_{next_id}",
            kind=kind,
            bbox=BoundingBox(float(x), float(y), float(bw), float(bh)),
            center=Point(float(cx), float(cy)),
            confidence=confidence,
            geometry={
                "path_points_px": points,
                "point_count_sampled": len(points),
                "perimeter_px": round(perimeter, 2),
                "chord_px": round(chord, 2),
                "straightness": round(straightness, 4),
                "mean_turning_abs_rad": round(turning_abs, 5),
                "closed": False,
                "representation": "sampled_polyline"
            },
            style={"stroke": "dark", "closed": False},
        )
        added.append(element)
        next_id += 1
        if len(added) >= 160:
            break

    scene.elements.extend(added)
    scene.image["curve_analysis"] = {"enabled": True, "added_candidates": len(added)}
    if added:
        scene.warnings.append("Curve candidates are sampled pixel paths; semantic interpretation as arc, spline, or domain-specific curve remains downstream work.")
    return scene
