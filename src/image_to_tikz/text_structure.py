from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

from .vir import BoundingBox, Relation, TextBlock, VisualScene


def enrich_text_structure(scene: VisualScene) -> VisualScene:
    """Extract deterministic typography-like structure without OCR."""
    _classify_regions(scene)
    _group_glyphs_into_runs(scene)
    _detect_baselines(scene)
    _detect_formula_like_layouts(scene)
    _attach_text_relations(scene)
    return scene


def _classify_regions(scene: VisualScene) -> None:
    for t in scene.texts:
        w, h = max(t.bbox.width, 1.0), max(t.bbox.height, 1.0)
        aspect = w / h
        area_ratio = (w * h) / max(scene.image.get("width", 1) * scene.image.get("height", 1), 1)
        if aspect > 5 and h < 0.08 * scene.image.get("height", 1):
            role = "text_line_candidate"
        elif aspect < 0.45 and h > 12:
            role = "vertical_label_candidate"
        elif area_ratio < 0.0008 and h < 0.05 * scene.image.get("height", 1):
            role = "small_label_or_symbol_candidate"
        else:
            role = "text_region_candidate"
        t.language = role


def _group_glyphs_into_runs(scene: VisualScene) -> None:
    regions = sorted(scene.texts, key=lambda t: (t.bbox.center.y, t.bbox.x))
    used: set[str] = set()
    for t in regions:
        if t.id in used:
            continue
        band = [t]
        used.add(t.id)
        for q in regions:
            if q.id in used:
                continue
            dy = abs(q.bbox.center.y - t.bbox.center.y)
            gap = q.bbox.x - (t.bbox.x + t.bbox.width)
            height = max(t.bbox.height, q.bbox.height)
            if dy <= 0.45 * height and -0.30 * height <= gap <= 2.5 * height:
                band.append(q)
                used.add(q.id)
        if len(band) >= 2:
            band.sort(key=lambda x: x.bbox.x)
            ids = [x.id for x in band]
            for a in band:
                a.language = f"glyph_run_candidate:{','.join(ids)}"


def _detect_baselines(scene: VisualScene) -> None:
    rows: defaultdict[int, list[TextBlock]] = defaultdict(list)
    for t in scene.texts:
        key = int(round(t.bbox.center.y / max(t.bbox.height, 1.0) * 4))
        rows[key].append(t)
    for members in rows.values():
        if len(members) < 2:
            continue
        centers = [m.bbox.center.y for m in members]
        baseline = sum(centers) / len(centers)
        for m in members:
            m.language = f"{m.language or 'text_region_candidate'};baseline_y≈{baseline:.1f}"


def _detect_formula_like_layouts(scene: VisualScene) -> None:
    texts = scene.texts
    for a in texts:
        for b in texts:
            if a.id >= b.id:
                continue
            dx = abs(a.bbox.center.x - b.bbox.center.x)
            dy = abs(a.bbox.center.y - b.bbox.center.y)
            if dx < max(a.bbox.width, b.bbox.width) * 0.65 and dy > max(a.bbox.height, b.bbox.height) * 0.55:
                scene.relations.append(Relation(a.id, "stacked_text_candidate", b.id, 0.52))
            if a.bbox.x < b.bbox.x and a.bbox.center.y < b.bbox.center.y and dx < max(a.bbox.width, b.bbox.width) * 1.8:
                scene.relations.append(Relation(a.id, "superscript_or_subscript_candidate", b.id, 0.42))


def _attach_text_relations(scene: VisualScene) -> None:
    for t in scene.texts:
        # Find nearest non-text primitive and emit a conservative label relation.
        candidates = []
        for e in scene.elements:
            dx = max(e.bbox.x - t.bbox.center.x, 0, t.bbox.center.x - (e.bbox.x + e.bbox.width))
            dy = max(e.bbox.y - t.bbox.center.y, 0, t.bbox.center.y - (e.bbox.y + e.bbox.height))
            d = math.hypot(dx, dy)
            candidates.append((d, e))
        if candidates:
            d, e = min(candidates, key=lambda x: x[0])
            threshold = max(24.0, 1.5 * max(t.bbox.width, t.bbox.height, e.bbox.width * 0.1, e.bbox.height * 0.1))
            if d <= threshold:
                scene.relations.append(Relation(t.id, "label_or_annotation_candidate", e.id, round(max(0.35, 1 - d / threshold), 3), {"distance_px": round(d, 2)}))
