from __future__ import annotations

from collections import defaultdict
from typing import Any

from .vir import BoundingBox, Point, Relation, VisualElement, VisualScene


def enrich_scene_grammar(scene: VisualScene) -> VisualScene:
    """Add deterministic region/layout evidence without semantic AI."""
    _local_regions(scene)
    _layout_bands(scene)
    _nested_containment(scene)
    _axis_candidates(scene)
    return scene


def _local_regions(scene: VisualScene) -> None:
    elems = scene.elements
    if not elems:
        return
    # Fixed spatial bucketing provides stable, model-free regional structure.
    cols = 3
    rows = 3
    w = float(scene.image.get("width", 1)); h = float(scene.image.get("height", 1))
    buckets: defaultdict[tuple[int, int], list[VisualElement]] = defaultdict(list)
    for e in elems:
        c = min(cols - 1, max(0, int((e.center.x / max(w, 1)) * cols)))
        r = min(rows - 1, max(0, int((e.center.y / max(h, 1)) * rows)))
        buckets[(r, c)].append(e)
    for (r, c), members in buckets.items():
        if len(members) < 2:
            continue
        rid = f"region_{r+1}_{c+1}"
        for e in members:
            e.geometry["spatial_region"] = rid


def _layout_bands(scene: VisualScene) -> None:
    elems = scene.elements
    if len(elems) < 2:
        return
    w = float(scene.image.get("width", 1)); h = float(scene.image.get("height", 1))
    bands = {"top": [], "middle": [], "bottom": [], "left": [], "center": [], "right": []}
    for e in elems:
        nx = e.center.x / max(w, 1); ny = e.center.y / max(h, 1)
        bands["top" if ny < .33 else "bottom" if ny > .67 else "middle"].append(e)
        bands["left" if nx < .33 else "right" if nx > .67 else "center"].append(e)
    for name, members in bands.items():
        if len(members) < 2:
            continue
        for e in members:
            e.geometry.setdefault("layout_bands", []).append(name)


def _nested_containment(scene: VisualScene) -> None:
    elems = scene.elements
    for a in elems:
        for b in elems:
            if a.id == b.id:
                continue
            if _contains(a.bbox, b.bbox) and a.bbox.width * a.bbox.height > b.bbox.width * b.bbox.height * 1.4:
                scene.relations.append(Relation(a.id, "contains", b.id, 0.76))


def _axis_candidates(scene: VisualScene) -> None:
    for e in scene.elements:
        if e.kind != "line_segment":
            continue
        if e.geometry.get("possible_role") != "axis_or_baseline":
            continue
        scene.relations.append(Relation(e.id, "global_reference_line_candidate", e.id, 0.55))


def _contains(a: BoundingBox, b: BoundingBox, tol: float = 3.0) -> bool:
    return (
        a.x - tol <= b.x and a.y - tol <= b.y
        and a.x + a.width + tol >= b.x + b.width
        and a.y + a.height + tol >= b.y + b.height
    )
