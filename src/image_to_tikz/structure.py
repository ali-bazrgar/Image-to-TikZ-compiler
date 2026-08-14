from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from .vir import Relation, VisualElement, VisualScene


def enrich_structure(scene: VisualScene) -> VisualScene:
    """Infer deterministic higher-level structure without semantic AI."""
    _stroke_features(scene)
    _topology(scene)
    _repetition(scene)
    _symmetry(scene)
    _groups(scene)
    return scene


def _stroke_features(scene: VisualScene) -> None:
    for e in scene.elements:
        if e.kind != "line_segment":
            continue
        g = e.geometry
        length = float(g.get("length_px", 0))
        angle = float(g.get("angle_deg", 0))
        g["stroke_orientation"] = g.get("orientation", "unknown")
        g["length_normalized"] = round(length / max(scene.image.get("width", 1), scene.image.get("height", 1)), 5)
        g["angle_mod_180_deg"] = round(angle % 180, 2)
        endpoints = g.get("endpoint_candidates", [])
        if endpoints:
            g["endpoint_evidence"] = len(endpoints)


def _topology(scene: VisualScene) -> None:
    shapes = [e for e in scene.elements if e.kind != "line_segment"]
    lines = [e for e in scene.elements if e.kind == "line_segment"]
    for line in lines:
        s = line.geometry.get("start_px", [0, 0])
        t = line.geometry.get("end_px", [0, 0])
        best: list[tuple[float, VisualElement, str]] = []
        for shape in shapes:
            ds = _point_to_box(s, shape.bbox)
            dt = _point_to_box(t, shape.bbox)
            if ds < 28:
                best.append((ds, shape, "start"))
            if dt < 28:
                best.append((dt, shape, "end"))
        for d, shape, endpoint in sorted(best, key=lambda x: x[0])[:4]:
            scene.relations.append(Relation(line.id, "endpoint_connects_candidate", shape.id, round(max(0.35, 1-d/28), 3), {"endpoint": endpoint, "distance_px": round(d,2)}))


def _repetition(scene: VisualScene) -> None:
    groups: defaultdict[str, list[VisualElement]] = defaultdict(list)
    for e in scene.elements:
        if e.kind == "line_segment":
            key = e.kind + ":" + str(e.geometry.get("orientation")) + ":" + str(round(e.bbox.width/max(e.bbox.height,1),1))
        else:
            key = e.kind + ":" + str(round(e.bbox.width/max(e.bbox.height,1),1))
        groups[key].append(e)
    for key, members in groups.items():
        if len(members) < 3:
            continue
        ordered = sorted(members, key=lambda e: (e.center.y, e.center.x))
        spread = math.hypot(ordered[-1].center.x-ordered[0].center.x, ordered[-1].center.y-ordered[0].center.y)
        if spread <= 0:
            continue
        for i, a in enumerate(ordered):
            for b in ordered[i+1:]:
                scene.relations.append(Relation(a.id, "same_visual_family_as", b.id, 0.72, {"family": key}))


def _symmetry(scene: VisualScene) -> None:
    elems = [e for e in scene.elements if e.kind != "line_segment"]
    if len(elems) < 2:
        return
    w = float(scene.image.get("width", 1)); h = float(scene.image.get("height", 1))
    cx = w/2; cy = h/2
    for a in elems:
        for b in elems:
            if a.id >= b.id:
                continue
            mx = abs((a.center.x+b.center.x)/2-cx)/w
            my = abs((a.center.y+b.center.y)/2-cy)/h
            dx = abs(abs(a.center.x-cx)-abs(b.center.x-cx))/w
            dy = abs(abs(a.center.y-cy)-abs(b.center.y-cy))/h
            if mx < .025 and dx < .025 and abs(a.bbox.width-b.bbox.width) < max(a.bbox.width,b.bbox.width)*.15:
                scene.relations.append(Relation(a.id, "approximately_mirrored_horizontally", b.id, 0.68))
            if my < .025 and dy < .025 and abs(a.bbox.height-b.bbox.height) < max(a.bbox.height,b.bbox.height)*.15:
                scene.relations.append(Relation(a.id, "approximately_mirrored_vertically", b.id, 0.68))


def _groups(scene: VisualScene) -> None:
    # Connected components over candidate connection/alignment relations.
    adjacency: dict[str, set[str]] = defaultdict(set)
    strong = {"endpoint_connects_candidate", "touch_or_connect_candidate"}
    for r in scene.relations:
        if r.relation in strong and r.confidence >= .5:
            adjacency[r.source].add(r.target); adjacency[r.target].add(r.source)
    seen: set[str] = set(); gid = 1
    by_id = {e.id: e for e in scene.elements}
    for seed in list(by_id):
        if seed in seen or seed not in adjacency:
            continue
        stack=[seed]; comp=[]; seen.add(seed)
        while stack:
            x=stack.pop(); comp.append(x)
            for y in adjacency[x]:
                if y not in seen:
                    seen.add(y); stack.append(y)
        if len(comp) >= 2:
            for eid in comp:
                by_id[eid].geometry["structure_group"] = f"group_{gid}"
            gid += 1


def _point_to_box(p: list[float], box: Any) -> float:
    x,y = p
    dx = max(box.x-x, 0, x-(box.x+box.width))
    dy = max(box.y-y, 0, y-(box.y+box.height))
    return math.hypot(dx,dy)
