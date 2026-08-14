from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from .vir import Relation, VisualElement, VisualScene


def enrich_structure(scene: VisualScene) -> VisualScene:
    """Infer deterministic higher-level structure without semantic AI."""
    _stroke_features(scene)
    _topology(scene)
    _line_style_features(scene)
    _arrow_and_dimension_candidates(scene)
    _repetition(scene)
    _symmetry(scene)
    _groups(scene)
    return scene


def _stroke_features(scene: VisualScene) -> None:
    scale = max(float(scene.image.get("width", 1)), float(scene.image.get("height", 1)))
    for e in scene.elements:
        if e.kind != "line_segment":
            continue
        g = e.geometry
        length = float(g.get("length_px", 0))
        angle = float(g.get("angle_deg", 0))
        g["stroke_orientation"] = g.get("orientation", "unknown")
        g["length_normalized"] = round(length / max(scale, 1.0), 5)
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
        for shape in shapes:
            ds = _point_to_box(s, shape.bbox)
            dt = _point_to_box(t, shape.bbox)
            if ds < 28:
                scene.relations.append(Relation(line.id, "endpoint_connects_candidate", shape.id, round(max(0.35, 1-ds/28), 3), {"endpoint": "start", "distance_px": round(ds,2)}))
            if dt < 28:
                scene.relations.append(Relation(line.id, "endpoint_connects_candidate", shape.id, round(max(0.35, 1-dt/28), 3), {"endpoint": "end", "distance_px": round(dt,2)}))

    # Line-to-line junctions and crossings are explicit graph evidence.
    for i, a in enumerate(lines):
        a0 = a.geometry.get("start_px", [0, 0]); a1 = a.geometry.get("end_px", [0, 0])
        for b in lines[i+1:]:
            b0 = b.geometry.get("start_px", [0, 0]); b1 = b.geometry.get("end_px", [0, 0])
            best = min((_dist(a0, b0), "a_start_b_start"), (_dist(a0, b1), "a_start_b_end"),
                       (_dist(a1, b0), "a_end_b_start"), (_dist(a1, b1), "a_end_b_end"), key=lambda x: x[0])
            if best[0] <= 18:
                scene.relations.append(Relation(a.id, "line_junction_candidate", b.id, round(max(0.4, 1-best[0]/18), 3), {"endpoint_pair": best[1], "distance_px": round(best[0],2)}))
            elif _segments_near_crossing(a0, a1, b0, b1, tolerance=10):
                scene.relations.append(Relation(a.id, "line_crossing_candidate", b.id, 0.62))


def _line_style_features(scene: VisualScene) -> None:
    for e in scene.elements:
        if e.kind != "line_segment":
            continue
        g = e.geometry
        length = float(g.get("length_px", 0))
        if length < 40:
            g.setdefault("stroke_style", "unknown")
        elif "dark_run_ratio" in g:
            ratio = float(g["dark_run_ratio"])
            g["stroke_style"] = "solid_candidate" if ratio > 0.72 else ("dashed_or_dotted_candidate" if ratio < 0.45 else "mixed_or_unknown")
        else:
            g.setdefault("stroke_style", "continuous_or_unknown")


def _arrow_and_dimension_candidates(scene: VisualScene) -> None:
    lines = [e for e in scene.elements if e.kind == "line_segment"]
    for line in lines:
        endpoints = line.geometry.get("endpoint_candidates", [])
        if endpoints:
            line.geometry["possible_arrowhead_count"] = min(2, len(endpoints))
            for ep in endpoints:
                ep["interpretations"] = ["junction", "arrowhead", "line_cap"]

    for i, a in enumerate(lines):
        for b in lines[i+1:]:
            oa = a.geometry.get("orientation"); ob = b.geometry.get("orientation")
            if oa != ob or oa not in {"horizontal", "vertical"}:
                continue
            sep = _center_distance_perpendicular(a, b, oa)
            if 12 <= sep <= max(scene.image.get("width", 1), scene.image.get("height", 1))*0.25:
                scene.relations.append(Relation(a.id, "parallel_dimension_candidate", b.id, 0.55, {"orientation": oa, "separation_px": round(sep,2)}))


def _repetition(scene: VisualScene) -> None:
    groups: defaultdict[str, list[VisualElement]] = defaultdict(list)
    for e in scene.elements:
        if e.kind == "line_segment":
            ratio = e.bbox.width/max(e.bbox.height,1)
            key = e.kind + ":" + str(e.geometry.get("orientation")) + ":" + str(round(ratio,1))
        else:
            key = e.kind + ":" + str(round(e.bbox.width/max(e.bbox.height,1),1))
        groups[key].append(e)
    for key, members in groups.items():
        if len(members) < 3:
            continue
        for i, a in enumerate(members):
            for b in members[i+1:]:
                scene.relations.append(Relation(a.id, "same_visual_family_as", b.id, 0.72, {"family": key}))


def _symmetry(scene: VisualScene) -> None:
    elems = [e for e in scene.elements if e.kind != "line_segment"]
    if len(elems) < 2:
        return
    w = float(scene.image.get("width", 1)); h = float(scene.image.get("height", 1))
    cx = w/2; cy = h/2
    for i, a in enumerate(elems):
        for b in elems[i+1:]:
            mx = abs((a.center.x+b.center.x)/2-cx)/w
            my = abs((a.center.y+b.center.y)/2-cy)/h
            dx = abs(abs(a.center.x-cx)-abs(b.center.x-cx))/w
            dy = abs(abs(a.center.y-cy)-abs(b.center.y-cy))/h
            if mx < .025 and dx < .025 and abs(a.bbox.width-b.bbox.width) < max(a.bbox.width,b.bbox.width)*.15:
                scene.relations.append(Relation(a.id, "approximately_mirrored_horizontally", b.id, 0.68))
            if my < .025 and dy < .025 and abs(a.bbox.height-b.bbox.height) < max(a.bbox.height,b.bbox.height)*.15:
                scene.relations.append(Relation(a.id, "approximately_mirrored_vertically", b.id, 0.68))


def _groups(scene: VisualScene) -> None:
    adjacency: dict[str, set[str]] = defaultdict(set)
    strong = {"endpoint_connects_candidate", "touch_or_connect_candidate", "line_junction_candidate"}
    for r in scene.relations:
        if r.relation in strong and r.confidence >= .5:
            adjacency[r.source].add(r.target)
            adjacency[r.target].add(r.source)
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


def _dist(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])


def _segments_near_crossing(a0: list[float], a1: list[float], b0: list[float], b1: list[float], tolerance: float) -> bool:
    def orient(p, q, r):
        return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
    oa = orient(a0, a1, b0); ob = orient(a0, a1, b1)
    oc = orient(b0, b1, a0); od = orient(b0, b1, a1)
    if oa * ob <= 0 and oc * od <= 0:
        return True
    return min(_point_to_segment(a0, b0, b1), _point_to_segment(a1, b0, b1), _point_to_segment(b0, a0, a1), _point_to_segment(b1, a0, a1)) <= tolerance


def _point_to_segment(p, a, b):
    vx, vy = b[0]-a[0], b[1]-a[1]
    wx, wy = p[0]-a[0], p[1]-a[1]
    den = vx*vx + vy*vy
    if den == 0:
        return math.hypot(wx, wy)
    t = max(0.0, min(1.0, (wx*vx + wy*vy)/den))
    qx, qy = a[0]+t*vx, a[1]+t*vy
    return math.hypot(p[0]-qx, p[1]-qy)


def _center_distance_perpendicular(a: VisualElement, b: VisualElement, orientation: str) -> float:
    return abs(a.center.y - b.center.y) if orientation == "horizontal" else abs(a.center.x - b.center.x)
