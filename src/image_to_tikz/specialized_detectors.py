from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .vir import Relation, VisualScene


DOMAIN_DETECTORS = {
    "flowchart": ("arrow_geometry", "connector_graph"),
    "engineering_diagram": ("arrow_geometry", "dimension_candidates", "axes"),
    "electrical_schematic": ("junctions", "arrow_geometry", "symbol_regions"),
    "plot_or_chart": ("axes", "ticks", "plot_regions"),
    "technical_drawing": ("dimensions", "centerlines", "projection_lines", "stroke_style"),
    "map_or_layout": ("regions", "alignment", "spatial_groups"),
    "geometry": ("angle_normalization", "symmetry"),
    "general_diagram": ("arrow_geometry", "axes", "dimension_candidates"),
}


def enrich_specialized_detectors(scene: VisualScene, image_path: str | Path) -> VisualScene:
    """Run deterministic domain-specific detectors selected from the top routed domain."""
    routing = scene.image.get("domain_routing", {})
    top = routing.get("top") or {}
    domain = top.get("domain", "general_diagram")
    requested = tuple(top.get("recommended_detectors") or DOMAIN_DETECTORS.get(domain, ()))
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        scene.warnings.append(f"Specialized detectors skipped: could not decode {image_path}")
        return scene

    run = set(requested)
    evidence: dict[str, int] = {}
    if "arrow_geometry" in run:
        evidence["arrow_candidates"] = _detect_arrows(scene, image)
    if "axes" in run:
        evidence["axis_candidates"] = _detect_axes(scene)
    if "dimension_candidates" in run or "dimensions" in run:
        evidence["dimension_candidates"] = _detect_dimensions(scene)
    if "ticks" in run:
        evidence["tick_candidates"] = _detect_ticks(scene)
    if "projection_lines" in run or "centerlines" in run:
        evidence["reference_line_candidates"] = _detect_reference_lines(scene)
    if "junctions" in run:
        evidence["junction_candidates"] = _detect_junctions(scene)
    if "symbol_regions" in run or "plot_regions" in run or "regions" in run:
        evidence["region_candidates"] = _detect_regions(scene)

    scene.image["specialized_detectors"] = {
        "domain": domain,
        "detectors": sorted(run),
        "evidence_counts": evidence,
        "method": "deterministic_domain_routed",
    }
    scene.semantic_summary = (
        scene.semantic_summary
        + "\nSPECIALIZED_DETECTORS: domain="
        + str(domain)
        + " detectors="
        + ",".join(sorted(run))
        + "."
    ).strip()
    return scene


def _detect_arrows(scene: VisualScene, gray: np.ndarray) -> int:
    edge = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 60, 160)
    lines = cv2.HoughLinesP(
        edge, 1, np.pi / 180,
        threshold=max(12, gray.shape[1] // 45),
        minLineLength=max(10, gray.shape[0] // 35),
        maxLineGap=max(4, gray.shape[0] // 120),
    )
    if lines is None:
        return 0
    arr = np.asarray(lines).reshape(-1, 4)
    candidates = 0
    elements = [e for e in scene.elements if e.kind == "line_segment"]
    for x1, y1, x2, y2 in arr:
        length = math.hypot(float(x2 - x1), float(y2 - y1))
        if length < 12 or not elements:
            continue
        for x, y, name in ((int(x1), int(y1), "start"), (int(x2), int(y2), "end")):
            density = _dark_neighbor_density(gray, x, y)
            if density <= 0.42:
                continue
            nearest = min(elements, key=lambda e: _endpoint_distance(e, x, y))
            endpoint = _closest_endpoint(nearest, x, y)
            candidates += 1
            nearest.geometry.setdefault("arrowhead_candidates", []).append(
                {"endpoint": endpoint, "x_px": x, "y_px": y, "density": round(density, 3)}
            )
            nearest.geometry["arrowhead_candidate"] = True
            nearest.geometry["arrowhead_evidence"] = max(
                float(nearest.geometry.get("arrowhead_evidence", 0.0)), min(0.99, density)
            )
            scene.relations.append(
                Relation(
                    nearest.id,
                    "arrowhead_candidate",
                    nearest.id,
                    min(0.95, max(0.4, density)),
                    {"endpoint": endpoint, "x_px": x, "y_px": y, "specialized_detector": "arrow_geometry"},
                )
            )
    return candidates


def _endpoint_distance(element, x: int, y: int) -> float:
    starts = element.geometry.get("start_px", [element.center.x, element.center.y])
    ends = element.geometry.get("end_px", [element.center.x, element.center.y])
    return min(math.hypot(float(starts[0]) - x, float(starts[1]) - y), math.hypot(float(ends[0]) - x, float(ends[1]) - y))


def _closest_endpoint(element, x: int, y: int) -> str:
    starts = element.geometry.get("start_px", [element.center.x, element.center.y])
    ends = element.geometry.get("end_px", [element.center.x, element.center.y])
    start_d = math.hypot(float(starts[0]) - x, float(starts[1]) - y)
    end_d = math.hypot(float(ends[0]) - x, float(ends[1]) - y)
    return "start" if start_d <= end_d else "end"


def _dark_neighbor_density(gray: np.ndarray, x: int, y: int) -> float:
    r = max(3, min(gray.shape) // 110)
    crop = gray[max(0, y - r): min(gray.shape[0], y + r + 1), max(0, x - r): min(gray.shape[1], x + r + 1)]
    return float((crop < 90).mean()) if crop.size else 0.0


def _detect_axes(scene: VisualScene) -> int:
    count = 0
    for e in scene.elements:
        if e.kind != "line_segment":
            continue
        length = float(e.geometry.get("length_px", 0))
        orientation = e.geometry.get("orientation")
        if orientation in {"horizontal", "vertical"} and length >= 0.20 * max(scene.image.get("width", 1), scene.image.get("height", 1)):
            e.geometry["axis_candidate"] = True
            count += 1
    return count


def _detect_dimensions(scene: VisualScene) -> int:
    count = 0
    for r in scene.relations:
        if r.relation == "parallel_dimension_candidate":
            r.evidence["specialized_detector"] = "dimension_candidates"
            count += 1
    return count


def _detect_ticks(scene: VisualScene) -> int:
    count = 0
    lines = [e for e in scene.elements if e.kind == "line_segment"]
    for a in lines:
        for b in lines:
            if a.id >= b.id:
                continue
            if a.geometry.get("orientation") == b.geometry.get("orientation"):
                continue
            if abs(a.center.x - b.center.x) < 12 or abs(a.center.y - b.center.y) < 12:
                count += 1
    return count


def _detect_reference_lines(scene: VisualScene) -> int:
    count = 0
    for e in scene.elements:
        if e.kind == "line_segment" and e.geometry.get("possible_role") == "axis_or_baseline":
            e.geometry["reference_line_candidate"] = True
            count += 1
    return count


def _detect_junctions(scene: VisualScene) -> int:
    count = 0
    for r in scene.relations:
        if r.relation in {"line_junction_candidate", "line_crossing_candidate", "touch_or_connect_candidate"}:
            r.evidence["specialized_detector"] = "junctions"
            count += 1
    return count


def _detect_regions(scene: VisualScene) -> int:
    count = 0
    for e in scene.elements:
        if e.kind in {"quadrilateral", "polygon", "circle_or_ellipse", "curve_path", "polyline_or_arc"}:
            e.geometry.setdefault("region_candidate", True)
            count += 1
    return count
