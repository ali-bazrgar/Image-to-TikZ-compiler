from __future__ import annotations

from collections import Counter
from typing import Any

from .vir import VisualScene


ELEMENT_SOURCE_BY_KIND = {
    "quadrilateral": "deterministic.geometry.contour",
    "polygon": "deterministic.geometry.contour",
    "circle_or_ellipse": "deterministic.geometry.circular_contour",
    "line_segment": "deterministic.geometry.hough_line",
    "curve_path": "deterministic.curves.contour_path",
    "polyline_or_arc": "deterministic.curves.contour_path",
}


def enrich_provenance(scene: VisualScene) -> VisualScene:
    """Attach stable provenance labels without modifying measured values."""
    element_sources: dict[str, str] = {}
    for element in scene.elements:
        source = ELEMENT_SOURCE_BY_KIND.get(element.kind, "deterministic.visual_element")
        if element.geometry.get("arrowhead_candidate"):
            source += "+deterministic.specialized.arrow_geometry"
        if element.geometry.get("axis_candidate") or element.geometry.get("reference_line_candidate"):
            source += "+deterministic.specialized.axis_reference"
        if element.geometry.get("region_candidate"):
            source += "+deterministic.specialized.region"
        element_sources[element.id] = source

    text_sources = {
        text.id: ("optional.ocr" if text.text else "deterministic.text_region")
        for text in scene.texts
    }

    relation_sources: list[dict[str, Any]] = []
    for relation in scene.relations:
        source = "deterministic.relation_inference"
        detector = relation.evidence.get("specialized_detector")
        if detector:
            source += f"+deterministic.specialized.{detector}"
        relation_sources.append({
            "source": relation.source,
            "relation": relation.relation,
            "target": relation.target,
            "provenance": source,
        })

    hypotheses = []
    if scene.image.get("domain_routing"):
        hypotheses.append({"type": "domain_routing", "provenance": "deterministic.domain_router"})
    if scene.image.get("specialized_detectors"):
        hypotheses.append({"type": "specialized_detector_summary", "provenance": "deterministic.specialized_detectors"})
    if scene.image.get("micro_vlm_hypotheses"):
        hypotheses.append({"type": "micro_vlm", "provenance": "optional.local_vlm"})

    scene.image["provenance"] = {
        "schema": "image-to-tikz/provenance",
        "version": "1.0",
        "element_sources": element_sources,
        "text_sources": text_sources,
        "relation_sources": relation_sources,
        "hypotheses": hypotheses,
        "summary": {
            "element_source_counts": dict(Counter(element_sources.values())),
            "text_source_counts": dict(Counter(text_sources.values())),
            "relation_count": len(relation_sources),
        },
    }
    return scene
