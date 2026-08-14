from __future__ import annotations

from dataclasses import asdict, dataclass

from .vir import VisualScene


@dataclass(frozen=True)
class DomainHypothesis:
    domain: str
    score: float
    evidence: tuple[str, ...]
    recommended_detectors: tuple[str, ...]


DOMAINS = (
    "geometry",
    "engineering_diagram",
    "electrical_schematic",
    "flowchart",
    "plot_or_chart",
    "technical_drawing",
    "map_or_layout",
    "general_diagram",
)


def route_domain(scene: VisualScene) -> list[DomainHypothesis]:
    """Score diagram families from deterministic visual evidence only."""
    counts: dict[str, int] = {}
    for e in scene.elements:
        counts[e.kind] = counts.get(e.kind, 0) + 1
    relations = scene.relations
    rels = {r.relation for r in relations}
    text_count = len(scene.texts)
    line_count = counts.get("line_segment", 0)
    curve_count = counts.get("curve_path", 0) + counts.get("polyline_or_arc", 0)
    shape_count = sum(counts.get(k, 0) for k in ("quadrilateral", "polygon", "circle_or_ellipse"))
    repeated_shape_evidence = "same_visual_family_as" in rels
    connector_evidence = rels & {
        "endpoint_connects_candidate",
        "line_junction_candidate",
        "line_crossing_candidate",
        "touch_or_connect_candidate",
    }
    arrow_evidence = 0
    for element in scene.elements:
        if element.kind != "line_segment":
            continue
        endpoints = element.geometry.get("endpoint_candidates", [])
        if any(ep.get("possible_junction_or_arrowhead") for ep in endpoints if isinstance(ep, dict)):
            arrow_evidence += 1

    scores = {d: (0.10, []) for d in DOMAINS}

    def add(domain: str, amount: float, reason: str) -> None:
        score, evidence = scores[domain]
        scores[domain] = (score + amount, evidence + [reason])

    if shape_count >= 2 and repeated_shape_evidence:
        add("geometry", 0.28, "multiple geometric primitives with repeated visual family evidence")

    if line_count >= 4 and ({"line_junction_candidate", "line_crossing_candidate"} & rels):
        add("engineering_diagram", 0.30, "line-heavy connected topology")
        add("electrical_schematic", 0.16, "junction/crossing topology is consistent with schematic structure")

    # Strong flowchart evidence: repeated box-like shapes connected by directed-looking lines.
    if shape_count >= 3 and (connector_evidence or arrow_evidence >= 1):
        add("flowchart", 0.28, "multiple shape nodes with connector/arrow evidence")
    if shape_count >= 3 and repeated_shape_evidence and line_count >= 2:
        add("flowchart", 0.12, "repeated box-like nodes arranged with line connectors")
    if text_count >= 3 and line_count >= 4:
        add("flowchart", 0.18, "many text regions combined with line connectors")

    if curve_count >= 1:
        add("plot_or_chart", 0.16, "curved/path primitives are present")

    if line_count >= 3 and "parallel_dimension_candidate" in rels:
        # Dimension candidates alone are weak; require additional technical-drawing evidence.
        dimension_evidence = any(
            "dimension" in str(e.geometry.get("possible_role", "")).lower()
            or "dimension" in str(e.geometry.get("stroke_role", "")).lower()
            for e in scene.elements
        )
        amount = 0.32 if dimension_evidence and shape_count >= 1 else 0.12
        add("technical_drawing", amount, "parallel dimension-line pattern candidate")

    if text_count >= 2 and shape_count >= 3:
        add("map_or_layout", 0.12, "dense labeled spatial primitives")

    add("general_diagram", 0.05, "fallback domain for mixed visual evidence")

    detector_map = {
        "geometry": ("angle_normalization", "polygon_metrics", "symmetry"),
        "engineering_diagram": ("topology", "stroke_style", "arrow_geometry", "dimension_candidates"),
        "electrical_schematic": ("topology", "junctions", "symbol_regions", "labels"),
        "flowchart": ("shape_semantics", "arrow_geometry", "text_regions", "connector_graph"),
        "plot_or_chart": ("axes", "ticks", "curve_paths", "labels", "plot_regions"),
        "technical_drawing": ("dimensions", "centerlines", "projection_lines", "stroke_style"),
        "map_or_layout": ("regions", "labels", "alignment", "spatial_groups"),
        "general_diagram": ("geometry", "topology", "text_regions", "spatial_relations"),
    }
    ranked = sorted(
        (
            DomainHypothesis(
                d,
                min(score, 0.99),
                tuple(dict.fromkeys(evidence)),
                detector_map[d],
            )
            for d, (score, evidence) in scores.items()
        ),
        key=lambda x: x.score,
        reverse=True,
    )
    return ranked


def enrich_domain_routing(scene: VisualScene) -> VisualScene:
    ranked = route_domain(scene)
    scene.image["domain_routing"] = {
        "method": "deterministic_evidence_scoring",
        "top": asdict(ranked[0]) if ranked else None,
        "ranked": [asdict(x) for x in ranked],
    }
    if ranked:
        scene.semantic_summary = (
            scene.semantic_summary
            + f"\nDOMAIN_ROUTE: top={ranked[0].domain} score={ranked[0].score:.2f}."
        ).strip()
    return scene
