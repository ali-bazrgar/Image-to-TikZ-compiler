from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .vir import VisualElement, VisualScene


def _j(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _bbox(e: VisualElement) -> str:
    b = e.bbox
    return f"({b.x:.1f},{b.y:.1f},{b.width:.1f},{b.height:.1f})"


def _compact_element(e: VisualElement) -> str:
    bits = [f"{e.id}:{e.kind}", f"b={_bbox(e)}", f"c=({e.center.x:.1f},{e.center.y:.1f})", f"q={e.confidence:.2f}"]
    if e.geometry:
        bits.append("g=" + _j(e.geometry))
    if e.style:
        bits.append("s=" + _j(e.style))
    if e.labels:
        bits.append("l=" + _j(e.labels))
    if e.text_refs:
        bits.append("t=" + _j(e.text_refs))
    return ";".join(bits)


def _clip_complete(items: list[str], budget: int, sep: str = "\n") -> tuple[list[str], int]:
    kept: list[str] = []
    used = 0
    for item in items:
        extra = len(item) if not kept else len(sep) + len(item)
        if used + extra > budget:
            break
        kept.append(item)
        used += extra
    return kept, used


def to_compact_llm_context(scene: VisualScene, max_chars: int = 32000) -> str:
    """Build a geometry-first, section-aware LLM record.

    No arbitrary character slicing is performed. Whole primitive records are
    retained or omitted, with secondary semantic evidence receiving the first
    truncation pressure. scene.json remains the lossless machine record.
    """
    img = scene.image
    w = float(img.get("width", img.get("width_px", 1)) or 1)
    h = float(img.get("height", img.get("height_px", 1)) or 1)
    kinds = Counter(e.kind for e in scene.elements)

    header = [
        "IMAGE_TO_TIKZ_VISUAL_RECORD v4",
        f"CANVAS {int(w)}x{int(h)}px; origin=(0,0) top-left; normalized=(x/W,y/H)",
        "AUTHORITY: measured element geometry/endpoints/curve points and text-region locations.",
        "SECONDARY: graph, spatial relations, routing, detector labels and VLM hypotheses.",
        "",
        "INVENTORY",
        *[f"{kind}={count}" for kind, count in sorted(kinds.items())],
    ]

    domain_lines: list[str] = []
    routing = img.get("domain_routing") or {}
    for item in (routing.get("ranked") or [])[:3]:
        evidence = ";".join(item.get("evidence", [])[:3])
        domain_lines.append(f"{item['domain']} score={float(item['score']):.2f} evidence={evidence}")

    graph_lines: list[str] = []
    graph = img.get("canonical_graph") or {}
    if graph:
        graph_lines.append(f"nodes={graph.get('node_count',0)} edges={graph.get('edge_count',0)} components={graph.get('component_count',0)}")
        for comp in graph.get("components", [])[:20]:
            graph_lines.append(f"component {comp['id']} nodes={','.join(comp['node_ids'])}")
        edges = [
            f"{edge['source']}-[{edge['relation']}]->{edge['target']}:{float(edge['confidence']):.2f}"
            for edge in graph.get("edges", [])[:80]
        ]
        if graph.get("edge_count", 0) > 80:
            edges.append(f"edges_truncated kept=80 total={graph.get('edge_count',0)}")
        if edges:
            graph_lines.append("edges " + " | ".join(edges))

    detector_lines: list[str] = []
    specialized = img.get("specialized_detectors") or {}
    if specialized:
        detector_lines.append(f"domain={specialized.get('domain')} method={specialized.get('method')}")
        counts = specialized.get("evidence_counts", {})
        if counts:
            detector_lines.append("counts=" + _j(counts))

    # Geometry is the highest-priority payload. Merged line segments are cheap
    # enough to retain broadly; noisy/low-confidence auxiliary elements come later.
    def priority(e: VisualElement) -> tuple:
        kind_bonus = 0 if e.kind == "line_segment" else (-1 if e.kind in {"quadrilateral", "circle_or_ellipse", "polygon", "curve_path", "polyline_or_arc"} else 1)
        return (kind_bonus, -float(e.confidence), -float(e.bbox.width * e.bbox.height))

    element_lines = [_compact_element(e) for e in sorted(scene.elements, key=priority)]
    element_header = ["ELEMENTS"]

    text_lines: list[str] = []
    for t in scene.texts:
        b = t.bbox
        text = t.text if t.text else "[undecoded]"
        text_lines.append(f"{t.id}:{text!r};role={t.role or 'unknown'};b=({b.x:.1f},{b.y:.1f},{b.width:.1f},{b.height:.1f});q={t.confidence:.2f}")

    graph_edges = {(e.get("source"), e.get("relation"), e.get("target")) for e in graph.get("edges", [])}
    relation_lines: list[str] = []
    for r in scene.relations:
        if (r.source, r.relation, r.target) not in graph_edges:
            suffix = f";ev={_j(r.evidence)}" if r.evidence else ""
            relation_lines.append(f"{r.source}-[{r.relation}]->{r.target}:{r.confidence:.2f}{suffix}")
            if len(relation_lines) >= 60:
                break

    vlm_lines: list[str] = []
    for item in (img.get("micro_vlm_hypotheses") or [])[:3]:
        text = str(item.get("text", "")).replace("\n", " ").strip()
        vlm_lines.append(f"crop={item.get('crop_id')} b={item.get('bbox_px')} p={float(item.get('priority',0)):.2f} hint={text[:400]}")

    summary_lines = [str(scene.semantic_summary).strip()[:800]] if scene.semantic_summary else []
    warning_lines = [f"- {w}" for w in scene.warnings[:8]] if scene.warnings else []

    base = header + ["", "ELEMENTS"]
    base_text = "\n".join(base)
    remaining = max(0, max_chars - len(base_text) - 1200)

    # Reserve enough budget for all sections before adding optional evidence.
    element_budget = int(remaining * 0.68)
    text_budget = int(remaining * 0.14)
    graph_budget = int(remaining * 0.10)
    misc_budget = remaining - element_budget - text_budget - graph_budget

    kept_elements, _ = _clip_complete(element_lines, element_budget)
    if len(kept_elements) < len(element_lines):
        kept_elements.append(f"ELEMENTS_TRUNCATED kept={len(kept_elements)} total={len(element_lines)}; see scene.json for full fidelity")

    kept_text, _ = _clip_complete(text_lines, text_budget)
    kept_graph, _ = _clip_complete(graph_lines, graph_budget)
    misc_items = []
    if domain_lines:
        misc_items += ["DOMAIN", *domain_lines]
    if detector_lines:
        misc_items += ["DETECTOR_EVIDENCE", *detector_lines]
    if vlm_lines:
        misc_items += ["VLM_HINTS", *vlm_lines]
    if summary_lines:
        misc_items += ["SUMMARY", *summary_lines]
    if warning_lines:
        misc_items += ["WARNINGS", *warning_lines]
    kept_misc, _ = _clip_complete(misc_items, misc_budget)

    sections = [base_text]
    sections.append("\n".join(["  " + line for line in kept_elements]))
    if kept_text:
        sections.append("\nTEXT\n" + "\n".join("  " + line for line in kept_text))
    if kept_graph:
        sections.append("\nGRAPH_DERIVED\n" + "\n".join("  " + line for line in kept_graph))
    if relation_lines:
        sections.append("\nRELATIONS_SECONDARY\n" + "\n".join("  " + line for line in relation_lines[: max(1, misc_budget // 90)]))
    if kept_misc:
        sections.append("\n" + "\n".join(kept_misc))
    sections.append(
        "\nDOWNSTREAM_TASK\n"
        "Reconstruct the measured drawing. Map every generated primitive to an element ID. "
        "Use derived graph/semantic evidence only to organize existing measured geometry; never invent geometry. "
        "Output one compilable TikZ figure."
    )
    return "\n".join(sections)
