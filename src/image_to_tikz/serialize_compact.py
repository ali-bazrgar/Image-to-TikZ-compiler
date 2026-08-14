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


def to_compact_llm_context(scene: VisualScene) -> str:
    img = scene.image
    w = float(img.get("width", img.get("width_px", 1)) or 1)
    h = float(img.get("height", img.get("height_px", 1)) or 1)
    kinds = Counter(e.kind for e in scene.elements)
    out: list[str] = [
        "IMAGE_TO_TIKZ_VISUAL_RECORD v2",
        f"CANVAS {int(w)}x{int(h)}px; origin=(0,0) top-left; normalized=(x/W,y/H)",
        "RULE: measured geometry/topology are authoritative; semantic labels are hypotheses.",
        "",
        "INVENTORY",
    ]
    out.extend(f"{kind}={count}" for kind, count in sorted(kinds.items()))

    routing = img.get("domain_routing") or {}
    ranked = routing.get("ranked") or []
    if ranked:
        out += ["", "DOMAIN"]
        for item in ranked[:3]:
            evidence = ";".join(item.get("evidence", [])[:3])
            out.append(f"{item['domain']} score={float(item['score']):.2f} evidence={evidence}")

    graph = img.get("canonical_graph") or {}
    if graph:
        out += ["", "GRAPH"]
        out.append(f"nodes={graph.get('node_count',0)} edges={graph.get('edge_count',0)} components={graph.get('component_count',0)}")
        for comp in graph.get("components", [])[:60]:
            out.append(f"component {comp['id']} nodes={','.join(comp['node_ids'])}")
        for edge in graph.get("edges", [])[:600]:
            out.append(f"edge {edge['source']} -[{edge['relation']}]-> {edge['target']} conf={float(edge['confidence']):.2f}")

    specialized = img.get("specialized_detectors") or {}
    if specialized:
        out += ["", "DETECTOR_EVIDENCE"]
        out.append(f"domain={specialized.get('domain')} method={specialized.get('method')}")
        counts = specialized.get("evidence_counts", {})
        if counts:
            out.append("counts=" + _j(counts))

    out += ["", "ELEMENTS"]
    for e in scene.elements:
        bits = [f"{e.id}: {e.kind}", f"bbox={_bbox(e)}", f"center=({e.center.x:.1f},{e.center.y:.1f})", f"conf={e.confidence:.2f}"]
        if e.geometry:
            bits.append("geom=" + _j(e.geometry))
        if e.style:
            bits.append("style=" + _j(e.style))
        if e.labels:
            bits.append("labels=" + _j(e.labels))
        if e.text_refs:
            bits.append("text_refs=" + _j(e.text_refs))
        out.append(" | ".join(bits))

    if scene.texts:
        out += ["", "TEXT"]
        for t in scene.texts:
            b = t.bbox
            text = t.text if t.text else "[undecoded]"
            out.append(f"{t.id}: text={text!r} role={t.role or 'unknown'} bbox=({b.x:.1f},{b.y:.1f},{b.width:.1f},{b.height:.1f}) conf={t.confidence:.2f}")

    graph_edges = {(e.get("source"), e.get("relation"), e.get("target")) for e in graph.get("edges", [])}
    extra_relations = []
    for r in scene.relations:
        if (r.source, r.relation, r.target) not in graph_edges:
            extra_relations.append(r)
    if extra_relations:
        out += ["", "RELATIONS_NOT_IN_GRAPH"]
        for r in extra_relations[:400]:
            suffix = f" evidence={_j(r.evidence)}" if r.evidence else ""
            out.append(f"{r.source} -[{r.relation}]-> {r.target} conf={r.confidence:.2f}{suffix}")

    hypotheses = img.get("micro_vlm_hypotheses") or []
    if hypotheses:
        out += ["", "VLM_HINTS"]
        for item in hypotheses[:4]:
            text = str(item.get("text", "")).replace("\n", " ").strip()
            out.append(f"crop={item.get('crop_id')} bbox={item.get('bbox_px')} priority={float(item.get('priority',0)):.2f} hint={text}")

    if scene.semantic_summary:
        out += ["", "SUMMARY", str(scene.semantic_summary).strip()]
    if scene.warnings:
        out += ["", "WARNINGS"]
        out.extend(f"- {w}" for w in scene.warnings[:20])

    out += [
        "",
        "DOWNSTREAM_TASK",
        "Infer diagram class from evidence. Reconstruct topology first, then geometry, then text placement/style. Output one compilable TikZ figure. Never invent unreadable labels or measured geometry.",
    ]
    return "\n".join(out)
