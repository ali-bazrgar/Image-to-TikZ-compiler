from __future__ import annotations

import json
from collections import Counter
from typing import Any, Iterable

from .vir import VisualElement, VisualScene


def _j(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _bbox(e: VisualElement) -> str:
    b = e.bbox
    return f"({b.x:.1f},{b.y:.1f},{b.width:.1f},{b.height:.1f})"


def _chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


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


def to_compact_llm_context(scene: VisualScene, max_chars: int = 32000) -> str:
    """Build a high-density LLM record while keeping scene.json lossless.

    Repetitive primitives are batched, evidence counts are bounded, and a
    deterministic character ceiling prevents runaway contexts. Full fidelity
    remains available in scene.json.
    """
    img = scene.image
    w = float(img.get("width", img.get("width_px", 1)) or 1)
    h = float(img.get("height", img.get("height_px", 1)) or 1)
    kinds = Counter(e.kind for e in scene.elements)
    out: list[str] = [
        "IMAGE_TO_TIKZ_VISUAL_RECORD v3",
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
        for comp in graph.get("components", [])[:40]:
            out.append(f"component {comp['id']} nodes={','.join(comp['node_ids'])}")
        edges = [
            f"{edge['source']}-[{edge['relation']}]->{edge['target']}:{float(edge['confidence']):.2f}"
            for edge in graph.get("edges", [])[:240]
        ]
        for batch in _chunks(edges, 10):
            out.append("edges " + " | ".join(batch))
        if graph.get("edge_count", 0) > 240:
            out.append(f"edges_truncated kept=240 total={graph.get('edge_count',0)}")

    specialized = img.get("specialized_detectors") or {}
    if specialized:
        out += ["", "DETECTOR_EVIDENCE"]
        out.append(f"domain={specialized.get('domain')} method={specialized.get('method')}")
        counts = specialized.get("evidence_counts", {})
        if counts:
            out.append("counts=" + _j(counts))

    out += ["", "ELEMENTS"]
    grouped: dict[str, list[str]] = {}
    for e in scene.elements:
        grouped.setdefault(e.kind, []).append(_compact_element(e))
    for kind in sorted(grouped):
        out.append(f"{kind} ({len(grouped[kind])})")
        for batch in _chunks(grouped[kind], 6):
            out.append("  " + " || ".join(batch))

    if scene.texts:
        out += ["", "TEXT"]
        texts: list[str] = []
        for t in scene.texts:
            b = t.bbox
            text = t.text if t.text else "[undecoded]"
            texts.append(f"{t.id}:{text!r};role={t.role or 'unknown'};b=({b.x:.1f},{b.y:.1f},{b.width:.1f},{b.height:.1f});q={t.confidence:.2f}")
        for batch in _chunks(texts, 5):
            out.append("  " + " || ".join(batch))

    graph_edges = {(e.get("source"), e.get("relation"), e.get("target")) for e in graph.get("edges", [])}
    extra_relations = [r for r in scene.relations if (r.source, r.relation, r.target) not in graph_edges]
    if extra_relations:
        out += ["", "RELATIONS_NOT_IN_GRAPH"]
        rels = []
        for r in extra_relations[:120]:
            suffix = f";ev={_j(r.evidence)}" if r.evidence else ""
            rels.append(f"{r.source}-[{r.relation}]->{r.target}:{r.confidence:.2f}{suffix}")
        for batch in _chunks(rels, 10):
            out.append("  " + " | ".join(batch))
        if len(extra_relations) > 120:
            out.append(f"relations_truncated kept=120 total={len(extra_relations)}")

    hypotheses = img.get("micro_vlm_hypotheses") or []
    if hypotheses:
        out += ["", "VLM_HINTS"]
        for item in hypotheses[:3]:
            text = str(item.get("text", "")).replace("\n", " ").strip()
            out.append(f"crop={item.get('crop_id')} b={item.get('bbox_px')} p={float(item.get('priority',0)):.2f} hint={text[:500]}")

    if scene.semantic_summary:
        out += ["", "SUMMARY", str(scene.semantic_summary).strip()[:1200]]
    if scene.warnings:
        out += ["", "WARNINGS"]
        out.extend(f"- {w}" for w in scene.warnings[:12])

    out += [
        "",
        "DOWNSTREAM_TASK",
        "Infer diagram class from evidence. Reconstruct topology first, then geometry, then text placement/style. Output one compilable TikZ figure. Never invent unreadable labels or measured geometry.",
    ]

    text = "\n".join(out)
    if len(text) <= max_chars:
        return text

    marker = f"\n\nCONTEXT_TRUNCATED chars={len(text)} limit={max_chars}; FULL_DATA_IN_SCENE_JSON"
    return text[: max_chars - len(marker)] + marker
