from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .vir import VisualElement, VisualScene


def to_json(scene: VisualScene, indent: int = 2) -> str:
    return json.dumps(scene.to_dict(), ensure_ascii=False, indent=indent)


def to_llm_context(scene: VisualScene) -> str:
    """Serialize deterministic observations as a canonical, model-agnostic visual record."""
    img = scene.image
    kinds = Counter(e.kind for e in scene.elements)
    w = float(img.get("width", img.get("width_px", 1)) or 1)
    h = float(img.get("height", img.get("height_px", 1)) or 1)
    lines = [
        "IMAGE_TO_TIKZ_VISUAL_RECORD v0.4",
        "STATUS: deterministic computer-vision observations; no semantic model was used.",
        f"CANVAS: width={int(w)}px height={int(h)}px",
        "COORDINATE_SYSTEM: origin=top-left; x→right; y→down; normalized=(x/width,y/height)",
        "",
        "INVENTORY:",
    ]
    for kind, count in sorted(kinds.items()):
        lines.append(f"- {kind}: {count}")
    if scene.semantic_summary:
        lines.extend(["", "SUMMARY:", scene.semantic_summary])
    if scene.elements:
        lines.extend(["", "ELEMENTS:"])
        for e in scene.elements:
            b = e.bbox
            nb = (b.x / w, b.y / h, b.width / w, b.height / h)
            lines.append(f"- {e.id}: kind={e.kind}; center_px=({e.center.x:.2f},{e.center.y:.2f}); bbox_px=({b.x:.2f},{b.y:.2f},{b.width:.2f},{b.height:.2f}); bbox_norm=({nb[0]:.5f},{nb[1]:.5f},{nb[2]:.5f},{nb[3]:.5f}); confidence={e.confidence:.2f}")
            if e.geometry:
                lines.append("  geometry=" + json.dumps(e.geometry, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            if e.labels:
                lines.append(f"  labels={e.labels}")
            if e.text_refs:
                lines.append(f"  nearby_text_regions={e.text_refs}")
    if scene.texts:
        lines.extend(["", "TEXT_REGIONS:"])
        for t in scene.texts:
            b = t.bbox
            content = t.text if t.text else "[undecoded]"
            lines.append(f"- {t.id}: content={content!r}; bbox_px=({b.x:.2f},{b.y:.2f},{b.width:.2f},{b.height:.2f}); center_px=({b.center.x:.2f},{b.center.y:.2f}); confidence={t.confidence}")
    if scene.relations:
        lines.extend(["", "RELATIONS:"])
        for r in scene.relations:
            evidence = "" if not r.evidence else " evidence=" + json.dumps(r.evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            lines.append(f"- {r.source} --{r.relation}--> {r.target}; confidence={r.confidence:.2f}{evidence}")
    lines.extend([
        "",
        "INTERPRETATION_RULES:",
        "- Treat geometry, coordinates, and detected text regions as observations.",
        "- Treat names such as axis_or_baseline, symmetry, connection, or group membership as hypotheses unless supported by multiple observations.",
        "- Preserve topology, repetition, alignment, relative scale, and spatial ordering before styling.",
        "- Never fabricate unreadable labels; keep their region as undecoded text when character content is unavailable.",
        "- When evidence conflicts, retain both candidates and lower confidence rather than silently choosing one.",
        "",
        "DOWNSTREAM_LLM_TASK:",
        "Infer the most plausible diagram semantics from this record, then map the measured geometry into TikZ. Clearly separate observations from semantic hypotheses."
    ])
    if scene.warnings:
        lines.extend(["", "WARNINGS:"])
        lines.extend(f"- {w}" for w in scene.warnings)
    return "\n".join(lines)


def to_compact_prompt(scene: VisualScene) -> str:
    return (
        "You are given a deterministic visual record produced without any AI model.\n"
        "First infer the diagram's semantic structure from the evidence. Do not invent measurements.\n"
        "Then generate compilable TikZ that preserves geometry and topology. Mark uncertain assumptions.\n\n"
        "<VISUAL_RECORD>\n" + to_llm_context(scene) + "\n</VISUAL_RECORD>"
    )


def _geometry_sentence(e: VisualElement) -> str:
    return json.dumps(e.geometry, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
