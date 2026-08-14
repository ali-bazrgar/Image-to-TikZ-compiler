from __future__ import annotations

import json
from typing import Any

from .vir import VisualElement, VisualScene


def to_json(scene: VisualScene, indent: int = 2) -> str:
    return json.dumps(scene.to_dict(), ensure_ascii=False, indent=indent)


def to_llm_context(scene: VisualScene) -> str:
    """Produce a model-agnostic textual scene description.

    The text is deliberately redundant with the JSON: weaker models often recover
    geometry and relationships more reliably from short declarative sentences.
    """
    lines: list[str] = []
    img = scene.image
    lines.append("VISUAL_SCENE_VIR v0.1")
    lines.append(f"CANVAS: width={img['width']}px height={img['height']}px")
    lines.append("COORDINATES: origin=top-left; x→right; y→down; normalized coordinates are x/width and y/height")
    if scene.semantic_summary:
        lines.append("SUMMARY: " + scene.semantic_summary)
    if scene.texts:
        lines.append("TEXT:")
        for t in scene.texts:
            b = t.bbox
            lines.append(f"- {t.id}: {t.text!r} at ({b.x:.1f},{b.y:.1f}) size ({b.width:.1f}x{b.height:.1f}) conf={t.confidence}")
    if scene.elements:
        lines.append("ELEMENTS:")
        for e in scene.elements:
            b = e.bbox
            cx, cy = e.center.x, e.center.y
            geom = _geometry_sentence(e)
            refs = f" labels={','.join(e.text_refs)}" if e.text_refs else ""
            lines.append(
                f"- {e.id}: type={e.kind}; center=({cx:.1f},{cy:.1f}); bbox=({b.x:.1f},{b.y:.1f},{b.width:.1f},{b.height:.1f}); confidence={e.confidence:.2f}; {geom}{refs}"
            )
    if scene.relations:
        lines.append("RELATIONS:")
        for r in scene.relations:
            ev = "" if not r.evidence else " " + json.dumps(r.evidence, ensure_ascii=False, separators=(",", ":"))
            lines.append(f"- {r.source} --{r.relation}--> {r.target} confidence={r.confidence:.2f}{ev}")
    if scene.warnings:
        lines.append("WARNINGS:")
        lines.extend(f"- {w}" for w in scene.warnings)
    lines.append("SEMANTIC_TASK: infer the likely diagram meaning only from the evidence above; distinguish observed facts from hypotheses.")
    return "\n".join(lines)


def to_compact_prompt(scene: VisualScene) -> str:
    """Prompt payload suitable for a generic text-only LLM before TikZ generation."""
    context = to_llm_context(scene)
    return (
        "You are given a deterministic visual intermediate representation of a raster diagram.\n"
        "Treat ELEMENTS, TEXT and RELATIONS as observations. Do not invent coordinates or objects without marking them as hypotheses.\n"
        "Infer the diagram's semantic structure, then generate TikZ that reproduces that structure.\n\n"
        "<VIR>\n" + context + "\n</VIR>\n"
        "Return: (1) a concise semantic interpretation, (2) TikZ code, (3) uncertainties."
    )


def _geometry_sentence(e: VisualElement) -> str:
    g: dict[str, Any] = e.geometry
    if e.kind == "line":
        return f"line start={g.get('start')} end={g.get('end')} angle={g.get('angle_deg')}deg orientation={g.get('orientation')}"
    if e.kind in {"rectangle", "polygon"}:
        return f"vertices={g.get('vertices', [])}"
    if e.kind == "circle_or_ellipse":
        return f"shape circularity={g.get('circularity')} area={g.get('area_px2')}"
    return "geometry=" + json.dumps(g, ensure_ascii=False, separators=(",", ":"))
