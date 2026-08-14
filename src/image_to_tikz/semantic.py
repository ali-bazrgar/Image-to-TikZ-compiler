from __future__ import annotations

from collections import Counter
from typing import Any

from .vir import VisualScene


def build_semantic_context(scene: VisualScene, vision: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fuse deterministic geometry and optional VLM observations without replacing measured geometry."""
    kinds = Counter(e.kind for e in scene.elements)
    context: dict[str, Any] = {
        "task": "reconstruct_image_as_tikz",
        "principles": [
            "Preserve topology and relative geometry before visual styling.",
            "Use normalized coordinates supplied by the scene as the source of measured geometry.",
            "Treat semantic vision observations as hypotheses unless supported by measured geometry.",
            "Never invent elements when evidence is absent; record uncertainty instead.",
        ],
        "scene": scene.to_dict(),
        "inventory": dict(kinds),
        "vision_observation": vision or None,
        "output_contract": {
            "goal": "Generate compilable TikZ reproducing the observed diagram.",
            "requirements": [
                "Preserve all detected text and labels when confidence is sufficient.",
                "Preserve line endpoints, relative alignment, containment, and connection topology.",
                "Prefer simple TikZ primitives over pixel tracing.",
                "Explain uncertainty in comments rather than silently changing geometry.",
            ],
        },
    }
    return context


def render_llm_context(scene: VisualScene, vision: dict[str, Any] | None = None) -> str:
    """Produce deterministic text that can be fed to a text-only LLM."""
    payload = build_semantic_context(scene, vision)
    lines = [
        "IMAGE-TO-TIKZ VISUAL INTERMEDIATE REPRESENTATION",
        f"Schema: {scene.schema} v{scene.version}",
        f"Image: {scene.image.get('width_px')}x{scene.image.get('height_px')} px",
        "Coordinate system: normalized x=0..1 left→right, y=0..1 top→bottom",
        "",
        "ELEMENTS:",
    ]
    for e in scene.elements:
        g = e.geometry
        line = f"- {e.id} | kind={e.kind} | bbox=({e.bbox.x:.4f},{e.bbox.y:.4f},{e.bbox.width:.4f},{e.bbox.height:.4f}) | confidence={e.confidence:.2f}"
        if g:
            line += f" | geometry={g}"
        if e.text_refs:
            line += f" | text_refs={e.text_refs}"
        lines.append(line)
    lines.append("")
    lines.append("TEXT:")
    for t in scene.texts:
        lines.append(f"- {t.id} | {t.text!r} | bbox=({t.bbox.x:.4f},{t.bbox.y:.4f},{t.bbox.width:.4f},{t.bbox.height:.4f})")
    lines.append("")
    lines.append("RELATIONS:")
    for r in scene.relations:
        lines.append(f"- {r.source} --{r.relation}--> {r.target} (confidence={r.confidence:.2f})")
    lines.extend(["", "SEMANTIC SUMMARY:", scene.semantic_summary or "No semantic summary available."])
    if vision:
        lines.extend(["", "VISION HYPOTHESES:", str(vision)])
    return "\n".join(lines)
