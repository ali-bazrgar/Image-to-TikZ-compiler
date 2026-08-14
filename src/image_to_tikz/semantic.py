from __future__ import annotations

from collections import Counter

from .vir import VisualScene


def render_llm_context(scene: VisualScene) -> str:
    """Create a deterministic text-only description for any downstream LLM.

    This function does no semantic inference itself. It exposes measured geometry,
    detected text regions, candidate relations, coordinate conventions, and
    uncertainty so a separate language model can reason from explicit evidence.
    """
    kinds = Counter(e.kind for e in scene.elements)
    img = scene.image
    lines = [
        "IMAGE_TO_TIKZ_VISUAL_RECORD v0.2",
        "PURPOSE: Reconstruct the pictured diagram in TikZ from deterministic image measurements.",
        f"CANVAS: width={img.get('width', img.get('width_px', 0))}px height={img.get('height', img.get('height_px', 0))}px",
        "COORDINATE_SYSTEM: origin=top-left; x increases rightward; y increases downward; geometry is measured in pixels; normalized coordinates are derived by dividing by canvas width/height.",
        "IMPORTANT: OBSERVED facts and HYPOTHESES must never be confused. The computer-vision stage does not understand domain semantics.",
        "",
        "INVENTORY:",
    ]
    for kind, count in sorted(kinds.items()):
        lines.append(f"- {kind}: {count}")

    if scene.texts:
        lines.extend(["", "TEXT_REGIONS:"])
        for t in scene.texts:
            b = t.bbox
            content = t.text if t.text else "[text content not decoded by the model-free pipeline]"
            lines.append(
                f"- {t.id}: content={content!r}; bbox_px=({b.x:.1f},{b.y:.1f},{b.width:.1f},{b.height:.1f}); center_px=({t.bbox.center.x:.1f},{t.bbox.center.y:.1f}); confidence={t.confidence}"
            )

    if scene.elements:
        lines.extend(["", "ELEMENTS:"])
        for e in scene.elements:
            b = e.bbox
            n = _normalized_box(b, img)
            lines.append(
                f"- {e.id}: kind={e.kind}; bbox_px=({b.x:.1f},{b.y:.1f},{b.width:.1f},{b.height:.1f}); bbox_norm=({n[0]:.5f},{n[1]:.5f},{n[2]:.5f},{n[3]:.5f}); center_px=({e.center.x:.1f},{e.center.y:.1f}); confidence={e.confidence:.2f}"
            )
            if e.geometry:
                lines.append(f"  geometry={e.geometry}")
            if e.text_refs:
                lines.append(f"  nearby_text_regions={e.text_refs}")

    if scene.relations:
        lines.extend(["", "RELATIONS:"])
        for r in scene.relations:
            evidence = f" evidence={r.evidence}" if r.evidence else ""
            lines.append(f"- {r.source} --{r.relation}--> {r.target}; confidence={r.confidence:.2f}{evidence}")

    lines.extend(["", "SEMANTIC_GUIDANCE:"])
    lines.append("- First infer what type of diagram is most consistent with the measured primitives and arrangement.")
    lines.append("- Preserve topology, alignment, repetition, relative scale, and spatial ordering before styling.")
    lines.append("- Do not invent unreadable text. Preserve a text-region placeholder unless the downstream LLM can infer the label from context.")
    lines.append("- When a primitive classification is ambiguous, retain the low-level geometry and mark the interpretation as uncertain.")

    if scene.semantic_summary:
        lines.extend(["", "COMPUTER_VISION_SUMMARY:", scene.semantic_summary])
    if scene.warnings:
        lines.extend(["", "WARNINGS:"])
        lines.extend(f"- {w}" for w in scene.warnings)
    return "\n".join(lines)


def build_semantic_context(scene: VisualScene) -> dict:
    """Return a model-independent dictionary describing the deterministic evidence."""
    return {
        "task": "reconstruct_image_as_tikz",
        "observations": scene.to_dict(),
        "instruction": "Infer semantics only from measured evidence; clearly mark hypotheses and preserve uncertainty.",
    }


def _normalized_box(box, image: dict) -> tuple[float, float, float, float]:
    w = float(image.get("width", image.get("width_px", 1)) or 1)
    h = float(image.get("height", image.get("height_px", 1)) or 1)
    return box.x / w, box.y / h, box.width / w, box.height / h
