from __future__ import annotations

import json
from collections import Counter

from .vir import VisualElement, VisualScene


def to_json(scene: VisualScene, indent: int = 2) -> str:
    return json.dumps(scene.to_dict(), ensure_ascii=False, indent=indent)


def to_llm_context(scene: VisualScene) -> str:
    """Serialize deterministic observations plus a spatial narrative for a text-only LLM."""
    img = scene.image
    kinds = Counter(e.kind for e in scene.elements)
    w = float(img.get("width", img.get("width_px", 1)) or 1)
    h = float(img.get("height", img.get("height_px", 1)) or 1)
    lines = [
        "IMAGE_TO_TIKZ_VISUAL_RECORD v0.5",
        "STATUS: deterministic computer-vision observations; no semantic model was used.",
        f"CANVAS: width={int(w)}px height={int(h)}px",
        "COORDINATE_SYSTEM: origin=top-left; x→right; y→down; normalized=(x/width,y/height)",
        "OBSERVATION_POLICY: measurements are evidence; semantic labels are hypotheses.",
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
            lines.append(
                f"- {e.id}: kind={e.kind}; center_px=({e.center.x:.2f},{e.center.y:.2f}); "
                f"bbox_px=({b.x:.2f},{b.y:.2f},{b.width:.2f},{b.height:.2f}); "
                f"bbox_norm=({nb[0]:.5f},{nb[1]:.5f},{nb[2]:.5f},{nb[3]:.5f}); confidence={e.confidence:.2f}"
            )
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
            lines.append(
                f"- {t.id}: content={content!r}; bbox_px=({b.x:.2f},{b.y:.2f},{b.width:.2f},{b.height:.2f}); "
                f"center_px=({b.center.x:.2f},{b.center.y:.2f}); confidence={t.confidence}"
            )

    if scene.relations:
        lines.extend(["", "RELATIONS:"])
        for r in scene.relations:
            evidence = "" if not r.evidence else " evidence=" + json.dumps(r.evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            lines.append(f"- {r.source} --{r.relation}--> {r.target}; confidence={r.confidence:.2f}{evidence}")

    narrative = _spatial_narrative(scene)
    if narrative:
        lines.extend(["", "SPATIAL_NARRATIVE:"])
        lines.extend(narrative)

    lines.extend([
        "",
        "INTERPRETATION_RULES:",
        "- Treat geometry, coordinates, text-region locations, containment and ordering as observations.",
        "- Treat axis, arrowhead, dimension, symmetry, group and semantic-role labels as hypotheses unless supported by multiple observations.",
        "- Reconstruct topology and relative placement before deciding stylistic details.",
        "- Never fabricate unreadable labels. Preserve an undecoded text region when character content is unavailable.",
        "- When evidence conflicts, retain alternatives and lower confidence rather than silently selecting one.",
        "",
        "DOWNSTREAM_LLM_TASK:",
        "First infer the most plausible diagram class and scene structure from the record. Then generate TikZ that preserves measured topology, geometry, repetition, containment and spatial ordering."
    ])
    if scene.warnings:
        lines.extend(["", "WARNINGS:"])
        lines.extend(f"- {w}" for w in scene.warnings)
    return "\n".join(lines)


def _spatial_narrative(scene: VisualScene) -> list[str]:
    elems = {e.id: e for e in scene.elements}
    out: list[str] = []
    for r in scene.relations:
        a = elems.get(r.source)
        b = elems.get(r.target)
        if not a or not b:
            continue
        relation = r.relation
        if relation == "contains":
            out.append(f"- {a.id} contains {b.id}.")
        elif relation == "horizontally_aligned_with":
            out.append(f"- {a.id} and {b.id} share an approximately common horizontal level.")
        elif relation == "vertically_aligned_with":
            out.append(f"- {a.id} and {b.id} share an approximately common vertical axis.")
        elif relation == "near":
            out.append(f"- {a.id} is spatially near {b.id}.")
        elif relation == "same_visual_family_as":
            out.append(f"- {a.id} and {b.id} have similar measured visual proportions/orientation.")
        elif relation in {"endpoint_connects_candidate", "line_junction_candidate", "line_crossing_candidate", "touch_or_connect_candidate"}:
            out.append(f"- {a.id} may connect or intersect with {b.id}; this is a geometric candidate, not a semantic certainty.")
        elif relation in {"approximately_mirrored_horizontally", "approximately_mirrored_vertically"}:
            out.append(f"- {a.id} and {b.id} show approximate {relation.replace('approximately_mirrored_', '')} mirror symmetry.")
        elif relation == "parallel_dimension_candidate":
            out.append(f"- {a.id} and {b.id} are approximately parallel and may form a dimension/extension-line pattern.")
        elif relation == "global_reference_line_candidate":
            out.append(f"- {a.id} is a long reference-line candidate (axis/baseline/dimension/connector are all possible).")
    # Layout language derived directly from measured centers.
    ordered = sorted(scene.elements, key=lambda e: (e.center.y, e.center.x))
    for i, e in enumerate(ordered[:80]):
        horiz = "left" if e.center.x < float(scene.image.get("width", 1)) / 3 else "right" if e.center.x > float(scene.image.get("width", 1)) * 2 / 3 else "center"
        vert = "top" if e.center.y < float(scene.image.get("height", 1)) / 3 else "bottom" if e.center.y > float(scene.image.get("height", 1)) * 2 / 3 else "middle"
        out.append(f"- {e.id} lies in the {vert}-{horiz} area of the canvas.")
    return out[:800]


def to_compact_prompt(scene: VisualScene) -> str:
    return (
        "You are given a deterministic visual record produced without any AI model.\n"
        "Treat every measurement as evidence and every semantic interpretation as a hypothesis.\n"
        "Infer the diagram class and topology first. Then generate compilable TikZ preserving geometry, ordering, containment and connections.\n\n"
        "<VISUAL_RECORD>\n" + to_llm_context(scene) + "\n</VISUAL_RECORD>"
    )


def _geometry_sentence(e: VisualElement) -> str:
    return json.dumps(e.geometry, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
