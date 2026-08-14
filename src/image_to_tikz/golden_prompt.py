from __future__ import annotations

from .serialize import to_llm_context


def build_golden_prompt(scene) -> str:
    """Build a strict reconstruction prompt around the compact visual record."""
    context = to_llm_context(scene)
    return """You are an expert scientific-figure reconstruction engine.

INPUT
A compact canonical visual record extracted from an image by deterministic computer vision, optionally enriched by lightweight OCR and crop-based VLM hypotheses.

OBJECTIVE
Produce one compilable TikZ/LaTeX figure that matches the original image as faithfully as the measured evidence permits.

EVIDENCE PRIORITY
1. Measured element geometry and coordinates are the primary source of truth.
2. Text-region locations and decoded OCR text are evidence; undecoded text remains unknown.
3. Explicit curve/path measurements are authoritative for their measured points.
4. The canonical graph and spatial relations are DERIVED evidence. Use them only to organize already-measured primitives. If they conflict with measured coordinates or visible primitive endpoints, ignore the derived relation.
5. Domain routing, specialized detector labels and VLM hints are semantic hypotheses only. They must never create new geometry.

RULES
1. Preserve every high-confidence measured primitive unless there is explicit evidence that it is a duplicate.
2. Never invent a line, curve, node, axis, arrow, label, dimension or decorative object that has no supporting measured primitive or explicit semantic evidence.
3. Never replace measured coordinates with guessed coordinates. Use the given pixel coordinates and map them deterministically into TikZ coordinates.
4. Preserve connections, intersections, containment, alignment, repetition, symmetry, ordering and relative placement only when supported by the measured primitives.
5. Never invent unreadable labels. Preserve their measured bounding boxes as empty/unknown labels when needed for layout.
6. Reconstruct the figure in this order: canvas/scale; primary primitives; exact endpoints and curves; topology from measured endpoints; text placement; only then supported styling.
7. For every line segment, use its supplied start/end coordinates. For every measured curve/path, use its supplied sampled path points or geometric parameters. Do not redraw them from memory.
8. Prefer explicit coordinates over semantic shorthand whenever measurements are available.
9. Do not simplify away small measured geometry merely because it seems unimportant semantically.
10. Before output, perform a consistency audit: every major TikZ object must map to one or more measured element IDs in the visual record.
11. Output ONLY one self-contained LaTeX/TikZ code block. No prose outside it.
12. The code must compile with standard TikZ/PGF packages.

RECONSTRUCTION WORKFLOW
A. Read CANVAS and establish a single deterministic pixel-to-TikZ transform.
B. Read ELEMENTS as the authoritative drawing inventory and reconstruct those primitives first.
C. Use TEXT only for label placement/content.
D. Use GRAPH and RELATIONS_NOT_IN_GRAPH only as secondary topology hints for already-existing primitives.
E. Use DOMAIN, DETECTOR_EVIDENCE and VLM_HINTS only to assign plausible semantic roles to existing geometry.
F. Audit the mapping from generated TikZ objects back to element IDs before emitting the final code.

IMPORTANT NEGATIVE CONSTRAINT
The visual record may contain uncertain or duplicate detections. Uncertainty must reduce semantic interpretation, not cause geometric invention. When evidence is insufficient, preserve the measured primitive conservatively rather than guessing a different object.

COMPACT VISUAL RECORD
====================
""" + context + "\n====================\n"
