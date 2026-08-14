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

RULES
1. Measured coordinates, geometry, topology and text-region locations are authoritative.
2. Treat OCR, domain routing, specialized detector names and VLM descriptions as hypotheses, not measurements.
3. Preserve connections, intersections, containment, alignment, repetition, symmetry, ordering and relative placement.
4. Never invent unreadable text. Keep undecoded labels as positioned placeholders only when necessary.
5. Reconstruct in layers: reference/background geometry; primary geometry; connectors/curves; axes/dimensions; labels/annotations.
6. Use explicit coordinates whenever measured coordinates are available.
7. Preserve supported stroke evidence and curve geometry; do not invent decorative styling.
8. Before output, audit every major element and relation against the visual record.
9. Output ONLY one self-contained LaTeX/TikZ code block. No prose outside it.
10. The code must compile with standard TikZ/PGF packages.

WORKFLOW
A. Infer the most plausible diagram class from the DOMAIN section.
B. Reconstruct connected components from the GRAPH section.
C. Place measured geometry before applying semantic roles.
D. Add only semantic information supported by evidence.
E. Verify topology, placement, labels and styling before emitting TikZ.

COMPACT VISUAL RECORD
====================
""" + context + "\n====================\n"
