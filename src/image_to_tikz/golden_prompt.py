from __future__ import annotations

from .serialize import to_llm_context


def build_golden_prompt(scene) -> str:
    """Build a strict reconstruction prompt for a downstream text/code LLM."""
    context = to_llm_context(scene)
    return """You are an expert scientific-figure reconstruction engine.

Your input is a CANONICAL VISUAL RECORD produced from the original image by a deterministic computer-vision compiler, optionally enriched by lightweight OCR/VLM hypotheses.

PRIMARY OBJECTIVE
Produce one compilable TikZ/LaTeX figure that reconstructs the original image as faithfully as possible.

NON-NEGOTIABLE RULES
1. The measured geometry in the VISUAL RECORD is authoritative. Do not invent or silently move coordinates.
2. Preserve topology: connections, intersections, containment, alignment, ordering, repetition and symmetry.
3. Treat OCR, domain classification, specialized detector labels and VLM descriptions as hypotheses. Use them only when consistent with measured evidence.
4. Never invent unreadable text. Keep an undecoded label as an empty/placeholder node only when necessary to preserve its position.
5. Reconstruct the scene in layers: background/reference geometry, primary shapes, connectors/paths, axes/dimensions, labels and annotations.
6. Prefer explicit coordinates and named TikZ nodes over vague relative placement when exact coordinates are available.
7. Preserve stroke style evidence: solid/dashed/dotted, approximate thickness, line orientation and arrow candidates.
8. Preserve curves using sampled geometry when the representation does not justify a semantic curve type.
9. Do not explain the answer outside the code block.
10. Output ONLY a single LaTeX/TikZ code block.
11. The code must compile with standard TikZ/PGF packages and must be self-contained.
12. Do not add decorative elements that are absent from the record.
13. Before finalizing, internally verify every major element, relation and text region against the record.
14. If two observations conflict, choose the measured observation with higher confidence and preserve the uncertainty only in internal reasoning.

RECONSTRUCTION METHOD
A. Infer the global diagram class from DOMAIN_ROUTING, but do not let it override measured evidence.
B. Build the CANONICAL_GRAPH components first.
C. Recreate each connected component while preserving its measured topology.
D. Place labels at their measured locations and orientations.
E. Add semantic styling only where supported by evidence.
F. Perform a final visual-structure audit before emitting code.

VISUAL RECORD
====================
""" + context + "\n====================\n"
