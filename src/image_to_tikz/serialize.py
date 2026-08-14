from __future__ import annotations

import json

from .serialize_compact import to_compact_llm_context
from .vir import VisualScene


def to_json(scene: VisualScene, indent: int = 2) -> str:
    """Full-fidelity machine-readable representation. Keep this detailed."""
    return json.dumps(scene.to_dict(), ensure_ascii=False, indent=indent)


def to_llm_context(scene: VisualScene) -> str:
    """Compact high-density representation for downstream LLMs.

    Detailed internal CV data belongs in scene.json; this text deliberately
    avoids repeating graph edges, relations, provenance and narrative copies.
    """
    return to_compact_llm_context(scene)


def to_compact_prompt(scene: VisualScene) -> str:
    return (
        "You are given a compact canonical visual record produced by a deterministic image-analysis compiler.\n"
        "Measured geometry, coordinates, topology and text-region locations are authoritative. Semantic detector/VLM labels are hypotheses.\n"
        "Reconstruct the diagram class and topology first, then geometry, text placement and supported styling.\n"
        "Never invent unreadable labels or move measured geometry.\n\n"
        "<VISUAL_RECORD>\n" + to_llm_context(scene) + "\n</VISUAL_RECORD>"
    )
