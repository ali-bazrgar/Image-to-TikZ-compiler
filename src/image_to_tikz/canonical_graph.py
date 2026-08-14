from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from .vir import VisualScene


def enrich_canonical_graph(scene: VisualScene) -> VisualScene:
    """Build one deterministic node/edge graph from the complete VIR scene."""
    nodes: list[dict[str, Any]] = []
    ids: set[str] = set()
    for e in scene.elements:
        nodes.append({
            "id": e.id,
            "node_type": "element",
            "kind": e.kind,
            "confidence": e.confidence,
            "center_px": [round(e.center.x, 2), round(e.center.y, 2)],
        })
        ids.add(e.id)
    for t in scene.texts:
        nodes.append({
            "id": t.id,
            "node_type": "text",
            "kind": "text_region",
            "role": t.role,
            "confidence": t.confidence,
            "center_px": [round(t.bbox.center.x, 2), round(t.bbox.center.y, 2)],
        })
        ids.add(t.id)

    edges: list[dict[str, Any]] = []
    adjacency: dict[str, set[str]] = defaultdict(set)
    for r in scene.relations:
        if r.source not in ids or r.target not in ids:
            continue
        edges.append({
            "source": r.source,
            "relation": r.relation,
            "target": r.target,
            "confidence": r.confidence,
            "evidence": r.evidence,
        })
        adjacency[r.source].add(r.target)
        adjacency[r.target].add(r.source)

    unseen = set(ids)
    components: list[list[str]] = []
    while unseen:
        start = min(unseen)
        queue = deque([start])
        unseen.remove(start)
        comp: list[str] = []
        while queue:
            current = queue.popleft()
            comp.append(current)
            for neighbor in sorted(adjacency.get(current, ())):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        components.append(sorted(comp))
    components.sort(key=lambda c: (-len(c), c))

    component_map: dict[str, str] = {}
    for idx, comp in enumerate(components, 1):
        for node_id in comp:
            component_map[node_id] = f"component_{idx}"
    for node in nodes:
        node["component"] = component_map[node["id"]]

    scene.image["canonical_graph"] = {
        "schema": "image-to-tikz/canonical-graph",
        "version": "1.0",
        "method": "deterministic_relation_projection",
        "node_count": len(nodes),
        "edge_count": len(edges),
        "component_count": len(components),
        "nodes": nodes,
        "edges": edges,
        "components": [
            {"id": f"component_{i}", "node_ids": comp, "size": len(comp)}
            for i, comp in enumerate(components, 1)
        ],
    }
    scene.semantic_summary = (
        scene.semantic_summary
        + f"\nCANONICAL_GRAPH: nodes={len(nodes)} edges={len(edges)} components={len(components)}."
    ).strip()
    return scene
