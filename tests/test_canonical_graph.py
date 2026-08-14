import cv2
import numpy as np

from image_to_tikz.pipeline import analyze_image


def test_canonical_graph_projects_valid_nodes_edges_and_components(tmp_path):
    image = np.full((260, 520, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (40, 90), (160, 180), (0, 0, 0), 3)
    cv2.rectangle(image, (360, 90), (480, 180), (0, 0, 0), 3)
    cv2.line(image, (160, 135), (360, 135), (0, 0, 0), 3)
    path = tmp_path / "graph.png"
    assert cv2.imwrite(str(path), image)

    scene, context = analyze_image(path, multiscale=False, ocr="off")
    graph = scene.image["canonical_graph"]

    node_ids = {node["id"] for node in graph["nodes"]}
    assert graph["schema"] == "image-to-tikz/canonical-graph"
    assert graph["node_count"] == len(graph["nodes"])
    assert graph["edge_count"] == len(graph["edges"])
    assert graph["component_count"] >= 1
    assert all(edge["source"] in node_ids and edge["target"] in node_ids for edge in graph["edges"])
    assert all(node["component"].startswith("component_") for node in graph["nodes"])
    assert "CANONICAL_GRAPH:" in scene.semantic_summary
    assert "CANONICAL_GRAPH:" in context
