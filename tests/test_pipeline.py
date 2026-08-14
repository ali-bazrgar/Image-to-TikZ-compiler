from __future__ import annotations

import cv2
import numpy as np

from image_to_tikz.pipeline import analyze_image


def test_pipeline_extracts_basic_geometry(tmp_path):
    image = np.full((300, 500, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (60, 80), (220, 200), (0, 0, 0), 3)
    cv2.circle(image, (360, 140), 45, (0, 0, 0), 3)
    cv2.line(image, (220, 140), (315, 140), (0, 0, 0), 3)
    path = tmp_path / "diagram.png"
    assert cv2.imwrite(str(path), image)

    scene, context = analyze_image(path)

    assert scene.image["width"] == 500
    assert scene.image["height"] == 300
    assert scene.elements
    assert "ELEMENTS:" in context
    assert "RELATIONS:" in context


def test_pipeline_exposes_model_free_structure(tmp_path):
    image = np.full((360, 600, 3), 255, dtype=np.uint8)
    cv2.line(image, (80, 180), (300, 180), (0, 0, 0), 3)
    cv2.line(image, (300, 180), (300, 90), (0, 0, 0), 3)
    cv2.line(image, (300, 180), (300, 270), (0, 0, 0), 3)
    cv2.rectangle(image, (40, 140), (80, 220), (0, 0, 0), 3)
    path = tmp_path / "topology.png"
    assert cv2.imwrite(str(path), image)

    scene, context = analyze_image(path)

    relation_names = {r.relation for r in scene.relations}
    assert relation_names & {"line_junction_candidate", "line_crossing_candidate"}
    assert "stroke_orientation" in next(e.geometry for e in scene.elements if e.kind == "line_segment")
    assert "INTERPRETATION_RULES:" in context
