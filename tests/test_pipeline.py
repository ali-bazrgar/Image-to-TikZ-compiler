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

    scene, context = analyze_image(path, enable_ocr=False)

    assert scene.image["width_px"] == 500
    assert scene.image["height_px"] == 300
    assert scene.elements
    assert "ELEMENTS:" in context
    assert "RELATIONS:" in context
