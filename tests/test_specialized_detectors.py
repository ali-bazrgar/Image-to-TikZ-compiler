import cv2
import numpy as np

from image_to_tikz.pipeline import analyze_image


def test_specialized_detectors_are_routed_and_recorded(tmp_path):
    image = np.full((360, 600, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (50, 140), (180, 220), (0, 0, 0), 3)
    cv2.rectangle(image, (380, 140), (530, 220), (0, 0, 0), 3)
    cv2.arrowedLine(image, (180, 180), (380, 180), (0, 0, 0), 3, tipLength=0.08)
    path = tmp_path / "specialized.png"
    assert cv2.imwrite(str(path), image)

    scene, context = analyze_image(path, multiscale=False, ocr="off")

    result = scene.image["specialized_detectors"]
    assert result["method"] == "deterministic_domain_routed"
    assert result["detectors"]
    assert "detectors=" in scene.semantic_summary
    assert "SPECIALIZED_DETECTORS" in context
