import cv2
import numpy as np

from image_to_tikz.pipeline import analyze_image


def test_structure_pipeline_adds_higher_level_evidence(tmp_path):
    image = np.full((500, 800, 3), 255, np.uint8)
    cv2.rectangle(image, (80, 180), (220, 320), (0, 0, 0), 4)
    cv2.rectangle(image, (580, 180), (720, 320), (0, 0, 0), 4)
    cv2.line(image, (220, 250), (580, 250), (0, 0, 0), 4)
    cv2.imwrite(str(tmp_path / "diagram.png"), image)

    scene, context = analyze_image(tmp_path / "diagram.png")
    assert scene.elements
    assert "IMAGE_TO_TIKZ_VISUAL_RECORD" in context
    assert any(e.geometry.get("structure_group") for e in scene.elements) or any(
        r.relation == "endpoint_connects_candidate" for r in scene.relations
    )
    assert any("orientation" in e.geometry for e in scene.elements if e.kind == "line_segment")
