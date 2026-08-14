import cv2
import numpy as np

from image_to_tikz.pipeline import analyze_image


def test_text_structure_is_model_free(tmp_path):
    image = np.full((260, 520, 3), 255, dtype=np.uint8)
    cv2.putText(image, "x2", (90, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, "y", (180, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.rectangle(image, (60, 60), (260, 190), (0, 0, 0), 2)
    path = tmp_path / "text.png"
    assert cv2.imwrite(str(path), image)

    scene, context = analyze_image(path, multiscale=False, ocr="off")

    assert scene.texts
    assert any(t.role and "candidate" in t.role for t in scene.texts)
    assert "TEXT" in context
    assert "DOWNSTREAM_TASK" in context
