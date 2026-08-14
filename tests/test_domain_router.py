import cv2
import numpy as np

from image_to_tikz.domain_router import route_domain
from image_to_tikz.pipeline import analyze_image


def test_domain_routing_is_deterministic_and_model_free(tmp_path):
    image = np.full((420, 700, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (60, 150), (180, 270), (0, 0, 0), 3)
    cv2.rectangle(image, (260, 150), (380, 270), (0, 0, 0), 3)
    cv2.rectangle(image, (460, 150), (580, 270), (0, 0, 0), 3)
    cv2.arrowedLine(image, (180, 210), (260, 210), (0, 0, 0), 3, tipLength=0.12)
    cv2.arrowedLine(image, (380, 210), (460, 210), (0, 0, 0), 3, tipLength=0.12)
    path = tmp_path / "flow.png"
    assert cv2.imwrite(str(path), image)

    scene, _ = analyze_image(path, multiscale=False, ocr="off")
    ranking = route_domain(scene)

    assert ranking
    assert ranking[0].domain in {"flowchart", "engineering_diagram", "general_diagram"}
    assert "domain_routing" in scene.image
    assert scene.image["domain_routing"]["method"] == "deterministic_evidence_scoring"
