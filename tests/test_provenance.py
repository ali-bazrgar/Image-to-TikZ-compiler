from __future__ import annotations

import cv2
import numpy as np

from image_to_tikz.pipeline import analyze_image


def test_provenance_is_complete_and_exposed_to_llm(tmp_path):
    image = np.full((240, 420, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (40, 80), (150, 160), (0, 0, 0), 3)
    cv2.line(image, (150, 120), (320, 120), (0, 0, 0), 3)
    path = tmp_path / "provenance.png"
    assert cv2.imwrite(str(path), image)

    scene, context = analyze_image(path, multiscale=False, ocr="off")
    provenance = scene.image["provenance"]

    assert provenance["schema"] == "image-to-tikz/provenance"
    assert provenance["element_sources"]
    assert provenance["relation_sources"]
    assert provenance["summary"]["relation_count"] == len(provenance["relation_sources"])
    assert "EVIDENCE_PROVENANCE:" in context
