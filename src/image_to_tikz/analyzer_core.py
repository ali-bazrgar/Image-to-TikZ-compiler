from __future__ import annotations

from pathlib import Path

import cv2

from .analyzer_model_free import ModelFreeImageAnalyzer


class ArrayImageAnalyzer(ModelFreeImageAnalyzer):
    """Run the existing deterministic analyzer directly on an in-memory image."""

    def analyze_array(self, image, *, virtual_path: str = "<array>"):
        if image is None or getattr(image, "size", 0) == 0:
            raise ValueError("Empty image array")
        h, w = image.shape[:2]
        # Encode/decode through OpenCV only to reuse the exact model-free implementation.
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("Could not encode image crop")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("Could not decode image crop")
        # ModelFreeImageAnalyzer is deterministic; preserve a descriptive virtual path.
        temp = Path(virtual_path)
        scene = self._analyze_decoded(decoded, temp)
        return scene

    def _analyze_decoded(self, image, path: Path):
        h, w = image.shape[:2]
        import cv2 as _cv2
        gray = _cv2.cvtColor(image, _cv2.COLOR_BGR2GRAY)
        scale = min(1.0, 2200 / max(w, h))
        work = _cv2.resize(gray, None, fx=scale, fy=scale, interpolation=_cv2.INTER_AREA)
        elements = self._shapes(work, scale)
        elements.extend(self._lines(work, scale))
        elements = self._dedupe(elements, w, h)
        text_regions = self._text_regions(gray)
        self._attach_text(elements, text_regions)
        self._mark_global_roles(elements, w, h)
        relations = self._relations(elements, w, h)
        from .vir import VisualScene
        return VisualScene(
            schema="image-to-tikz/vir",
            version="0.4",
            image={"path": str(path), "width": w, "height": h, "channels": 3, "analysis_scale": scale},
            coordinate_system={"origin": "top-left", "x_axis": "right", "y_axis": "down", "units": "pixels", "normalization": "divide x by width and y by height"},
            elements=elements,
            texts=text_regions,
            relations=relations,
            semantic_summary=self._summary(w, h, elements, text_regions, relations),
            warnings=["Text glyphs are not decoded: this core intentionally has no OCR or AI model dependency."],
        )
