from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .vir import BoundingBox, Point, Relation, TextBlock, VisualElement, VisualScene

try:
    import pytesseract
except Exception:  # optional dependency
    pytesseract = None


class ImageAnalyzer:
    """Convert a raster diagram into a deterministic, model-agnostic scene graph."""

    def __init__(self, enable_ocr: bool = True, min_contour_area_ratio: float = 0.00008):
        self.enable_ocr = enable_ocr
        self.min_contour_area_ratio = min_contour_area_ratio

    def analyze(self, image_path: str | Path) -> VisualScene:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(path)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode image: {path}")
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scale = self._analysis_scale(w, h)
        work = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        elements, warnings = self._extract_geometry(work, w, h, scale)
        texts = self._extract_text(image, w, h)
        self._attach_text_refs(elements, texts)
        relations = self._infer_relations(elements, w, h)
        summary = self._make_summary(w, h, elements, texts, relations)
        return VisualScene(
            schema="image-to-tikz/vir",
            version="0.1",
            image={
                "path": str(path),
                "width": w,
                "height": h,
                "channels": 3,
                "analysis_scale": scale,
            },
            coordinate_system={
                "origin": "top-left",
                "x_axis": "right",
                "y_axis": "down",
                "units": "pixels",
                "normalization": "x'=x/width, y'=y/height",
            },
            elements=elements,
            texts=texts,
            relations=relations,
            semantic_summary=summary,
            warnings=warnings,
        )

    @staticmethod
    def _analysis_scale(width: int, height: int) -> float:
        target = 1800
        longest = max(width, height)
        return min(1.0, target / max(1, longest))

    def _extract_geometry(self, gray: np.ndarray, original_w: int, original_h: int, scale: float):
        den = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(den, 50, 150, apertureSize=3)
        kernel = np.ones((2, 2), np.uint8)
        clean = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        area_limit = gray.shape[0] * gray.shape[1] * self.min_contour_area_ratio
        elements: list[VisualElement] = []
        serial = 1

        # Closed contours are used for circles, rectangles, polygons and generic closed shapes.
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < area_limit:
                continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw < 4 or bh < 4:
                continue
            cx = x + bw / 2
            cy = y + bh / 2
            ratio = bw / max(1, bh)
            circularity = 4 * math.pi * area / max(peri * peri, 1e-9)
            if len(approx) == 4 and 0.5 < ratio < 2.0:
                kind = "rectangle"
            elif circularity > 0.72:
                kind = "circle_or_ellipse"
            elif len(approx) >= 3 and len(approx) <= 10:
                kind = "polygon"
            else:
                continue
            points = [[round(float(p[0][0] / scale), 2), round(float(p[0][1] / scale), 2)] for p in approx]
            elements.append(
                VisualElement(
                    id=f"shape_{serial}",
                    kind=kind,
                    bbox=self._bbox(x, y, bw, bh, scale),
                    center=Point(round(cx / scale, 2), round(cy / scale, 2)),
                    confidence=0.72,
                    geometry={"area_px2": round(area / (scale * scale), 2), "vertices": points, "circularity": round(circularity, 4)},
                )
            )
            serial += 1

        # Long line segments give the LLM explicit edge/stroke information even when there is no closed contour.
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(18, int(gray.shape[1] * 0.04)), minLineLength=max(15, int(gray.shape[1] * 0.05)), maxLineGap=8)
        if lines is not None:
            existing = [(e.center.x, e.center.y) for e in elements]
            for line in lines[:, 0, :]:
                x1, y1, x2, y2 = map(int, line)
                length = math.hypot(x2 - x1, y2 - y1)
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                if abs(angle) < 3 or abs(abs(angle) - 180) < 3:
                    orientation = "horizontal"
                elif abs(abs(angle) - 90) < 3:
                    orientation = "vertical"
                else:
                    orientation = "diagonal"
                elements.append(
                    VisualElement(
                        id=f"line_{serial}",
                        kind="line",
                        bbox=self._line_bbox(x1, y1, x2, y2, scale),
                        center=Point(round((x1 + x2) / 2 / scale, 2), round((y1 + y2) / 2 / scale, 2)),
                        confidence=0.68,
                        geometry={
                            "start": [round(x1 / scale, 2), round(y1 / scale, 2)],
                            "end": [round(x2 / scale, 2), round(y2 / scale, 2)],
                            "length": round(length / scale, 2),
                            "angle_deg": round(angle, 2),
                            "orientation": orientation,
                        },
                        style={"stroke": "dark", "closed": False},
                    )
                )
                serial += 1
                if serial > 220:
                    break

        # Remove near-duplicate lines and very large contours that are likely page borders.
        elements = self._deduplicate(elements, original_w, original_h)
        warnings = []
        if not elements:
            warnings.append("No reliable geometric primitives were detected; a vision-language model may be needed for semantic interpretation.")
        return elements, warnings

    @staticmethod
    def _bbox(x: int, y: int, w: int, h: int, scale: float) -> BoundingBox:
        return BoundingBox(round(x / scale, 2), round(y / scale, 2), round(w / scale, 2), round(h / scale, 2))

    @staticmethod
    def _line_bbox(x1: int, y1: int, x2: int, y2: int, scale: float) -> BoundingBox:
        return BoundingBox(round(min(x1, x2) / scale, 2), round(min(y1, y2) / scale, 2), round(abs(x2 - x1) / scale, 2), round(abs(y2 - y1) / scale, 2))

    def _extract_text(self, image: np.ndarray, width: int, height: int) -> list[TextBlock]:
        if not self.enable_ocr or pytesseract is None:
            return []
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config="--psm 11")
        except Exception:
            return []
        texts: list[TextBlock] = []
        idx = 1
        for i, raw in enumerate(data.get("text", [])):
            text = (raw or "").strip()
            try:
                conf = float(data["conf"][i]) / 100.0
            except Exception:
                conf = 0.0
            if not text or conf < 0.25:
                continue
            x, y, w, h = [int(data[k][i]) for k in ("left", "top", "width", "height")]
            texts.append(TextBlock(f"text_{idx}", text, BoundingBox(x, y, w, h), round(conf, 3)))
            idx += 1
        return texts

    @staticmethod
    def _attach_text_refs(elements: list[VisualElement], texts: list[TextBlock]) -> None:
        for text in texts:
            tc = text.bbox.center
            candidates = []
            for el in elements:
                dx = tc.x - el.center.x
                dy = tc.y - el.center.y
                d = math.hypot(dx, dy)
                if d < max(el.bbox.width, el.bbox.height) * 1.5 + 60:
                    candidates.append((d, el))
            if candidates:
                candidates.sort(key=lambda x: x[0])
                candidates[0][1].text_refs.append(text.id)

    @staticmethod
    def _infer_relations(elements: list[VisualElement], width: int, height: int) -> list[Relation]:
        relations: list[Relation] = []
        diag = math.hypot(width, height)
        for i, a in enumerate(elements):
            for b in elements[i + 1 :]:
                dx = b.center.x - a.center.x
                dy = b.center.y - a.center.y
                d = math.hypot(dx, dy)
                if d < diag * 0.045:
                    relations.append(Relation(a.id, "near", b.id, round(max(0.35, 1 - d / (diag * 0.045)), 3), {"center_distance_px": round(d, 2)}))
                if abs(dx) < max(a.bbox.width, b.bbox.width) * 0.20:
                    relations.append(Relation(a.id, "vertically_aligned_with", b.id, 0.78))
                if abs(dy) < max(a.bbox.height, b.bbox.height) * 0.20:
                    relations.append(Relation(a.id, "horizontally_aligned_with", b.id, 0.78))
                if a.kind == "line" and b.kind in {"rectangle", "circle_or_ellipse", "polygon"}:
                    if ImageAnalyzer._distance_to_bbox(a.center, b.bbox) < 28:
                        relations.append(Relation(a.id, "touches_or_connects", b.id, 0.58))
        return relations[:1200]

    @staticmethod
    def _distance_to_bbox(p: Point, box: BoundingBox) -> float:
        dx = max(box.x - p.x, 0, p.x - (box.x + box.width))
        dy = max(box.y - p.y, 0, p.y - (box.y + box.height))
        return math.hypot(dx, dy)

    @staticmethod
    def _deduplicate(elements: list[VisualElement], width: int, height: int) -> list[VisualElement]:
        out: list[VisualElement] = []
        for e in sorted(elements, key=lambda z: (z.kind != "line", -z.bbox.width * z.bbox.height)):
            if e.bbox.width > width * 0.92 and e.bbox.height > height * 0.92:
                continue
            duplicate = False
            for q in out:
                if e.kind != q.kind:
                    continue
                d = math.hypot(e.center.x - q.center.x, e.center.y - q.center.y)
                if d < 6 and abs(e.bbox.width - q.bbox.width) < 12 and abs(e.bbox.height - q.bbox.height) < 12:
                    duplicate = True
                    break
            if not duplicate:
                out.append(e)
        return out[:260]

    @staticmethod
    def _make_summary(width: int, height: int, elements: list[VisualElement], texts: list[TextBlock], relations: list[Relation]) -> str:
        counts: dict[str, int] = {}
        for e in elements:
            counts[e.kind] = counts.get(e.kind, 0) + 1
        parts = [f"Canvas {width}x{height}px."]
        if counts:
            parts.append("Detected primitives: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
        if texts:
            parts.append("Visible text: " + ", ".join(t.text for t in texts[:24]) + (" ..." if len(texts) > 24 else ""))
        if relations:
            parts.append(f"Spatial/structural relations: {len(relations)} candidate relations.")
        return " ".join(parts)
