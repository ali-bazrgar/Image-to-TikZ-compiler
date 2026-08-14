from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .vir import BoundingBox, Point, Relation, TextBlock, VisualElement, VisualScene


class ModelFreeImageAnalyzer:
    """Deterministic diagram analysis using only OpenCV/NumPy."""

    def __init__(self, min_area_ratio: float = 0.00005) -> None:
        self.min_area_ratio = min_area_ratio

    def analyze(self, image_path: str | Path) -> VisualScene:
        path = Path(image_path)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode image: {path}")
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scale = min(1.0, 2200 / max(w, h))
        work = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        elements = self._shapes(work, scale)
        elements.extend(self._lines(work, scale))
        elements = self._dedupe(elements, w, h)
        text_regions = self._text_regions(gray)
        self._attach_text(elements, text_regions)
        self._mark_global_roles(elements, w, h)
        relations = self._relations(elements, w, h)
        warnings = ["Text glyphs are not decoded: this core intentionally has no OCR or AI model dependency."]
        return VisualScene(
            schema="image-to-tikz/vir",
            version="0.4",
            image={"path": str(path), "width": w, "height": h, "channels": 3, "analysis_scale": scale},
            coordinate_system={"origin": "top-left", "x_axis": "right", "y_axis": "down", "units": "pixels", "normalization": "divide x by width and y by height"},
            elements=elements,
            texts=text_regions,
            relations=relations,
            semantic_summary=self._summary(w, h, elements, text_regions, relations),
            warnings=warnings,
        )

    def _shapes(self, gray: np.ndarray, scale: float) -> list[VisualElement]:
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 140, L2gradient=True)
        contours, _ = cv2.findContours(cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8)), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        limit = gray.shape[0] * gray.shape[1] * self.min_area_ratio
        out: list[VisualElement] = []
        for i, c in enumerate(contours, 1):
            area = cv2.contourArea(c)
            if area < limit:
                continue
            peri = cv2.arcLength(c, True)
            if not peri:
                continue
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            x, y, w, h = cv2.boundingRect(c)
            if w < 5 or h < 5:
                continue
            circularity = 4 * math.pi * area / max(peri * peri, 1e-9)
            extent = area / max(w * h, 1)
            ratio = w / max(h, 1)
            if len(approx) == 4 and 0.15 < ratio < 6.5 and extent > 0.35:
                kind, conf = "quadrilateral", 0.80
            elif circularity > 0.70 and 0.5 < ratio < 2.0:
                kind, conf = "circle_or_ellipse", 0.86
            elif 3 <= len(approx) <= 12 and extent > 0.20:
                kind, conf = "polygon", 0.74
            else:
                continue
            vertices = [[round(float(p[0][0] / scale), 2), round(float(p[0][1] / scale), 2)] for p in approx]
            out.append(VisualElement(
                id=f"shape_{i}", kind=kind,
                bbox=BoundingBox(round(x / scale, 2), round(y / scale, 2), round(w / scale, 2), round(h / scale, 2)),
                center=Point(round((x + w / 2) / scale, 2), round((y + h / 2) / scale, 2)),
                confidence=conf,
                geometry={"vertices_px": vertices, "vertex_count": len(vertices), "area_px2": round(area / (scale * scale), 2), "circularity": round(circularity, 4), "extent": round(extent, 4)},
            ))
        return out

    def _lines(self, gray: np.ndarray, scale: float) -> list[VisualElement]:
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 140, L2gradient=True)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=max(16, gray.shape[1] // 35),
            minLineLength=max(12, min(gray.shape[:2]) // 28),
            maxLineGap=max(5, min(gray.shape[:2]) // 120),
        )
        out: list[VisualElement] = []
        if lines is None:
            return out
        # OpenCV versions may return shape (N, 1, 4) or (N, 4).
        rows = np.asarray(lines).reshape(-1, 4)
        for i, row in enumerate(rows, 1):
            x1, y1, x2, y2 = map(int, row)
            length = math.hypot(x2 - x1, y2 - y1)
            if length < min(gray.shape[:2]) * 0.025:
                continue
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            a = angle % 180
            orient = "horizontal" if min(a, abs(a - 180)) < 3 else ("vertical" if abs(a - 90) < 3 else "diagonal")
            out.append(VisualElement(
                id=f"line_{i}", kind="line_segment",
                bbox=BoundingBox(round(min(x1, x2) / scale, 2), round(min(y1, y2) / scale, 2), round(abs(x2 - x1) / scale, 2), round(abs(y2 - y1) / scale, 2)),
                center=Point(round((x1 + x2) / (2 * scale), 2), round((y1 + y2) / (2 * scale), 2)),
                confidence=0.72,
                geometry={"start_px": [round(x1 / scale, 2), round(y1 / scale, 2)], "end_px": [round(x2 / scale, 2), round(y2 / scale, 2)], "length_px": round(length / scale, 2), "angle_deg": round(angle, 2), "orientation": orient, "endpoint_candidates": self._endpoint_candidates(gray, x1, y1, x2, y2, scale)},
            ))
        return out

    @staticmethod
    def _endpoint_candidates(gray: np.ndarray, x1: int, y1: int, x2: int, y2: int, scale: float) -> list[dict]:
        h, w = gray.shape[:2]
        r = max(4, min(h, w) // 90)
        out = []
        for x, y, name in ((x1, y1, "start"), (x2, y2, "end")):
            crop = gray[max(0, y-r):min(h, y+r+1), max(0, x-r):min(w, x+r+1)]
            if crop.size and float((crop < 95).mean()) > 0.28:
                out.append({"endpoint": name, "x_px": round(x / scale, 2), "y_px": round(y / scale, 2), "possible_junction_or_arrowhead": True})
        return out

    @staticmethod
    def _text_regions(gray: np.ndarray) -> list[TextBlock]:
        h, w = gray.shape[:2]
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 11)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2)))
        n, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
        boxes = []
        for i in range(1, n):
            x, y, ww, hh, area = map(int, stats[i])
            if 5 <= area <= w * h * 0.01 and hh >= 4 and ww >= 2 and not (ww > w * 0.35 and hh > h * 0.15):
                density = area / max(1, ww * hh)
                if 0.02 <= density <= 0.75:
                    boxes.append((x, y, ww, hh))
        boxes.sort(key=lambda b: (b[1], b[0]))
        return [TextBlock(f"text_region_{i}", "", BoundingBox(x, y, ww, hh), 0.45) for i, (x, y, ww, hh) in enumerate(boxes[:160], 1)]

    @staticmethod
    def _attach_text(elements: list[VisualElement], texts: list[TextBlock]) -> None:
        for t in texts:
            p = t.bbox.center
            candidates = [(math.hypot(p.x - e.center.x, p.y - e.center.y), e) for e in elements]
            if candidates:
                candidates.sort(key=lambda z: z[0])
                candidates[0][1].text_refs.append(t.id)

    @staticmethod
    def _relations(elements: list[VisualElement], w: int, h: int) -> list[Relation]:
        relations: list[Relation] = []
        diag = math.hypot(w, h)
        for i, a in enumerate(elements):
            for b in elements[i + 1:]:
                dx, dy = b.center.x - a.center.x, b.center.y - a.center.y
                d = math.hypot(dx, dy)
                if d < diag * 0.045:
                    relations.append(Relation(a.id, "near", b.id, round(1 - d / (diag * 0.045), 3)))
                if abs(dy) < max(a.bbox.height, b.bbox.height) * 0.20:
                    relations.append(Relation(a.id, "horizontally_aligned_with", b.id, 0.78))
                if abs(dx) < max(a.bbox.width, b.bbox.width) * 0.20:
                    relations.append(Relation(a.id, "vertically_aligned_with", b.id, 0.78))
                if a.kind == "line_segment" and b.kind != "line_segment":
                    if ModelFreeImageAnalyzer._point_to_box(a.center, b.bbox) < 32:
                        relations.append(Relation(a.id, "touch_or_connect_candidate", b.id, 0.58))
        return relations[:1800]

    @staticmethod
    def _point_to_box(p: Point, b: BoundingBox) -> float:
        return math.hypot(max(b.x - p.x, 0, p.x - b.x - b.width), max(b.y - p.y, 0, p.y - b.y - b.height))

    @staticmethod
    def _dedupe(elements: list[VisualElement], w: int, h: int) -> list[VisualElement]:
        out: list[VisualElement] = []
        for e in sorted(elements, key=lambda x: (x.kind != "line_segment", -x.bbox.width * x.bbox.height)):
            if e.bbox.width > w * 0.94 and e.bbox.height > h * 0.94:
                continue
            if any(e.kind == q.kind and math.hypot(e.center.x - q.center.x, e.center.y - q.center.y) < 7 for q in out):
                continue
            out.append(e)
        return out[:320]

    @staticmethod
    def _mark_global_roles(elements: list[VisualElement], w: int, h: int) -> None:
        for e in elements:
            if e.kind != "line_segment":
                continue
            if e.geometry.get("length_px", 0) > w * 0.25 and e.geometry.get("orientation") == "horizontal":
                e.geometry["possible_role"] = "axis_or_baseline"
            elif e.geometry.get("length_px", 0) > h * 0.25 and e.geometry.get("orientation") == "vertical":
                e.geometry["possible_role"] = "axis_or_baseline"

    @staticmethod
    def _summary(w: int, h: int, elements: list[VisualElement], texts: list[TextBlock], relations: list[Relation]) -> str:
        counts: dict[str, int] = {}
        for e in elements:
            counts[e.kind] = counts.get(e.kind, 0) + 1
        return f"Canvas {w}x{h}px. Measured primitives: " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())) + f". Text-like regions: {len(texts)}. Candidate relations: {len(relations)}."
