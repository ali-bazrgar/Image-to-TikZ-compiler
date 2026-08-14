from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .analyzer_api import ImageAnalyzer
from .vir import BoundingBox, Point, VisualElement, VisualScene


@dataclass(frozen=True)
class AnalysisWindow:
    x: int
    y: int
    width: int
    height: int
    name: str


class MultiscaleAnalyzer:
    """Run deterministic analysis globally and on overlapping local windows, then fuse results."""

    def __init__(self, overlap: float = 0.18, min_window: int = 320) -> None:
        self.overlap = max(0.05, min(overlap, 0.45))
        self.min_window = min_window
        self.base = ImageAnalyzer()

    def analyze(self, image_path: str | Path) -> VisualScene:
        path = Path(image_path)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode image: {path}")
        h, w = image.shape[:2]

        base_scene = self.base.analyze(path)
        extra: list[VisualElement] = []
        window_records: list[dict[str, Any]] = []

        for window in self._windows(w, h):
            crop = image[window.y:window.y + window.height, window.x:window.x + window.width]
            if crop.size == 0:
                continue
            crop_name = f"{window.name}.png"
            crop_path = path.with_name(f".__multiscale_{crop_name}")
            try:
                ok = cv2.imwrite(str(crop_path), crop)
                if not ok:
                    continue
                local_scene = self.base.analyze(crop_path)
                window_records.append({"name": window.name, "bbox_px": [window.x, window.y, window.width, window.height], "elements": len(local_scene.elements)})
                extra.extend(self._translate(local_scene.elements, window, local_scene.image))
            finally:
                try:
                    crop_path.unlink(missing_ok=True)
                except Exception:
                    pass

        base_scene.elements = self._fuse_elements(base_scene.elements + extra, w, h)
        base_scene.relations = []
        base_scene.relations.extend(self._spatial_relations(base_scene.elements, w, h))
        base_scene.image["analysis_mode"] = "global_plus_overlapping_local_windows"
        base_scene.image["analysis_windows"] = window_records
        base_scene.warnings.append("Multiscale fusion uses deterministic duplicate suppression; local windows improve small-detail recall but can still produce ambiguous classifications.")
        base_scene.semantic_summary = self._summary(base_scene, len(window_records))
        return base_scene

    def _windows(self, w: int, h: int) -> list[AnalysisWindow]:
        if max(w, h) <= 1600:
            return [AnalysisWindow(0, 0, w, h, "full")]
        target_w = max(self.min_window, min(1400, w // 2 + int(w * 0.05)))
        target_h = max(self.min_window, min(1400, h // 2 + int(h * 0.05)))
        sx = max(1, int(target_w * (1 - self.overlap)))
        sy = max(1, int(target_h * (1 - self.overlap)))
        xs = list(range(0, max(1, w - target_w + 1), sx))
        ys = list(range(0, max(1, h - target_h + 1), sy))
        if not xs or xs[-1] != max(0, w - target_w):
            xs.append(max(0, w - target_w))
        if not ys or ys[-1] != max(0, h - target_h):
            ys.append(max(0, h - target_h))
        out = []
        for yi, y in enumerate(ys):
            for xi, x in enumerate(xs):
                out.append(AnalysisWindow(x, y, min(target_w, w - x), min(target_h, h - y), f"window_{yi}_{xi}"))
        return out

    @staticmethod
    def _translate(elements: list[VisualElement], window: AnalysisWindow, image: dict[str, Any]) -> list[VisualElement]:
        out: list[VisualElement] = []
        ww = float(image.get("width", window.width) or window.width)
        hh = float(image.get("height", window.height) or window.height)
        for e in elements:
            b = e.bbox
            shifted = BoundingBox(b.x + window.x, b.y + window.y, b.width, b.height)
            c = Point(e.center.x + window.x, e.center.y + window.y)
            geometry = dict(e.geometry)
            for key in ("start_px", "end_px"):
                if key in geometry and geometry[key]:
                    geometry[key] = [geometry[key][0] + window.x, geometry[key][1] + window.y]
            geometry["source_window"] = window.name
            out.append(VisualElement(id=f"{e.id}@{window.name}", kind=e.kind, bbox=shifted, center=c, confidence=min(0.95, e.confidence * 0.97), geometry=geometry, style=dict(e.style), labels=list(e.labels), text_refs=list(e.text_refs)))
        return out

    @staticmethod
    def _fuse_elements(elements: list[VisualElement], w: int, h: int) -> list[VisualElement]:
        result: list[VisualElement] = []
        for e in sorted(elements, key=lambda item: (-item.confidence, item.kind != "line_segment")):
            duplicate = False
            for q in result:
                if e.kind != q.kind:
                    continue
                center_distance = float(np.hypot(e.center.x - q.center.x, e.center.y - q.center.y))
                scale = max(8.0, min(e.bbox.width + e.bbox.height, q.bbox.width + q.bbox.height) * 0.08)
                if center_distance <= scale and abs(e.bbox.width - q.bbox.width) <= max(10.0, q.bbox.width * 0.15) and abs(e.bbox.height - q.bbox.height) <= max(10.0, q.bbox.height * 0.15):
                    duplicate = True
                    break
            if not duplicate:
                result.append(e)
        return result[:700]

    @staticmethod
    def _spatial_relations(elements: list[VisualElement], w: int, h: int):
        from .vir import Relation
        relations = []
        diag = float(np.hypot(w, h))
        for i, a in enumerate(elements):
            for b in elements[i + 1:]:
                d = float(np.hypot(a.center.x - b.center.x, a.center.y - b.center.y))
                if d < diag * 0.035:
                    relations.append(Relation(a.id, "near", b.id, round(1 - d / (diag * 0.035), 3), {"source": "multiscale"}))
                if abs(a.center.y - b.center.y) < max(a.bbox.height, b.bbox.height) * 0.16:
                    relations.append(Relation(a.id, "horizontally_aligned_with", b.id, 0.82, {"source": "multiscale"}))
                if abs(a.center.x - b.center.x) < max(a.bbox.width, b.bbox.width) * 0.16:
                    relations.append(Relation(a.id, "vertically_aligned_with", b.id, 0.82, {"source": "multiscale"}))
        return relations[:2400]

    @staticmethod
    def _summary(scene: VisualScene, windows: int) -> str:
        kinds: dict[str, int] = {}
        for e in scene.elements:
            kinds[e.kind] = kinds.get(e.kind, 0) + 1
        inventory = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items()))
        return f"Canvas {scene.image['width']}x{scene.image['height']}px analyzed globally and across {windows} overlapping local windows. Fused primitives: {inventory}. Candidate relations: {len(scene.relations)}."
