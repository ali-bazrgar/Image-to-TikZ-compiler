from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .analyzer_api import ImageAnalyzer
from .analyzer_core import ArrayImageAnalyzer
from .vir import BoundingBox, Point, Relation, VisualElement, VisualScene


@dataclass(frozen=True)
class AnalysisWindow:
    x: int
    y: int
    width: int
    height: int
    name: str


class MultiscaleAnalyzer:
    """Analyze globally and on overlapping local windows, then fuse deterministically."""

    def __init__(self, overlap: float = 0.18, min_window: int = 320) -> None:
        self.overlap = max(0.05, min(overlap, 0.45))
        self.min_window = min_window
        self.base = ImageAnalyzer()
        self.array_base = ArrayImageAnalyzer()

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
            local_scene = self.array_base.analyze_array(crop, virtual_path=f"<{window.name}>")
            window_records.append({
                "name": window.name,
                "bbox_px": [window.x, window.y, window.width, window.height],
                "elements": len(local_scene.elements),
            })
            extra.extend(self._translate(local_scene.elements, window))

        base_scene.elements = self._fuse_elements(base_scene.elements + extra)
        base_scene.relations = self._spatial_relations(base_scene.elements, w, h)
        base_scene.image["analysis_mode"] = "global_plus_overlapping_local_windows"
        base_scene.image["analysis_windows"] = window_records
        base_scene.warnings.append(
            "Multiscale fusion uses deterministic duplicate suppression; local windows improve small-detail recall but can still produce ambiguous classifications."
        )
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
        return [
            AnalysisWindow(x, y, min(target_w, w - x), min(target_h, h - y), f"window_{yi}_{xi}")
            for yi, y in enumerate(ys)
            for xi, x in enumerate(xs)
        ]

    @staticmethod
    def _translate(elements: list[VisualElement], window: AnalysisWindow) -> list[VisualElement]:
        out: list[VisualElement] = []
        for e in elements:
            b = e.bbox
            shifted = BoundingBox(b.x + window.x, b.y + window.y, b.width, b.height)
            c = Point(e.center.x + window.x, e.center.y + window.y)
            geometry = dict(e.geometry)
            for key in ("start_px", "end_px"):
                if key in geometry and geometry[key]:
                    geometry[key] = [geometry[key][0] + window.x, geometry[key][1] + window.y]
            geometry["source_window"] = window.name
            out.append(VisualElement(
                id=f"{e.id}@{window.name}",
                kind=e.kind,
                bbox=shifted,
                center=c,
                confidence=min(0.95, e.confidence * 0.97),
                geometry=geometry,
                style=dict(e.style),
                labels=list(e.labels),
                text_refs=list(e.text_refs),
            ))
        return out

    @staticmethod
    def _fuse_elements(elements: list[VisualElement]) -> list[VisualElement]:
        result: list[VisualElement] = []
        for e in sorted(elements, key=lambda item: (-item.confidence, item.kind != "line_segment")):
            duplicate = False
            for q in result:
                if e.kind != q.kind:
                    continue
                center_distance = float(np.hypot(e.center.x - q.center.x, e.center.y - q.center.y))
                scale = max(8.0, min(e.bbox.width + e.bbox.height, q.bbox.width + q.bbox.height) * 0.08)
                if (
                    center_distance <= scale
                    and abs(e.bbox.width - q.bbox.width) <= max(10.0, q.bbox.width * 0.15)
                    and abs(e.bbox.height - q.bbox.height) <= max(10.0, q.bbox.height * 0.15)
                ):
                    duplicate = True
                    break
            if not duplicate:
                result.append(e)
        return result[:700]

    @staticmethod
    def _spatial_relations(elements: list[VisualElement], w: int, h: int):
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
        return (
            f"Canvas {scene.image['width']}x{scene.image['height']}px analyzed globally and across "
            f"{windows} overlapping local windows. Fused primitives: {inventory}. "
            f"Candidate relations: {len(scene.relations)}."
        )
