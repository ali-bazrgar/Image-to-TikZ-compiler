from __future__ import annotations

from pathlib import Path

import cv2

from .vir import VisualScene


def render_debug(image_path: str | Path, scene: VisualScene, output_path: str | Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {image_path}")

    for e in scene.elements:
        x1 = int(round(e.bbox.x))
        y1 = int(round(e.bbox.y))
        x2 = int(round(e.bbox.x + e.bbox.width))
        y2 = int(round(e.bbox.y + e.bbox.height))
        cv2.rectangle(image, (x1, y1), (x2, y2), (180, 80, 40), 1)
        label = f"{e.id}:{e.kind}"
        cv2.putText(image, label, (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 60, 200), 1, cv2.LINE_AA)
        cx, cy = int(round(e.center.x)), int(round(e.center.y))
        cv2.circle(image, (cx, cy), 2, (40, 150, 40), -1)
        if e.kind == "line":
            start = e.geometry.get("start")
            end = e.geometry.get("end")
            if start and end:
                cv2.circle(image, (int(start[0]), int(start[1])), 4, (40, 100, 220), -1)
                cv2.circle(image, (int(end[0]), int(end[1])), 4, (220, 100, 40), -1)

    for t in scene.texts:
        b = t.bbox
        cv2.rectangle(image, (int(b.x), int(b.y)), (int(b.x + b.width), int(b.y + b.height)), (40, 180, 180), 1)
        cv2.putText(image, f"{t.id}:{t.text}", (int(b.x), int(b.y + b.height + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 180, 180), 1, cv2.LINE_AA)

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), image):
        raise OSError(f"Could not write debug image: {target}")
