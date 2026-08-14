from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class VerificationResult:
    compiled: bool
    score: float | None
    pdf_path: str | None
    rendered_image_path: str | None
    log: str
    error: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)


def extract_tikz_code(text: str) -> str:
    blocks = re.findall(r"```(?:latex|tex|tikz)?\s*(.*?)```", text, flags=re.I | re.S)
    if blocks:
        return max(blocks, key=len).strip()
    return text.strip()


def compile_and_compare(
    original_image: str | Path,
    tikz_text: str,
    *,
    pdflatex: str | Path = "pdflatex",
    pdftoppm: str | Path = "pdftoppm",
    work_dir: str | Path | None = None,
    timeout: float = 60.0,
) -> VerificationResult:
    original = Path(original_image)
    code = extract_tikz_code(tikz_text)
    root = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="image_to_tikz_verify_"))
    root.mkdir(parents=True, exist_ok=True)
    tex = root / "reconstruction.tex"
    pdf = root / "reconstruction.pdf"
    rendered = root / "reconstruction-1.png"
    tex.write_text(
        "\\documentclass[border=2pt]{standalone}\n"
        "\\usepackage{tikz}\n"
        "\\usetikzlibrary{arrows.meta,calc,decorations.pathreplacing,positioning,shapes.geometric}\n"
        "\\begin{document}\n"
        + code
        + "\n\\end{document}\n",
        encoding="utf-8",
    )

    try:
        proc = subprocess.run(
            [str(pdflatex), "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(root), str(tex)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=root,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return VerificationResult(False, None, None, None, "", f"pdflatex failed to start/finish: {exc}")
    if proc.returncode != 0 or not pdf.exists():
        return VerificationResult(False, None, None, None, proc.stdout[-12000:] + "\n" + proc.stderr[-12000:], "LaTeX compilation failed")

    if not Path(str(pdftoppm)).exists() and shutil.which(str(pdftoppm)) is None:
        return VerificationResult(True, None, str(pdf), None, proc.stdout[-12000:], "pdftoppm not found; compilation succeeded but image comparison was skipped")

    try:
        image_proc = subprocess.run(
            [str(pdftoppm), "-png", "-singlefile", "-r", "150", str(pdf), str(rendered.with_suffix(""))],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=root,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return VerificationResult(True, None, str(pdf), None, proc.stdout[-12000:], f"rendering failed: {exc}")
    if image_proc.returncode != 0 or not rendered.exists():
        return VerificationResult(True, None, str(pdf), None, image_proc.stdout[-12000:] + "\n" + image_proc.stderr[-12000:], "PDF rendered unsuccessfully")

    score, metrics = compare_images_detailed(original, rendered)
    return VerificationResult(True, score, str(pdf), str(rendered), image_proc.stdout[-12000:], metrics=metrics)


def compare_images(reference: str | Path, candidate: str | Path) -> float:
    score, _ = compare_images_detailed(reference, candidate)
    return score


def compare_images_detailed(reference: str | Path, candidate: str | Path) -> tuple[float, dict[str, float]]:
    a = cv2.imread(str(reference), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        raise ValueError("Could not read reference or candidate image")

    original_shapes = {"reference_h": float(a.shape[0]), "reference_w": float(a.shape[1]), "candidate_h": float(b.shape[0]), "candidate_w": float(b.shape[1])}
    target_h = max(a.shape[0], b.shape[0])
    target_w = max(a.shape[1], b.shape[1])
    a = _fit_canvas(a, target_w, target_h)
    b = _fit_canvas(b, target_w, target_h)

    a_blur = cv2.GaussianBlur(a, (5, 5), 0)
    b_blur = cv2.GaussianBlur(b, (5, 5), 0)
    mae = float(np.mean(cv2.absdiff(a_blur, b_blur))) / 255.0

    edges_a = cv2.Canny(a_blur, 50, 150)
    edges_b = cv2.Canny(b_blur, 50, 150)
    edge_mae = float(np.mean(cv2.absdiff(edges_a, edges_b))) / 255.0

    edge_overlap = _edge_overlap(edges_a, edges_b)
    bbox_similarity = _foreground_bbox_similarity(a, b)
    centroid_similarity = _centroid_similarity(edges_a, edges_b)

    pixel_score = max(0.0, min(1.0, 1.0 - mae))
    edge_score = max(0.0, min(1.0, 1.0 - edge_mae))
    score = max(
        0.0,
        min(
            1.0,
            0.40 * pixel_score
            + 0.25 * edge_score
            + 0.20 * edge_overlap
            + 0.10 * bbox_similarity
            + 0.05 * centroid_similarity,
        ),
    )
    metrics = {
        **original_shapes,
        "pixel_score": pixel_score,
        "edge_score": edge_score,
        "edge_overlap": edge_overlap,
        "bbox_similarity": bbox_similarity,
        "centroid_similarity": centroid_similarity,
        "score": score,
    }
    return score, metrics


def _edge_overlap(a: np.ndarray, b: np.ndarray) -> float:
    if not np.any(a) and not np.any(b):
        return 1.0
    if not np.any(a) or not np.any(b):
        return 0.0
    kernel = np.ones((3, 3), np.uint8)
    dil_a = cv2.dilate(a, kernel, iterations=1) > 0
    dil_b = cv2.dilate(b, kernel, iterations=1) > 0
    a_hit = float(np.logical_and(a > 0, dil_b).sum()) / max(float((a > 0).sum()), 1.0)
    b_hit = float(np.logical_and(b > 0, dil_a).sum()) / max(float((b > 0).sum()), 1.0)
    return 2.0 * a_hit * b_hit / max(a_hit + b_hit, 1e-9)


def _foreground_bbox_similarity(a: np.ndarray, b: np.ndarray) -> float:
    def bbox(image: np.ndarray) -> tuple[int, int, int, int] | None:
        mask = image < 245
        ys, xs = np.where(mask)
        if xs.size == 0 or ys.size == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    ba, bb = bbox(a), bbox(b)
    if ba is None and bb is None:
        return 1.0
    if ba is None or bb is None:
        return 0.0
    ax, ay, ax2, ay2 = ba
    bx, by, bx2, by2 = bb
    aw, ah = max(ax2 - ax, 1), max(ay2 - ay, 1)
    bw, bh = max(bx2 - bx, 1), max(by2 - by, 1)
    center_a = ((ax + ax2) / 2.0, (ay + ay2) / 2.0)
    center_b = ((bx + bx2) / 2.0, (by + by2) / 2.0)
    center_dist = np.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1])
    diag = np.hypot(a.shape[1], a.shape[0])
    center_score = max(0.0, 1.0 - float(center_dist) / max(float(diag) * 0.5, 1.0))
    size_score = max(0.0, 1.0 - (abs(aw - bw) / max(aw, bw) + abs(ah - bh) / max(ah, bh)) / 2.0)
    return 0.5 * center_score + 0.5 * size_score


def _centroid_similarity(a: np.ndarray, b: np.ndarray) -> float:
    def centroid(image: np.ndarray) -> tuple[float, float] | None:
        ys, xs = np.where(image > 0)
        if xs.size == 0 or ys.size == 0:
            return None
        return float(xs.mean()), float(ys.mean())

    ca, cb = centroid(a), centroid(b)
    if ca is None and cb is None:
        return 1.0
    if ca is None or cb is None:
        return 0.0
    dist = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))
    diag = float(np.hypot(a.shape[1], a.shape[0]))
    return max(0.0, 1.0 - dist / max(diag * 0.25, 1.0))


def _fit_canvas(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width), 255, dtype=np.uint8)
    h, w = image.shape[:2]
    y = max(0, (height - h) // 2)
    x = max(0, (width - w) // 2)
    crop_h = min(h, height - y)
    crop_w = min(w, width - x)
    canvas[y:y + crop_h, x:x + crop_w] = image[:crop_h, :crop_w]
    return canvas
