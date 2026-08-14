from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
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

    score = compare_images(original, rendered)
    return VerificationResult(True, score, str(pdf), str(rendered), image_proc.stdout[-12000:])


def compare_images(reference: str | Path, candidate: str | Path) -> float:
    a = cv2.imread(str(reference), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        raise ValueError("Could not read reference or candidate image")
    target_h = max(a.shape[0], b.shape[0])
    target_w = max(a.shape[1], b.shape[1])
    a = _fit_canvas(a, target_w, target_h)
    b = _fit_canvas(b, target_w, target_h)
    a = cv2.GaussianBlur(a, (5, 5), 0)
    b = cv2.GaussianBlur(b, (5, 5), 0)
    mae = float(np.mean(cv2.absdiff(a, b))) / 255.0
    edges_a = cv2.Canny(a, 50, 150)
    edges_b = cv2.Canny(b, 50, 150)
    edge_mae = float(np.mean(cv2.absdiff(edges_a, edges_b))) / 255.0
    return max(0.0, min(1.0, 1.0 - (0.7 * mae + 0.3 * edge_mae)))


def _fit_canvas(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width), 255, dtype=np.uint8)
    h, w = image.shape[:2]
    y = max(0, (height - h) // 2)
    x = max(0, (width - w) // 2)
    crop_h = min(h, height - y)
    crop_w = min(w, width - x)
    canvas[y:y + crop_h, x:x + crop_w] = image[:crop_h, :crop_w]
    return canvas
