from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

import image_to_tikz.pipeline as pipeline_module
from image_to_tikz.micro_vlm import MicroVLMError, validate_model_directory
from image_to_tikz.ocr import LightweightOCRError, _normalize_result
from image_to_tikz.pipeline import analyze_image


def test_pipeline_extracts_basic_geometry(tmp_path):
    image = np.full((300, 500, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (60, 80), (220, 200), (0, 0, 0), 3)
    cv2.circle(image, (360, 140), 45, (0, 0, 0), 3)
    cv2.line(image, (220, 140), (315, 140), (0, 0, 0), 3)
    path = tmp_path / "diagram.png"
    assert cv2.imwrite(str(path), image)

    scene, context = analyze_image(path, multiscale=False, ocr="off")
    assert scene.image["width"] == 500
    assert scene.image["height"] == 300
    assert scene.elements
    assert "ELEMENTS" in context
    assert "GRAPH" in context


def test_pipeline_exposes_model_free_structure(tmp_path):
    image = np.full((360, 600, 3), 255, dtype=np.uint8)
    cv2.line(image, (80, 180), (300, 180), (0, 0, 0), 3)
    cv2.line(image, (300, 180), (300, 90), (0, 0, 0), 3)
    cv2.line(image, (300, 180), (300, 270), (0, 0, 0), 3)
    cv2.rectangle(image, (40, 140), (80, 220), (0, 0, 0), 3)
    path = tmp_path / "topology.png"
    assert cv2.imwrite(str(path), image)

    scene, context = analyze_image(path, multiscale=False, ocr="off")
    relation_names = {r.relation for r in scene.relations}
    assert relation_names & {"line_junction_candidate", "line_crossing_candidate"}
    assert "stroke_orientation" in next(e.geometry for e in scene.elements if e.kind == "line_segment")
    assert "DOWNSTREAM_TASK" in context


def test_pipeline_detects_curved_path(tmp_path):
    image = np.full((320, 520, 3), 255, dtype=np.uint8)
    pts = np.array([[40 + x, 170 + int(55 * np.sin(x / 28.0))] for x in range(420)], np.int32)
    cv2.polylines(image, [pts], False, (0, 0, 0), 3)
    path = tmp_path / "curve.png"
    assert cv2.imwrite(str(path), image)

    scene, context = analyze_image(path, multiscale=False, ocr="off")
    curve_kinds = {"curve_path", "polyline_or_arc"}
    assert any(e.kind in curve_kinds for e in scene.elements)
    assert "curve" in context.lower() or "polyline" in context.lower()


def test_ocr_auto_falls_back_cleanly_when_optional_engine_is_missing(tmp_path, monkeypatch):
    image = np.full((120, 240, 3), 255, dtype=np.uint8)
    cv2.putText(image, "A1", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    path = tmp_path / "text.png"
    assert cv2.imwrite(str(path), image)

    def missing_engine(*args, **kwargs):
        raise LightweightOCRError("test: optional OCR unavailable")

    monkeypatch.setattr(pipeline_module, "enrich_scene_with_ocr", missing_engine)
    scene, _ = analyze_image(path, multiscale=False, ocr="auto")
    assert scene.image["ocr"]["enabled"] is False
    assert any("optional OCR unavailable" in warning for warning in scene.warnings)


def test_ocr_on_requires_optional_engine(tmp_path, monkeypatch):
    image = np.full((120, 240, 3), 255, dtype=np.uint8)
    path = tmp_path / "text.png"
    assert cv2.imwrite(str(path), image)
    monkeypatch.setattr(
        pipeline_module,
        "enrich_scene_with_ocr",
        lambda *args, **kwargs: (_ for _ in ()).throw(LightweightOCRError("missing")),
    )
    try:
        analyze_image(path, multiscale=False, ocr="on")
    except LightweightOCRError as exc:
        assert str(exc) == "missing"
    else:
        raise AssertionError("ocr='on' must fail when the optional engine is unavailable")


def test_ocr_result_normalization_supports_current_and_legacy_shapes():
    box = [[[1, 2], [20, 2], [20, 12], [1, 12]]]
    current = SimpleNamespace(boxes=box, txts=("label",), scores=(0.91,))
    assert _normalize_result(current)[0][1:] == ("label", 0.91)
    legacy = [(box[0], "label", 0.87)]
    assert _normalize_result((legacy, [0.01, 0.02]))[0][1:] == ("label", 0.87)


def test_micro_vlm_model_policy_accepts_small_and_rejects_over_3gb(tmp_path):
    good = tmp_path / "good"
    good.mkdir()
    (good / "model.safetensors").write_bytes(b"x" * 1024)
    assert validate_model_directory(good) == 1024

    bad = tmp_path / "bad"
    bad.mkdir()
    with (bad / "model.safetensors").open("wb") as handle:
        handle.truncate(3_000_000_001)
    try:
        validate_model_directory(bad)
    except MicroVLMError as exc:
        assert "exceeding" in str(exc)
    else:
        raise AssertionError("The 3GB hard model-size policy must be enforced")
