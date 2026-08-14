# Image-to-TikZ-compiler

A model-agnostic image understanding front-end for diagram-to-TikZ systems.

## Core idea

The goal is not image captioning. A TikZ-capable LLM needs a recoverable representation of **what is drawn, where it is drawn, how objects connect, what text is present, and which observations are uncertain**.

This repository therefore builds a Visual Intermediate Representation (VIR):

```text
image
  ↓
normalization / denoising
  ↓
classical CV: contours + lines + geometry
  ↓
OCR (optional)
  ↓
spatial + connection relations
  ↓
VIR JSON + canonical LLM text
  ↓
(optional) multimodal VLM semantic enrichment
  ↓
any text LLM / code model
  ↓
TikZ
```

The representation intentionally contains both machine-readable geometry and a declarative text rendering. This allows small or text-only models to reason over the scene without receiving the original pixels.

## Current components

- `vir.py`: canonical scene schema (elements, text blocks, relations, coordinate system).
- `analyzer.py`: deterministic image analysis using OpenCV plus optional Tesseract OCR.
- `graph.py`: line extraction, duplicate suppression, alignment/proximity relations and conservative arrowhead candidates.
- `semantic.py`: fuses measured geometry with optional VLM hypotheses without replacing the measured geometry.
- `vision.py`: adapter for any OpenAI-compatible multimodal endpoint.
- `pipeline.py`: end-to-end deterministic analysis pipeline.
- `serialize.py`: JSON, canonical context, and ready-to-use LLM prompt serialization.
- `render.py`: debug visualization of detections.

## Install

```bash
pip install -e .
```

For OCR and multimodal enrichment:

```bash
pip install -e '.[all]'
```

Tesseract itself must also be installed on the operating system for OCR.

## Basic usage

The original lightweight analyzer remains available:

```bash
image-to-vir diagram.png --pretty -o scene.json --context scene.txt --prompt tikz-prompt.txt
```

For the richer geometry/relationship pipeline:

```bash
image-to-vir diagram.png --full-pipeline --pretty \
  -o scene.json \
  --context scene.txt \
  --debug-image debug.png
```

The outputs have distinct jobs:

- `scene.json`: authoritative structured intermediate representation.
- `scene.txt`: canonical, model-readable description of detected facts and relations.
- `tikz-prompt.txt`: ready-to-send instruction for a text-only LLM.
- `debug.png`: visual audit of what the deterministic detector actually found.

## Optional vision-model enrichment

Any OpenAI-compatible multimodal server can act as a second semantic observer:

```bash
image-to-vir diagram.png --full-pipeline \
  --vision-url http://127.0.0.1:8080 \
  --vision-model YOUR_MODEL \
  --vision-json vision.json
```

The vision layer is deliberately **not** asked to produce TikZ. It reports scene type, semantic roles, approximate positions, relations, and uncertainties. The deterministic measurements remain the geometric source of truth.

## Design principles

1. **Observation vs. interpretation**: measured geometry and OCR are observations; semantic VLM statements are hypotheses.
2. **Geometry is explicit**: every visual element has a bounding box and center, with primitive-specific geometry when available.
3. **Relations are first-class**: alignment, proximity and connection candidates are represented explicitly.
4. **Coordinates are explicit**: normalized coordinates use `x=0..1` left→right and `y=0..1` top→bottom.
5. **Redundancy is intentional**: JSON is authoritative; canonical text makes the same information easier for weaker language models to consume.
6. **Model independence**: the image-understanding backend and TikZ-generating LLM can be changed independently.

## Verification direction

The next stage is not simply “make the description longer”. It is a closed reconstruction loop:

```text
VIR → TikZ → LaTeX render → PNG/SVG
                    ↓
             compare with input
                    ↓
             discrepancy map
                    ↓
          repair constraints / VIR
                    ↓
                  TikZ
```

This provides an objective signal for geometry and topology errors and is the foundation for later automatic critique/correction.

## Tests

```bash
pip install -e '.[test]'
pytest
```
