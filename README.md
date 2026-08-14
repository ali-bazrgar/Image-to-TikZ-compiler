# Image-to-TikZ-compiler

A model-agnostic image understanding front-end for diagram-to-TikZ systems.

## What this project does

The core problem is not simply image captioning. A TikZ-capable LLM needs a recoverable representation of **what is drawn and how the pieces relate to one another**. This repository therefore creates a Visual Intermediate Representation (VIR):

```text
image
  ↓
normalization / denoising
  ↓
classical CV: contours + lines + geometry
  ↓
OCR (optional)
  ↓
spatial relations
  ↓
VIR JSON + canonical LLM text
  ↓
(optional) multimodal VLM semantic enrichment
  ↓
any text LLM / code model
  ↓
TikZ
```

The representation intentionally contains both machine-readable geometry and a short declarative description. This lets small or text-only models reason over the scene without needing to see the original image.

## Install

```bash
pip install -e .
```

For OCR:

```bash
pip install -e '.[ocr]'
```

Tesseract itself must also be installed on the operating system.

## Basic usage

```bash
image-to-vir diagram.png --pretty -o scene.json --context scene.txt --prompt tikz-prompt.txt
```

The three outputs have different jobs:

* `scene.json`: full structured intermediate representation.
* `scene.txt`: canonical, model-readable description of all detected facts.
* `tikz-prompt.txt`: ready-to-send instruction for a text-only LLM.

## Optional vision-model enrichment

Any OpenAI-compatible multimodal server can be used as a second semantic observer. This is deliberately optional; the deterministic CV layer remains usable without a model.

Example with a local OpenAI-compatible endpoint:

```bash
image-to-vir diagram.png --vision-url http://127.0.0.1:8080 --vision-model YOUR_MODEL --vision-json vision.json
```

The enrichment layer asks the vision model for scene type, semantic roles, approximate positions, relations, and uncertainties. It does **not** ask the model for TikZ. This separation is important: image interpretation and code generation become independent stages.

## VIR design principles

1. **Observation vs. interpretation**: detected lines, boxes and text are observations; semantic hypotheses are separate.
2. **Geometry is explicit**: every object has a bounding box and center, with detailed primitive geometry where available.
3. **Relations are first-class**: alignment, proximity and connection candidates are represented explicitly rather than forcing an LLM to infer them from raw coordinates.
4. **Coordinate system is stated**: no hidden image-axis assumptions.
5. **Redundancy is intentional**: JSON is authoritative; the textual serialization makes the same information easier for weaker language models to consume.
6. **No model lock-in**: the core pipeline uses standard computer vision and optional OCR. A VLM is an enrichment backend, not a required dependency.

## What this does not claim yet

This first implementation is a strong front-end rather than a finished image-to-TikZ system. Classical CV cannot reliably infer every semantic object in engineering diagrams, arrows, mathematical notation, topology, or chart semantics. The optional VLM layer and a future domain-specific semantic parser are intended to address those gaps.

The next major stage should add: arrow-head detection, connected-component graph construction, axis/plot recognition, dimension-line recognition, shape-specific classifiers, crop-level analysis, confidence calibration, and a renderer-based verification loop (VIR → TikZ → PDF/SVG → image comparison).
