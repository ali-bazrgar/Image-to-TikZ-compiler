# Image-to-TikZ-compiler

A strictly model-free image-analysis front-end for diagram-to-TikZ systems.

## Core contract

This repository does **not** run an AI model. Its job ends at a deterministic, text-readable representation of the input image. A downstream LLM is responsible for semantic interpretation and TikZ generation.

```text
image
  ↓
normalization / denoising
  ↓
OpenCV geometry extraction
  ↓
shapes + line segments + endpoints
  ↓
text-like region detection
  ↓
spatial / alignment / connection candidates
  ↓
higher-level structure: groups + repetition + symmetry + stroke features
  ↓
Visual Intermediate Representation (VIR)
  ↓
canonical JSON + deterministic natural-language context
  ↓
ANY external text LLM
  ↓
TikZ
```

The important design goal is **recoverability**, not captioning: the generated representation exposes what the computer can measure about the image, where it occurs, how elements relate, recurring structure, and where uncertainty remains.

## Runtime dependencies

Required runtime libraries are only:

- OpenCV
- NumPy
- Pillow

No model weights, inference server, API key, OCR engine, or network service is required.

## What is measured

The model-free analyzer currently extracts:

- quadrilateral, polygon, circle/ellipse candidates
- line segments with endpoints, length, angle, and orientation
- possible junction/arrowhead endpoint candidates
- long horizontal/vertical axis or baseline candidates
- text-like regions (location and size, without inventing character content)
- horizontal and vertical alignment
- proximity
- line-to-shape connection candidates
- endpoint-to-shape connection candidates
- structural groups formed from connection evidence
- repeated visual families
- approximate horizontal/vertical mirror symmetry
- stroke orientation and normalized length features
- normalized coordinates and confidence values

The core deliberately does **not** claim that a measured primitive has a domain-specific meaning. For example, a long horizontal line may be an axis, baseline, dimension line, or ordinary connector. The downstream LLM receives the evidence and decides among hypotheses.

## Install

```bash
pip install -e .
```

For tests:

```bash
pip install -e '.[test]'
```

## Usage

```bash
image-to-vir diagram.png \
  --pretty \
  -o scene.json \
  --context scene.txt \
  --prompt tikz-prompt.txt \
  --debug-image debug.png
```

Outputs:

- `scene.json`: authoritative structured VIR.
- `scene.txt`: deterministic natural-language representation designed for a text-only LLM.
- `tikz-prompt.txt`: a wrapper prompt around the same evidence.
- `debug.png`: visual audit of detected primitives and text regions.

## LLM hand-off

The downstream LLM should treat the generated data as evidence, not as a pre-decided semantic interpretation. It should first infer the diagram class and topology, then produce TikZ while preserving measured geometry and explicitly marking uncertain assumptions.

The compiler therefore remains independent of the choice of GPT, Qwen, Llama, Gemma, or any other external model.

## Verification direction

A later stage can close the reconstruction loop without changing the model-free contract:

```text
VIR → LLM-generated TikZ → LaTeX/SVG render
                         ↓
                  image comparison
                         ↓
                  discrepancy report
                         ↓
                  revised LLM prompt
```

The comparison stage is an evaluator; it does not become part of the image-understanding core.

## Tests

```bash
pytest
```
