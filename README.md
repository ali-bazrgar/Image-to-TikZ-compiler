# Image-to-TikZ-compiler

A model-independent image-analysis front-end for diagram-to-TikZ systems.

## Core contract

The deterministic core never requires AI. An optional lightweight OCR adapter may be enabled, but this project deliberately rejects the use of models larger than 1 GB. The downstream LLM is outside the compiler and consumes the generated Visual Intermediate Representation (VIR).

```text
image
  ↓
global + overlapping local analysis
  ↓
geometry / curves / paths
  ↓
stroke + topology analysis
  ↓
text-region structure
  ↓
(optional) lightweight OCR < 1 GB
  ↓
groups + repetition + symmetry + scene grammar
  ↓
Visual Intermediate Representation
  ↓
canonical JSON + spatial natural-language context
  ↓
ANY downstream text LLM
  ↓
TikZ
```

## Runtime dependencies

Required runtime libraries:

- OpenCV
- NumPy
- Pillow

Optional lightweight OCR:

```bash
pip install -e '.[ocr]'
```

The optional OCR path uses RapidOCR/ONNX Runtime mobile-style models. Current RapidOCR/PaddleOCR documentation lists mobile detection and recognition models in the single- and tens-of-megabytes range, far below the project's 1 GB model limit. citeturn968545search2turn968545search3

No large VLM, large language model, inference server, API key, or network service is required by the compiler.

## Current analysis layers

The deterministic engine measures:

- quadrilateral, polygon, circle/ellipse candidates
- line segments with endpoints, length, angle and orientation
- curve/path candidates and sampled geometry
- endpoint, junction and crossing candidates
- possible arrowhead, axis/baseline and dimension-line patterns
- stroke orientation and conservative stroke-style evidence
- text-like regions, glyph runs and baseline candidates
- stacked/superscript/subscript-like layout candidates
- label/annotation relationships
- horizontal/vertical alignment and proximity
- connection topology and structure groups
- repeated visual families
- approximate mirror symmetry
- spatial regions and scene layout
- normalized coordinates and confidence values

When OCR is available, recognized text is merged back into the same VIR without replacing the measured geometry.

## Installation

Core only:

```bash
pip install -e .
```

Core + tests:

```bash
pip install -e '.[test]'
```

Core + lightweight OCR:

```bash
pip install -e '.[ocr]'
```

## Usage

Without optional OCR:

```bash
image-to-vir diagram.png \
  --ocr off \
  --pretty \
  -o scene.json \
  --context scene.txt \
  --prompt tikz-prompt.txt \
  --debug-image debug.png
```

Automatic lightweight OCR when installed:

```bash
image-to-vir diagram.png \
  --ocr auto \
  --pretty \
  -o scene.json \
  --context scene.txt
```

Require OCR and fail clearly when it is not installed:

```bash
image-to-vir diagram.png --ocr on
```

Disable multiscale analysis for faster debugging:

```bash
image-to-vir diagram.png --no-multiscale --ocr off
```

## Output

`scene.json` is the authoritative structured representation.

`scene.txt` is the canonical natural-language representation intended for any text-only LLM. It includes pixel coordinates, normalized coordinates, geometry, relations, topology, spatial narrative, uncertainty and optional OCR text.

`tikz-prompt.txt` wraps the same record in a generic reconstruction instruction.

`debug.png` provides a visual audit of what the computer-vision stages detected.

## Important design rule

Observed geometry is never silently replaced by semantic guesses. Candidates such as "arrowhead", "axis", "dimension line", "label", "symmetry" and "connection" remain explicitly marked as hypotheses with evidence/confidence.

The compiler is independent of the downstream model. GPT, Qwen, Llama, Gemma or another text/code model can consume the same VIR. The only model-size restriction applies to optional models shipped/used by this compiler: **1 GB maximum**.

## Verification direction

The eventual reconstruction loop can remain outside the model-free analysis contract:

```text
VIR → downstream LLM → TikZ → LaTeX/SVG render
                           ↓
                    image comparison
                           ↓
                    discrepancy report
```

## Tests

```bash
pytest
```
