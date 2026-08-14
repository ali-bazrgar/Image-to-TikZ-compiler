# Image-to-TikZ-compiler

A model-independent image-analysis front-end for diagram-to-TikZ systems.

## Core contract

The deterministic core never requires AI. Optional lightweight OCR and a local lightweight VLM can enrich the representation, but **any model used inside this compiler must have total local weight size <= 1 GB**. The downstream LLM that turns VIR into TikZ is outside the compiler and is not subject to this internal observer limit.

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
(optional) lightweight OCR <= 1 GB
  ↓
groups + repetition + symmetry + scene grammar
  ↓
Visual Intermediate Representation
  ↓
canonical JSON + spatial natural-language context
  ↓
(optional) lightweight semantic VLM hypothesis <= 1 GB
  ↓
ANY downstream text/code LLM
  ↓
TikZ
```

## Recommended lightweight semantic observer

The first integrated micro-VLM adapter targets `HuggingFaceTB/SmolVLM-256M-Instruct`. Its current Hugging Face model weight file is about 513 MB, safely below the 1 GB compiler limit. It is designed for compact multimodal tasks including image description, document QA and basic visual reasoning. The adapter uses `transformers` locally and does not call an inference API. citeturn964582search1turn501629search1turn352237search0

The VLM is **not authoritative**. It contributes only a `MICRO_VLM_HYPOTHESIS`; measured coordinates, topology and geometry from the deterministic CV pipeline always remain authoritative.

## Runtime dependencies

Required runtime libraries:

- OpenCV
- NumPy
- Pillow

Optional lightweight OCR:

```bash
pip install -e '.[ocr]'
```

Optional lightweight semantic VLM:

```bash
pip install -e '.[micro-vlm]'
```

The OCR and VLM adapters are optional. The core pipeline works without either.

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

When OCR is available, recognized text is merged back into the same VIR without replacing measured geometry.

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

Core + lightweight semantic VLM:

```bash
pip install -e '.[micro-vlm]'
```

## Usage

Without optional models:

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
image-to-vir diagram.png --ocr auto --pretty -o scene.json --context scene.txt
```

Use a local SmolVLM-256M-Instruct checkout as a semantic observer:

```bash
image-to-vir diagram.png \
  --ocr auto \
  --micro-vlm-dir ./models/SmolVLM-256M-Instruct \
  --micro-vlm-device auto \
  --pretty \
  -o scene.json \
  --context scene.txt
```

The adapter validates the total local weight size before loading and rejects anything above 1 GB.

## Output

`scene.json` is the authoritative structured representation.

`scene.txt` is the canonical natural-language representation intended for any text-only LLM. It includes pixel coordinates, normalized coordinates, geometry, relations, topology, spatial narrative, uncertainty, optional OCR text and optional semantic hypotheses.

`tikz-prompt.txt` wraps the same record in a generic reconstruction instruction.

`debug.png` provides a visual audit of what the computer-vision stages detected.

## Important design rule

Observed geometry is never silently replaced by semantic guesses. Candidates such as "arrowhead", "axis", "dimension line", "label", "symmetry" and "connection" remain explicitly marked as hypotheses with evidence/confidence.

The compiler is independent of the downstream model. GPT, Qwen, Llama, Gemma or another text/code model can consume the same VIR. The internal model-size restriction applies only to optional models used by this compiler: **1 GB maximum**.

## Verification direction

The eventual reconstruction loop can remain outside the image-understanding contract:

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
