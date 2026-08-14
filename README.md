# Image-to-TikZ-compiler

A model-independent image-analysis front-end for diagram-to-TikZ systems.

## Core contract

The deterministic core never requires AI. Optional lightweight OCR and a local lightweight VLM can enrich the representation. For a machine with a 3 GB GPU, the compiler enforces a **3 GB hard ceiling on optional model weights** and defaults to a **2.5 GB recommended budget** to leave VRAM for runtime overhead. The downstream LLM that turns VIR into TikZ is outside this internal GPU-oriented budget.

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
(optional) lightweight OCR
  ↓
groups + repetition + symmetry + scene grammar
  ↓
Visual Intermediate Representation
  ↓
deterministic semantic-crop selection
  ↓
(optional) lightweight VLM on only high-value crops
  ↓
canonical JSON + spatial natural-language context
  ↓
ANY downstream text/code LLM
  ↓
TikZ
```

## Recommended lightweight semantic observer

The first integrated micro-VLM adapter targets `HuggingFaceTB/SmolVLM-256M-Instruct`. Its current Hugging Face model weight file is about 513 MB, so it is comfortably inside the recommended GPU budget. The adapter uses `transformers` locally and does not call an inference API.

The VLM is **not authoritative**. It contributes only semantic hypotheses; measured coordinates, topology and geometry from the deterministic CV pipeline remain authoritative.

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

## GPU budget policy

For the user's 3 GB GPU target:

- hard weight ceiling: **3.0 GB**
- default/recommended weight ceiling: **2.5 GB**
- smaller models are preferred for speed
- the VLM is run only on selected high-value crops rather than repeatedly on the whole image
- deterministic OpenCV analysis remains the main source of geometry and topology

The size check measures local weight files before loading. The policy is about model weights; actual VRAM usage can be higher because of framework buffers, activations, processor state and CUDA overhead.

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

## Semantic crop selection

The optional VLM is not blindly run on every image. A deterministic crop selector chooses high-value regions such as:

- text-heavy regions
- low-confidence primitives
- curves and ambiguous paths
- junction/arrow candidates
- dense connected structure groups

The selector produces crop IDs, global bounding boxes, reasons and priorities. The VLM then analyzes only those crops and the results are attached to `scene.image.micro_vlm_hypotheses`.

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

Use a local semantic VLM:

```bash
image-to-vir diagram.png \
  --ocr auto \
  --micro-vlm-dir ./models/SmolVLM-256M-Instruct \
  --micro-vlm-device auto \
  --micro-vlm-max-crops 8 \
  --micro-vlm-max-model-gb 2.5 \
  --pretty \
  -o scene.json \
  --context scene.txt
```

A model above the configured budget is rejected before loading. The absolute hard ceiling is 3 GB.

## Output

`scene.json` is the authoritative structured representation.

`scene.txt` is the canonical natural-language representation intended for any text-only LLM. It includes pixel coordinates, normalized coordinates, geometry, relations, topology, spatial narrative, uncertainty, optional OCR text and optional semantic hypotheses.

`tikz-prompt.txt` wraps the same record in a generic reconstruction instruction.

`debug.png` provides a visual audit of what the computer-vision stages detected.

## Important design rule

Observed geometry is never silently replaced by semantic guesses. Candidates such as "arrowhead", "axis", "dimension line", "label", "symmetry" and "connection" remain explicitly marked as hypotheses with evidence/confidence.

The compiler is independent of the downstream model. GPT, Qwen, Llama, Gemma or another text/code model can consume the same VIR. The internal GPU budget applies only to optional models used by this compiler; it does not restrict the final downstream LLM.

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
