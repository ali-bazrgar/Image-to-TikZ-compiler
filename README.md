# Image-to-TikZ-compiler

A model-independent image-analysis front-end for diagram-to-TikZ systems.

## Core contract

The deterministic core never requires AI. Optional lightweight OCR and local lightweight VLM observers can enrich the representation. For a machine with a 3 GB GPU, the compiler enforces a **3 GB hard ceiling on optional model weights** and defaults to a **2.5 GB recommended budget** to leave VRAM for runtime overhead. The downstream LLM that turns VIR into TikZ is outside this internal GPU-oriented budget.

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
canonical scene graph
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

## Lightweight semantic observers

Two optional local backends are supported:

1. A Transformers backend for small Hugging Face VLM checkpoints.
2. A **llama.cpp server backend** for GGUF multimodal models such as `ggml-org/SmolVLM2-2.2B-Instruct-GGUF`.

The SmolVLM2 GGUF repository currently provides a 1.11 GB `Q4_K_M` text model and a 593 MB `mmproj` projector. llama.cpp explicitly lists SmolVLM2 as supported multimodal input and supports supplying a custom `--mmproj` file. citeturn854509search0turn150514search5

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

Optional Transformers VLM:

```bash
pip install -e '.[micro-vlm]'
```

The llama.cpp GGUF backend uses only Python's standard library for its HTTP/base64 adapter; the actual multimodal inference is performed by a locally running llama.cpp server.

## GPU budget policy

For the 3 GB GPU target:

- hard weight ceiling: **3.0 GB combined model + multimodal projector**
- default/recommended weight ceiling: **2.5 GB**
- smaller models are preferred for speed
- the VLM is run only on selected high-value crops rather than repeatedly on the whole image
- deterministic OpenCV analysis remains the main source of geometry and topology

The size check measures local weight files. Actual VRAM usage can be higher because of framework buffers, activations, context/KV cache and CUDA overhead.

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
- canonical node/edge graph and connected components
- normalized coordinates, confidence values and evidence provenance

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

Core + Transformers semantic VLM:

```bash
pip install -e '.[micro-vlm]'
```

No Python package is required for the llama.cpp backend itself; you need a llama.cpp build that includes `llama-server`/multimodal support.

## SmolVLM2 GGUF setup

For your GTX 1060 3 GB target, the recommended pair is:

```text
SmolVLM2-2.2B-Instruct-Q4_K_M.gguf
mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf
```

The current Hugging Face files are approximately 1.11 GB and 593 MB respectively. citeturn854509search0

Start llama.cpp with:

```bat
llama-server.exe ^
  -m "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf" ^
  --mmproj "mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf" ^
  --host 127.0.0.1 ^
  --port 8080 ^
  -c 4096 ^
  -ngl 99
```

llama.cpp documents multimodal image input through `llama-server` and `--mmproj`; it also lists SmolVLM2 among supported vision models. citeturn150514search5

Then run the compiler against selected crops:

```bat
image-to-vir diagram.png ^
  --ocr auto ^
  --micro-vlm-backend llama-server ^
  --micro-vlm-model-path "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf" ^
  --micro-vlm-mmproj-path "mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf" ^
  --micro-vlm-base-url http://127.0.0.1:8080/v1 ^
  --micro-vlm-model-name SmolVLM2-2.2B-Instruct ^
  --micro-vlm-max-crops 8 ^
  --micro-vlm-max-model-gb 2.5 ^
  --pretty ^
  -o scene.json ^
  --context scene.txt
```

The compiler validates the combined model + `mmproj` file size before sending any image to the server. The hard ceiling remains 3 GB.

## Usage without optional models

```bash
image-to-vir diagram.png \
  --ocr off \
  --pretty \
  -o scene.json \
  --context scene.txt \
  --prompt tikz-prompt.txt \
  --debug-image debug.png
```

## Output

`scene.json` is the authoritative structured representation.

`scene.txt` is the canonical natural-language representation intended for any text-only LLM. It includes pixel coordinates, normalized coordinates, geometry, relations, topology, the canonical graph, spatial narrative, evidence provenance, uncertainty, optional OCR text and optional semantic hypotheses.

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
