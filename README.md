# Image-to-TikZ-compiler

A model-independent image-understanding front-end whose output is designed for a downstream LLM that generates TikZ.

## Final pipeline

```text
original image
    ↓
Python/OpenCV deterministic multi-scale analysis
    ↓
geometry / curves / paths / strokes / topology / text regions
    ↓
groups / repetition / symmetry / scene grammar / canonical graph
    ↓
(optional) lightweight OCR
    ↓
(optional) SmolVLM2 semantic inspection of only selected crops
    ↓
canonical Visual Intermediate Representation
    ↓
Golden Prompt + machine-readable visual context
    ↓
commercial text/code LLM
    ↓
TikZ/LaTeX
    ↓
(optional) local compile + render + similarity verification
```

The deterministic measurements are authoritative. OCR, domain routing, specialized-detector roles and SmolVLM2 are semantic evidence/hypotheses and must never silently overwrite measured geometry or topology.

## Recommended local vision model

For a 3 GB GPU, use the GGUF SmolVLM2 backend through llama.cpp:

```text
SmolVLM2-2.2B-Instruct-Q4_K_M.gguf
mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf
```

The local model + mmproj are budget-checked before use. The hard model-weight ceiling is 3.0 GB and the default recommended budget is 2.5 GB, leaving room for runtime overhead.

llama.cpp provides a multimodal OpenAI-compatible server and supports a `--mmproj` projector for multimodal models. citeturn518984search1turn518984search6

## Installation

Core:

```bash
pip install -e .
```

Tests:

```bash
pip install -e '.[test]'
```

OCR:

```bash
pip install -e '.[ocr]'
```

Optional Transformers VLM:

```bash
pip install -e '.[micro-vlm]'
```

Desktop Studio:

```bash
pip install -e '.[gui]'
```

## Desktop Studio

Launch:

```bash
image-to-tikz-studio
```

The Studio GUI provides one place for:

- source image
- llama.cpp executable
- SmolVLM2 GGUF and mmproj
- llama.cpp port, context and GPU layers
- number of semantic crops
- OCR and multiscale settings
- pdflatex / pdftoppm paths for verification
- commercial OpenAI-compatible endpoint, model and session-only API key
- output directory
- Visual Context, Golden Prompt, TikZ and logs

The API key is intentionally not saved to disk.

## SmolVLM2 GGUF setup

For the GTX 1060 3 GB target, use:

```text
SmolVLM2-2.2B-Instruct-Q4_K_M.gguf
mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf
```

Start llama.cpp:

```bat
llama-server.exe ^
  -m "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf" ^
  --mmproj "mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf" ^
  --host 127.0.0.1 ^
  --port 8080 ^
  -c 4096 ^
  -ngl 99 ^
  --temp 0.1 ^
  --top-p 0.9
```

llama.cpp documents multimodal image input through its OpenAI-compatible server and `--mmproj`. citeturn518984search1turn518984search6

## CLI

Without AI observers:

```bash
image-to-vir diagram.png --ocr off --pretty \
  -o scene.json \
  --context llm_context.txt \
  --prompt golden_prompt.txt
```

With SmolVLM2 through llama.cpp:

```bash
image-to-vir diagram.png \
  --ocr auto \
  --micro-vlm-backend llama-server \
  --micro-vlm-model-path "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf" \
  --micro-vlm-mmproj-path "mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf" \
  --micro-vlm-base-url http://127.0.0.1:8080/v1 \
  --micro-vlm-model-name SmolVLM2-2.2B-Instruct \
  --micro-vlm-max-crops 8 \
  --micro-vlm-max-model-gb 2.5 \
  --pretty \
  -o scene.json \
  --context llm_context.txt \
  --prompt golden_prompt.txt
```

## Golden Prompt

`golden_prompt.txt` is a strict reconstruction contract for a downstream text/code LLM that does not receive the image. It requires the model to:

- preserve measured coordinates and topology
- reconstruct canonical graph components first
- preserve curves, stroke evidence, text placement and ordering
- treat OCR/domain/VLM semantics as hypotheses
- never invent unreadable labels
- output only one compilable TikZ/LaTeX code block
- perform an internal final structural audit before emitting code

The same Golden Prompt is produced by both the CLI and Studio.

## Commercial LLM

Studio can send the Golden Prompt directly to an OpenAI-compatible `/v1/chat/completions` endpoint. For providers with a different API contract, use the exported prompt manually or configure a compatible gateway.

## Verification

The Studio can take the generated TikZ, compile it with `pdflatex`, render it with `pdftoppm`, and calculate a structural image-similarity score against the original. This score is a useful verification signal, not a mathematical proof of pixel-perfect equality.

The intended high-accuracy loop is:

```text
image
  ↓
rich VIR + Golden Prompt
  ↓
downstream LLM
  ↓
TikZ
  ↓
LaTeX render
  ↓
image comparison
  ↓
repair prompt / second LLM pass
```

## Tests

```bash
pytest -q
```

GitHub Actions runs the regression suite on every push and pull request.
