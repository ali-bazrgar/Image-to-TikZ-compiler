# Model-free architecture

The core compiler has no AI model dependency and performs all image analysis with deterministic Python libraries.

Pipeline:

1. Read and normalize the raster image.
2. Denoise and generate multiple edge/binary representations.
3. Detect contours and classify geometric primitives using deterministic measurements.
4. Detect line segments and endpoint/junction candidates.
5. Detect text-like regions structurally without guessing their characters.
6. Infer alignment, proximity, containment/connection candidates and global axes/baselines.
7. Serialize all observations into JSON and canonical natural-language text.
8. A downstream LLM is external to this project and may consume the resulting text to reason about semantics and generate TikZ.

The compiler itself never calls an LLM, VLM, OCR model, remote API, or model-serving process.
