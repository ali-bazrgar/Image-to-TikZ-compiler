# Strictly model-free core

The compiler itself performs no AI inference.

Required runtime libraries are OpenCV, NumPy and Pillow. The pipeline measures pixels, edges, contours, line segments, text-like regions, coordinates, repetition, alignment, proximity, and connection candidates. It serializes these observations to JSON and deterministic natural language.

The downstream LLM is outside this repository's runtime. Its job is semantic interpretation and TikZ generation from the generated representation.

Text glyphs are not invented by the core. A future non-core OCR adapter may supply character content, but the core remains usable without one.
