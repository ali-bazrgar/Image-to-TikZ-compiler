from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyzer import ImageAnalyzer
from .serialize import to_compact_prompt, to_json, to_llm_context


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a raster diagram into an LLM-readable Visual Intermediate Representation.")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("-o", "--output", help="Output JSON path; defaults to stdout")
    parser.add_argument("--context", help="Write the canonical LLM text context to this file")
    parser.add_argument("--prompt", help="Write a ready-to-use text-only LLM prompt to this file")
    parser.add_argument("--no-ocr", action="store_true", help="Disable optional Tesseract OCR")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args()

    scene = ImageAnalyzer(enable_ocr=not args.no_ocr).analyze(args.image)
    payload = to_json(scene, indent=2 if args.pretty else None)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload)
    if args.context:
        Path(args.context).write_text(to_llm_context(scene), encoding="utf-8")
    if args.prompt:
        Path(args.prompt).write_text(to_compact_prompt(scene), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
