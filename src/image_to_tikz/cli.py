from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .analyzer import ImageAnalyzer
from .serialize import to_compact_prompt, to_json, to_llm_context


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a raster diagram into an LLM-readable Visual Intermediate Representation.")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("-o", "--output", help="Output JSON path; defaults to stdout")
    parser.add_argument("--context", help="Write the canonical LLM text context to this file")
    parser.add_argument("--prompt", help="Write a ready-to-use text-only LLM prompt to this file")
    parser.add_argument("--debug-image", help="Write an annotated PNG showing detected objects, boxes and line endpoints")
    parser.add_argument("--no-ocr", action="store_true", help="Disable optional Tesseract OCR")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    parser.add_argument("--vision-url", help="Optional OpenAI-compatible multimodal endpoint, e.g. http://127.0.0.1:8080")
    parser.add_argument("--vision-model", help="Model name used by the multimodal endpoint")
    parser.add_argument("--vision-api-key", default=os.getenv("VISION_API_KEY"), help="Optional API key")
    parser.add_argument("--vision-json", help="Write semantic VLM enrichment to this JSON file")
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
    if args.debug_image:
        from .render import render_debug
        render_debug(args.image, scene, args.debug_image)

    if args.vision_url or args.vision_json:
        if not (args.vision_url and args.vision_model):
            parser.error("--vision-url and --vision-model are both required when vision enrichment is enabled")
        from .vision import VisionEnricher
        enriched = VisionEnricher(args.vision_url, args.vision_model, args.vision_api_key).analyze(args.image)
        target = args.vision_json or "vision-enrichment.json"
        Path(target).write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
