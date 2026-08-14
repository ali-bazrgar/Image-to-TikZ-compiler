from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import analyze_image
from .serialize import to_compact_prompt, to_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert an image into a visual representation for any downstream LLM."
    )
    parser.add_argument("image", help="Input image path")
    parser.add_argument("-o", "--output", help="Output JSON path; defaults to stdout")
    parser.add_argument("--context", help="Write the canonical LLM-readable text context")
    parser.add_argument("--prompt", help="Write a ready-to-use text-only LLM prompt")
    parser.add_argument("--debug-image", help="Write an annotated PNG showing detected primitives")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    parser.add_argument("--no-multiscale", action="store_true", help="Disable global+local multi-scale analysis")
    parser.add_argument("--ocr", choices=("auto", "on", "off"), default="auto", help="Lightweight RapidOCR mode")
    parser.add_argument("--ocr-score", type=float, default=0.35, help="Minimum OCR confidence")
    parser.add_argument("--micro-vlm-dir", help="Local directory for an optional sub-1GB SmolVLM-256M-Instruct model")
    parser.add_argument("--micro-vlm-device", default="auto", choices=("auto", "cpu", "cuda"), help="Device for optional micro-VLM")
    args = parser.parse_args()

    scene, context = analyze_image(
        args.image,
        multiscale=not args.no_multiscale,
        ocr=args.ocr,
        ocr_score_threshold=args.ocr_score,
        micro_vlm_dir=args.micro_vlm_dir,
        micro_vlm_device=args.micro_vlm_device,
    )
    payload = to_json(scene, indent=2 if args.pretty else None)

    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload)

    if args.context:
        Path(args.context).write_text(context, encoding="utf-8")
    if args.prompt:
        Path(args.prompt).write_text(to_compact_prompt(scene), encoding="utf-8")
    if args.debug_image:
        from .render import render_debug
        render_debug(args.image, scene, args.debug_image)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
