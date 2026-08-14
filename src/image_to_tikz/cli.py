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
    parser.add_argument("--micro-vlm-backend", choices=("none", "transformers", "llama-server"), default="none", help="Optional semantic observer backend")
    parser.add_argument("--micro-vlm-dir", help="Local Transformers model directory")
    parser.add_argument("--micro-vlm-model-path", help="Local GGUF language-model file for llama.cpp")
    parser.add_argument("--micro-vlm-mmproj-path", help="Local GGUF multimodal projector for llama.cpp")
    parser.add_argument("--micro-vlm-base-url", default="http://127.0.0.1:8080/v1", help="llama.cpp OpenAI-compatible base URL")
    parser.add_argument("--micro-vlm-model-name", default="SmolVLM2-2.2B-Instruct", help="Model name sent to the llama.cpp server")
    parser.add_argument("--micro-vlm-device", default="auto", choices=("auto", "cpu", "cuda"), help="Device for optional Transformers VLM")
    parser.add_argument("--micro-vlm-max-crops", type=int, default=8, help="Maximum high-value crops inspected by the optional VLM")
    parser.add_argument("--micro-vlm-max-model-gb", type=float, default=2.5, help="Maximum combined model-weight size in GB; hard ceiling is 3.0 GB")
    args = parser.parse_args()

    scene, context = analyze_image(
        args.image,
        multiscale=not args.no_multiscale,
        ocr=args.ocr,
        ocr_score_threshold=args.ocr_score,
        micro_vlm_backend=args.micro_vlm_backend,
        micro_vlm_dir=args.micro_vlm_dir,
        micro_vlm_device=args.micro_vlm_device,
        micro_vlm_model_path=args.micro_vlm_model_path,
        micro_vlm_mmproj_path=args.micro_vlm_mmproj_path,
        micro_vlm_base_url=args.micro_vlm_base_url,
        micro_vlm_model_name=args.micro_vlm_model_name,
        micro_vlm_max_crops=args.micro_vlm_max_crops,
        micro_vlm_max_model_bytes=int(args.micro_vlm_max_model_gb * 1_000_000_000),
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
