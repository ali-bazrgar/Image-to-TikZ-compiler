import cv2
import numpy as np

from image_to_tikz.golden_prompt import build_golden_prompt
from image_to_tikz.gui import run as gui_run
from image_to_tikz.llm_client import chat_completion
from image_to_tikz.llama_server_vlm import build_llama_server_command, validate_gguf_pair
from image_to_tikz.pipeline import analyze_image
from image_to_tikz.tikz_verifier import extract_tikz_code


def test_golden_prompt_contains_strict_reconstruction_contract(tmp_path):
    image = np.full((180, 260, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (30, 40), (120, 130), (0, 0, 0), 3)
    path = tmp_path / "prompt.png"
    assert cv2.imwrite(str(path), image)
    scene, _ = analyze_image(path, multiscale=False, ocr="off")
    prompt = build_golden_prompt(scene)
    assert "expert scientific-figure reconstruction engine" in prompt
    assert "Output ONLY a single LaTeX/TikZ code block." in prompt
    assert "VISUAL RECORD" in prompt
    assert "COMPACT VISUAL RECORD" in prompt


def test_tikz_extractor_prefers_longest_code_block():
    text = "before\n```txt\nignore\n```\n```latex\n\\begin{tikzpicture}\nX\n\\end{tikzpicture}\n```"
    assert "tikzpicture" in extract_tikz_code(text)


def test_gguf_pair_validation_and_server_command(tmp_path):
    model = tmp_path / "model.gguf"; mmproj = tmp_path / "mmproj.gguf"
    model.write_bytes(b"m" * 100); mmproj.write_bytes(b"p" * 50)
    info = validate_gguf_pair(model, mmproj)
    assert info.total_bytes == 150
    cmd = build_llama_server_command("llama-server.exe", model, mmproj, context=4096, gpu_layers=99)
    assert "--mmproj" in cmd and "-ngl" in cmd


def test_gui_entrypoint_is_importable_without_loading_qt():
    assert callable(gui_run)
