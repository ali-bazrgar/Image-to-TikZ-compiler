from image_to_tikz.llama_server_vlm import build_llama_server_command


def test_qprocess_command_does_not_duplicate_executable() -> None:
    args = build_llama_server_command(
        r"D:/llama/llama-server.exe",
        r"D:/models/model.gguf",
        r"D:/models/mmproj.gguf",
        no_mmproj_offload=True,
    )

    assert args[0:2] == ["-m", r"D:/models/model.gguf"]
    assert args.count(r"D:/llama/llama-server.exe") == 0
    assert "--mmproj" in args
    assert "--no-mmproj-offload" in args
