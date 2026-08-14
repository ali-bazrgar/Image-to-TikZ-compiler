from image_to_tikz.serialize import to_llm_context
from image_to_tikz.vir import VisualScene


def test_llm_context_is_compact_and_has_no_duplicate_wrapper():
    scene = VisualScene(image={"width": 1000, "height": 800})
    context = to_llm_context(scene)
    assert context.startswith("IMAGE_TO_TIKZ_VISUAL_RECORD v2")
    assert "DOWNSTREAM_TASK" in context
    assert "<VISUAL_RECORD>" not in context
    assert len(context.splitlines()) < 120
