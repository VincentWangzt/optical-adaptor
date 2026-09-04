from pathlib import Path

from PIL import Image

from optical_agent.evaluate_ocr import _source_portion
from optical_agent.infer_ocr import ModelConfig, count_deepseek_image_tokens


def test_source_portion_uses_inclusive_line_numbers() -> None:
    portion, end = _source_portion("one\ntwo\nthree\n", 2, 3)
    assert portion == "two\nthree\n"
    assert end == 3


def test_deepseek_visual_token_counts(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (2550, 3300)).save(image_path)
    deepseek_1 = ModelConfig(
        model_id="deepseek-ai/DeepSeek-OCR",
        revision=None,
        prompt=None,
        backend="deepseek",
        base_size=1024,
        image_size=640,
        max_crops=6,
    )
    deepseek_2 = ModelConfig(
        model_id="deepseek-ai/DeepSeek-OCR-2",
        revision=None,
        prompt=None,
        backend="deepseek",
        base_size=1024,
        image_size=768,
        max_crops=6,
    )
    deepseek_1_small_no_crop = ModelConfig(
        model_id="deepseek-ai/DeepSeek-OCR",
        revision=None,
        prompt=None,
        backend="deepseek",
        base_size=640,
        image_size=640,
        crop_mode=False,
        max_crops=6,
    )

    assert count_deepseek_image_tokens(image_path, deepseek_1) == 903
    assert count_deepseek_image_tokens(image_path, deepseek_2) == 1121
    assert count_deepseek_image_tokens(image_path, deepseek_1_small_no_crop) == 111
