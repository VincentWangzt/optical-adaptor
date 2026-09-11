import base64
import io
from pathlib import Path

import pytest
from PIL import Image

from optical_adaptor.benchmark.compare import lcb_answer
from optical_adaptor.benchmark.prepare import join_lines, source_lines
from optical_adaptor.benchmark.run import case_messages, score, task_decoding
from optical_adaptor.inference.backend import (
    GREEDY_DECODING,
    expand_image_tokens,
    normalize_messages,
)
from optical_adaptor.inference.messages import chat_ids
from optical_adaptor.training.config import file_sha256, load_pipeline


def test_image_order_and_message_boundaries():
    def part(color):
        stream = io.BytesIO()
        Image.new("RGB", (4, 4), color).save(stream, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()
        return {"type": "image_url", "image_url": {"url": url}}

    messages, images = normalize_messages(
        [
            {"role": "user", "content": [part("red"), part("blue")]},
            {"role": "assistant", "content": "previous response"},
            {"role": "user", "content": [part("green")]},
        ]
    )
    assert [im.getpixel((0, 0)) for im in images] == [(255, 0, 0), (0, 0, 255), (0, 128, 0)]
    assert [r["role"] for r in messages] == ["user", "assistant", "user"]
    assert messages[0]["content"] == [{"type": "image"}, {"type": "image"}]
    assert expand_image_tokens([1, 9, 2, 1, 9, 2], 9, [2, 3]) == [1, 9, 9, 2, 1, 9, 9, 9, 2]
    with pytest.raises(ValueError, match="placeholders"):
        expand_image_tokens([1, 9, 2], 9, [111, 111])


def test_original_source_lines_preserve_blank_line_at_boundary():
    for final in ["last source line", ""]:
        lines = ["x" * 201] * 79 + [final]
        assert source_lines(join_lines(lines)) == lines


def test_request_controls_never_include_repository_or_answer(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (10, 10)).save(path)
    case = {
        "task": "qa",
        "reference": "D",
        "before_images": "Goal: ",
        "after_images": "Question and options",
        "text_prompt": "Goal: REPO Question and options",
        "no_image_prompt": "Goal: Question and options",
        "images": [{"path": path.name, "sha256": file_sha256(path)}],
    }
    assert case_messages(case, "no-image", tmp_path)[0]["content"] == [
        {"type": "text", "text": "Goal: Question and options"}
    ]
    image_content = case_messages(case, "images", tmp_path)[0]["content"]
    assert [r["type"] for r in image_content] == ["text", "image_url", "text"]
    assert "REPO" in case_messages(case, "text", tmp_path)[0]["content"][0]["text"]
    assert score(case, "D")["correct"]
    assert not score(case, "A or D")["correct"]
    assert score(case, "Answer unavailable")["invalid_answer"]


def test_upstream_lcb_answer_policy_is_reported_separately():
    assert lcb_answer("Final Answer: c") == "C"
    assert lcb_answer("The answer is B") == "B"
    assert lcb_answer("A) an explanation") is None
    assert lcb_answer("import hashlib") is None


def test_reconstruction_decoding_matches_training_while_qa_stays_greedy():
    pipeline = load_pipeline(Path(__file__).resolve().parents[1] / "configs/training.yaml")
    reconstruction = task_decoding("reconstruction", pipeline)
    assert reconstruction.to_vllm_kwargs() == {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
    }
    assert task_decoding("qa", pipeline) == GREEDY_DECODING


def test_qwen_template_matches_adjacent_vision_blocks():
    from transformers import AutoTokenizer

    pipeline = load_pipeline(Path(__file__).resolve().parents[1] / "configs/training.yaml")
    model = pipeline.config.models
    tokenizer = AutoTokenizer.from_pretrained(
        model.qwen_id, revision=model.qwen_revision, local_files_only=True
    )
    for count in (2, 4, 8):
        ids = chat_ids(
            tokenizer,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe."},
                        *[{"type": "image"}] * count,
                    ],
                }
            ],
        )
        image = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        first = ids.index(start)
        assert ids[first : first + 3 * count] == [start, image, end] * count
        expanded = expand_image_tokens(ids, image, [111] * count)
        assert expanded.count(image) == 111 * count
        assert expanded.count(start) == expanded.count(end) == count
