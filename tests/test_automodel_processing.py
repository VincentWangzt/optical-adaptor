"""Focused CPU checks; run on the server with uv run pytest."""

import copy
import random

import pytest
from transformers import AutoTokenizer

from optical_adaptor.automodel.config import read_config
from optical_adaptor.automodel.conversations import (
    make_record,
    naive_messages,
    normalize_messages,
    parse_visual_areas,
    stack_records,
    swe_records,
    visual_spans,
)
from optical_adaptor.automodel.processing import ConversationCompiler, RejectedSample


@pytest.fixture(scope="module")
def settings():
    return read_config("configs/automodel.yaml")[1]


@pytest.fixture(scope="module")
def tokenizer(settings):
    return AutoTokenizer.from_pretrained(
        settings.llm["model_id"], revision=settings.llm["revision"]
    )


def naive(settings, seed=42):
    text = "  def f():\n      return '你好'\n" + "x = 1\n" * 104
    text = text.rstrip("\n")
    messages, tools, areas = naive_messages(
        text, text, "reconstruction", settings.prepare, random.Random(seed), 100
    )
    return make_record(
        messages=messages,
        tools=tools,
        areas=areas,
        source=settings.prepare.sources[0],
        group="example/repository",
        origin_id="document",
        task="reconstruction",
        view="front",
        thinking=False,
        turn_count=1,
        seed=42,
        config=settings.prepare,
    )


@pytest.mark.parametrize("seed", range(4))
def test_reconstruction_mask_preserves_whitespace_and_multibyte(settings, tokenizer, seed):
    record = naive(settings, seed)
    pair = ConversationCompiler(tokenizer, settings.processing, 111).compile(record)
    target = record["messages"][-1]["content"]
    assert tokenizer.decode(pair.targets) == target + "<|im_end|>"
    assert len(pair.image_positions) == 3
    assert [pair.teacher_ids[i + 1] for i in pair.teacher_positions] == pair.targets
    assert [pair.student_ids[i + 1] for i in pair.student_positions] == pair.targets
    assert len(pair.teacher_ids) != len(pair.student_ids)
    prefix = tokenizer.decode(pair.student_ids[: pair.generation_prefix_length])
    assert prefix.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert not set(pair.student_positions) & {i for slot in pair.image_positions for i in slot[:-1]}


def test_all_assistants_retain_reasoning_before_later_user(settings, tokenizer):
    record = naive(settings)
    record["task"], record["thinking"] = "next_action", True
    record["messages"][-1] = {
        "role": "assistant",
        "content": "action A",
        "reasoning_content": "earlier reasoning",
    }
    record["messages"].extend(
        [
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": "action B", "reasoning_content": "later reasoning"},
        ]
    )
    all_pair = ConversationCompiler(tokenizer, settings.processing, 111).compile(record)
    last_pair = ConversationCompiler(
        tokenizer, settings.processing.model_copy(update={"assistant_loss": "last"}), 111
    ).compile(record)
    all_targets, last_targets = (
        tokenizer.decode(all_pair.targets),
        tokenizer.decode(last_pair.targets),
    )
    assert "earlier reasoning" in all_targets and "later reasoning" in all_targets
    assert "earlier reasoning" not in last_targets and "later reasoning" in last_targets
    assert "earlier reasoning" in tokenizer.decode(last_pair.teacher_ids)


def test_overlength_rejects_whole_record(settings, tokenizer):
    record = naive(settings)
    before = copy.deepcopy(record)
    config = settings.processing.model_copy(update={"max_teacher_tokens": 8})
    with pytest.raises(RejectedSample, match="teacher_length"):
        ConversationCompiler(tokenizer, config, 111).compile(record)
    assert record == before


def test_visual_offsets_cover_every_non_newline_character():
    text = "a" * 233 + "\n\n" + "  second line\n" * 10
    spans = visual_spans(text, 3, 20)
    covered = [i for start, end in spans for i in range(start, end)]
    assert len(covered) == len(set(covered))
    assert all(i in covered for i, char in enumerate(text) if char != "\n")


def test_full_swe_record_preserves_complete_history_and_split(settings):
    messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]
    for step in range(5):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": f"action {step}",
                    "reasoning_content": f"thought {step}",
                },
                {"role": "tool", "content": (f"observation {step}\n" * 70).rstrip("\n")},
            ]
        )
    messages.append({"role": "assistant", "content": "done", "reasoning_content": "last thought"})
    row = {"repo": "example/repository", "trajectory_id": "trajectory", "tools": []}
    limited = list(
        swe_records(row, messages, settings.prepare.sources[1], settings.prepare, 42, 100)
    )
    assert limited
    assert all(record["task"] in {"reconstruction", "continuation"} for record in limited)
    assert all(record["image_count"] <= 2 for record in limited)
    prepare = settings.prepare.model_copy(
        update={
            "tasks": ["reconstruction", "continuation", "next_action"],
            "max_images": None,
        }
    )
    records = list(swe_records(row, messages, settings.prepare.sources[1], prepare, 42, 100))
    full = next(record for record in records if record["origin"]["view"] == "full")
    assert parse_visual_areas(full["messages"])[0] == messages
    assert full["turn_count"] == 5 and full["image_count"] == 10
    assert full["source"].endswith("_full")
    assert any(record["turn_count"] == 1 and record["task"] == "next_action" for record in records)


def test_normalize_tool_arguments():
    original = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "function": {"name": "bash", "arguments": '{"command":"ls"}'},
                }
            ],
        }
    ]
    normalized = normalize_messages(original)
    assert normalized[0]["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}
    assert original[0]["tool_calls"][0]["function"]["arguments"] == '{"command":"ls"}'


def test_short_stack_documents_still_produce_reconstruction(settings):
    text = "def short_file():\n    return 42"
    row = {"repository_name": "example/repository", "path": "short.py", "content": text}
    records = list(stack_records(row, text, settings.prepare.sources[0], settings.prepare, 42, 100))
    assert len(records) == 1
    assert records[0]["task"] == "reconstruction"
    assert records[0]["messages"][-1]["content"] == text
