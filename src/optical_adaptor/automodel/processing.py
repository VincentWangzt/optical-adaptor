"""Compile one conversation into two sequences with identical supervised token identities."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass

import torch

from optical_adaptor.automodel.config import ProcessingConfig
from optical_adaptor.automodel.conversations import validate_record
from optical_adaptor.renderer import RenderConfig, render_pages


class RejectedSample(ValueError):
    """A counted, explicit data rejection; never silently truncate a trajectory."""


@dataclass
class PairedTokens:
    teacher_ids: list[int]
    student_ids: list[int]
    teacher_positions: list[int]
    student_positions: list[int]
    targets: list[int]
    image_positions: list[list[int]]
    visual_texts: list[str]
    generation_prefix_length: int


def training_template(template: str) -> str:
    """Annotate Qwen's native tool/chat format while preserving historical reasoning.

    Qwen's inference template drops thoughts before the last user query and trims
    content. Neither behavior is suitable for exact transcription SFT. Fail on an
    unknown template rather than silently discarding masks or reasoning.
    """
    content_assignment = "{%- set content = render_content(message.content, true) %}"
    reasoning_assignment = "{%- set reasoning_content = reasoning_content %}"
    replacements = {
        "{%- if loop.index0 > ns.last_query_index %}": "{%- if true %}",
        "{%- set content = render_content(message.content, true)|trim %}": content_assignment,
        "{%- set reasoning_content = reasoning_content|trim %}": reasoning_assignment,
        '{%- elif message.role == "assistant" %}': '{%- elif message.role == "assistant" %}'
        '{{- oa_prefix ~ "as:" ~ loop.index0 ~ ":END" }}',
        '{%- elif message.role == "tool" %}': '{{- oa_prefix ~ "ae:" ~ loop.index0 ~ ":END" }}'
        '{%- elif message.role == "tool" %}',
    }
    for old, new in replacements.items():
        if template.count(old) != 1:
            raise ValueError(f"Unsupported chat template: expected exactly one {old!r}")
        template = template.replace(old, new)
    return template


class ConversationCompiler:
    def __init__(self, tokenizer, config: ProcessingConfig, tokens_per_image: int):
        self.tokenizer = tokenizer
        self.config = config
        self.tokens_per_image = tokens_per_image
        self.template = training_template(tokenizer.chat_template)
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            raise ValueError("Tokenizer must define a padding token")

    def compile(self, record: dict) -> PairedTokens:
        validate_record(record)
        messages = copy.deepcopy(record["messages"])
        last_assistant = max(i for i, m in enumerate(messages) if m["role"] == "assistant")
        messages = messages[: last_assistant + 1]
        prefix = "OPTICAL_INTERNAL_" + record["sample_id"] + ":"
        for message in messages:
            for value in (message["content"], message.get("reasoning_content", "")):
                if prefix in value or any(s in value for s in self.tokenizer.all_special_tokens):
                    raise RejectedSample("reserved_token_collision")
        visual_texts = [
            record["messages"][area["message"]]["content"][area["start"] : area["end"]]
            for area in record["visual_areas"]
        ]
        for index in reversed(range(len(record["visual_areas"]))):
            area = record["visual_areas"][index]
            if area["message"] >= len(messages):
                raise ValueError("Visual areas must precede the last assistant target")
            content = messages[area["message"]]["content"]
            messages[area["message"]]["content"] = (
                content[: area["start"]] + prefix + f"v:{index}:END" + content[area["end"] :]
            )
        annotated = self.tokenizer.apply_chat_template(
            messages,
            tools=record["tools"] or None,
            tokenize=False,
            chat_template=self.template,
            add_generation_prompt=False,
            enable_thinking=record["thinking"],
            oa_prefix=prefix,
        )
        event = re.compile(re.escape(prefix) + r"(as|ae|v):(\d+):END")
        pieces, assistant_ranges, visual_ranges = [], {}, []
        cursor = length = 0
        for match in event.finditer(annotated):
            text = annotated[cursor : match.start()]
            pieces.append(text)
            length += len(text)
            kind, index = match[1], int(match[2])
            if kind == "v":
                start_marker, end_marker = self.config.vision_start, self.config.vision_end
                visual = visual_texts[index]
                visual_ranges.append(
                    (length + len(start_marker), length + len(start_marker) + len(visual), index)
                )
                text = start_marker + visual + end_marker
                pieces.append(text)
                length += len(text)
            elif kind == "as":
                assistant_ranges[index] = [length, None]
            else:
                assistant_ranges[index][1] = length
            cursor = match.end()
        pieces.append(annotated[cursor:])
        text = "".join(pieces)
        if prefix in text or len(visual_ranges) != len(visual_texts):
            raise ValueError("Lost a conversation annotation during chat formatting")
        selected = set(assistant_ranges)
        if self.config.assistant_loss == "last" or record["task"] != "next_action":
            selected = {last_assistant}
        payload_ranges = []
        generation_start = None
        for index, (start, end) in assistant_ranges.items():
            header = "<|im_start|>assistant\n<think>\n"
            if not record["thinking"]:
                if messages[index].get("reasoning_content"):
                    raise ValueError("Non-thinking source unexpectedly contains reasoning")
                header += "\n</think>\n\n"
            if not text[start:end].startswith(header) or not text[start:end].endswith(
                "<|im_end|>\n"
            ):
                raise ValueError("Unexpected assistant serialization; cannot establish target mask")
            if index == last_assistant:
                generation_start = start + len(header)
            if index in selected:
                payload_ranges.append((start + len(header), end - 1))
        # Tokenize at supervised boundaries so a whitespace BPE token cannot
        # straddle the assistant prefix and the first target characters. The two
        # branches share these exact text token IDs; no alignment by token counts.
        cuts = sorted(
            {
                0,
                len(text),
                generation_start,
                *(point for bounds in payload_ranges for point in bounds),
                *(point for low, high, _ in visual_ranges for point in (low, high)),
            }
        )
        ids, offsets = [], []
        for low, high in zip(cuts[:-1], cuts[1:], strict=True):
            encoded = self.tokenizer(
                text[low:high], add_special_tokens=False, return_offsets_mapping=True
            )
            ids.extend(encoded["input_ids"])
            if len(ids) > self.config.max_teacher_tokens:
                raise RejectedSample("teacher_length")
            offsets.extend((start + low, end + low) for start, end in encoded["offset_mapping"])
        target_indices = [
            i
            for i, (start, end) in enumerate(offsets)
            if end > start and any(start >= low and end <= high for low, high in payload_ranges)
        ]
        if not target_indices or target_indices[0] == 0:
            raise RejectedSample("empty_supervision")
        image_ranges = []
        for low, high, index in visual_ranges:
            indices = [
                i
                for i, (start, end) in enumerate(offsets)
                if end > low and start < high and end > start
            ]
            if not indices or offsets[indices[0]][0] < low or offsets[indices[-1]][1] > high:
                raise ValueError("Visual markers must isolate token boundaries")
            image_ranges.append((indices[0], indices[-1] + 1, index))
        image_ranges.sort()
        student_ids, old_to_new, image_positions = [], {}, []
        cursor = 0
        for start, end, index in image_ranges:
            if index != len(image_positions):
                raise ValueError("Chat formatting changed visual area order")
            for old in range(cursor, start):
                old_to_new[old] = len(student_ids)
                student_ids.append(ids[old])
            image_positions.append(
                list(range(len(student_ids), len(student_ids) + self.tokens_per_image))
            )
            student_ids.extend([self.pad_id] * self.tokens_per_image)
            cursor = end
        for old in range(cursor, len(ids)):
            old_to_new[old] = len(student_ids)
            student_ids.append(ids[old])
        if len(student_ids) > self.config.max_student_tokens:
            raise RejectedSample("student_length")
        # A prediction at p-1 supervises the token at p, independently in each
        # sequence. Labels are compact paired targets, never image/padding slots.
        student_indices = [old_to_new[i] for i in target_indices]
        targets = [ids[i] for i in target_indices]
        if targets != [student_ids[i] for i in student_indices]:
            raise AssertionError("Teacher/student target-token identities do not match")
        generation_token = next(
            i for i, (start, end) in enumerate(offsets) if start >= generation_start and end > start
        )
        return PairedTokens(
            ids,
            student_ids,
            [i - 1 for i in target_indices],
            [i - 1 for i in student_indices],
            targets,
            image_positions,
            visual_texts,
            old_to_new[generation_token],
        )


def collate_records(records: list[dict], compiler: ConversationCompiler) -> dict:
    compiled = [compiler.compile(record) for record in records]
    labels = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(pair.targets, dtype=torch.long) for pair in compiled],
        batch_first=True,
        padding_value=-100,
    )
    return {"records": records, "compiled": compiled, "labels": labels}


def render_batch(compiled: list[PairedTokens], render_config: RenderConfig):
    images = []
    for pair in compiled:
        for text in pair.visual_texts:
            pages, _, truncated = render_pages(text, config=render_config)
            if truncated or len(pages) != 1:
                raise ValueError("Each visual area must render to exactly one complete image")
            images.append(pages[0])
    return images
