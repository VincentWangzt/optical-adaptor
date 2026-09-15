"""SFT conversations and offset-addressed visual areas, independent of tokenization."""

from __future__ import annotations

import copy
import json
import random
from collections.abc import Iterator

from optical_adaptor.automodel.config import PrepareConfig, Source, bin_name, fingerprint


def normalize_messages(messages: list[dict]) -> list[dict]:
    result = []
    for original in messages:
        message = {
            key: copy.deepcopy(value)
            for key, value in original.items()
            if value is not None
            and key
            in {"role", "content", "reasoning_content", "tool_calls", "tool_call_id", "name"}
        }
        if message["role"] not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported message role: {message['role']}")
        content = message.get("content", "")
        if isinstance(content, list):
            if any(part.get("type") != "text" for part in content):
                raise ValueError("Source trajectories must contain text-only content")
            content = "".join(part["text"] for part in content)
        if not isinstance(content, str):
            raise TypeError("Message content must be text")
        message["content"] = content
        # The HF template requires dictionaries, whereas Open-SWE stores JSON strings.
        for call in message.get("tool_calls", []):
            function = call["function"]
            if isinstance(function["arguments"], str):
                function["arguments"] = json.loads(function["arguments"])
            if not isinstance(function["arguments"], dict):
                raise TypeError("Tool arguments must be a JSON object")
        result.append(message)
    return result


def visual_spans(text: str, max_rows: int, line_width: int) -> list[tuple[int, int]]:
    """Paginate at display-row boundaries without discarding any non-newline character.

    Newlines between pages remain ordinary text between the corresponding images.
    Very long logical lines may span pages; no ellipsis or truncation is inserted.
    """
    rows = []
    offset = 0
    for line in text.splitlines(keepends=True):
        body = line.removesuffix("\n")
        for start in range(0, max(len(body), 1), line_width):
            rows.append((offset + start, offset + min(start + line_width, len(body))))
        offset += len(line)
    spans = []
    for start in range(0, len(rows), max_rows):
        chunk = rows[start : start + max_rows]
        first, last = chunk[0][0], chunk[-1][1]
        if first < last:
            spans.append((first, last))
    return spans


def validate_record(record: dict) -> None:
    messages = record["messages"]
    if not any(m["role"] == "assistant" for m in messages):
        raise ValueError("Every SFT sample requires a genuine assistant target")
    previous = (-1, -1)
    for area in record["visual_areas"]:
        index, start, end = area["message"], area["start"], area["end"]
        if not 0 <= index < len(messages):
            raise ValueError("Visual message index out of range")
        if messages[index]["role"] not in {"user", "tool"}:
            raise ValueError("Visual areas are only allowed in conditioning user/tool messages")
        if not 0 <= start < end <= len(messages[index]["content"]):
            raise ValueError("Invalid visual text offsets")
        if index < previous[0] or (index == previous[0] and start < previous[1]):
            raise ValueError("Visual areas must be ordered and non-overlapping")
        previous = (index, end)
    if not record["visual_areas"] or len(record["visual_areas"]) != record["image_count"]:
        raise ValueError("image_count must equal the number of visual areas")


def make_record(
    *,
    messages: list[dict],
    tools: list[dict],
    areas: list[dict],
    source: Source,
    group: str,
    origin_id: str,
    task: str,
    view: str,
    thinking: bool,
    turn_count: int,
    seed: int,
    config: PrepareConfig,
) -> dict:
    split_value = int(fingerprint([seed, "repository", group.casefold()])[:16], 16) / 2**64
    record = {
        "schema_version": 1,
        "sample_id": fingerprint([source.name, origin_id, task, view, messages, areas])[:24],
        "group_id": group.casefold(),
        "source": source.name,
        "task": task,
        "view": view,
        "thinking": thinking,
        "messages": messages,
        "tools": tools,
        "visual_areas": areas,
        "image_count": len(areas),
        "turn_count": turn_count,
        "image_bin": bin_name(len(areas), config.image_bins),
        "turn_bin": bin_name(turn_count, config.turn_bins),
        "split": "eval" if split_value < config.eval_fraction else "train",
        "origin": {
            "dataset_id": source.dataset_id,
            "revision": source.revision,
            "record_id": origin_id,
        },
    }
    validate_record(record)
    return record


def naive_messages(
    text: str,
    target: str,
    task: str,
    config: PrepareConfig,
    rng: random.Random,
    line_width: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    prompts = (
        config.reconstruction_prompts if task == "reconstruction" else (config.continuation_prompts)
    )
    prompt = rng.choice(prompts).format(lines=config.continuation_lines)
    instruction_role = rng.choice(config.instruction_roles)
    image_role = rng.choice(config.image_roles)
    messages = []
    if instruction_role == "system":
        messages.append({"role": "system", "content": prompt})
    tools = []
    if image_role == "tool":
        messages.append(
            {
                "role": "user",
                "content": prompt
                if instruction_role == "user"
                else "Please read the supplied document.",
            }
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_document",
                    "description": "Return the supplied document pages.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "document",
                        "type": "function",
                        "function": {
                            "name": "read_document",
                            "arguments": {},
                        },
                    },
                ],
            }
        )
        visual_index, prefix = len(messages), ""
        messages.append({"role": "tool", "tool_call_id": "document", "content": text})
    else:
        prefix = prompt + "\n\n" if instruction_role == "user" else ""
        visual_index = len(messages)
        messages.append({"role": "user", "content": prefix + text})
    areas = [
        {"message": visual_index, "start": len(prefix) + start, "end": len(prefix) + end}
        for start, end in visual_spans(text, config.lines_per_image, line_width)
    ]
    messages.append({"role": "assistant", "content": target})
    return messages, tools, areas


def stack_records(
    row: dict,
    text: str,
    source: Source,
    config: PrepareConfig,
    seed: int,
    line_width: int,
) -> Iterator[dict]:
    group = row["max_stars_repo_name"]
    if not group:
        raise ValueError("The Stack record has no repository identity")
    origin_id = row.get("hexsha") or fingerprint(row["content"])
    pages = visual_spans(text, config.lines_per_image, line_width)
    rng = random.Random(fingerprint([seed, source.name, origin_id]))
    for images in config.stack_image_counts:
        if len(pages) < images:
            continue
        # Front and middle variants have the same supervised SFT structure.
        for position in ("front", "middle"):
            if position == "middle" and len(pages) == images:
                continue
            page_start = 0 if position == "front" else rng.randrange(1, len(pages) - images + 1)
            start, end = pages[page_start][0], pages[page_start + images - 1][1]
            visual = text[start:end]
            following = text[end:].removeprefix("\n").splitlines()[: config.continuation_lines]
            if len(following) < config.continuation_lines:
                continue
            for task in ("reconstruction", "continuation"):
                target = visual if task == "reconstruction" else "\n".join(following)
                messages, tools, areas = naive_messages(
                    visual, target, task, config, rng, line_width
                )
                if len(areas) != images:
                    raise AssertionError("Stack page selection changed the requested image count")
                yield make_record(
                    messages=messages,
                    tools=tools,
                    areas=areas,
                    source=source,
                    group=group,
                    origin_id=origin_id,
                    task=task,
                    view=f"{position}_{images}",
                    thinking=False,
                    turn_count=1,
                    seed=seed,
                    config=config,
                )


def swe_records(
    row: dict,
    messages: list[dict],
    source: Source,
    config: PrepareConfig,
    seed: int,
    line_width: int,
) -> Iterator[dict]:
    group, origin_id = row["repo"], row["trajectory_id"]
    if not group or not origin_id:
        raise ValueError("SWE record needs repository and trajectory identities")
    tools = [json.loads(tool) if isinstance(tool, str) else tool for tool in row["tools"]]
    rng = random.Random(fingerprint([seed, source.name, origin_id]))
    areas = [
        {"message": index, "start": start, "end": end}
        for index, message in enumerate(messages)
        if message["role"] == "tool" and len(message["content"]) >= config.min_observation_chars
        for start, end in visual_spans(message["content"], config.lines_per_image, line_width)
    ]
    if not areas:
        return
    # A turn is an assistant action plus its following observations. Several tool
    # responses from one action count as one turn, even if they produce many pages.
    owner, owner_by_message = -1, {}
    for index, message in enumerate(messages):
        if message["role"] == "assistant":
            owner = index
        owner_by_message[index] = owner
    last_assistant = max(i for i, m in enumerate(messages) if m["role"] == "assistant")
    usable_areas = [a for a in areas if a["message"] < last_assistant]
    if not usable_areas:
        return
    turns = {owner_by_message[a["message"]] for a in usable_areas}
    # Keep every original message in the full record, including any trailing tool
    # responses. Processing stops after the last supervised action, not mid-turn.
    yield make_record(
        messages=messages,
        tools=tools,
        areas=usable_areas,
        source=source,
        group=group,
        origin_id=origin_id,
        task="next_action",
        view="full",
        thinking=source.thinking,
        turn_count=len(turns),
        seed=seed,
        config=config,
    )
    targets = [
        i
        for i, m in enumerate(messages)
        if m["role"] == "assistant" and any(a["message"] < i for a in usable_areas)
    ]
    if config.max_actions_per_trajectory is not None:
        targets = sorted(rng.sample(targets, min(len(targets), config.max_actions_per_trajectory)))
    first_assistant = next(i for i, m in enumerate(messages) if m["role"] == "assistant")
    for target in targets:
        before = [a for a in usable_areas if a["message"] < target]
        owners = sorted({owner_by_message[a["message"]] for a in before})
        if owners[0] < 0:
            raise ValueError("Tool observations must follow an assistant action")
        for window in config.window_turns:
            if len(owners) < window:
                continue
            start = owners[-window]
            selected_indices = list(range(first_assistant)) + list(range(start, target + 1))
            remap = {old: new for new, old in enumerate(selected_indices)}
            selected_areas = [
                {**a, "message": remap[a["message"]]} for a in before if a["message"] >= start
            ]
            selected_messages = [messages[i] for i in selected_indices]
            yield make_record(
                messages=selected_messages,
                tools=tools,
                areas=selected_areas,
                source=source,
                group=group,
                origin_id=origin_id,
                task="next_action",
                view=f"window_{window}_action_{target}",
                thinking=source.thinking,
                turn_count=window,
                seed=seed,
                config=config,
            )
    observations = sorted({a["message"] for a in usable_areas})
    observations = rng.sample(
        observations, min(len(observations), config.reconstruction_observations_per_trajectory)
    )
    for index in observations:
        text = messages[index]["content"]
        for task in ("reconstruction", "continuation"):
            if task == "continuation":
                lines = text.splitlines()
                if len(lines) <= config.continuation_lines + 4:
                    continue
                visual = "\n".join(lines[: -config.continuation_lines])
                target = "\n".join(lines[-config.continuation_lines :])
            else:
                visual = target = text
            naive, naive_tools, naive_areas = naive_messages(
                visual, target, task, config, rng, line_width
            )
            yield make_record(
                messages=naive,
                tools=naive_tools,
                areas=naive_areas,
                source=source,
                group=group,
                origin_id=origin_id,
                task=task,
                view=f"observation_{index}",
                thinking=False,
                turn_count=1,
                seed=seed,
                config=config,
            )


def slice_key(record: dict) -> str:
    view = (
        "full"
        if record["view"] == "full"
        else ("window" if record["view"].startswith("window_") else record["view"].split("_")[0])
    )
    return "/".join(
        (
            record["source"],
            record["task"],
            view,
            "images-" + record["image_bin"],
            "turns-" + record["turn_bin"],
        )
    )
