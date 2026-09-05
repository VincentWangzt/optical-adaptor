from __future__ import annotations

import argparse
import hashlib
import html
import json
import multiprocessing
import os
import re
import unicodedata
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from huggingface_hub import HfApi, hf_hub_download

from optical_adaptor.renderer import RenderConfigError, font_codepoints, render_pages
from optical_adaptor.token_utils import load_tokenizer
from optical_adaptor.training.config import (
    SPLITS,
    Pipeline,
    file_sha256,
    fingerprint,
    load_credentials,
    load_pipeline,
    write_json,
)


@dataclass(frozen=True)
class CanonicalText:
    text: str
    source_boundaries: tuple[int, ...]


def canonicalize(source: str, coverage: frozenset[int], tab_width: int) -> CanonicalText:
    pieces, boundaries = [], [0]
    for match in re.finditer(r"([^\r\n]*)(\r\n|\r|\n|$)", source):
        line, newline = match.groups()
        if not line and not newline:
            continue
        column = 0
        for offset, char in enumerate(line.rstrip(" \t")):
            if char == "\t":
                rendered = " " * (tab_width - column % tab_width)
            elif ord(char) not in coverage or unicodedata.category(char).startswith("C"):
                rendered = f"\\u{ord(char):04x}" if ord(char) <= 0xFFFF else f"\\U{ord(char):08x}"
            else:
                rendered = char
            pieces.append(rendered)
            boundaries.extend([match.start() + offset + 1] * len(rendered))
            column += len(rendered)
        if newline:
            pieces.append("\n")
            boundaries.append(match.end())
    return CanonicalText("".join(pieces), tuple(boundaries))


def inspection_markup(text: str, start: int, end: int) -> str:
    if not 0 <= start <= end <= len(text):
        raise ValueError("invalid visual offsets")
    return (
        html.escape(text[:start])
        + "<vision>"
        + html.escape(text[start:end])
        + ("</vision>" + html.escape(text[end:]))
    )


def repository_split(repository: str, seed: int, sizes: dict[str, int]) -> str:
    value = int(fingerprint([seed, "repository", repository.casefold()]), 16) % sum(sizes.values())
    boundary = 0
    for split in SPLITS:
        boundary += sizes[split]
        if value < boundary:
            return split
    raise AssertionError("unreachable split boundary")


def encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def bounded_tokens(tokenizer: Any, text: str, count: int, *, suffix: bool = False):
    """Read enough source for a token boundary without tokenizing a giant remaining file."""
    characters = min(len(text), max(2048, count * 16))
    while True:
        offset = len(text) - characters if suffix else 0
        value = tokenizer(
            text[offset : offset + characters],
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        if len(value["input_ids"]) > count + 32 or characters == len(text):
            return value, offset
        characters = min(len(text), characters * 2)


def make_record(
    row: dict, split: str, pipeline: Pipeline, tokenizer: Any, coverage: frozenset[int]
) -> tuple[dict | None, str]:
    config = pipeline.config
    raw = row["content"]
    try:
        raw.encode("utf-8", errors="strict")
    except UnicodeError:
        return None, "decoding"
    canonical = canonicalize(raw, coverage, pipeline.render.text.tab_width)
    text = canonical.text
    starts = [0] + [match.end() for match in re.finditer("\n", text)]
    if text.endswith("\n"):
        starts.pop()
    first = len(starts) // 2 if split == "middle_continuation" else 0
    identity = fingerprint([row["repository_name"].casefold(), row["path"]])
    counts = sorted(
        range(config.data.min_lines, config.data.max_lines + 1),
        key=lambda n: fingerprint([config.seed, identity, n]),
    )
    reason = "length"
    for count in counts:
        if first + count >= len(starts):
            continue
        start, end = starts[first], starts[first + count] - 1
        visual = text[start:end]
        if visual.endswith("\n"):
            reason = "invisible_terminal_line"
            continue
        visual_ids = encode(tokenizer, visual)
        if not visual_ids or len(visual_ids) > config.data.max_visual_tokens:
            reason = "visual_tokens"
            continue
        following, _ = bounded_tokens(tokenizer, text[end:], config.data.continuation_tokens)
        continuation_ids = following["input_ids"][: config.data.continuation_tokens]
        if len(continuation_ids) != config.data.continuation_tokens:
            reason = "continuation_tokens"
            continue
        continuation_end = end + following["offset_mapping"][config.data.continuation_tokens - 1][1]
        continuation = text[end:continuation_end]
        # Reject partial UTF-8/codepoint token boundaries instead of silently changing targets.
        if encode(tokenizer, continuation) != continuation_ids:
            reason = "token_boundary"
            continue
        prefix, prefix_ids, prefix_start = "", [], start
        if first:
            preceding, preceding_offset = bounded_tokens(
                tokenizer, text[:start], config.data.prefix_tokens, suffix=True
            )
            prefix_start = preceding["offset_mapping"][
                max(0, len(preceding["input_ids"]) - config.data.prefix_tokens)
            ][0]
            prefix_start += preceding_offset
            prefix = text[prefix_start:start]
            prefix_ids = encode(tokenizer, prefix)
            if len(prefix_ids) > config.data.prefix_tokens:
                reason = "prefix_boundary"
                continue
        display_lines = sum(
            max(
                1,
                (len(line) + pipeline.render.text.line_width - 1)
                // pipeline.render.text.line_width,
            )
            for line in visual.split("\n")
        )
        if display_lines > config.data.max_display_lines:
            reason = "display_lines"
            continue
        try:
            images, actual_lines, truncated = render_pages(visual, config=pipeline.render)
        except RenderConfigError:
            reason = "renderability"
            continue
        if len(images) != 1 or truncated or actual_lines != display_lines:
            raise RuntimeError("training renderer did not preserve the complete span")
        width, height = images[0].size
        record = {
            "record_id": identity,
            "split": split,
            "repository": row["repository_name"],
            "path": row["path"],
            "language": row["lang"],
            "licenses": row["licenses"],
            "source_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "canonical_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "visual_sha256": hashlib.sha256(visual.encode()).hexdigest(),
            "prefix": prefix,
            "visual": visual,
            "continuation": continuation,
            "prefix_ids": prefix_ids,
            "visual_ids": visual_ids,
            "continuation_ids": continuation_ids,
            "prefix_start": prefix_start,
            "visual_start": start,
            "visual_end": end,
            "continuation_end": continuation_end,
            "source_visual_start": canonical.source_boundaries[start],
            "source_visual_end": canonical.source_boundaries[end],
            "source_continuation_end": canonical.source_boundaries[continuation_end],
            "logical_lines": count,
            "display_lines": display_lines,
            "width": width,
            "height": height,
            "aspect_ratio": width / height,
            "visual_token_count": len(visual_ids),
        }
        record["record_hash"] = fingerprint(record)
        return record, "accepted"
    return None, reason


def load_manifest(pipeline: Pipeline) -> list[dict]:
    metadata_path = pipeline.manifest.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    if metadata["data_fingerprint"] != pipeline.data_fingerprint:
        raise ValueError("manifest configuration fingerprint mismatch")
    if file_sha256(pipeline.manifest) != metadata["manifest_sha256"]:
        raise ValueError("manifest checksum mismatch")
    records = pl.read_parquet(pipeline.manifest).to_dicts()
    if Counter(row["split"] for row in records) != pipeline.config.data.split_sizes:
        raise ValueError("manifest split counts mismatch")
    if len({row["record_id"] for row in records}) != len(records):
        raise ValueError("duplicate source file in manifest")
    repositories: dict[str, str] = {}
    for row in records:
        repo = row["repository"].casefold()
        if repo in repositories and repositories[repo] != row["split"]:
            raise ValueError("repository appears in multiple splits")
        repositories[repo] = row["split"]
    return records


_worker_state = None


def initialize_selection_worker(pipeline: Pipeline) -> None:
    global _worker_state
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = load_tokenizer(
        pipeline.config.models.qwen_id, revision=pipeline.config.models.qwen_revision
    )
    coverage = frozenset().union(
        *(
            font_codepoints(path)
            for path in (pipeline.render.text.font, *pipeline.render.text.fallback_fonts)
        )
    )
    _worker_state = pipeline, tokenizer, coverage


def select_candidate(candidate: tuple[str, str, int], split: str):
    if _worker_state is None:
        raise RuntimeError("selection worker was not initialized")
    _, source_path, offset = candidate
    with Path(source_path).open("rb") as stream:
        stream.seek(offset)
        row = json.loads(stream.readline())
    pipeline, tokenizer, coverage = _worker_state
    return make_record(row, split, pipeline, tokenizer, coverage)


def selected_candidates(pipeline: Pipeline, ordered, split: str, start: int, workers: int):
    """Bound outstanding work while consuming results in deterministic candidate order."""
    pool = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_selection_worker,
        initargs=(pipeline,),
    )
    pending = deque()
    submitted = start
    try:
        while submitted < min(len(ordered), start + workers * 2):
            pending.append((submitted, pool.submit(select_candidate, ordered[submitted], split)))
            submitted += 1
        while pending:
            index, future = pending.popleft()
            yield index, future.result()
            if submitted < len(ordered):
                pending.append(
                    (submitted, pool.submit(select_candidate, ordered[submitted], split))
                )
                submitted += 1
    finally:
        for _, future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


def prepare(pipeline: Pipeline, workers: int) -> None:
    config = pipeline.config
    load_credentials(pipeline, wandb=False)
    if pipeline.manifest.exists():
        records = load_manifest(pipeline)
        print(f"verified existing manifest: {len(records)} records", flush=True)
        return
    directory = pipeline.manifest.parent
    directory.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info(config.data.dataset_id, revision=config.data.revision)
    files = sorted(
        item.rfilename for item in info.siblings if item.rfilename.endswith("/data.json")
    )
    if len(files) != 30:
        raise ValueError(f"expected 30 dataset languages, found {len(files)}")
    source_paths = [
        hf_hub_download(
            config.data.dataset_id, name, repo_type="dataset", revision=config.data.revision
        )
        for name in files
    ]
    # Index only identities/line offsets in memory, then seek selected records in seeded order.
    candidates: dict[str, list[tuple[str, str, int]]] = {split: [] for split in SPLITS}
    for source_path in source_paths:
        with Path(source_path).open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                row = json.loads(line)
                split = repository_split(
                    row["repository_name"], config.seed, config.data.split_sizes
                )
                key = fingerprint([config.seed, row["repository_name"], row["path"]])
                candidates[split].append((key, source_path, offset))
    records, audit = [], {}
    seen = set()
    for split in SPLITS:
        partial = directory / f"{split}.selection.json"
        chosen, counts, cursor = [], Counter(), 0
        if partial.exists():
            state = json.loads(partial.read_text())
            if state["fingerprint"] != pipeline.data_fingerprint:
                raise ValueError("partial selection fingerprint mismatch")
            chosen, counts, cursor = state["records"], Counter(state["counts"]), state["cursor"]
        seen.update(row["record_id"] for row in chosen)
        ordered = sorted(candidates[split])
        resume_cursor = cursor
        candidates_iterator = selected_candidates(pipeline, ordered, split, resume_cursor, workers)
        for index, (record, reason) in candidates_iterator:
            if len(chosen) == config.data.split_sizes[split]:
                break
            cursor = index + 1
            counts[reason] += 1
            if record is not None and record["record_id"] not in seen:
                seen.add(record["record_id"])
                chosen.append(record)
            if cursor % 250 == 0:
                write_json(
                    partial,
                    {
                        "fingerprint": pipeline.data_fingerprint,
                        "records": chosen,
                        "counts": dict(counts),
                        "cursor": cursor,
                    },
                )
                print(f"{split}: selected={len(chosen)} scanned={cursor}", flush=True)
        candidates_iterator.close()
        write_json(
            partial,
            {
                "fingerprint": pipeline.data_fingerprint,
                "records": chosen,
                "counts": dict(counts),
                "cursor": cursor,
            },
        )
        audit[split] = {
            "candidates": len(ordered),
            "selected": len(chosen),
            "eligibility": dict(counts),
        }
        write_json(directory / "eligibility.json", audit)
        if len(chosen) != config.data.split_sizes[split]:
            raise RuntimeError(f"insufficient eligible records: {audit}")
        records.extend(chosen)
    teacher_row = 0
    for index, record in enumerate(records):
        record["encoder_shard"] = index // config.data.shard_rows
        record["encoder_row"] = index % config.data.shard_rows
        needs_teacher = record["split"] != "reconstruction"
        record["teacher_shard"] = teacher_row // config.data.shard_rows if needs_teacher else -1
        record["teacher_row"] = teacher_row % config.data.shard_rows if needs_teacher else -1
        teacher_row += int(needs_teacher)
    frame = pl.DataFrame(records, infer_schema_length=None)
    overlaps = {}
    for key in ("source_sha256", "canonical_sha256", "visual_sha256"):
        overlaps[key] = (
            frame.group_by(key)
            .agg(pl.col("split").n_unique().alias("splits"))
            .filter(pl.col("splits") > 1)
            .height
        )
    audit["cross_split_exact_hash_groups"] = overlaps
    write_json(directory / "eligibility.json", audit)
    temporary = pipeline.manifest.with_suffix(".parquet.tmp")
    frame.write_parquet(temporary)
    temporary.replace(pipeline.manifest)
    write_json(
        pipeline.manifest.with_suffix(".json"),
        {
            "data_fingerprint": pipeline.data_fingerprint,
            "manifest_sha256": file_sha256(pipeline.manifest),
            "records": len(records),
        },
    )
    for record in records[: config.data.debug_images]:
        images, _, _ = render_pages(record["visual"], config=pipeline.render)
        images[0].save(directory / f"debug-{record['record_id']}.png")
        text = record["prefix"] + record["visual"] + record["continuation"]
        (directory / f"debug-{record['record_id']}.xml").write_text(
            inspection_markup(
                text, len(record["prefix"]), len(record["prefix"]) + len(record["visual"])
            ),
            encoding="utf-8",
        )
    print(f"prepared {len(load_manifest(pipeline))} records; audit={audit}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic optical-adapter manifests")
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    prepare(load_pipeline(args.config), args.workers)


if __name__ == "__main__":
    main()
