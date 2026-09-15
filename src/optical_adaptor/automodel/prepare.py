"""Prepare repository-disjoint SFT slices. Pixels and model states stay online."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import polars as pl
from datasets import load_dataset
from dotenv import load_dotenv

from optical_adaptor.automodel.config import fingerprint, read_config
from optical_adaptor.automodel.conversations import (
    normalize_messages,
    slice_key,
    stack_records,
    swe_records,
)
from optical_adaptor.renderer import font_codepoints, load_render_config
from optical_adaptor.training.data import canonicalize


def prepare(config_path: str, source_limit: int | None = None) -> dict:
    raw, optical = read_config(config_path)
    config, seed = optical.prepare, raw["seed"]
    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.parquet"
    if manifest.exists() or (root / "preparation.incomplete").exists():
        raise FileExistsError(f"Choose a fresh data output directory: {root}")
    (root / "preparation.incomplete").write_text("Preparation has not finished.\n")
    render = load_render_config(optical.render_config)
    coverage = frozenset().union(
        *(font_codepoints(path) for path in (render.text.font, *render.text.fallback_fonts))
    )
    rows, counts, originals, seen = [], Counter(), Counter(), set()
    for source in config.sources:
        dataset = load_dataset(
            source.dataset_id,
            data_files=source.data_files,
            revision=source.revision,
            split=source.split,
            streaming=True,
        )
        raw_path = root / "originals" / f"{source.name}.jsonl"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        limit = source_limit if source_limit is not None else source.max_records
        with raw_path.open("w", encoding="utf-8", newline="\n") as raw_output:
            for row in dataset:
                if limit is not None and originals[source.name] >= limit:
                    break
                if source.kind == "stack":
                    text = canonicalize(row["content"], coverage, render.text.tab_width).text
                    records = stack_records(
                        row, text.rstrip("\n"), source, config, seed, render.text.line_width
                    )
                else:
                    messages = normalize_messages(row["messages"])
                    for message in messages:
                        if message["role"] == "tool":
                            message["content"] = canonicalize(
                                message["content"], coverage, render.text.tab_width
                            ).text.rstrip("\n")
                    records = swe_records(
                        row, messages, source, config, seed, render.text.line_width
                    )
                # Retain original complete trajectories/documents separately.
                # Metadata (including reference patches) never enters SFT input.
                raw_output.write(json.dumps(row, ensure_ascii=False) + "\n")
                originals[source.name] += 1
                for record in records:
                    sample_id = record["sample_id"]
                    if sample_id in seen:
                        counts["duplicate"] += 1
                        continue
                    seen.add(sample_id)
                    key = slice_key(record)
                    path = Path("slices") / key / f"{record['split']}.jsonl"
                    output = root / path
                    output.parent.mkdir(parents=True, exist_ok=True)
                    payload = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
                    with output.open("ab") as handle:
                        offset = handle.tell()
                        handle.write(payload)
                    rows.append(
                        {
                            **{
                                key: record[key]
                                for key in (
                                    "sample_id",
                                    "group_id",
                                    "source",
                                    "task",
                                    "view",
                                    "thinking",
                                    "image_count",
                                    "turn_count",
                                    "image_bin",
                                    "turn_bin",
                                    "split",
                                )
                            },
                            "slice": key,
                            "path": str(path.as_posix()),
                            "offset": offset,
                            "length": len(payload),
                            "sha256": fingerprint(record),
                        }
                    )
                    counts[f"{record['split']}/{key}"] += 1
                if originals[source.name] % 100 == 0:
                    print(f"{source.name}: {originals[source.name]} source records", flush=True)
    if not rows:
        raise ValueError("No eligible records were found")
    frame = pl.DataFrame(rows)
    leakage = frame.group_by("group_id").agg(pl.col("split").n_unique()).filter(pl.col("split") > 1)
    if leakage.height:
        raise AssertionError("Repository leakage between train and eval")
    frame.write_parquet(manifest)
    summary = {
        "preparation_fingerprint": fingerprint(
            {
                "seed": seed,
                "prepare": config.model_dump(),
                "render": Path(optical.render_config).read_text(),
            }
        ),
        "original_records": dict(originals),
        "counts": dict(sorted(counts.items())),
        "samples": len(rows),
        "dataset_fingerprint": fingerprint(rows),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (root / "preparation.incomplete").unlink()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/automodel.yaml")
    parser.add_argument("--source-limit", type=int)
    args = parser.parse_args()
    load_dotenv()
    prepare(args.config, args.source_limit)


if __name__ == "__main__":
    main()
