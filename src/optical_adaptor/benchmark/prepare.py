from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from huggingface_hub import hf_hub_download

from optical_adaptor.artifacts import file_sha256, fingerprint, load_credentials, write_json
from optical_adaptor.automodel.config import FilterConfig
from optical_adaptor.automodel.conversations import parse_visual_areas
from optical_adaptor.automodel.data import ConversationDataset
from optical_adaptor.benchmark.config import BenchmarkConfig, load_benchmark
from optical_adaptor.inference.messages import chat_ids, text_message
from optical_adaptor.renderer import FontChain, font_codepoints, render_pages
from optical_adaptor.text import canonicalize
from optical_adaptor.token_utils import load_tokenizer


def source_lines(text: str) -> list[str]:
    # splitlines drops exactly one terminal newline, as does the shared renderer.
    return text.splitlines()


def join_lines(lines: list[str]) -> str:
    text = "\n".join(lines)
    return text + "\n" if lines and lines[-1] == "" else text


def save_image(text: str, directory: Path, pipeline) -> dict:
    key = fingerprint([pipeline.data_fingerprint, text])
    path = directory / "images" / f"{key}.png"
    metadata_path = path.with_suffix(".json")
    if path.exists() and metadata_path.exists():
        return json.loads(metadata_path.read_text())
    images, display_lines, truncated = render_pages(text, config=pipeline.render)
    if len(images) != 1 or truncated:
        raise ValueError("benchmark image must contain the entire supplied source span")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp.png")
    images[0].save(temporary)
    temporary.replace(path)
    metadata = {
        "path": str(path.relative_to(directory)),
        "sha256": file_sha256(path),
        "source_lines": len(source_lines(text)),
        "display_lines": display_lines,
        "width": images[0].width,
        "height": images[0].height,
    }
    write_json(metadata_path, metadata)
    return metadata


def benchmark_records(pipeline):
    filters = FilterConfig(
        sources=None, tasks=None, image_bins=None, turn_bins=None, min_images=1, max_images=None
    )
    result = []
    for split in ("train", "eval"):
        dataset = ConversationDataset(str(pipeline.manifest.parent), split, filters)
        for index, row in enumerate(dataset.rows):
            record = dataset[index]
            messages, areas = parse_visual_areas(record["messages"])
            visuals = [messages[a["message"]]["content"][a["start"] : a["end"]] for a in areas]
            result.append({**row, "visuals": visuals})
    return result


def reconstruction_sets(records, config, pipeline, directory):
    heldout = sorted(
        (r for r in records if r["split"] == "eval" and r["task"] == "reconstruction"),
        key=lambda r: fingerprint([config.seed, r["sample_id"]]),
    )
    training_texts = {text for r in records if r["split"] == "train" for text in r["visuals"]}
    heldout = [r for r in heldout if not training_texts.intersection(r["visuals"])]
    pages = [(r, text) for r in heldout for text in r["visuals"]]
    cases, audit = [], {"validation_scope": "sample-level", "eligible_samples": len(heldout)}
    for count in config.multi_image_counts:
        usable = len(pages) // count * count
        for offset in range(0, usable, count):
            group = pages[offset : offset + count]
            ids = [r["sample_id"] for r, _ in group]
            visuals = [text for _, text in group]
            cases.append(
                {
                    "id": fingerprint(["multi", ids, visuals]),
                    "suite": f"reconstruction-{count}-images",
                    "task": "reconstruction",
                    "source_record_ids": ids,
                    "repositories": [r["group_id"] for r, _ in group],
                    "images": [save_image(text, directory, pipeline) for text in visuals],
                    "reference": "\n".join(visuals),
                    "instruction": config.reconstruction_instruction,
                }
            )
        audit[f"multi_{count}"] = {"cases": usable // count, "unused_pages": len(pages) - usable}
    long_count = 0
    for row in heldout:
        lines = source_lines("\n".join(row["visuals"]))
        if len(lines) < config.long_source_lines:
            continue
        reference = join_lines(lines[: config.long_source_lines])
        cases.append(
            {
                "id": fingerprint(["long", row["sample_id"], config.long_source_lines]),
                "suite": f"reconstruction-{config.long_source_lines}-lines",
                "task": "reconstruction",
                "source_record_ids": [row["sample_id"]],
                "repositories": [row["group_id"]],
                "images": [save_image(reference, directory, pipeline)],
                "reference": reference,
                "instruction": config.reconstruction_instruction,
            }
        )
        long_count += 1
    audit["long"] = {"cases": long_count}
    return cases, audit


def lcb_cases(config, pipeline, directory, tokenizer, pool):
    archive = Path(
        hf_hub_download(
            config.lcb_dataset,
            "LongCodeQA.zip",
            repo_type="dataset",
            revision=config.lcb_revision,
            local_dir=directory / "lcb-download",
        )
    )
    cases, excluded, seen = [], Counter(), set()
    coverage = frozenset().union(
        *(
            font_codepoints(p)
            for p in (pipeline.render.text.font, *pipeline.render.text.fallback_fonts)
        )
    )
    with zipfile.ZipFile(archive) as source:
        for name in sorted(source.namelist()):
            if not name.endswith(".json"):
                continue
            for index, row in enumerate(json.loads(source.read(name))):
                # Filter every shard using the pinned target tokenizer, never shard labels.
                tokens = len(chat_ids(tokenizer, text_message(row["prompt"])))
                if tokens >= config.prompt_token_limit:
                    excluded["prompt_at_or_above_limit"] += 1
                    continue
                key = fingerprint([row["prompt"], row["correct_letter"]])
                if key in seen:
                    excluded["duplicate"] += 1
                    continue
                seen.add(key)
                if row["prompt"].count(row["repo_text"]) != 1:
                    raise ValueError("LCB prompt must contain repo_text exactly once")
                before, after = row["prompt"].split(row["repo_text"])
                font = FontChain(pipeline.render.text)
                escaped = []
                for char in set(row["repo_text"]):
                    if ord(char) > 127 and ord(char) in coverage:
                        _, top, _, bottom = font.getbbox(char)
                        if top < 0 or bottom > pipeline.render.text.line_height:
                            escaped.append(ord(char))
                canonical = canonicalize(
                    row["repo_text"], coverage.difference(escaped), pipeline.render.text.tab_width
                ).text
                lines = source_lines(canonical)
                pages = [
                    join_lines(lines[start : start + config.qa_source_lines_per_image])
                    for start in range(0, len(lines), config.qa_source_lines_per_image)
                ]
                images = list(
                    pool.map(partial(save_image, directory=directory, pipeline=pipeline), pages)
                )
                if len(images) > config.backend.max_images:
                    raise ValueError(
                        f"LCB {name}:{index} needs {len(images)} images; raise max_images"
                    )
                cases.append(
                    {
                        "id": key,
                        "suite": "longcodeqa-under-32k",
                        "task": "qa",
                        "source_shard": name,
                        "source_index": index,
                        "repository": row["repo"],
                        "images": images,
                        "reference": row["correct_letter"].strip().upper(),
                        "text_prompt": row["prompt"],
                        "before_images": before,
                        "after_images": after,
                        # Remove only the repository; keep the question, options and instructions.
                        "no_image_prompt": before + after,
                        "text_prompt_tokens": tokens,
                        "canonical_render_changed": canonical != row["repo_text"],
                        "escaped_vertical_glyph_codepoints": sorted(escaped),
                    }
                )
                if len(cases) % 10 == 0:
                    print(
                        f"LCB rendered {len(cases)} cases; latest repository={row['repo']}",
                        flush=True,
                    )
            print(f"LCB: {name}; retained={len(cases)} excluded={dict(excluded)}", flush=True)
    return cases, {
        "cases": len(cases),
        "excluded": dict(excluded),
        "archive_sha256": file_sha256(archive),
        "revision": config.lcb_revision,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    config, pipeline, directory = load_benchmark(args.config)
    load_credentials(pipeline, wandb=False)
    directory.mkdir(parents=True, exist_ok=True)
    identity = {
        "config": config.data_settings(),
        "source_manifest": file_sha256(pipeline.manifest),
        "data_fingerprint": pipeline.data_fingerprint,
    }
    state_path = directory / "preparation.json"
    if state_path.exists():
        previous = json.loads(state_path.read_text())["identity"]
        if "backend" in previous["config"]:
            previous["config"] = BenchmarkConfig.model_validate(previous["config"]).data_settings()
        if previous != identity:
            raise ValueError("preparation configuration changed; select a new output directory")
    write_json(state_path, {"identity": identity})
    reconstruction_path = directory / "reconstruction.json"
    if reconstruction_path.exists():
        saved = json.loads(reconstruction_path.read_text())
        cases, reconstruction_audit = saved["cases"], saved["audit"]
    else:
        cases, reconstruction_audit = reconstruction_sets(
            benchmark_records(pipeline), config, pipeline, directory
        )
        write_json(reconstruction_path, {"cases": cases, "audit": reconstruction_audit})
    tokenizer = load_tokenizer(
        pipeline.optical.llm["model_id"], revision=pipeline.optical.llm["revision"]
    )
    qa_path = directory / "longcodeqa.json"
    if qa_path.exists():
        saved = json.loads(qa_path.read_text())
        qa, qa_audit = saved["cases"], saved["audit"]
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            qa, qa_audit = lcb_cases(config, pipeline, directory, tokenizer, pool)
        write_json(qa_path, {"cases": qa, "audit": qa_audit})
    cases.extend(qa)
    for case in cases:
        if case["task"] == "reconstruction":
            case["reference_tokens"] = len(
                tokenizer.encode(case["reference"], add_special_tokens=False)
            )
            if case["reference_tokens"] >= config.reconstruction_max_tokens:
                raise ValueError("reconstruction output budget is smaller than a reference")
    write_json(directory / "manifest.json", {"identity": identity, "cases": cases})
    write_json(
        directory / "audit.json",
        {
            "reconstruction": reconstruction_audit,
            "lcb": qa_audit,
            "suite_counts": dict(Counter(r["suite"] for r in cases)),
        },
    )
    print(json.dumps(json.loads((directory / "audit.json").read_text()), indent=2), flush=True)


if __name__ == "__main__":
    main()
