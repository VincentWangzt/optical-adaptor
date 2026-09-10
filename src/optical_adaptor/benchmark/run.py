from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

from optical_adaptor.benchmark.config import load_benchmark
from optical_adaptor.edit_distance import evaluate_edit_distance
from optical_adaptor.inference.backend import BACKENDS, ChatRequest, build_backend
from optical_adaptor.inference.messages import text_message
from optical_adaptor.training.config import file_sha256, load_credentials, write_json


def case_messages(case: dict, mode: str, directory: Path) -> list[dict]:
    if mode in {"text", "no-image"}:
        if case["task"] != "qa":
            raise ValueError("text and no-image controls are defined only for QA")
        return text_message(case["text_prompt" if mode == "text" else "no_image_prompt"])
    if mode != "images":
        raise ValueError(f"unknown evaluation condition: {mode}")
    before = case["instruction"] if case["task"] == "reconstruction" else case["before_images"]
    content = [{"type": "text", "text": before}]
    for metadata in case["images"]:
        path = directory / metadata["path"]
        if file_sha256(path) != metadata["sha256"]:
            raise ValueError(f"image checksum mismatch: {path}")
        content.append({"type": "image_url", "image_url": {"url": path.resolve().as_uri()}})
    if case["task"] == "qa":
        content.append({"type": "text", "text": case["after_images"]})
    return [{"role": "user", "content": content}]


def score(case: dict, text: str) -> dict:
    if case["task"] == "qa":
        match = re.fullmatch(r"([A-D])[.)]?", text.strip())
        answer = match[1] if match else None
        return {
            "answer": answer,
            "correct": answer == case["reference"],
            "invalid_answer": answer is None,
        }
    characters = evaluate_edit_distance(case["reference"], text, unit="character")
    words = evaluate_edit_distance(case["reference"], text, unit="word")
    return {
        "exact_match": case["reference"] == text,
        "character_distance": characters["distance"],
        "reference_characters": characters["reference_units"],
        "word_distance": words["distance"],
        "reference_words": words["reference_units"],
        "character_error_rate": characters["error_rate"],
    }


def report(cases, modes, destination):
    expected, groups = Counter(), defaultdict(list)
    for case in cases:
        for mode in modes:
            if case["task"] != "qa" and mode != "images":
                continue
            group = f"{case['suite']}/{mode}"
            expected[group] += 1
            path = destination / mode / f"{case['id']}.json"
            if path.exists():
                groups[group].append(json.loads(path.read_text()))
    summary = {}
    for name, count in expected.items():
        rows = groups[name]
        values = {"expected": count, "completed": len(rows), "complete": len(rows) == count}
        if rows:
            values.update(
                {
                    "length_stops": sum(r["response"]["finish_reason"] == "length" for r in rows),
                    "mean_prompt_tokens": sum(r["response"]["prompt_tokens"] for r in rows)
                    / len(rows),
                    "mean_image_tokens": sum(r["response"]["image_tokens"] for r in rows)
                    / len(rows),
                    "mean_completion_tokens": sum(r["response"]["completion_tokens"] for r in rows)
                    / len(rows),
                }
            )
            if "correct" in rows[0]["score"]:
                values["accuracy_on_completed"] = sum(r["score"]["correct"] for r in rows) / len(
                    rows
                )
                values["invalid_answers"] = sum(r["score"]["invalid_answer"] for r in rows)
            else:
                values["character_error_rate"] = sum(
                    r["score"]["character_distance"] for r in rows
                ) / max(1, sum(r["score"]["reference_characters"] for r in rows))
                values["word_error_rate"] = sum(r["score"]["word_distance"] for r in rows) / max(
                    1, sum(r["score"]["reference_words"] for r in rows)
                )
                values["exact_match_on_completed"] = sum(
                    r["score"]["exact_match"] for r in rows
                ) / len(rows)
        summary[name] = values
    write_json(destination / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--modes", nargs="+", choices=["images", "text", "no-image"])
    parser.add_argument("--suites", nargs="+")
    parser.add_argument("--limit-per-suite", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    config, pipeline, directory = load_benchmark(args.config)
    modes = args.modes or (
        ["images"] if args.backend == "vllm-adapter" else ["no-image", "text", "images"]
    )
    if args.backend == "vllm-adapter" and modes != ["images"]:
        raise ValueError("QA text controls use the unchanged native Qwen language model")
    destination = args.output or directory / "results" / args.backend
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["identity"]["config"] != config.model_dump():
        raise ValueError("benchmark configuration changed after data preparation")
    cases = manifest["cases"]
    if args.suites:
        missing = set(args.suites) - {r["suite"] for r in cases}
        if missing:
            raise ValueError(f"unknown suites: {missing}")
        cases = [r for r in cases if r["suite"] in args.suites]
    if args.limit_per_suite is not None:
        if args.limit_per_suite < 1:
            raise ValueError("limit-per-suite must be positive")
        counts, chosen = Counter(), []
        for case in cases:
            if counts[case["suite"]] < args.limit_per_suite:
                chosen.append(case)
                counts[case["suite"]] += 1
        cases = chosen
    if args.report_only:
        print(json.dumps(report(cases, modes, destination), indent=2))
        return
    load_credentials(pipeline, wandb=False)
    identity = {
        "manifest_sha256": file_sha256(manifest_path),
        "backend": args.backend,
        "checkpoint_sha256": file_sha256(pipeline.repo / config.checkpoint / "adapter.safetensors")
        if args.backend == "vllm-adapter"
        else None,
        "vllm": version("vllm"),
        "transformers": version("transformers"),
        "torch": version("torch"),
        "config": config.model_dump(),
    }
    identity_path = destination / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("inference provenance changed; choose a new result directory")
    write_json(identity_path, identity)
    backend = build_backend(
        args.backend,
        pipeline,
        config.backend,
        checkpoint=pipeline.repo / config.checkpoint,
        seed=config.seed,
    )
    write_json(destination / "backend.json", backend.identity)
    try:
        for mode in modes:
            pending = [
                r
                for r in cases
                if (mode == "images" or r["task"] == "qa")
                and not (destination / mode / f"{r['id']}.json").exists()
            ]
            # Group similarly sized requests in each suite to reduce padding and stragglers.
            pending.sort(key=lambda r: (r["suite"], len(r["images"]), r["id"]))
            for start in range(0, len(pending), config.batch_size):
                batch = pending[start : start + config.batch_size]
                requests = [
                    ChatRequest(
                        case_messages(r, mode, directory),
                        config.qa_max_tokens
                        if r["task"] == "qa"
                        else config.reconstruction_max_tokens,
                        config.seed,
                    )
                    for r in batch
                ]
                responses = backend.generate(requests)
                for case, response in zip(batch, responses, strict=True):
                    write_json(
                        destination / mode / f"{case['id']}.json",
                        {
                            "id": case["id"],
                            "suite": case["suite"],
                            "mode": mode,
                            "response": response.to_dict(),
                            "score": score(case, response.text),
                        },
                    )
                summary = report(cases, modes, destination)
                print(
                    json.dumps(
                        {
                            "mode": mode,
                            "finished": start + len(batch),
                            "pending_at_start": len(pending),
                            "last_suite": batch[-1]["suite"],
                        }
                    ),
                    flush=True,
                )
        summary = report(cases, modes, destination)
        write_json(destination / "complete.json", {"identity": identity, "summary": summary})
    finally:
        backend.close()


if __name__ == "__main__":
    main()
