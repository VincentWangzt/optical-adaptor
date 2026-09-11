"""Compare saved QA responses without changing or repeating model inference."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from optical_adaptor.benchmark.config import load_benchmark
from optical_adaptor.benchmark.run import score
from optical_adaptor.training.config import write_json


def lcb_answer(text: str) -> str | None:
    """Match upstream LongCodeQA's final-answer or terminal-letter policy."""
    final = re.search(r"Final Answer:\s*([ABCD])", text, re.IGNORECASE)
    terminal = re.search(r"\b([ABCD])\s*$", text.strip(), re.IGNORECASE)
    match = final or terminal
    return match[1].upper() if match else None


def compare(directory: Path, results_directory: Path):
    cases = [
        r
        for r in json.loads((directory / "manifest.json").read_text())["cases"]
        if r["task"] == "qa"
    ]
    report = {}
    for backend, mode in [
        ("vllm-adapter", "images"),
        ("vllm-native", "images"),
        ("vllm-native", "text"),
        ("vllm-native", "no-image"),
    ]:
        rows = []
        for case in cases:
            path = results_directory / backend / mode / f"{case['id']}.json"
            if path.exists():
                response = json.loads(path.read_text())["response"]
                answer = lcb_answer(response["text"])
                rows.append(
                    {
                        "id": case["id"],
                        "lcb_answer": answer,
                        "lcb_correct": answer == case["reference"],
                        "strict_correct": score(case, response["text"])["correct"],
                        "response": response,
                    }
                )
        result = {
            "expected": len(cases),
            "completed": len(rows),
            "complete": len(rows) == len(cases),
        }
        if rows:
            result.update(
                {
                    "lcb_correct": sum(r["lcb_correct"] for r in rows),
                    "lcb_accuracy_on_completed": sum(r["lcb_correct"] for r in rows) / len(rows),
                    "strict_correct": sum(r["strict_correct"] for r in rows),
                    "strict_accuracy_on_completed": sum(r["strict_correct"] for r in rows)
                    / len(rows),
                    "lcb_unparseable": sum(r["lcb_answer"] is None for r in rows),
                    "length_stops": sum(r["response"]["finish_reason"] == "length" for r in rows),
                    "mean_prompt_tokens": sum(r["response"]["prompt_tokens"] for r in rows)
                    / len(rows),
                    "mean_image_tokens": sum(r["response"]["image_tokens"] for r in rows)
                    / len(rows),
                }
            )
        report[f"{backend}/{mode}"] = result
    write_json(results_directory / "qa-comparison.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    args = parser.parse_args()
    config, _, directory = load_benchmark(args.config)
    print(json.dumps(compare(directory, directory / config.results_subdir), indent=2))


if __name__ == "__main__":
    main()
