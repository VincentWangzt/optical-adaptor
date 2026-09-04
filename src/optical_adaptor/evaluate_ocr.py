from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from optical_adaptor.edit_distance import evaluate_edit_distance
from optical_adaptor.infer_ocr import (
    InferenceConfigError,
    load_inference_config,
    run_ocr_images,
    validate_inference_runtime,
)
from optical_adaptor.renderer import RenderConfigError, load_render_config, render_source
from optical_adaptor.token_utils import count_and_truncate, count_tokens, load_tokenizer


def _positive_int(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _memory_utilization(value: float, name: str) -> float:
    if not 0 < value <= 1:
        raise ValueError(f"{name} must be greater than 0 and at most 1")
    return value


def _source_portion(text: str, start_line: int, end_line: int | None) -> tuple[str, int]:
    lines = text.splitlines(keepends=True)
    if not lines:
        lines = [""]
    if start_line > len(lines):
        raise ValueError(
            f"--start-line {start_line} exceeds the file's {len(lines)} line(s)"
        )
    stop = len(lines) if end_line is None else min(end_line, len(lines))
    if stop < start_line:
        raise ValueError("--end-line must be greater than or equal to --start-line")
    return "".join(lines[start_line - 1 : stop]), stop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select part of a source file, optionally truncate it by model tokens, "
            "render it, run OCR, and report text/image token counts."
        )
    )
    parser.add_argument("source", type=Path, help="UTF-8/code file to evaluate")
    parser.add_argument(
        "--inference-config",
        type=Path,
        default=Path("configs/inference.qwen3.5-4b.yaml"),
        help="model inference YAML (default: configs/inference.qwen3.5-4b.yaml)",
    )
    parser.add_argument(
        "--render-config",
        type=Path,
        default=Path("configs/render.yaml"),
        help="render YAML (default: configs/render.yaml)",
    )
    parser.add_argument("--model", help="model alias; inferred for a single-model config")
    parser.add_argument("--start-line", type=int, default=1, help="first line, inclusive")
    parser.add_argument("--end-line", type=int, help="last line, inclusive")
    parser.add_argument(
        "--max-source-tokens",
        type=int,
        help="truncate the selected source portion at this many model tokens",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation"),
        help="render, OCR, and report directory (default: outputs/evaluation)",
    )
    parser.add_argument("--prompt", help="override the configured OCR prompt")
    parser.add_argument("--gpu", type=int, help="override inference.gpu")
    parser.add_argument("--max-new-tokens", type=int, help="override generation limit")
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable or disable vLLM eager execution",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        help="override the fraction of GPU memory available to this vLLM instance",
    )
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable or disable Qwen thinking mode",
    )
    return parser


def _run(args: argparse.Namespace) -> Path:
    if args.start_line <= 0:
        raise ValueError("--start-line must be a positive integer")
    if args.end_line is not None and args.end_line <= 0:
        raise ValueError("--end-line must be a positive integer")
    if args.max_source_tokens is not None:
        _positive_int(args.max_source_tokens, "--max-source-tokens")
    if args.max_new_tokens is not None:
        _positive_int(args.max_new_tokens, "--max-new-tokens")
    if args.gpu_memory_utilization is not None:
        _memory_utilization(args.gpu_memory_utilization, "--gpu-memory-utilization")
    if args.gpu is not None and args.gpu < 0:
        raise ValueError("--gpu must be a non-negative integer")

    source_path = args.source.expanduser().resolve()
    render_config = load_render_config(args.render_config)
    try:
        full_source = source_path.read_text(encoding=render_config.text.encoding)
    except UnicodeError as exc:
        raise ValueError(
            f"cannot decode {source_path} with {render_config.text.encoding!r}: {exc}"
        ) from exc
    selected_source, actual_end_line = _source_portion(
        full_source, args.start_line, args.end_line
    )

    inference, models = load_inference_config(args.inference_config)
    if args.model is None:
        if len(models) != 1:
            choices = ", ".join(models)
            raise ValueError(f"--model is required; configured aliases: {choices}")
        alias = next(iter(models))
    else:
        alias = args.model
    if alias not in models:
        choices = ", ".join(models)
        raise ValueError(f"unknown model {alias!r}; configured aliases: {choices}")
    model_config = models[alias]
    validate_inference_runtime(model_config)
    inference = replace(
        inference,
        max_new_tokens=args.max_new_tokens or inference.max_new_tokens,
        enable_thinking=(
            args.thinking if args.thinking is not None else inference.enable_thinking
        ),
        gpu=args.gpu if args.gpu is not None else inference.gpu,
        enforce_eager=(
            args.enforce_eager
            if args.enforce_eager is not None
            else inference.enforce_eager
        ),
        gpu_memory_utilization=(
            args.gpu_memory_utilization
            if args.gpu_memory_utilization is not None
            else inference.gpu_memory_utilization
        ),
    )

    # Set visibility before tokenizer/Transformers imports torch.
    if inference.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(inference.gpu)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    tokenizer = load_tokenizer(
        model_config.model_id,
        revision=model_config.revision,
        trust_remote_code=inference.trust_remote_code,
    )
    source_tokens = count_and_truncate(
        selected_source, tokenizer, args.max_source_tokens
    )

    output_dir = args.output_dir.expanduser().resolve()
    render_dir = output_dir / "rendered"
    ocr_dir = output_dir / "ocr"
    result = render_source(
        source_tokens.content,
        stem=source_path.stem,
        output_dir=render_dir,
        config=render_config,
    )
    page_results = run_ocr_images(
        alias=alias,
        model_config=model_config,
        inference=inference,
        image_paths=result.page_paths,
        output_dir=ocr_dir,
        prompt_override=args.prompt,
    )

    page_texts = [
        Path(page["output"]).read_text(encoding="utf-8").rstrip("\n")
        for page in page_results
    ]
    combined_ocr = "\n".join(page_texts)
    combined_path = output_dir / f"{source_path.stem}.{alias}.ocr.md"
    combined_path.write_text(combined_ocr + "\n", encoding="utf-8")

    report: dict[str, Any] = {
        "source": str(source_path),
        "model": alias,
        "model_id": model_config.model_id,
        "line_range": {"start": args.start_line, "end": actual_end_line},
        "source_characters": len(source_tokens.content),
        "source_tokens_before_truncation": source_tokens.original_token_count,
        "text_tokens": source_tokens.token_count,
        "source_truncated_by_tokens": source_tokens.truncated,
        "rendered_pages": len(result.page_paths),
        "render_truncated_by_page_limit": result.truncated,
        "image_tokens": sum(page["image_tokens"] for page in page_results),
        "ocr_output_tokens": count_tokens(combined_ocr, tokenizer),
        "combined_ocr_output": str(combined_path),
        "edit_distance": {
            "character": evaluate_edit_distance(
                source_tokens.content.rstrip("\n"), combined_ocr, unit="character"
            ),
            "word": evaluate_edit_distance(
                source_tokens.content.rstrip("\n"), combined_ocr, unit="word"
            ),
        },
        "pages": page_results,
    }
    report_path = output_dir / f"{source_path.stem}.{alias}.evaluation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Text tokens: {report['text_tokens']}")
    print(f"Image tokens: {report['image_tokens']}")
    print(f"OCR output tokens: {report['ocr_output_tokens']}")
    print(f"Character edit distance: {report['edit_distance']['character']['distance']}")
    print(f"Report: {report_path}")
    return report_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _run(args)
    except (
        FileNotFoundError,
        InferenceConfigError,
        OSError,
        RenderConfigError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ocr-evaluate: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
