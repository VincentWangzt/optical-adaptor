from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml

from optical_agent.token_utils import count_tokens


class InferenceConfigError(ValueError):
    """Raised when an inference YAML is invalid."""


@dataclass(frozen=True)
class InferenceConfig:
    prompt: str
    dtype: str
    max_new_tokens: int
    enable_thinking: bool
    gpu: int | None
    enforce_eager: bool
    gpu_memory_utilization: float
    trust_remote_code: bool


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    revision: str | None
    prompt: str | None
    backend: str = "qwen"
    base_size: int | None = None
    image_size: int | None = None
    crop_mode: bool = True
    max_crops: int | None = None
    max_model_len: int = 8192


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InferenceConfigError(f"{name} must be a YAML mapping")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InferenceConfigError(f"{name} must be a positive integer")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise InferenceConfigError(f"{name} must be true or false")
    return value


def _memory_utilization(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InferenceConfigError(f"{name} must be a number greater than 0 and at most 1")
    result = float(value)
    if not 0 < result <= 1:
        raise InferenceConfigError(f"{name} must be greater than 0 and at most 1")
    return result


def load_inference_config(path: str | Path) -> tuple[InferenceConfig, dict[str, ModelConfig]]:
    config_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InferenceConfigError(f"cannot read inference config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise InferenceConfigError(f"invalid YAML in {config_path}: {exc}") from exc
    root = _as_mapping(raw, "config")
    inference_raw = _as_mapping(root.get("inference"), "inference")
    models_raw = _as_mapping(root.get("models"), "models")

    prompt = inference_raw.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise InferenceConfigError("inference.prompt must be a non-empty string")
    dtype = str(inference_raw.get("dtype", "bfloat16"))
    if dtype not in {"bfloat16", "float16"}:
        raise InferenceConfigError("inference.dtype must be bfloat16 or float16")
    gpu = inference_raw.get("gpu", 0)
    if gpu is not None and (isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0):
        raise InferenceConfigError("inference.gpu must be null or a non-negative integer")

    inference = InferenceConfig(
        prompt=prompt,
        dtype=dtype,
        max_new_tokens=_positive_int(
            inference_raw.get("max_new_tokens", 8192), "inference.max_new_tokens"
        ),
        enable_thinking=_boolean(
            inference_raw.get("enable_thinking", False), "inference.enable_thinking"
        ),
        gpu=gpu,
        enforce_eager=_boolean(
            inference_raw.get("enforce_eager", True), "inference.enforce_eager"
        ),
        gpu_memory_utilization=_memory_utilization(
            inference_raw.get("gpu_memory_utilization", 0.92),
            "inference.gpu_memory_utilization",
        ),
        trust_remote_code=_boolean(
            inference_raw.get("trust_remote_code", False), "inference.trust_remote_code"
        ),
    )
    models: dict[str, ModelConfig] = {}
    for alias, value in models_raw.items():
        model_raw = _as_mapping(value, f"models.{alias}")
        model_id = model_raw.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            raise InferenceConfigError(f"models.{alias}.model_id must be a non-empty string")
        revision = model_raw.get("revision")
        if revision is not None and not isinstance(revision, str):
            raise InferenceConfigError(f"models.{alias}.revision must be null or a string")
        model_prompt = model_raw.get("prompt")
        if model_prompt is not None and (
            not isinstance(model_prompt, str) or not model_prompt.strip()
        ):
            raise InferenceConfigError(f"models.{alias}.prompt must be null or a non-empty string")
        backend = str(model_raw.get("backend", "qwen"))
        if backend not in {"qwen", "deepseek"}:
            raise InferenceConfigError(
                f"models.{alias}.backend must be qwen or deepseek"
            )
        base_size = model_raw.get("base_size")
        image_size = model_raw.get("image_size")
        max_crops = model_raw.get("max_crops")
        crop_mode = model_raw.get("crop_mode", True)
        max_model_len = _positive_int(
            model_raw.get("max_model_len", 8192), f"models.{alias}.max_model_len"
        )
        if backend == "deepseek":
            base_size = _positive_int(base_size, f"models.{alias}.base_size")
            image_size = _positive_int(image_size, f"models.{alias}.image_size")
            max_crops = _positive_int(max_crops, f"models.{alias}.max_crops")
            crop_mode = _boolean(crop_mode, f"models.{alias}.crop_mode")
            if "<image>" not in (model_prompt or prompt):
                raise InferenceConfigError(
                    f"models.{alias} effective prompt must contain <image>"
                )
        elif any(
            key in model_raw for key in ("base_size", "image_size", "crop_mode", "max_crops")
        ):
            raise InferenceConfigError(
                f"models.{alias} DeepSeek image options require backend: deepseek"
            )
        models[str(alias)] = ModelConfig(
            model_id=model_id,
            revision=revision,
            prompt=model_prompt,
            backend=backend,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            max_crops=max_crops,
            max_model_len=max_model_len,
        )
    if not models:
        raise InferenceConfigError("models must contain at least one model")
    return inference, models


def _import_vllm_stack() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from PIL import Image
        from vllm import LLM, SamplingParams
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "vLLM inference dependencies are missing; run `uv sync`"
        ) from exc
    return torch, Image, LLM, SamplingParams


def _effective_prompt(
    model_config: ModelConfig,
    inference: InferenceConfig,
    prompt_override: str | None,
) -> str:
    prompt = (
        prompt_override
        if prompt_override is not None
        else model_config.prompt or inference.prompt
    )
    if not prompt.strip():
        raise InferenceConfigError("the effective prompt must be non-empty")
    return prompt.strip()


def _run_model(
    *,
    alias: str,
    model_config: ModelConfig,
    inference: InferenceConfig,
    image_path: Path,
    output_dir: Path,
    prompt_override: str | None = None,
) -> dict[str, Any]:
    return _run_models(
        alias=alias,
        model_config=model_config,
        inference=inference,
        image_paths=(image_path,),
        output_dir=output_dir,
        prompt_override=prompt_override,
    )[0]


def _run_models(
    *,
    alias: str,
    model_config: ModelConfig,
    inference: InferenceConfig,
    image_paths: tuple[Path, ...],
    output_dir: Path,
    prompt_override: str | None = None,
) -> list[dict[str, Any]]:
    validate_inference_runtime(model_config)
    return _run_vllm_models(
        alias=alias,
        model_config=model_config,
        inference=inference,
        image_paths=image_paths,
        output_dir=output_dir,
        prompt_override=prompt_override,
    )


def validate_inference_runtime(model_config: ModelConfig) -> None:
    """Fail early unless the unified, pinned vLLM runtime is installed."""

    try:
        vllm_version = version("vllm")
    except PackageNotFoundError as exc:
        raise RuntimeError("vLLM inference dependencies are missing; run `uv sync`") from exc
    if vllm_version != "0.28.0":
        raise RuntimeError(
            f"this project requires vLLM 0.28.0, but {vllm_version} is installed; "
            "run `uv sync`"
        )


def _check_accelerator(torch: Any, inference: InferenceConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("OCR inference requires an NVIDIA CUDA GPU")


def _base_result(
    *,
    alias: str,
    model_config: ModelConfig,
    inference: InferenceConfig,
    image_path: Path,
    text_path: Path,
    prompt: str,
    load_seconds: float,
    infer_seconds: float,
    peak_gpu_memory_gib: float,
) -> dict[str, Any]:
    return {
        "model": alias,
        "backend": "vllm",
        "model_family": model_config.backend,
        "model_id": model_config.model_id,
        "image": str(image_path),
        "output": str(text_path),
        "load_seconds": round(load_seconds, 3),
        "inference_seconds": round(infer_seconds, 3),
        "peak_gpu_memory_gib": round(peak_gpu_memory_gib, 3),
        "revision": model_config.revision,
        "prompt": prompt,
        "dtype": inference.dtype,
        "physical_gpu": inference.gpu,
        "max_new_tokens": inference.max_new_tokens,
        "max_model_len": model_config.max_model_len,
        "enforce_eager": inference.enforce_eager,
        "gpu_memory_utilization": inference.gpu_memory_utilization,
        "vllm_version": version("vllm"),
    }


def _deepseek_crop_grid(
    width: int,
    height: int,
    *,
    image_size: int,
    max_crops: int,
) -> tuple[int, int]:
    if width <= image_size and height <= image_size:
        return (1, 1)
    ratios = {
        (columns, rows)
        for area in range(2, max_crops + 1)
        for columns in range(1, area + 1)
        for rows in range(1, area + 1)
        if 2 <= columns * rows <= max_crops
    }
    candidates = sorted(ratios, key=lambda ratio: ratio[0] * ratio[1])
    aspect = width / height
    image_area = width * height
    best = (1, 1)
    best_difference = float("inf")
    for candidate in candidates:
        difference = abs(aspect - candidate[0] / candidate[1])
        if difference < best_difference:
            best_difference = difference
            best = candidate
        elif difference == best_difference:
            candidate_area = image_size * image_size * candidate[0] * candidate[1]
            if image_area > 0.5 * candidate_area:
                best = candidate
    return best


def count_deepseek_image_tokens(image_path: Path, model_config: ModelConfig) -> int:
    """Count visual embeddings produced by a configured DeepSeek OCR encoder."""

    from PIL import Image

    if model_config.base_size is None or model_config.image_size is None:
        raise InferenceConfigError("DeepSeek model is missing base_size or image_size")
    with Image.open(image_path) as image:
        width, height = image.size
    patch_size, downsample_ratio = 16, 4
    base_queries = (
        model_config.base_size // patch_size + downsample_ratio - 1
    ) // downsample_ratio
    crop_queries = (
        model_config.image_size // patch_size + downsample_ratio - 1
    ) // downsample_ratio
    is_v2 = "ocr-2" in model_config.model_id.lower()
    global_tokens = base_queries**2 if is_v2 else base_queries * (base_queries + 1)
    local_tokens = 0
    if model_config.crop_mode and (
        width > model_config.image_size or height > model_config.image_size
    ):
        columns, rows = _deepseek_crop_grid(
            width,
            height,
            image_size=model_config.image_size,
            max_crops=model_config.max_crops or 1,
        )
        if is_v2:
            local_tokens = columns * rows * crop_queries**2
        else:
            local_tokens = rows * crop_queries * (columns * crop_queries + 1)
    # Both native vLLM implementations append one learned view-separator token.
    return local_tokens + global_tokens + 1


def _deepseek_mm_processor_kwargs(model_config: ModelConfig) -> dict[str, Any]:
    if model_config.base_size is None or model_config.image_size is None:
        raise InferenceConfigError("DeepSeek model is missing image processor settings")
    return {
        "base_size": model_config.base_size,
        "image_size": model_config.image_size,
        "crop_mode": model_config.crop_mode,
        "max_crops": model_config.max_crops,
        "strategy": "v2" if "ocr-2" in model_config.model_id.lower() else "v1",
    }


def _configure_deepseek_vllm(model_config: ModelConfig) -> None:
    """Keep vLLM's native model constants aligned with processor overrides."""

    if model_config.base_size is None or model_config.image_size is None:
        raise InferenceConfigError("DeepSeek model is missing image processor settings")

    from vllm.model_executor.models import deepseek_ocr as v1_model
    from vllm.transformers_utils.processors import deepseek_ocr as processor

    processor.BASE_SIZE = model_config.base_size
    processor.IMAGE_SIZE = model_config.image_size
    processor.CROP_MODE = model_config.crop_mode

    if "ocr-2" in model_config.model_id.lower():
        from vllm.model_executor.models import deepseek_ocr2 as v2_model

        v2_model.BASE_SIZE = model_config.base_size
        v2_model.IMAGE_SIZE = model_config.image_size
        v2_model.CROP_MODE = model_config.crop_mode
    else:
        v1_model.BASE_SIZE = model_config.base_size
        v1_model.IMAGE_SIZE = model_config.image_size
        v1_model.CROP_MODE = model_config.crop_mode


def _shutdown_vllm(llm: Any) -> None:
    """Release an in-process vLLM engine so another model can use the GPU."""

    engine = getattr(llm, "llm_engine", None)
    engine_core = getattr(engine, "engine_core", None)
    if engine_core is not None:
        engine_core.shutdown()


def _run_vllm_models(
    *,
    alias: str,
    model_config: ModelConfig,
    inference: InferenceConfig,
    image_paths: tuple[Path, ...],
    output_dir: Path,
    prompt_override: str | None,
) -> list[dict[str, Any]]:
    torch, Image, LLM, SamplingParams = _import_vllm_stack()
    _check_accelerator(torch, inference)
    prompt = _effective_prompt(model_config, inference, prompt_override)
    is_deepseek = model_config.backend == "deepseek"
    if is_deepseek and prompt.count("<image>") != 1:
        raise InferenceConfigError("a DeepSeek prompt must contain exactly one <image>")

    engine_options: dict[str, Any] = {
        "model": model_config.model_id,
        "revision": model_config.revision,
        "dtype": inference.dtype,
        "trust_remote_code": inference.trust_remote_code,
        "tensor_parallel_size": 1,
        "max_model_len": model_config.max_model_len,
        "max_num_seqs": 1,
        "limit_mm_per_prompt": {"image": 1},
        "enforce_eager": inference.enforce_eager,
        "gpu_memory_utilization": inference.gpu_memory_utilization,
    }
    if is_deepseek:
        from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

        _configure_deepseek_vllm(model_config)
        engine_options["mm_processor_kwargs"] = _deepseek_mm_processor_kwargs(model_config)
        engine_options["logits_processors"] = [NGramPerReqLogitsProcessor]
    else:
        # This only selects the current Transformers behavior for video inputs;
        # setting it explicitly also suppresses an irrelevant image-only warning.
        engine_options["mm_processor_kwargs"] = {"cap_pixels_per_frame": False}

    print(f"[{alias}] Loading {model_config.model_id} ...", file=sys.stderr, flush=True)
    started = time.perf_counter()
    llm = LLM(**engine_options)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    tokenizer = llm.get_tokenizer()
    image_token = "<image>" if is_deepseek else "<|image_pad|>"
    image_token_id = tokenizer.convert_tokens_to_ids(image_token)
    sampling_options: dict[str, Any] = {
        "temperature": 0.0,
        "max_tokens": inference.max_new_tokens,
        "skip_special_tokens": not is_deepseek,
    }
    if is_deepseek:
        sampling_options["extra_args"] = {
            # DeepSeek's official vLLM recipes use 30 for OCR v1 and 20 for OCR v2.
            "ngram_size": 20 if "ocr-2" in model_config.model_id.lower() else 30,
            "window_size": 90,
            "whitelist_token_ids": {128821, 128822},
        }
    sampling_params = SamplingParams(**sampling_options)

    results: list[dict[str, Any]] = []
    try:
        for index, image_path in enumerate(image_paths):
            torch.cuda.reset_peak_memory_stats()
            print(f"[{alias}] Running OCR on {image_path} ...", file=sys.stderr, flush=True)
            infer_started = time.perf_counter()
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
                if is_deepseek:
                    request = {
                        "prompt": prompt,
                        "multi_modal_data": {"image": rgb_image},
                    }
                    request_output = llm.generate(
                        request, sampling_params, use_tqdm=False
                    )[0]
                else:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_pil", "image_pil": rgb_image},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                    request_output = llm.chat(
                        messages,
                        sampling_params,
                        use_tqdm=False,
                        chat_template_kwargs={
                            "enable_thinking": inference.enable_thinking
                        },
                    )[0]
            torch.cuda.synchronize()
            infer_seconds = time.perf_counter() - infer_started

            completion = request_output.outputs[0]
            text = completion.text.rstrip()
            text_path = output_dir / f"{image_path.stem}.{alias}.md"
            text_path.write_text(text + "\n", encoding="utf-8")
            prompt_token_ids = list(request_output.prompt_token_ids)
            result = _base_result(
                alias=alias,
                model_config=model_config,
                inference=inference,
                image_path=image_path,
                text_path=text_path,
                prompt=prompt,
                load_seconds=load_seconds if index == 0 else 0.0,
                infer_seconds=infer_seconds,
                peak_gpu_memory_gib=torch.cuda.max_memory_allocated() / 1024**3,
            )
            result.update(
                {
                    "enable_thinking": inference.enable_thinking,
                    "input_tokens": len(prompt_token_ids),
                    "prompt_tokens": count_tokens(
                        prompt.replace("<image>", "") if is_deepseek else prompt,
                        tokenizer,
                    ),
                    "image_tokens": prompt_token_ids.count(image_token_id),
                    "output_tokens": len(completion.token_ids),
                    "finish_reason": completion.finish_reason,
                    "stop_reason": completion.stop_reason,
                }
            )
            if is_deepseek:
                result.update(
                    {
                        "base_size": model_config.base_size,
                        "image_size": model_config.image_size,
                        "crop_mode": model_config.crop_mode,
                        "max_crops": model_config.max_crops,
                    }
                )
            results.append(result)
    finally:
        _shutdown_vllm(llm)
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    return results


def run_ocr_image(
    *,
    alias: str,
    model_config: ModelConfig,
    inference: InferenceConfig,
    image_path: str | Path,
    output_dir: str | Path,
    prompt_override: str | None = None,
) -> dict[str, Any]:
    """Run one configured OCR model on one image and persist its text result."""

    return run_ocr_images(
        alias=alias,
        model_config=model_config,
        inference=inference,
        image_paths=(image_path,),
        output_dir=output_dir,
        prompt_override=prompt_override,
    )[0]


def run_ocr_images(
    *,
    alias: str,
    model_config: ModelConfig,
    inference: InferenceConfig,
    image_paths: tuple[str | Path, ...],
    output_dir: str | Path,
    prompt_override: str | None = None,
) -> list[dict[str, Any]]:
    """Run one loaded OCR model over multiple images and persist every result."""

    if not image_paths:
        raise ValueError("image_paths must contain at least one image")
    images = tuple(Path(path).expanduser().resolve() for path in image_paths)
    for image in images:
        if not image.is_file():
            raise FileNotFoundError(f"image does not exist: {image}")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    return _run_models(
        alias=alias,
        model_config=model_config,
        inference=inference,
        image_paths=images,
        output_dir=destination,
        prompt_override=prompt_override,
    )


def build_parser(
    default_config: Path = Path("configs/inference.yaml"),
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a configured Qwen3.5 or DeepSeek OCR model on one document image."
    )
    parser.add_argument("image", type=Path, help="PNG/JPEG document image")
    parser.add_argument(
        "--model",
        default="all",
        help="model alias from the YAML or 'all' (default: all)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help=f"inference YAML (default: {default_config})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/inference"),
        help="result directory (default: outputs/inference)",
    )
    parser.add_argument("--prompt", help="override the configured prompt")
    parser.add_argument("--gpu", type=int, help="override inference.gpu")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="override inference.max_new_tokens",
    )
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable or disable Qwen thinking mode",
    )
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
    return parser


def main(
    argv: list[str] | None = None,
    *,
    default_config: Path = Path("configs/inference.yaml"),
) -> int:
    args = build_parser(default_config).parse_args(argv)
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        print(f"ocr-infer: error: image does not exist: {image_path}", file=sys.stderr)
        return 2
    try:
        inference, models = load_inference_config(args.config)
        if args.max_new_tokens is not None:
            _positive_int(args.max_new_tokens, "--max-new-tokens")
        if args.gpu_memory_utilization is not None:
            _memory_utilization(args.gpu_memory_utilization, "--gpu-memory-utilization")
        if args.gpu is not None and args.gpu < 0:
            raise InferenceConfigError("--gpu must be a non-negative integer")
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
        selected = list(models) if args.model == "all" else [args.model]
        missing = [alias for alias in selected if alias not in models]
        if missing:
            choices = ", ".join(models)
            raise InferenceConfigError(
                f"unknown model {missing[0]!r}; configured aliases: {choices}"
            )

        # Restrict visibility before importing torch so the selected physical GPU is cuda:0.
        if inference.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(inference.gpu)
        # Offline inference runs in-process so metrics and deterministic cleanup
        # work consistently across all three model families.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        results = [
            _run_model(
                alias=alias,
                model_config=models[alias],
                inference=inference,
                image_path=image_path,
                output_dir=output_dir,
                prompt_override=args.prompt,
            )
            for alias in selected
        ]
        summary_path = output_dir / f"{image_path.stem}.inference.json"
        summary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    except (InferenceConfigError, RuntimeError, OSError, ValueError) as exc:
        print(f"ocr-infer: error: {exc}", file=sys.stderr)
        return 2

    for result in results:
        print(result["output"])
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
