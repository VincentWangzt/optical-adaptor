from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

from optical_adaptor.renderer import RenderConfig, load_render_config

PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
Revision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
SPLITS = ("train", "front_continuation", "middle_continuation", "reconstruction")
AdapterKind = Literal["mlp", "transformer"]


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ModelsConfig(StrictConfig):
    encoder_id: str
    encoder_revision: Revision
    qwen_id: str
    qwen_revision: Revision
    encoder_tensor_count: PositiveInt
    image_size: PositiveInt


class DataConfig(StrictConfig):
    dataset_id: str
    revision: Revision
    split_sizes: dict[str, PositiveInt]
    min_lines: PositiveInt
    max_lines: PositiveInt
    max_visual_tokens: PositiveInt
    max_display_lines: PositiveInt
    continuation_tokens: PositiveInt
    prefix_tokens: PositiveInt
    shard_rows: PositiveInt
    debug_images: Annotated[int, Field(ge=0)]


class AdapterConfig(StrictConfig):
    input_dim: PositiveInt
    output_dim: PositiveInt
    sequence_length: PositiveInt
    transformer_layers: PositiveInt
    transformer_heads: PositiveInt
    transformer_ffn_dim: PositiveInt
    dropout: Annotated[float, Field(ge=0, lt=1)]


class TrainingConfig(StrictConfig):
    epochs: PositiveInt
    global_pairs: PositiveInt
    lr: PositiveFloat
    betas: Annotated[list[Annotated[float, Field(ge=0, lt=1)]], Field(min_length=2, max_length=2)]
    weight_decay: Annotated[float, Field(ge=0)]
    epsilon: PositiveFloat
    warmup_ratio: Annotated[float, Field(ge=0, lt=1)]
    max_grad_norm: PositiveFloat
    loss_chunk_tokens: PositiveInt
    microbatch_candidates: list[PositiveInt]
    memory_limit_gib: PositiveFloat
    profile_warmup_updates: PositiveInt
    profile_measured_updates: PositiveInt
    checkpoint_every: PositiveInt
    keep_periodic: PositiveInt


class EvaluationConfig(StrictConfig):
    every: PositiveInt
    generation_every: PositiveInt
    generation_subset: PositiveInt
    max_new_tokens: PositiveInt
    instruction: Annotated[str, Field(min_length=1)]


class LoggingConfig(StrictConfig):
    entity: Annotated[str, Field(min_length=1)]
    project: Annotated[str, Field(min_length=1)]


class PipelineConfig(StrictConfig):
    seed: Annotated[int, Field(ge=0)]
    output_dir: str
    render_config: str
    models: ModelsConfig
    data: DataConfig
    adapter: AdapterConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig

    @model_validator(mode="after")
    def consistent(self):
        if set(self.data.split_sizes) != set(SPLITS):
            raise ValueError(f"split_sizes must contain exactly {SPLITS}")
        if self.data.min_lines > self.data.max_lines:
            raise ValueError("min_lines exceeds max_lines")
        if self.adapter.input_dim % self.adapter.transformer_heads:
            raise ValueError("input_dim must be divisible by transformer_heads")
        if (
            (self.adapter.input_dim, self.adapter.output_dim, self.adapter.sequence_length)
            != (1280, 2560, 111)
            or self.models.image_size != 640
            or self.adapter.dropout != 0
        ):
            raise ValueError("v1 requires 640px images, 111x1280 -> 111x2560, and zero dropout")
        if not self.training.microbatch_candidates:
            raise ValueError("microbatch_candidates must not be empty")
        if self.evaluation.generation_subset > self.data.split_sizes["reconstruction"]:
            raise ValueError("generation_subset exceeds reconstruction split")
        return self

    @property
    def total_updates(self) -> int:
        return self.training.epochs * math.ceil(
            self.data.split_sizes["train"] / self.training.global_pairs
        )

    @property
    def warmup_updates(self) -> int:
        return math.ceil(self.total_updates * self.training.warmup_ratio)


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class Pipeline:
    config: PipelineConfig
    repo: Path
    output: Path
    render: RenderConfig
    data_fingerprint: str

    @property
    def manifest(self) -> Path:
        return self.output / "data" / "manifest.parquet"


def load_pipeline(path: str | Path) -> Pipeline:
    path = Path(path).resolve()
    config = PipelineConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    repo = path.parent.parent
    render_path = repo / config.render_config
    render = load_render_config(render_path)
    font_paths = [render.text.font, *render.text.fallback_fonts]
    if not render.text.font or render.pages.resolution != (1280, None):
        raise ValueError("training rendering requires an explicit font and [1280, auto]")
    data_hash = fingerprint(
        {
            "schema": 1,
            "seed": config.seed,
            "data": config.data.model_dump(),
            "models": config.models.model_dump(),
            "render": yaml.safe_load(render_path.read_text(encoding="utf-8")),
            "fonts": [file_sha256(Path(item)) for item in font_paths],
        }
    )
    return Pipeline(config, repo, repo / config.output_dir, render, data_hash)


def load_credentials(pipeline: Pipeline, *, wandb: bool) -> None:
    load_dotenv(pipeline.repo / ".env", override=False)
    keys = ["HF_TOKEN"] + (["WANDB_API_KEY"] if wandb else [])
    missing = [key for key in keys if not os.environ.get(key)]
    if missing:
        raise RuntimeError(f"missing remote credentials: {', '.join(missing)}")
