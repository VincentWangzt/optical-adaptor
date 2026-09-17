from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Bin(StrictConfig):
    name: str
    minimum: int = Field(ge=0)
    maximum: int | None


def bin_name(value: int, bins: list[Bin]) -> str:
    matches = [
        b.name for b in bins if value >= b.minimum and (b.maximum is None or value <= b.maximum)
    ]
    if len(matches) != 1:
        raise ValueError(f"Value {value} must belong to exactly one bin, got {matches}")
    return matches[0]


class Source(StrictConfig):
    name: str
    kind: Literal["stack", "swe"]
    dataset_id: str
    revision: str
    data_files: str
    split: str
    thinking: bool
    max_records: int | None


class PrepareConfig(StrictConfig):
    output_dir: str
    sources: list[Source]
    eval_fraction: float = Field(gt=0, lt=1)
    tasks: list[Literal["reconstruction", "continuation", "next_action"]] = Field(min_length=1)
    max_images: int | None = Field(gt=0)
    min_observation_chars: int = Field(gt=0)
    lines_per_image: int = Field(gt=0)
    stack_image_counts: list[int]
    continuation_lines: int = Field(gt=0)
    window_turns: list[int]
    max_actions_per_trajectory: int | None
    reconstruction_observations_per_trajectory: int = Field(ge=0)
    image_bins: list[Bin]
    turn_bins: list[Bin]
    instruction_roles: list[Literal["system", "user"]]
    image_roles: list[Literal["user", "tool"]]
    reconstruction_prompts: list[str]
    continuation_prompts: list[str]


class FilterConfig(StrictConfig):
    sources: list[str] | None
    tasks: list[str] | None
    image_bins: list[str] | None
    turn_bins: list[str] | None
    min_images: int = Field(ge=1)
    max_images: int | None


class ProcessingConfig(StrictConfig):
    assistant_loss: Literal["all", "last"]
    max_teacher_tokens: int = Field(gt=0)
    max_student_tokens: int = Field(gt=0)
    overlength: Literal["error", "drop"]
    vision_start: str
    vision_end: str
    preserve_all_reasoning: bool
    image_microbatch_size: int = Field(gt=0)


class DataConfig(StrictConfig):
    train_filter: FilterConfig
    eval_filter: FilterConfig
    weights: dict[str, float]
    samples_per_epoch: int | None
    eval_samples: dict[str, int]
    num_workers: int = Field(ge=0)


class EvaluationConfig(StrictConfig):
    generation_every: int = Field(gt=0)
    generation_samples: dict[str, int]
    max_new_tokens: dict[str, int]
    tasks: list[str]


class OpticalConfig(StrictConfig):
    deterministic: bool
    render_config: str
    llm: dict
    vision: dict
    adapter: dict
    prepare: PrepareConfig
    processing: ProcessingConfig
    data: DataConfig
    evaluation: EvaluationConfig

    @model_validator(mode="after")
    def check_contract(self):
        if self.adapter["input_dim"] != self.vision["output_dim"]:
            raise ValueError("Adapter input_dim must equal vision output_dim")
        if self.processing.vision_start == self.processing.vision_end:
            raise ValueError("Visual boundaries must be distinct")
        if not self.processing.preserve_all_reasoning:
            raise ValueError("Dropping historical reasoning is unsupported in this training recipe")
        names = [source.name for source in self.prepare.sources]
        if len(names) != len(set(names)):
            raise ValueError("Source names must be unique")
        return self


def read_config(path: str | Path) -> tuple[dict, OpticalConfig]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return raw, OpticalConfig.model_validate(raw["optical"])


def fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def preparation_fingerprint(optical: OpticalConfig, seed: int) -> str:
    prepare = optical.prepare.model_dump(exclude={"output_dir"})
    return fingerprint(
        {
            "preparation_version": 3,
            "seed": seed,
            "prepare": prepare,
            "render": Path(optical.render_config).read_text(encoding="utf-8"),
        }
    )
