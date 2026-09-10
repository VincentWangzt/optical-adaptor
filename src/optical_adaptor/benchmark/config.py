from __future__ import annotations

from pathlib import Path

import yaml

from optical_adaptor.training.config import (
    PositiveInt,
    Revision,
    StrictConfig,
    load_pipeline,
)


class BackendConfig(StrictConfig):
    max_model_len: PositiveInt
    max_num_seqs: PositiveInt
    max_num_batched_tokens: PositiveInt
    max_images: PositiveInt
    vision_batch_size: PositiveInt
    gpu_memory_utilization: float
    enforce_eager: bool


class BenchmarkConfig(StrictConfig):
    training_config: str
    output_dir: str
    checkpoint: str
    seed: int
    multi_image_counts: list[PositiveInt]
    long_source_lines: PositiveInt
    qa_source_lines_per_image: PositiveInt
    lcb_dataset: str
    lcb_revision: Revision
    prompt_token_limit: PositiveInt
    reconstruction_max_tokens: PositiveInt
    qa_max_tokens: PositiveInt
    batch_size: PositiveInt
    backend: BackendConfig
    reconstruction_instruction: str

    def data_settings(self) -> dict:
        return self.model_dump(
            include={
                "training_config",
                "seed",
                "multi_image_counts",
                "long_source_lines",
                "qa_source_lines_per_image",
                "lcb_dataset",
                "lcb_revision",
                "prompt_token_limit",
                "reconstruction_instruction",
            }
        )


def load_benchmark(path: Path):
    path = path.resolve()
    config = BenchmarkConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    pipeline = load_pipeline(path.parent.parent / config.training_config)
    return config, pipeline, pipeline.repo / config.output_dir
