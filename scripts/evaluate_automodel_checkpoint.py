"""Evaluate an exported optical adapter without resuming optimizer updates.

Launch with ``uv run torchrun --standalone --nproc-per-node=N`` and select idle
devices with ``CUDA_VISIBLE_DEVICES``. The source run's config controls data,
model and teacher-forced evaluation; only generation quotas and batch size vary.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from nemo_automodel.components.config.loader import ConfigNode
from safetensors.torch import load_file

from optical_adaptor.automodel.config import OpticalConfig
from optical_adaptor.automodel.recipe import OpticalKDRecipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--generation-samples-per-slice", type=int, required=True)
    parser.add_argument("--generation-batch-size", type=int, required=True)
    args = parser.parse_args()
    if args.checkpoint_step < 1:
        raise ValueError("checkpoint-step must be positive")
    if args.generation_samples_per_slice < 1 or args.generation_batch_size < 1:
        raise ValueError("Generation sample and batch sizes must be positive")

    source = yaml.safe_load(args.source_config.read_text(encoding="utf-8"))
    exported = json.loads((args.adapter.parent / "optical-model.json").read_text())
    if exported["step"] != args.checkpoint_step:
        raise ValueError("Adapter export step does not match checkpoint-step")
    if exported["model"] != {key: source["optical"][key] for key in ("llm", "vision", "adapter")}:
        raise ValueError("Adapter export and source model configuration differ")

    rank = int(os.environ["RANK"])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if rank == 0 and any(output.iterdir()):
        raise FileExistsError(f"Evaluation output directory is occupied: {output}")
    source["checkpoint"].update(
        enabled=False, checkpoint_dir=str(output / "checkpoints"), restore_from=None
    )
    source["wandb"] = None
    evaluation = source["optical"]["evaluation"]
    evaluation["generation_samples"] = {"": args.generation_samples_per_slice}
    evaluation["generation_batch_size"] = args.generation_batch_size
    OpticalConfig.model_validate(source["optical"])
    if source["step_scheduler"]["max_steps"] != args.checkpoint_step:
        raise ValueError("Source training max_steps does not match checkpoint-step")
    if rank == 0:
        (output / "config.yaml").write_text(
            yaml.safe_dump(source, sort_keys=False), encoding="utf-8"
        )
        (output / "source.json").write_text(
            json.dumps(
                {
                    "source_config": str(args.source_config.resolve()),
                    "adapter": str(args.adapter.resolve()),
                    "checkpoint_step": args.checkpoint_step,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    started = time.perf_counter()
    trainer = OpticalKDRecipe(ConfigNode(source))
    trainer.setup()
    model = trainer.student_model()
    model.adapter.load_state_dict(load_file(args.adapter, device="cpu"), strict=True)
    if trainer.val_dataloader is None:
        raise ValueError("Source config has no validation data")
    device = trainer.dist_env.device
    torch.cuda.synchronize(device)
    setup_seconds = time.perf_counter() - started
    setup_peak_mib = torch.cuda.max_memory_reserved(device) / 2**20
    torch.cuda.reset_peak_memory_stats(device)

    # The source checkpoint was saved after its final step; no optimizer step is run.
    trainer.step_scheduler.step = args.checkpoint_step - 1
    evaluation_started = time.perf_counter()
    result = trainer._run_validation_epoch(trainer.val_dataloader)
    torch.cuda.synchronize(device)
    evaluation_seconds = time.perf_counter() - evaluation_started
    per_rank = {
        "rank": dist.get_rank(),
        "gpu": torch.cuda.current_device(),
        "setup_seconds": setup_seconds,
        "evaluation_seconds": evaluation_seconds,
        "setup_peak_reserved_mib": setup_peak_mib,
        "evaluation_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "evaluation_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
    }
    reports = [None] * dist.get_world_size()
    dist.all_gather_object(reports, per_rank)
    if trainer.dist_env.is_main:
        (output / "results.json").write_text(
            json.dumps(
                {
                    "metrics": {key: float(value) for key, value in result.metrics.items()},
                    "ranks": reports,
                    "wall_seconds": time.perf_counter() - started,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    trainer.metric_logger_train.close()
    trainer.metric_logger_valid.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
