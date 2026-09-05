"""Run explicitly under two-rank Accelerate; real Qwen/cache checkpoint replay."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from transformers import get_constant_schedule_with_warmup

from optical_adaptor.training.cache import TensorCache
from optical_adaptor.training.config import load_credentials, load_pipeline, write_json
from optical_adaptor.training.data import load_manifest
from optical_adaptor.training.models import FrozenQwen, build_adapter
from optical_adaptor.training.sampling import epoch_windows
from optical_adaptor.training.trainer import run_update


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    accelerator = Accelerator(
        mixed_precision="bf16",
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[DistributedDataParallelKwargs(broadcast_buffers=False)],
    )
    if accelerator.device.type != "cuda" or accelerator.num_processes != 2:
        raise ValueError("run this validation on two CUDA ranks with Accelerate")
    if args.output.exists():
        raise FileExistsError(args.output)
    pipeline = load_pipeline("configs/training.yaml")
    load_credentials(pipeline, wandb=False)
    set_seed(pipeline.config.seed)
    records = load_manifest(pipeline)
    cache = TensorCache(pipeline, records)
    rows = [row for row in records if row["split"] == "train"][:8]
    if len({len(row["visual_ids"]) for row in rows}) < 2:
        raise ValueError("resume validation needs unequal reconstruction lengths")
    qwen = FrozenQwen(pipeline, accelerator.device)
    adapter = build_adapter("mlp", pipeline.config.adapter)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=2e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01
    )
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=29)
    adapter, optimizer = accelerator.prepare(adapter, optimizer)
    accelerator.register_for_checkpointing(scheduler)

    def update(epoch):
        window = next(
            epoch_windows(
                rows,
                seed=pipeline.config.seed,
                epoch=epoch,
                global_pairs=32,
                rank=accelerator.process_index,
                world_size=accelerator.num_processes,
            )
        )
        statistics, _ = run_update(
            accelerator,
            adapter,
            qwen,
            cache,
            optimizer,
            window,
            microbatch_size=1,
            chunk_size=128,
            max_grad_norm=1.0,
        )
        scheduler.step()
        return (
            statistics,
            [row["record_id"] for row in window.continuation],
            [row["record_id"] for row in window.reconstruction],
        )

    update(0)
    saved_weights = [parameter.detach().clone() for parameter in adapter.parameters()]
    saved_moments = [state["exp_avg"].clone() for state in optimizer.state.values()]
    saved_lr = scheduler.get_last_lr()
    accelerator.save_state(str(args.output / "after-one"))
    expected_statistics, expected_c, expected_r = update(1)
    expected_weights = [parameter.detach().clone() for parameter in adapter.parameters()]
    expected_moments = [state["exp_avg"].clone() for state in optimizer.state.values()]
    expected_random = torch.rand(4, device=accelerator.device)
    accelerator.load_state(str(args.output / "after-one"))
    assert scheduler.get_last_lr() == saved_lr
    for parameter, saved in zip(adapter.parameters(), saved_weights, strict=True):
        torch.testing.assert_close(parameter, saved, rtol=0, atol=0)
    for state, saved in zip(optimizer.state.values(), saved_moments, strict=True):
        torch.testing.assert_close(state["exp_avg"], saved, rtol=0, atol=0)
    actual_statistics, actual_c, actual_r = update(1)
    diagnostic = {
        "expected_statistics": expected_statistics.tolist(),
        "actual_statistics": actual_statistics.tolist(),
        "weights_max_abs": [
            float((parameter - expected).abs().max())
            for parameter, expected in zip(adapter.parameters(), expected_weights, strict=True)
        ],
        "moment_relative_l2": [
            float((state["exp_avg"] - expected).norm() / expected.norm().clamp_min(1e-12))
            for state, expected in zip(optimizer.state.values(), expected_moments, strict=True)
        ],
    }
    write_json(args.output / f"diagnostic-{accelerator.process_index}.json", diagnostic)
    assert actual_c == expected_c and actual_r == expected_r
    assert scheduler.last_epoch == 2
    assert all(parameter.dtype == torch.float32 for parameter in adapter.parameters())
    assert all(parameter.grad is None for parameter in qwen.model.parameters())
    error = 0.0
    for parameter, expected in zip(adapter.parameters(), expected_weights, strict=True):
        torch.testing.assert_close(parameter, expected, rtol=0, atol=2e-6)
        error = max(error, float((parameter - expected).abs().max()))
    for state, expected in zip(optimizer.state.values(), expected_moments, strict=True):
        assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
        torch.testing.assert_close(state["exp_avg"], expected, rtol=1e-4, atol=1e-7)
    torch.testing.assert_close(actual_statistics, expected_statistics, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        torch.rand(4, device=accelerator.device), expected_random, rtol=0, atol=0
    )
    result = {
        "rank": accelerator.process_index,
        "weight_max_abs": error,
        "statistics_max_abs": float((actual_statistics - expected_statistics).abs().max()),
        "weight_atol": 2e-6,
        "statistics_rtol": 1e-4,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    write_json(args.output / f"rank-{accelerator.process_index}.json", result)
    accelerator.print("GPU_DDP_CHECKPOINT_REPLAY_OK", result)
    accelerator.end_training()


if __name__ == "__main__":
    main()
