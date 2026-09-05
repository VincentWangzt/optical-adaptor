"""Run remotely with torchrun --nproc_per_node=2; Accelerate uses CPU/Gloo here."""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from accelerate import Accelerator
from torch import nn
from transformers import get_constant_schedule_with_warmup

from optical_adaptor.training.objectives import task_loss
from optical_adaptor.training.sampling import epoch_windows
from optical_adaptor.training.trainer import run_update


class TinyCache:
    def batch(self, records, kind, device):
        return torch.stack([row[kind] for row in records]).to(device)


class TinyQwen:
    def __init__(self):
        self.model = SimpleNamespace(lm_head=nn.Linear(3, 11, bias=False).requires_grad_(False))

    def student_hidden(self, records, adapted, task):
        targets, masks, values = [], [], []
        for row, value in zip(records, adapted, strict=True):
            target = row["continuation_ids"] if task == "continuation" else row["visual_ids"] + [10]
            targets.extend(target)
            masks.extend([True] * (len(target) - 1) + [task == "continuation"])
            values.append(value.mean(0).expand(len(target), -1))
        return torch.cat(values), torch.tensor(targets), torch.tensor(masks)


def validate_distributed(output: Path):
    accelerator = Accelerator(cpu=True)
    assert accelerator.num_processes == 2
    torch.manual_seed(17)
    qwen, cache = TinyQwen(), TinyCache()
    adapter = nn.Linear(4, 3)
    reference = copy.deepcopy(adapter)
    rows = [
        {
            "encoder": torch.randn(2, 4),
            "teacher": torch.randn(3, 3),
            "continuation_ids": [1, 2, 3],
            "visual_ids": [i % 9] * (i + 1),
        }
        for i in range(8)
    ]
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.002)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.002)
    scheduler = get_constant_schedule_with_warmup(optimizer, 1)
    reference_scheduler = get_constant_schedule_with_warmup(reference_optimizer, 1)
    adapter, optimizer = accelerator.prepare(adapter, optimizer)
    accelerator.register_for_checkpointing(scheduler)
    windows = list(
        epoch_windows(
            rows, seed=42, epoch=0, global_pairs=6, rank=accelerator.process_index, world_size=2
        )
    )
    full_windows = list(epoch_windows(rows, seed=42, epoch=0, global_pairs=6, rank=0, world_size=1))

    def update(window):
        statistics, _ = run_update(
            accelerator,
            adapter,
            qwen,
            cache,
            optimizer,
            window,
            microbatch_size=1,
            chunk_size=2,
            max_grad_norm=1000.0,
        )
        scheduler.step()
        return statistics

    def reference_update(window):
        reference_optimizer.zero_grad(set_to_none=True)
        for task in ("continuation", "reconstruction"):
            batch = getattr(window, task)
            hidden, targets, mask = qwen.student_hidden(
                batch, reference(cache.batch(batch, "encoder", "cpu")), task
            )
            teacher = cache.batch(batch, "teacher", "cpu") if task == "continuation" else None
            result = task_loss(
                hidden, targets, mask, head=qwen.model.lm_head, teacher=teacher, chunk_size=99
            )
            (0.5 * result.loss / getattr(window, f"{task}_tokens")).backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1000.0)
        reference_optimizer.step()
        reference_scheduler.step()

    update(windows[0])
    reference_update(full_windows[0])
    # Compare gradients themselves: AdamW parameter deltas alone can hide a uniform
    # scaling bug because its first/second moments normalize that scale away.
    reference_parameters = dict(reference.named_parameters())
    for name, value in accelerator.unwrap_model(adapter).named_parameters():
        torch.testing.assert_close(
            value.grad, reference_parameters[name].grad, rtol=1e-5, atol=1e-7
        )
    accelerator.save_state(str(output), safe_serialization=True)
    next_rng = (random.random(), np.random.rand(), torch.rand(4))
    expected_statistics = update(windows[1])
    reference_update(full_windows[1])
    expected = copy.deepcopy(accelerator.unwrap_model(adapter).state_dict())
    for name, value in expected.items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=1e-5, atol=1e-7)
    accelerator.load_state(str(output))
    assert random.random() == next_rng[0] and np.random.rand() == next_rng[1]
    torch.testing.assert_close(torch.rand(4), next_rng[2], rtol=0, atol=0)
    actual_statistics = update(windows[1])
    torch.testing.assert_close(actual_statistics, expected_statistics, rtol=0, atol=0)
    for name, value in accelerator.unwrap_model(adapter).state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    assert scheduler.last_epoch == 2 and scheduler.get_last_lr() == [0.002]
    flat = torch.cat([value.flatten() for value in adapter.parameters()])
    gathered = accelerator.gather(flat).reshape(2, -1)
    torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
    if accelerator.is_main_process:
        print("DDP_WEIGHTING_AND_EXACT_RESUME_OK", flush=True)
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    validate_distributed(parser.parse_args().output)
