from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml
from transformers import get_constant_schedule_with_warmup

from optical_adaptor.training.config import fingerprint
from optical_adaptor.training.resume import (
    checkpoint_warmup,
    resolve_continuation,
    validate_checkpoint,
    validate_restored_state,
)


def identities():
    config = yaml.safe_load(Path("configs/training.yaml").read_text())
    source = {
        "config": copy.deepcopy(config),
        "data": "data-hash",
        "kind": "mlp",
        "world_size": 2,
        "mode": "train",
        "microbatch": 2,
        "total_updates": 939,
        "train_records": "records-hash",
        "torch": "runtime-version",
    }
    source["config"]["evaluation"].pop("sampling")
    metadata = {
        "identity": fingerprint(source),
        "resolved": source,
        "kind": "mlp",
        "step": 939,
        "epoch": 3,
        "update": 0,
        "wandb_id": "parent",
    }
    current = copy.deepcopy(source)
    current["config"] = config
    current["config"]["training"]["epochs"] = 10
    current["total_updates"] = 3130
    return metadata, current


def extend(metadata, current, *, allow_sampling_change=True):
    return resolve_continuation(
        current,
        metadata,
        Path("parent/final"),
        extend=True,
        allow_sampling_change=allow_sampling_change,
    )


def test_extension_and_exact_resume_preserve_original_schedule():
    metadata, current = identities()
    extended = extend(metadata, current)
    assert checkpoint_warmup(extended) == 29
    assert extended["continuation"]["source_step"] == 939
    assert extended["continuation"]["sampling_changed"]
    assert current["total_updates"] - metadata["step"] == 2191
    saved = {
        **metadata,
        "identity": fingerprint(extended),
        "resolved": extended,
        "step": 1000,
        "epoch": 3,
        "update": 61,
        "wandb_id": "child",
    }
    resumed = resolve_continuation(
        current, saved, Path("child/step-001000"), extend=False, allow_sampling_change=False
    )
    assert resumed == extended
    assert checkpoint_warmup(resumed) == 29
    with pytest.raises(ValueError, match="completed"):
        extend(saved, current)


def test_sampling_change_is_explicit_and_exact_resume_stays_strict():
    metadata, current = identities()
    with pytest.raises(ValueError, match="unapproved"):
        extend(metadata, current, allow_sampling_change=False)
    with pytest.raises(ValueError, match="mismatch"):
        resolve_continuation(
            current, metadata, Path("parent/final"), extend=False, allow_sampling_change=False
        )


@pytest.mark.parametrize("changed", ["data", "world_size", "microbatch", "train_records", "torch"])
def test_extension_rejects_changed_training_identity(changed):
    metadata, current = identities()
    current[changed] = "changed"
    with pytest.raises(ValueError, match="changed"):
        extend(metadata, current)


@pytest.mark.parametrize("section,key", [("training", "lr"), ("evaluation", "max_new_tokens")])
def test_extension_rejects_other_config_changes(section, key):
    metadata, current = identities()
    current["config"][section][key] *= 2
    with pytest.raises(ValueError, match="changed"):
        extend(metadata, current)


def test_corrupt_progress_is_rejected():
    metadata, _ = identities()
    metadata["update"] = 1
    with pytest.raises(ValueError, match="cursor"):
        validate_checkpoint(metadata)


def test_restored_adam_and_lr_continue_without_restarting_warmup():
    metadata, current = identities()
    identity = extend(metadata, current)
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=2e-4)
    scheduler = get_constant_schedule_with_warmup(
        optimizer, checkpoint_warmup(metadata["resolved"])
    )
    for _ in range(939):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
    optimizer_state, scheduler_state = copy.deepcopy(optimizer.state_dict()), scheduler.state_dict()
    restored = torch.optim.AdamW([parameter], lr=2e-4)
    continued = get_constant_schedule_with_warmup(restored, checkpoint_warmup(identity))
    restored.load_state_dict(optimizer_state)
    continued.load_state_dict(scheduler_state)
    validate_restored_state(restored, continued, 939)
    assert continued.get_last_lr() == [2e-4]
    restored.step()
    continued.step()
    validate_restored_state(restored, continued, 940)
    assert continued.get_last_lr() == [2e-4]
    with pytest.raises(ValueError, match="scheduler"):
        validate_restored_state(restored, continued, 0)
