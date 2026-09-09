from __future__ import annotations

import copy
import math
from pathlib import Path

from optical_adaptor.training.config import fingerprint


def checkpoint_warmup(resolved: dict) -> int:
    if "continuation" in resolved:
        return resolved["continuation"]["warmup_updates"]
    return math.ceil(resolved["total_updates"] * resolved["config"]["training"]["warmup_ratio"])


def validate_checkpoint(metadata: dict) -> None:
    resolved = metadata["resolved"]
    if metadata["identity"] != fingerprint(resolved) or metadata["kind"] != resolved["kind"]:
        raise ValueError("checkpoint metadata fingerprint/kind mismatch")
    config = resolved["config"]
    updates_per_epoch = math.ceil(
        config["data"]["split_sizes"]["train"] / config["training"]["global_pairs"]
    )
    if resolved["mode"] == "train" and (
        metadata["step"] != metadata["epoch"] * updates_per_epoch + metadata["update"]
        or not 0 <= metadata["update"] < updates_per_epoch
        or resolved["total_updates"] != config["training"]["epochs"] * updates_per_epoch
    ):
        raise ValueError("checkpoint epoch/update cursor is inconsistent")
    if not 0 <= metadata["step"] <= resolved["total_updates"]:
        raise ValueError("checkpoint step is outside its training budget")


def resolve_continuation(
    identity: dict,
    metadata: dict,
    checkpoint: Path,
    *,
    extend: bool,
    allow_sampling_change: bool,
) -> dict:
    """Validate an exact resume or a deliberate extension of a completed run."""
    validate_checkpoint(metadata)
    source = metadata["resolved"]
    current = copy.deepcopy(identity)
    if not extend:
        if "continuation" in source:
            current["continuation"] = copy.deepcopy(source["continuation"])
        if fingerprint(current) != metadata["identity"]:
            raise ValueError("checkpoint data/configuration/runtime/topology mismatch")
        return current
    if (
        current["mode"] != "train"
        or source["mode"] != "train"
        or metadata["step"] != source["total_updates"]
        or current["config"]["training"]["epochs"] <= source["config"]["training"]["epochs"]
    ):
        raise ValueError(
            "extension requires a completed training run and a larger total epoch count"
        )
    expected = copy.deepcopy(source)
    expected.pop("continuation", None)
    expected["total_updates"] = current["total_updates"]
    expected["config"]["training"]["epochs"] = current["config"]["training"]["epochs"]
    if allow_sampling_change:
        expected["config"]["evaluation"]["sampling"] = current["config"]["evaluation"]["sampling"]
    if expected != current:
        raise ValueError(
            "extension changed data, training, runtime, or unapproved evaluation settings"
        )
    current["continuation"] = {
        "source_checkpoint": str(checkpoint.resolve()),
        "source_identity": metadata["identity"],
        "source_wandb_id": metadata["wandb_id"],
        "source_step": metadata["step"],
        "source_total_updates": source["total_updates"],
        "warmup_updates": checkpoint_warmup(source),
        "sampling_changed": source["config"]["evaluation"].get("sampling")
        != current["config"]["evaluation"]["sampling"],
    }
    return current


def validate_restored_state(optimizer, scheduler, step: int) -> None:
    if scheduler.last_epoch != step:
        raise ValueError("restored scheduler step does not match checkpoint")
    if scheduler.get_last_lr() != [group["lr"] for group in optimizer.param_groups]:
        raise ValueError("restored optimizer and scheduler learning rates differ")
    if not optimizer.state or any(int(state["step"]) != step for state in optimizer.state.values()):
        raise ValueError("restored AdamW step does not match checkpoint")
