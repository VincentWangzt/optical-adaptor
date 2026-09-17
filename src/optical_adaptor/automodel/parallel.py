"""AutoModel parallel strategies and compact-target context parallelism."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from nemo_automodel.components.distributed.context_parallel.sharder import (
    ContextParallelSharder,
    ShardLayout,
    round_robin_local_indices,
    shard_sequence_for_cp_round_robin,
)
from nemo_automodel.components.distributed.context_parallel.utils import (
    attach_context_parallel_hooks,
    create_context_parallel_ctx,
    get_train_context,
)
from nemo_automodel.components.distributed.mesh_utils import get_fsdp_dp_mesh
from nemo_automodel.components.distributed.parallelizer import (
    ParallelizationStrategy,
    Qwen3_5ParallelizationStrategy,
    register_parallel_strategy,
)
from torch.distributed.fsdp import fully_shard
from torch.distributed.nn.functional import all_reduce


def select_context_targets(hidden: torch.Tensor, positions: torch.Tensor, cp_mesh) -> torch.Tensor:
    """Redistribute selected hidden states from input owners to target owners.

    Args:
        hidden: Local hidden states [batch, padded_input_sequence / CP, hidden].
        positions: Global input positions [batch, targets], with -1 for padding.
        cp_mesh: Round-robin context mesh, or None for an unsharded sequence.

    Returns:
        Selected hidden states [batch, padded_targets / CP, hidden]. Only these
        rows are projected to vocabulary logits. Communication uses hidden width,
        not vocabulary width, and backward sums contributions at input owners.
    """
    if cp_mesh is None or cp_mesh.size() == 1:
        return hidden.gather(1, positions.clamp_min(0)[..., None].expand(-1, -1, hidden.shape[-1]))
    sequence_length = hidden.shape[1] * cp_mesh.size()
    indices = round_robin_local_indices(cp_mesh, sequence_length, hidden.device)
    inverse = torch.full((sequence_length,), -1, dtype=torch.long, device=hidden.device)
    inverse[indices] = torch.arange(indices.numel(), device=hidden.device)
    local_positions = inverse[positions.clamp_min(0)]
    owned = (positions >= 0) & (local_positions >= 0)
    selected = hidden.gather(
        1, local_positions.clamp_min(0)[..., None].expand(-1, -1, hidden.shape[-1])
    )
    selected = selected * owned[..., None]
    selected = all_reduce(selected, group=cp_mesh.get_group())
    return shard_sequence_for_cp_round_robin(cp_mesh, selected)[0]


def shard_target_batch(cp_mesh, tp_mesh, batch, *, loss_mask=None, padding_token_id=0):
    """Shard compact supervision while keeping each branch's input coordinates.

    Args:
        cp_mesh: Context mesh.
        tp_mesh: Unused tensor mesh; vocabulary materialization belongs to loss.
        batch: Inputs [batch, input_sequence], positions [batch, targets], labels
            [batch, targets], and optional teacher logits [batch, targets, vocab].
            Image tensors remain full and are embedded inside the model forward.
        loss_mask: Unsupported additional supervision mask.
        padding_token_id: Unused; input padding belongs to the model forward.

    Returns:
        CP context factory, batch with labels and teacher logits sharded on their
        target axis, and the padded target layout. The input tensors are unchanged.
    """
    del tp_mesh, padding_token_id
    if loss_mask is not None:
        raise ValueError("Optical KD uses labels=-100 rather than a separate loss mask")
    targets = batch["labels"].shape[1]
    batch = dict(batch)
    batch["labels"], _, padded = shard_sequence_for_cp_round_robin(
        cp_mesh, batch["labels"], pad_value=-100
    )
    if "teacher_logits" in batch:
        batch["teacher_logits"] = shard_sequence_for_cp_round_robin(
            cp_mesh, batch["teacher_logits"]
        )[0]
    context = create_context_parallel_ctx(cp_mesh, [], [], set(), "allgather")
    return (
        get_train_context(False, False, context),
        batch,
        ShardLayout(original_seq_len=targets, padded_seq_len=padded),
    )


def target_sharder() -> ContextParallelSharder:
    """Construct the model-owned sharder used by the existing KD recipe."""
    return ContextParallelSharder(
        shard_batch=shard_target_batch, local_token_global_indices=round_robin_local_indices
    )


@register_parallel_strategy(name="OpticalModel")
class OpticalParallelizationStrategy(ParallelizationStrategy):
    """Reuse Qwen's TP/FSDP/CP policy, with separate adapter precision ownership."""

    def parallelize(self, model, device_mesh, **kwargs):
        """Shard the language model, adapter, and always-executed outer root."""
        language = model.resources.language
        Qwen3_5ParallelizationStrategy().parallelize(language, device_mesh, **kwargs)
        model.cp_mesh = language.cp_mesh
        if model.cp_mesh is not None:
            attach_context_parallel_hooks(language)
        dp_mesh = get_fsdp_dp_mesh(device_mesh, "dp_replicate", "dp_shard_cp")
        shard_options = {
            "mesh": dp_mesh,
            "mp_policy": kwargs["mp_policy"],
            "offload_policy": kwargs["offload_policy"],
            "reshard_after_forward": False,
        }
        if model.role == "student":
            fully_shard(model.adapter, **shard_options)
        # Frozen vision belongs to this root: variable image microbatch counts do
        # not create a different sequence of collectives across DP ranks.
        return fully_shard(model, **shard_options)


def generation_context(cp_mesh):
    """Use the same CP attention context for standalone generation forwards."""
    if cp_mesh is None:
        return nullcontext()
    return get_train_context(
        False, False, create_context_parallel_ctx(cp_mesh, [], [], set(), "allgather")
    )()
