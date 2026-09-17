"""AutoModel parallel strategies and compact-target context parallelism."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
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
from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh, get_fsdp_dp_mesh
from nemo_automodel.components.distributed.parallelizer import (
    ParallelizationStrategy,
    Qwen3_5ParallelizationStrategy,
    register_parallel_strategy,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.nn.functional import all_reduce

if TYPE_CHECKING:
    from optical_adaptor.automodel.model import OpticalModel


def select_context_targets(
    hidden: torch.Tensor, positions: torch.Tensor, cp_mesh: DeviceMesh | None
) -> torch.Tensor:
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


def shard_target_batch(
    cp_mesh: DeviceMesh,
    tp_mesh: DeviceMesh | None,
    batch: dict[str, Any],
    *,
    loss_mask: torch.Tensor | None = None,
    padding_token_id: int = 0,
) -> tuple[Callable[[], AbstractContextManager], dict[str, Any], ShardLayout]:
    """Shard compact supervision while keeping each branch's input coordinates.

    Args:
        cp_mesh: Context mesh.
        tp_mesh: Unused tensor mesh; vocabulary materialization belongs to loss.
        batch: Inputs [batch, input_sequence], positions [batch, targets], labels
            [batch, targets], and optional teacher logits [batch, targets, vocab].
            Image tensors remain full and are embedded inside the model forward.
        loss_mask: Unsupported supervision mask [batch, targets]; must be None.
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
    context = _attention_context(cp_mesh)
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

    def parallelize(
        self, model: OpticalModel, device_mesh: DeviceMesh, **kwargs: Any
    ) -> OpticalModel:
        """Shard the language model, adapter, and always-executed outer root."""
        if model.role == "student":
            # The recipe deliberately seeds model construction by global rank.
            # Frozen towers are restored from one checkpoint, but the new adapter
            # has no pretrained state. FSDP does not synchronize initialization:
            # align it over the existing DP/CP and TP groups before sharding.
            # Otherwise TP replicas start with different weights and checkpoint
            # deduplication mixes parameters from different initializations.
            with torch.no_grad():
                for mesh in (get_flat_mesh(device_mesh, "dp_cp"), device_mesh["tp"]):
                    if mesh.size() > 1:
                        source = int(mesh.mesh.flatten()[0])
                        for parameter in model.adapter.parameters():
                            dist.broadcast(parameter, src=source, group=mesh.get_group())
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


def generation_context(cp_mesh: DeviceMesh | None) -> AbstractContextManager:
    """Use the same CP attention context for standalone generation forwards."""
    if cp_mesh is None:
        return nullcontext()
    return get_train_context(False, False, _attention_context(cp_mesh))()


def _attention_context(cp_mesh):
    # Torch 2.13's public buffer API reads buffers[0].device even when the
    # model owns sequence sharding. Supply a minimal device anchor, leaving the
    # real inputs and targets under their explicit, differentiable layouts.
    anchor = torch.zeros(2 * cp_mesh.size(), device=cp_mesh.device_type)
    return create_context_parallel_ctx(cp_mesh, [anchor], [0], {anchor}, "allgather")
