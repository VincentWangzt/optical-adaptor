"""Opt-in native optical mesh parity: launch with torchrun, one or two GPUs."""

import argparse
import copy
import json
import os
import time
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.distributed.ddp import DDPManager
from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.loss.kd_loss import KDLoss
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.models.common.utils import BackendConfig
from nemo_automodel.components.optim.optimizer import AdamWConfig
from nemo_automodel.recipes.vlm import kd
from torch.distributed.tensor import DTensor
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from optical_adaptor.adapters import MLPAdapter
from optical_adaptor.automodel.backbone import OpticalQwen3_5ForCausalLM
from optical_adaptor.automodel.model import FrozenResources, OpticalModel


def tiny_model(role):
    torch.manual_seed(51)
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=256,
        tie_word_embeddings=True,
    )
    language = OpticalQwen3_5ForCausalLM(
        config, backend=BackendConfig(attn="sdpa", linear="torch", rms_norm="torch_fp32")
    )
    language.initialize_weights(buffer_device=torch.device("cuda"), dtype=torch.bfloat16)
    language = language.cuda().requires_grad_(False).eval()
    model = OpticalModel.__new__(OpticalModel)
    torch.nn.Module.__init__(model)
    model.resources = FrozenResources(language, torch.nn.Identity() if role == "student" else None)
    if role == "student":
        model.adapter = MLPAdapter(8, 64, 64).cuda()
    model.config, model.role = config, role
    model.cp_mesh = model.device_mesh = None
    model.supports_inline_generation = True
    model.image_microbatch_size = 2
    model.stage_timer = lambda name: nullcontext()
    return model


def batch_for(rank, micro):
    # Uneven target counts and positions in opposite CP input chunks; the one
    # target case leaves a CP rank without supervised tokens.
    targets = 1 if micro == 0 else 5 + rank
    batch = {"labels": (torch.arange(targets, device="cuda")[None] + 11) % 64}
    for role, length, offset in (("student", 19 + rank, 3), ("teacher", 27 + rank, 9)):
        ids = (torch.arange(length, device="cuda")[None] + micro + rank) % 64
        positions = torch.arange(targets, device="cuda")[None] + offset
        batch[role] = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids, dtype=torch.bool),
            "position_ids": torch.arange(length, device="cuda")[None],
            "loss_positions": positions,
        }
    batch["student"].update(
        pixel_values=torch.arange(16, device="cuda", dtype=torch.bfloat16).reshape(1, 2, 8) / 16,
        image_positions=torch.tensor([[[0, 1], [0, 2]]], device="cuda"),
    )
    return batch


def full(tensor):
    """Materialize a parameter or gradient of arbitrary shape from DP/TP shards."""
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def run(axis, checkpointing):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    env = initialize_distributed("nccl")
    world = dist.get_world_size()
    distributed = {
        "strategy": "ddp" if axis == "ddp" else "fsdp2",
        "tp_size": world if axis == "tp" else 1,
        "cp_size": world if axis == "cp" else 1,
        "activation_checkpointing": checkpointing,
    }
    recipe = kd.KnowledgeDistillationRecipeForVLM(ConfigNode({"distributed": distributed}))
    recipe.dist_env = env
    setup = recipe._create_distributed_setup()
    recipe.device_mesh = setup.mesh_context.device_mesh
    recipe.distributed_config = setup.strategy_config
    recipe.pp_enabled = False
    recipe._offload_teacher_model = False
    recipe.kd_ratio = 0.4
    recipe.kd_loss_fn, recipe.loss_fn = KDLoss(chunk_size=0), MaskedCrossEntropy()
    recipe._ce_loss_buffer, recipe._kd_loss_buffer = [], []
    model, teacher = tiny_model("student"), tiny_model("teacher")
    reference, reference_teacher = copy.deepcopy(model), copy.deepcopy(teacher)
    # Model builders use ranked RNG. Deliberately differ before wrapping to test
    # initialization synchronization, not only gradients from identical weights.
    with torch.no_grad():
        for parameter in model.adapter.parameters():
            parameter.add_(dist.get_rank() * 0.125)
    if axis == "ddp":
        manager = DDPManager(setup.strategy_config)
        model = manager.parallelize(model)
    else:
        manager = FSDP2Manager(setup.strategy_config, recipe.device_mesh)
        model = manager.parallelize(model)
        teacher = manager.parallelize(teacher)
    recipe.model_parts, recipe.teacher_model = [model], teacher
    plain = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    optimizer_config = AdamWConfig(lr=0.0002, betas=(0.9, 0.95), weight_decay=0.01)
    recipe.optimizer = optimizer_config.build(model)
    reference_optimizer = optimizer_config.build(reference)[0]
    recipe.lr_scheduler = recipe.moe_mesh = None
    recipe.checkpointer = SimpleNamespace(maybe_wait_for_staging=lambda: None)
    recipe.step_scheduler = SimpleNamespace(step=0, epoch=0)
    recipe.timestamp = time.perf_counter()
    dp, rank = recipe._get_dp_group_size(), recipe._get_dp_rank()
    results = []
    original_clip = kd.scale_grads_and_clip_grad_norm
    for step, accumulation in enumerate((2, 1)):
        recipe.step_scheduler.step = step
        denominator = sum(
            batch_for(r, m)["labels"].numel() for r in range(dp) for m in range(accumulation)
        )
        for r in range(dp):
            for micro in range(accumulation):
                batch = batch_for(r, micro)
                student_logits = reference(**batch["student"]).logits
                teacher_logits = reference_teacher(**batch["teacher"]).logits
                loss = 0.6 * recipe.loss_fn(
                    student_logits, batch["labels"], num_label_tokens=denominator
                )
                loss = loss + 0.4 * recipe.kd_loss_fn(
                    student_logits, teacher_logits, batch["labels"], num_batch_labels=denominator
                )
                loss.backward()
        expected = torch.cat([p.grad.flatten() for p in reference.adapter.parameters()])
        captured = {}

        def capture(*args, captured=captured, **kwargs):
            captured["gradient"] = torch.cat(
                [full(p.grad).flatten() for p in plain.adapter.parameters()]
            )
            return original_clip(*args, **kwargs)

        with patch.object(kd, "scale_grads_and_clip_grad_norm", capture):
            report = recipe._run_train_optim_step(
                [batch_for(rank, m) for m in range(accumulation)], max_grad_norm=0.1
            )
        actual = captured["gradient"]
        relative = ((actual - expected).norm() / expected.norm()).item()
        assert torch.isfinite(actual).all()
        assert relative < 0.025, (
            axis,
            step,
            relative,
            actual.norm().item(),
            expected.norm().item(),
        )
        norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.1)
        torch.testing.assert_close(report.metrics["grad_norm"].float(), norm, rtol=0.025, atol=1e-7)
        torch.testing.assert_close(
            torch.tensor(report.metrics["loss"]),
            torch.tensor(0.6 * report.metrics["ce_loss"] + 0.4 * report.metrics["kd_loss"]),
        )
        reference_optimizer.step()
        reference_optimizer.zero_grad(set_to_none=True)
        delta = max(
            (full(a) - b).abs().max().item()
            for a, b in zip(plain.adapter.parameters(), reference.adapter.parameters(), strict=True)
        )
        assert delta < 0.00004, (axis, step, delta)
        # Full adapter values must agree exactly across DP/TP/CP ranks, even
        # where BF16 distributed arithmetic differs slightly from serial.
        weights = torch.cat([full(p).detach().flatten() for p in plain.adapter.parameters()])
        replicas = [torch.empty_like(weights) for _ in range(world)]
        dist.all_gather(replicas, weights)
        assert all(torch.equal(weights, replica) for replica in replicas)
        assert all(p.dtype == torch.float32 for p in plain.adapter.parameters())
        assert all(p.grad is None for p in plain.adapter.parameters())
        results.append(
            {
                "step": step,
                "gradient_relative_l2": relative,
                "reference_grad_norm": norm.item(),
                "parameter_max_difference": delta,
                "replicas_bitwise_equal": True,
                "metrics": report.metrics,
            }
        )
    if dist.get_rank() == 0:
        print(
            json.dumps(
                {"axis": axis, "checkpointing": checkpointing, "results": results}, default=str
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", choices=("ddp", "dp", "tp", "cp"), required=True)
    parser.add_argument("--checkpointing", action="store_true")
    args = parser.parse_args()
    run(args.axis, args.checkpointing)
