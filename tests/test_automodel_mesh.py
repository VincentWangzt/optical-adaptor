"""Exercise the real KD bridge and accumulation against a serial CPU reference."""

import socket
import time
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.distributed.ddp import DDPManager
from nemo_automodel.components.loss.kd_loss import KDLoss
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.optim.optimizer import AdamWConfig
from nemo_automodel.recipes.kd_utils import STOP_TEACHER
from nemo_automodel.recipes.vlm.kd import KnowledgeDistillationRecipeForVLM


class SelectedModel(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.head = torch.nn.Linear(3, 7, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.arange(21).reshape(7, 3) * scale)

    def forward(self, input_ids, loss_positions):
        """Project selected [batch, target, feature] rows to [batch, target, vocab]."""
        selected = input_ids.gather(1, loss_positions[..., None].expand(-1, -1, 3))
        return SimpleNamespace(logits=self.head(selected))


def batch_for(rank, micro):
    generator = torch.Generator().manual_seed(rank * 13 + micro)
    # Paired branches have different full lengths but share compact targets.
    targets = 1 + rank + micro
    return {
        "student": {
            "input_ids": torch.randn(1, targets + 1, 3, generator=generator),
            "loss_positions": torch.arange(targets)[None],
        },
        "teacher": {
            "input_ids": torch.randn(1, targets + 3, 3, generator=generator),
            "loss_positions": (torch.arange(targets) + 2)[None],
        },
        "labels": torch.arange(targets)[None] % 7,
    }


def worker(rank, world_size, student_size, port):
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size
    )
    cfg = ConfigNode(
        {
            "separate_meshes": True,
            "distributed": {"strategy": "ddp", "dp_size": student_size},
            "teacher_distributed": {"strategy": "ddp", "dp_size": world_size - student_size},
        }
    )
    recipe = KnowledgeDistillationRecipeForVLM(cfg)
    recipe.dist_env = SimpleNamespace(world_size=world_size, device=torch.device("cpu"))
    setup = recipe._create_distributed_setup()
    recipe.device_mesh = setup.mesh_context.device_mesh
    recipe.distributed_config = setup.strategy_config
    recipe.pp_enabled = False
    recipe._offload_teacher_model = False
    recipe.kd_ratio = 0.4
    recipe.kd_loss_fn = KDLoss(chunk_size=0)
    recipe.loss_fn = MaskedCrossEntropy()
    recipe._ce_loss_buffer, recipe._kd_loss_buffer = [], []
    if recipe.kd_mesh_bridge.is_teacher:
        recipe.teacher_model = SelectedModel(0.03).requires_grad_(False)
        recipe._run_teacher_worker()
    else:
        model = DDPManager(
            setup.strategy_config, process_group=recipe.kd_mesh_bridge.student_group
        ).parallelize(SelectedModel(0.02))
        recipe.model_parts = [model]
        recipe.teacher_model = None
        assert recipe._get_dp_group_size() == student_size
        denominator = sum(
            batch_for(r, m)["labels"].numel() for r in range(student_size) for m in range(2)
        )
        for micro in range(2):
            recipe._forward_backward_step(
                micro,
                batch_for(rank, micro),
                loss_buffer=[],
                num_label_tokens=denominator,
                num_batches=2,
            )
        reference, teacher = SelectedModel(0.02), SelectedModel(0.03).requires_grad_(False)
        for r in range(student_size):
            for micro in range(2):
                batch = batch_for(r, micro)
                logits, truth = (
                    reference(**batch["student"]).logits,
                    teacher(**batch["teacher"]).logits,
                )
                loss = 0.6 * recipe.loss_fn(
                    logits, batch["labels"], num_label_tokens=denominator
                ) + 0.4 * recipe.kd_loss_fn(
                    logits, truth, batch["labels"], num_batch_labels=denominator
                )
                loss.backward()
        unwrapped = (
            model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        )
        torch.testing.assert_close(unwrapped.head.weight.grad, reference.head.weight.grad)
        # Exercise the actual optimizer boundary, not just forward/backward:
        # clipping, Adam moments, parameter references, and replica agreement.
        optimizer_config = AdamWConfig(lr=0.002, betas=(0.9, 0.95), weight_decay=0.01)
        recipe.optimizer = optimizer_config.build(model)
        reference_optimizer = optimizer_config.build(reference)[0]
        recipe.lr_scheduler = None
        recipe.moe_mesh = None
        recipe.checkpointer = SimpleNamespace(maybe_wait_for_staging=lambda: None)
        recipe.step_scheduler = SimpleNamespace(step=0, epoch=0)
        recipe.timestamp = time.perf_counter()
        recipe._ce_loss_buffer.clear()
        recipe._kd_loss_buffer.clear()
        recipe.optimizer[0].zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        for step in range(2):
            recipe.step_scheduler.step = step
            recipe._run_train_optim_step(
                [batch_for(rank, micro) for micro in range(2)], max_grad_norm=0.1
            )
            for r in range(student_size):
                for micro in range(2):
                    batch = batch_for(r, micro)
                    logits = reference(**batch["student"]).logits
                    truth = teacher(**batch["teacher"]).logits
                    loss = 0.6 * recipe.loss_fn(
                        logits, batch["labels"], num_label_tokens=denominator
                    ) + 0.4 * recipe.kd_loss_fn(
                        logits, truth, batch["labels"], num_batch_labels=denominator
                    )
                    loss.backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.1)
            reference_optimizer.step()
            reference_optimizer.zero_grad(set_to_none=True)
            torch.testing.assert_close(unwrapped.head.weight, reference.head.weight)
            actual_state = recipe.optimizer[0].state[unwrapped.head.weight]
            expected_state = reference_optimizer.state[reference.head.weight]
            for key in expected_state:
                torch.testing.assert_close(actual_state[key], expected_state[key])
            replicas = [torch.empty_like(unwrapped.head.weight) for _ in range(student_size)]
            dist.all_gather(
                replicas, unwrapped.head.weight.detach(), group=recipe.kd_mesh_bridge.student_group
            )
            assert all(torch.equal(replicas[0], weight) for weight in replicas)
            assert unwrapped.head.weight.grad is None
        recipe.kd_mesh_bridge.broadcast_command(STOP_TEACHER)
    dist.destroy_process_group()


@pytest.mark.parametrize("student_size", [1, 2])
def test_different_dp_meshes_preserve_accumulated_gradient(student_size):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(3, student_size, port), nprocs=3, join=True)
