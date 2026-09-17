"""Regression checks for the shared KD route and empty CP supervision."""

from contextlib import nullcontext
from types import SimpleNamespace

import torch
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.loss.kd_loss import KDLoss
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.recipes.vlm import kd


def test_empty_targets_preserve_backward_graph():
    student = torch.randn(1, 3, 7, requires_grad=True)
    loss = KDLoss(chunk_size=0)(student, torch.randn_like(student), torch.full((1, 3), -100))
    loss.backward()
    assert torch.equal(student.grad, torch.zeros_like(student))


def test_unpaired_shared_teacher_uses_sharded_input(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(3, 7)
            self.seen = []

        def forward(self, input_ids):
            self.seen.append(input_ids.shape[1])
            return SimpleNamespace(logits=self.head(input_ids))

    class HalfSharder:
        def __init__(self, *args, **kwargs):
            pass

        def shard(self, batch):
            return nullcontext, {key: value[:, :2] for key, value in batch.items()}

    recipe = kd.KnowledgeDistillationRecipeForVLM(ConfigNode({}))
    recipe.dist_env = SimpleNamespace(device=torch.device("cpu"))
    recipe.device_mesh = None
    recipe.distributed_config = SimpleNamespace(defer_fsdp_grad_sync=True)
    recipe.pp_enabled = False
    recipe.separate_meshes = False
    recipe._offload_teacher_model = False
    recipe.model_parts, recipe.teacher_model = [Model()], Model().requires_grad_(False)
    recipe.kd_ratio = 0.5
    recipe.loss_fn, recipe.kd_loss_fn = MaskedCrossEntropy(), KDLoss(chunk_size=0)
    recipe._ce_loss_buffer, recipe._kd_loss_buffer = [], []
    monkeypatch.setattr(kd, "ContextParallelSharder", HalfSharder)
    recipe._forward_backward_step(
        0,
        {"input_ids": torch.randn(1, 4, 3), "labels": torch.tensor([[1, 2, 3, 4]])},
        loss_buffer=[],
        num_label_tokens=2,
        num_batches=1,
        is_train=False,
    )
    assert recipe.model_parts[0].seen == recipe.teacher_model.seen == [2]
