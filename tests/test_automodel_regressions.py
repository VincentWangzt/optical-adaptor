"""Regression checks for the shared KD route and empty CP supervision."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from nemo_automodel._transformers import model_init
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.loss.kd_loss import KDLoss
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.models.common.utils import BackendConfig
from nemo_automodel.recipes.vlm import kd
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from optical_adaptor.automodel.backbone import (
    OpticalQwen3_5ForCausalLM,
    OpticalTextCheckpointAdapter,
)
from optical_adaptor.automodel.recipe import OpticalKDRecipe


@pytest.mark.parametrize(
    ("step", "generation_ids", "error", "message"),
    [
        (0, {"selected"}, RuntimeError, "validation batch reached"),
        (499, set(), RuntimeError, "validation batch reached"),
        (499, {"selected"}, ValueError, "Inline generation evaluation requires DDP"),
    ],
)
def test_fsdp_inline_generation_preflight_only_for_selected_samples(
    step, generation_ids, error, message
):
    model = SimpleNamespace(eval=lambda: None, supports_inline_generation=False)
    recipe = SimpleNamespace(
        model_parts=[model],
        eval_data=SimpleNamespace(rows=[{"slice": "stack/reconstruction"}]),
        dist_env=SimpleNamespace(device=torch.device("cpu")),
        generation_ids=generation_ids,
        step_scheduler=SimpleNamespace(step=step, is_last_step=False),
        optical=SimpleNamespace(evaluation=SimpleNamespace(generation_every=500)),
        student_model=lambda: model,
    )

    class StopAtBatch:
        def __iter__(self):
            raise RuntimeError("validation batch reached")

    with pytest.raises(error, match=message):
        OpticalKDRecipe._run_validation_epoch(recipe, StopAtBatch())


def test_explicit_null_disables_gradient_clipping(monkeypatch):
    recipe = object.__new__(OpticalKDRecipe)
    recipe.raw = {"clip_grad_norm": {"max_norm": None}}
    recipe.max_grad_norm = 1.0
    monkeypatch.setattr(kd.KnowledgeDistillationRecipeForVLM, "setup", lambda self: None)
    monkeypatch.setattr(OpticalKDRecipe, "_should_setup_training_components", lambda self: False)
    recipe.setup()
    assert recipe.max_grad_norm is None


def test_native_cpu_float32_attention_forward_and_input_gradient_parity():
    torch.manual_seed(17)
    config = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        layer_types=["full_attention"],
        tie_word_embeddings=True,
    )
    native = OpticalQwen3_5ForCausalLM(
        config, backend=BackendConfig(attn="sdpa", linear="torch", rms_norm="torch_fp32")
    )
    native.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    native.requires_grad_(False).eval()
    reference = Qwen3_5ForCausalLM(config).requires_grad_(False).eval()
    state = {
        key.replace("model.language_model.", "model."): value
        for key, value in native.state_dict_adapter.to_hf(native.state_dict()).items()
    }
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    reference.load_state_dict(state, strict=True)
    embeddings = torch.randn(2, 12, 32)
    positions = torch.tensor([[1, 5, 10], [2, 4, 6]])
    mask = torch.arange(12)[None] < torch.tensor([[12], [8]])
    outputs, gradients = [], []
    for model in (reference, native):
        inputs = embeddings.clone().requires_grad_()
        if model is native:
            logits = model(
                inputs_embeds=inputs,
                attention_mask=mask,
                position_ids=torch.arange(12)[None].expand(2, -1),
                loss_positions=positions,
            ).logits
        else:
            all_logits = model(inputs_embeds=inputs, attention_mask=mask, use_cache=False).logits
            logits = all_logits.gather(1, positions[..., None].expand(-1, -1, config.vocab_size))
        logits.square().mean().backward()
        outputs.append(logits)
        gradients.append(inputs.grad)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=1e-4, atol=1e-7)


def test_explicit_native_config_is_consumed_once_and_mtp_is_disabled():
    config = Qwen3_5TextConfig(
        vocab_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        layer_types=["full_attention"],
        num_nextn_predict_layers=1,
        architectures=["OpticalQwen3_5ForCausalLM"],
    )
    custom, model = model_init.__init_model(
        None,
        config,
        "sdpa",
        torch.float32,
        None,
        False,
        config=config,
        backend=BackendConfig(attn="sdpa", linear="torch", rms_norm="torch_fp32"),
    )
    assert custom and model.config is config
    assert model.mtp is None


def test_native_text_checkpoint_destinations_preserve_storage_and_ties():
    adapter = OpticalTextCheckpointAdapter(tied_embeddings=True)
    embedding = torch.zeros(11, 7, dtype=torch.bfloat16)
    gate = torch.zeros(3, dtype=torch.float32)
    state = {
        "model.embed_tokens.weight": embedding,
        "lm_head.weight": embedding,
        "model.layers.0.linear_attn._fp32_params.A_log": gate,
    }
    destinations = adapter.to_hf(state)
    destinations["model.language_model.embed_tokens.weight"].fill_(3)
    destinations["model.language_model.layers.0.linear_attn.A_log"].fill_(7)
    assert torch.equal(embedding, torch.full_like(embedding, 3))
    assert torch.equal(gate, torch.full_like(gate, 7))
    destinations["model.visual.unused.weight"] = torch.ones(4)
    restored = adapter.from_hf(destinations)
    assert restored.keys() == state.keys()
    assert restored["lm_head.weight"] is restored["model.embed_tokens.weight"]
    assert adapter.supports_low_memory_dcp_load
    for name, tensor in state.items():
        assert (
            dict(adapter.convert_single_tensor_to_hf(name, tensor)).keys()
            == adapter.to_hf({name: tensor}).keys()
        )


def test_empty_targets_preserve_backward_graph():
    student = torch.randn(1, 3, 7, dtype=torch.bfloat16, requires_grad=True)
    loss = KDLoss(chunk_size=0)(student, torch.randn_like(student), torch.full((1, 3), -100))
    loss.backward()
    assert loss.dtype == torch.float32
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
