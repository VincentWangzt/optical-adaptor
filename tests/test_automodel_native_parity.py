"""Opt-in pinned-checkpoint HF/native weight, logit and input-gradient comparison."""

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from huggingface_hub import snapshot_download
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from optical_adaptor.automodel.model import OpticalModel


def compare(config_path, output):
    initialize_distributed("nccl")
    torch.use_deterministic_algorithms(True)
    raw = yaml.safe_load(Path(config_path).read_text())["optical"]
    native = OpticalModel(
        pretrained_model_name_or_path=raw["llm"]["model_id"],
        llm=raw["llm"],
        vision=raw["vision"],
        adapter=raw["adapter"],
        role="teacher",
        image_microbatch_size=raw["processing"]["image_microbatch_size"],
        device=torch.device("cuda"),
    ).resources.language
    snapshot = snapshot_download(raw["llm"]["model_id"], revision=raw["llm"]["revision"])
    reference = (
        Qwen3_5ForCausalLM.from_pretrained(
            snapshot,
            config=copy.deepcopy(native.config),
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .cuda()
        .requires_grad_(False)
        .eval()
    )
    actual_state = native.state_dict_adapter.to_hf(native.state_dict())
    expected_state = reference.state_dict()
    for name, actual in actual_state.items():
        key = name.replace("model.language_model.", "model.")
        assert torch.equal(actual, expected_state[key]), (
            name,
            actual.dtype,
            expected_state[key].dtype,
        )
    assert len(actual_state) == len(expected_state) - 1  # tied output-head alias
    assert native.lm_head.weight is native.model.embed_tokens.weight
    ids = torch.arange(96, device="cuda").reshape(2, 48) + 100
    positions = torch.tensor([[2, 17, 31, 44], [3, 9, 15, 21]], device="cuda")
    mask = torch.arange(48, device="cuda")[None] < torch.tensor([[48], [24]], device="cuda")
    embeddings = native.get_input_embeddings()(ids).detach()
    results = []
    gradients, logits = [], []
    for model in (reference, native):
        inputs = embeddings.clone().requires_grad_()
        with sdpa_kernel(SDPBackend.MATH):
            if model is native:
                scores = model(
                    inputs_embeds=inputs,
                    attention_mask=mask,
                    position_ids=torch.arange(48, device="cuda")[None].expand(2, -1),
                    loss_positions=positions,
                ).logits
            else:
                hidden = model.model(
                    inputs_embeds=inputs, attention_mask=mask, use_cache=False
                ).last_hidden_state
                scores = model.lm_head(
                    hidden.gather(1, positions[..., None].expand(-1, -1, hidden.shape[-1]))
                )
            scores.float().square().mean().backward()
        logits.append(scores.detach().float())
        gradients.append(inputs.grad.float())
    for name, values in (("logits", logits), ("input_gradients", gradients)):
        expected, actual = values
        relative = ((actual - expected).norm() / expected.norm()).item()
        assert torch.isfinite(actual).all() and relative < 0.05, (name, relative)
        results.append(
            {
                "tensor": name,
                "relative_l2": relative,
                "max_difference": (actual - expected).abs().max().item(),
            }
        )
    report = {"identical_checkpoint_tensors": len(actual_state), "comparisons": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    compare(args.config, args.output)
