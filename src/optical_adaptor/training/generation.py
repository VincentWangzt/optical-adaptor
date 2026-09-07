from __future__ import annotations

import torch
from transformers import PreTrainedModel

from optical_adaptor.training.config import Pipeline, fingerprint
from optical_adaptor.training.models import FrozenQwen


@torch.no_grad()
def generate_tokens(
    pipeline: Pipeline,
    qwen: FrozenQwen,
    model: PreTrainedModel,
    records: list[dict],
    inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Use the same decoding policy and isolated RNG for native and adapted Qwen."""
    evaluation = pipeline.config.evaluation
    seed = int(
        fingerprint(
            [pipeline.config.seed, "reconstruction-generation", [r["record_id"] for r in records]]
        ),
        16,
    ) % (2**63)
    device = qwen.device
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        return model.generate(
            **inputs,
            **evaluation.sampling.model_dump(exclude={"presence_penalty"}),
            use_cache=True,
            max_new_tokens=evaluation.max_new_tokens,
            eos_token_id=qwen.assistant_end,
            pad_token_id=qwen.tokenizer.pad_token_id,
        )
