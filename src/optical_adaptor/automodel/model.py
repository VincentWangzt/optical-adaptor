"""Unified optical VLM with selected-position logits and adapter-only checkpoints."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components.distributed.config import DDPConfig
from nemo_automodel.components.distributed.ddp import DDPManager
from torch import nn
from transformers import AutoConfig, AutoTokenizer

from optical_adaptor.adapters import MLPAdapter
from optical_adaptor.automodel.vision import DeepSeekOCRVision


class FrozenResources(nn.Module):
    """Registered frozen modules; the recipe checkpoints only the adapter view."""

    def __init__(self, language: nn.Module, vision: nn.Module | None):
        super().__init__()
        self.language = language
        self.vision = vision


class OpticalModel(nn.Module):
    def __init__(
        self,
        *,
        pretrained_model_name_or_path,
        llm,
        vision,
        adapter,
        role,
        image_microbatch_size,
        device,
    ):
        super().__init__()
        if role not in {"teacher", "student"}:
            raise ValueError(f"Invalid optical model role: {role}")
        full_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, revision=llm["revision"]
        )
        config = (
            getattr(full_config, llm["text_config_key"]) if llm["text_config_key"] else full_config
        )
        language = NeMoAutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            revision=llm["revision"],
            config=config,
            dtype=torch.bfloat16,
            attn_implementation=llm["attn_implementation"],
            force_hf=True,
            use_liger_kernel=False,
            use_sdpa_patching=False,
        ).to(device)
        language.requires_grad_(False).eval()
        language.config.use_cache = False
        if role == "student" and llm["gradient_checkpointing"]:
            language.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        encoder = None
        if role == "student":
            encoder = DeepSeekOCRVision(**{k: v for k, v in vision.items() if k != "_target_"})
            encoder.to(device=device, dtype=torch.bfloat16)
            self.adapter = MLPAdapter(**{k: v for k, v in adapter.items() if k != "_target_"}).to(
                device
            )
        self.resources = FrozenResources(language, encoder)
        self.config = language.config
        self.role = role
        self.image_microbatch_size = image_microbatch_size
        self.stage_timer = lambda name: nullcontext()
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, revision=llm["revision"]
        )
        if language.get_input_embeddings().weight.shape[1] != adapter["output_dim"]:
            raise ValueError("Backbone and adapter embedding dimensions differ")

    @classmethod
    def from_pretrained(
        cls,
        *,
        pretrained_model_name_or_path,
        llm,
        vision,
        adapter,
        role,
        image_microbatch_size,
        distributed_setup,
        peft_config,
        freeze_config,
    ):
        if peft_config is not None or freeze_config is not None:
            raise ValueError("Optical trainability is fixed: only the adapter is trainable")
        if not isinstance(distributed_setup.strategy_config, DDPConfig):
            raise ValueError(
                "The optical Qwen wrapper currently supports DDP; TP/CP/PP are unvalidated"
            )
        if distributed_setup.activation_checkpointing:
            raise ValueError("Configure language activation checkpointing under optical.llm")
        model = cls(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            llm=llm,
            vision=vision,
            adapter=adapter,
            role=role,
            image_microbatch_size=image_microbatch_size,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        if role == "teacher":
            return model
        group = distributed_setup.mesh_context.process_group
        model = DDPManager(distributed_setup.strategy_config, process_group=group).parallelize(
            model
        )
        # The framework single-rank DDP path casts parameters; retain FP32 master weights.
        unwrapped = (
            model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
        )
        unwrapped.adapter.float()
        return model

    def train(self, mode=True):
        super().train(mode)
        self.resources.language.train(mode and self.role == "student")
        for module in self.resources.language.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
        if self.resources.vision is not None:
            self.resources.vision.eval()
        return self

    def embed(self, input_ids, pixel_values=None, image_positions=None):
        embeddings = self.resources.language.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            if self.role != "student" or image_positions is None:
                raise ValueError("Only the student accepts pixels, with explicit image positions")
            with self.stage_timer("vision_adapter"):
                features = torch.cat(
                    [
                        self.resources.vision(pixels)
                        for pixels in pixel_values.split(self.image_microbatch_size)
                    ]
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    adapted = self.adapter(features).to(embeddings.dtype)
                # [image, token, (batch, position)] preserves original image/target ordering.
                positions = image_positions.flatten(0, 1)
                flat_indices = positions[:, 0] * input_ids.shape[1] + positions[:, 1]
                embeddings = (
                    embeddings.flatten(0, 1)
                    .index_copy(0, flat_indices, adapted.flatten(0, 1))
                    .view(*input_ids.shape, -1)
                )
        return embeddings

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        loss_positions,
        pixel_values=None,
        image_positions=None,
    ):
        inputs = self.embed(input_ids, pixel_values, image_positions)
        hidden = self.resources.language.base_model(
            inputs_embeds=inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        ).last_hidden_state
        selected = hidden.gather(
            1, loss_positions.clamp_min(0).unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        )
        # One full-vocabulary projection; no target-position chunk/checkpoint loop.
        logits = self.resources.language.get_output_embeddings()(selected)
        return SimpleNamespace(logits=logits)

    @torch.no_grad()
    def generate(
        self, *, input_ids, attention_mask, pixel_values, image_positions, max_new_tokens, **kwargs
    ):
        self.eval()
        embeddings = self.embed(input_ids, pixel_values, image_positions)
        return self.resources.language.generate(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            **kwargs,
        )


@torch.no_grad()
def logit_statistics(student, teacher, labels, kd_loss):
    """Unnormalized CE/KL/teacher CE/agreement/accuracy/target counts."""
    valid = labels != -100
    student, teacher, labels = student[valid].float(), teacher[valid].float(), labels[valid]
    return torch.stack(
        (
            F.cross_entropy(student, labels, reduction="sum"),
            kd_loss(student, teacher, labels, num_batch_labels=1),
            F.cross_entropy(teacher, labels, reduction="sum"),
            (student.argmax(-1) == teacher.argmax(-1)).sum(),
            (student.argmax(-1) == labels).sum(),
            labels.new_tensor(labels.numel()),
        )
    )
