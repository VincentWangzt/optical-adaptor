"""Unified optical VLM with selected-position logits and adapter-only checkpoints."""

from __future__ import annotations

import copy
from contextlib import nullcontext

import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components.distributed.config import DDPConfig, FSDP2Config
from nemo_automodel.components.distributed.context_parallel.utils import cp_dispatcher_suspended
from nemo_automodel.components.distributed.ddp import DDPManager
from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager
from nemo_automodel.components.models.common.utils import BackendConfig
from torch import nn
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from optical_adaptor.adapters import MLPAdapter
from optical_adaptor.automodel.backbone import OpticalQwen3_5ForCausalLM
from optical_adaptor.automodel.parallel import target_sharder
from optical_adaptor.automodel.vision import DeepSeekOCRVision


class FrozenResources(nn.Module):
    """Registered frozen modules; the recipe checkpoints only the adapter view."""

    def __init__(self, language: nn.Module, vision: nn.Module | None):
        super().__init__()
        self.language = language
        self.vision = vision


class HFGenerationBackend:
    """Unsharded cached HF decoder, loaded only while generative evaluation runs."""

    def __init__(self, pretrained_model_name_or_path, llm, device):
        full_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, revision=llm["revision"]
        )
        config = (
            getattr(full_config, llm["text_config_key"])
            if llm["text_config_key"]
            else full_config
        )
        self.language = NeMoAutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            revision=llm["revision"],
            config=config,
            dtype=torch.bfloat16,
            attn_implementation=llm["attn_implementation"],
            force_hf=True,
            use_liger_kernel=False,
            use_sdpa_patching=False,
        ).to(device)
        self.language.requires_grad_(False).eval()

    @torch.no_grad()
    def generate(
        self,
        *,
        input_ids,
        attention_mask,
        image_embeds,
        image_positions,
        generation_config,
        max_new_tokens,
        pad_token_id,
        eos_token_id,
    ):
        embeddings = self.language.get_input_embeddings()(input_ids)
        positions = image_positions.flatten(0, 1)
        flat = positions[:, 0] * embeddings.shape[1] + positions[:, 1]
        embeddings = (
            embeddings.flatten(0, 1)
            .index_copy(0, flat, image_embeds.to(embeddings.dtype).flatten(0, 1))
            .view_as(embeddings)
        )
        config = copy.deepcopy(generation_config)
        config.max_new_tokens = max_new_tokens
        config.pad_token_id = pad_token_id
        config.eos_token_id = eos_token_id
        return self.language.generate(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            generation_config=config,
        )


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
        config.architectures = [OpticalQwen3_5ForCausalLM.__name__]
        snapshot = snapshot_download(
            pretrained_model_name_or_path,
            revision=llm["revision"],
            allow_patterns=["*.json", "*.safetensors"],
        )
        language = NeMoAutoModelForCausalLM.from_pretrained(
            snapshot,
            config=config,
            dtype=torch.bfloat16,
            attn_implementation=llm["attn_implementation"],
            force_hf=False,
            backend=BackendConfig(
                attn=llm["attn_implementation"], linear="torch", rms_norm="torch_fp32"
            ),
            use_liger_kernel=False,
            use_sdpa_patching=False,
        ).to(device)
        language.requires_grad_(False).eval()
        language.config.use_cache = False
        encoder = None
        if role == "student":
            encoder = DeepSeekOCRVision(**{k: v for k, v in vision.items() if k != "_target_"})
            encoder.to(device=device, dtype=torch.bfloat16)
            self.adapter = MLPAdapter(**{k: v for k, v in adapter.items() if k != "_target_"}).to(
                device
            )
        self.resources = FrozenResources(language, encoder)
        self.config = language.config
        self.cp_mesh = None
        self.device_mesh = None
        self.generation_group = None
        self.role = role
        self.image_microbatch_size = image_microbatch_size
        self.stage_timer = lambda name: nullcontext()
        self._generation_backend = None
        self._generation_backend_args = (
            pretrained_model_name_or_path,
            copy.deepcopy(llm),
            device,
        )
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
        strategy = distributed_setup.strategy_config
        if not isinstance(strategy, (DDPConfig, FSDP2Config)):
            raise ValueError("Optical KD supports AutoModel DDP and FSDP2 strategies")
        mesh = distributed_setup.mesh_context
        if mesh.pp_size > 1 or mesh.ep_size > 1:
            raise ValueError("Optical KD has no pipeline schedule or experts; PP and EP must be 1")
        if isinstance(strategy, FSDP2Config) and strategy.sequence_parallel:
            raise ValueError(
                "AutoModel's Qwen3.5 TP plan keeps recurrent layers replicated; "
                "disable sequence_parallel"
            )
        model = cls(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            llm=llm,
            vision=vision,
            adapter=adapter,
            role=role,
            image_microbatch_size=image_microbatch_size,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        model.device_mesh = mesh.device_mesh
        if isinstance(strategy, FSDP2Config):
            model.generation_group = mesh.process_group or dist.group.WORLD
            model = FSDP2Manager(strategy, device_mesh=mesh.device_mesh).parallelize(model)
        elif role == "student":
            model = DDPManager(strategy, process_group=mesh.process_group).parallelize(model)
        return model

    @property
    def model(self):
        """Expose native decoder layers to AutoModel activation checkpointing."""
        return self.resources.language.model

    def prepare_model_inputs_for_cp(self, batch, num_chunks=1):
        """Keep branch inputs full; shard labels [batch, targets] in target order."""
        del batch, num_chunks
        return {"cp_sharder": target_sharder()}

    def train(self, mode=True):
        super().train(mode)
        self.resources.language.train(mode and self.role == "student")
        for module in self.resources.language.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
        if self.resources.vision is not None:
            self.resources.vision.eval()
        return self

    def encode_images(self, pixel_values):
        """Map images [images, channels, height, width] to [images, tokens, hidden]."""
        with self.stage_timer("vision_adapter"), cp_dispatcher_suspended(self.cp_mesh):
            features = torch.cat(
                [
                    self.resources.vision(pixels)
                    for pixels in pixel_values.split(self.image_microbatch_size)
                ]
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self.adapter(features)

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        loss_positions,
        pixel_values=None,
        image_positions=None,
    ):
        if pixel_values is not None and (self.role != "student" or image_positions is None):
            raise ValueError("Only the student accepts pixels, with explicit image positions")
        images = self.encode_images(pixel_values) if pixel_values is not None else None
        return self.resources.language(
            input_ids=input_ids,
            image_embeds=images,
            image_positions=image_positions,
            attention_mask=attention_mask,
            position_ids=position_ids,
            loss_positions=loss_positions,
        )

    @torch.no_grad()
    def generate(
        self,
        *,
        input_ids,
        attention_mask,
        pixel_values,
        image_positions,
        generation_config: GenerationConfig,
        max_new_tokens,
    ):
        """Generate with the ordinary cache-enabled Hugging Face implementation.

        Args:
            input_ids: Prompt tokens [1, sequence].
            attention_mask: Prompt validity [1, sequence].
            pixel_values: Images [images, channels, height, width].
            image_positions: Image slots [images, image_tokens, 2] in prompt coordinates.
            generation_config: Standard Hugging Face generation configuration.
            max_new_tokens: Output-token cap for this evaluation slice.

        Returns:
            Generated tokens [1, generated_sequence], excluding the prompt.
        """
        if input_ids.shape[0] != 1:
            raise ValueError("Optical evaluation uses single-sample generation")
        if self.generation_group is not None:
            raise ValueError(
                "The unsharded Hugging Face generation backend requires DDP evaluation"
            )
        self.eval()
        if self._generation_backend is None:
            self._generation_backend = HFGenerationBackend(*self._generation_backend_args)
        images = self.encode_images(pixel_values)
        return self._generation_backend.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_embeds=images,
            image_positions=image_positions,
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
        )

    def release_generation_backend(self):
        """Release the temporary full HF decoder before training resumes."""
        self._generation_backend = None
        torch.cuda.empty_cache()
