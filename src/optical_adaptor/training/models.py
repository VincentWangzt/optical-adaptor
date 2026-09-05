from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import nn
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM

from optical_adaptor.training.config import AdapterConfig, AdapterKind, Pipeline


class MLPAdapter(nn.Module):
    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config
        self.projection = nn.Sequential(
            nn.LayerNorm(config.input_dim),
            nn.Linear(config.input_dim, config.output_dim),
            nn.GELU(),
            nn.Linear(config.output_dim, config.output_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 3 or embeddings.shape[1:] != (
            self.config.sequence_length,
            self.config.input_dim,
        ):
            raise ValueError("adapter input shape must be [batch,111,1280]")
        return self.projection(embeddings)


class TransformerAdapter(nn.Module):
    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.positions = nn.Parameter(torch.empty(config.sequence_length, config.input_dim))
        nn.init.normal_(self.positions, std=0.02)
        block = nn.TransformerEncoderLayer(
            config.input_dim,
            config.transformer_heads,
            config.transformer_ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            block, config.transformer_layers, enable_nested_tensor=False
        )
        # PyTorch clones the template block; initialize layers independently.
        for layer in self.encoder.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    module.reset_parameters()
        self.mlp = MLPAdapter(config)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.encoder(embeddings + self.positions))


def build_adapter(kind: AdapterKind, config: AdapterConfig) -> nn.Module:
    if kind == "mlp":
        return MLPAdapter(config)
    if kind == "transformer":
        return TransformerAdapter(config)
    raise ValueError(f"unknown adapter kind: {kind}")


class FrozenQwen:
    def __init__(self, pipeline: Pipeline, device: torch.device):
        config = pipeline.config.models
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.qwen_id, revision=config.qwen_revision
        )
        full_config = AutoConfig.from_pretrained(config.qwen_id, revision=config.qwen_revision)
        self.model = Qwen3_5ForCausalLM.from_pretrained(
            config.qwen_id,
            revision=config.qwen_revision,
            config=full_config.text_config,
            dtype=torch.bfloat16,
            device_map={"": str(device)},
            attn_implementation="sdpa",
        )
        self.model.requires_grad_(False)
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.model.eval()
        if (
            self.model.get_input_embeddings().weight.data_ptr()
            != self.model.lm_head.weight.data_ptr()
        ):
            raise RuntimeError("Qwen output head is not tied to token embeddings")
        self.vision_start = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self.assistant_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": pipeline.config.evaluation.instruction},
                    {"type": "image"},
                ],
            }
        ]
        rendered_template = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        template = self.tokenizer.encode(rendered_template, add_special_tokens=False)
        image_token = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if template.count(image_token) != 1:
            raise ValueError("expected one image placeholder in Qwen reconstruction template")
        boundary = template.index(image_token)
        self.reconstruction_before = template[:boundary]
        self.reconstruction_after = template[boundary + 1 :]

    def embed(self, ids: list[int]) -> torch.Tensor:
        return self.model.get_input_embeddings()(torch.tensor(ids, device=self.device))

    def hidden(
        self, inputs: list[torch.Tensor], *, student: bool, target_lengths: list[int]
    ) -> torch.Tensor:
        # CheckpointingLayer checks .training, independent of requires_grad.
        self.model.train(student)
        lengths = [len(value) for value in inputs]
        padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        attention_mask = (
            torch.arange(padded.shape[1], device=self.device)[None]
            < torch.tensor(lengths, device=self.device)[:, None]
        )
        position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)
        hidden = self.model.model(
            inputs_embeds=padded,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        ).last_hidden_state
        # Inputs contain the conditioning prefix plus targets[:-1].
        return torch.cat(
            [
                hidden[i, length - count : length]
                for i, (length, count) in enumerate(zip(lengths, target_lengths, strict=True))
            ]
        )

    @torch.no_grad()
    def teacher_hidden(self, records: list[dict]) -> torch.Tensor:
        targets = [len(row["continuation_ids"]) for row in records]
        inputs = [
            self.embed(row["prefix_ids"] + row["visual_ids"] + row["continuation_ids"][:-1])
            for row in records
        ]
        hidden = self.hidden(inputs, student=False, target_lengths=targets)
        return hidden.reshape(len(records), targets[0], -1)

    def student_hidden(
        self, records: list[dict], adapted: torch.Tensor, task: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, targets, text_mask, lengths = [], [], [], []
        for record, visual in zip(records, adapted, strict=True):
            if task == "continuation":
                before = record["prefix_ids"] + [self.vision_start]
                after = [self.vision_end]
                target = record["continuation_ids"]
                mask = [True] * len(target)
            elif task == "reconstruction":
                before, after = self.reconstruction_before, self.reconstruction_after
                target = record["visual_ids"] + [self.assistant_end]
                mask = [True] * (len(target) - 1) + [False]
            else:
                raise ValueError(f"unknown task: {task}")
            inputs.append(torch.cat([self.embed(before), visual, self.embed(after + target[:-1])]))
            targets.extend(target)
            text_mask.extend(mask)
            lengths.append(len(target))
        return (
            self.hidden(inputs, student=torch.is_grad_enabled(), target_lengths=lengths),
            torch.tensor(targets, device=self.device),
            torch.tensor(text_mask, device=self.device),
        )


class DeepSeekVision(nn.Module):
    """Pinned vLLM vision modules, with no DeepSeek language model instantiated."""

    def __init__(self, pipeline: Pipeline, device: torch.device):
        super().__init__()
        from transformers import CLIPVisionConfig
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed import init_distributed_environment, initialize_model_parallel
        from vllm.model_executor.models.deepencoder import (
            DeepCLIPVisionTransformer,
            build_sam_vit_b,
        )
        from vllm.model_executor.models.deepseek_vl2 import MlpProjector
        from vllm.transformers_utils.configs.deepseek_vl2 import MlpProjectorConfig
        from vllm.transformers_utils.processors.deepseek_ocr import ImageTransform

        if torch.distributed.is_initialized():
            raise RuntimeError("extract DeepSeek in a standalone process, outside training DDP")
        self.rendezvous = tempfile.TemporaryDirectory(prefix="optical-encoder-")
        self.vllm_config = VllmConfig()
        with set_current_vllm_config(self.vllm_config):
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=device.index or 0,
                distributed_init_method=f"file://{self.rendezvous.name}/rendezvous",
                backend="nccl",
            )
            initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
            self.sam_model = build_sam_vit_b()
            self.vision_model = DeepCLIPVisionTransformer(
                CLIPVisionConfig(
                    hidden_size=1024,
                    intermediate_size=4096,
                    num_attention_heads=16,
                    num_hidden_layers=24,
                    image_size=224,
                    patch_size=14,
                    projection_dim=512,
                    layer_norm_eps=1e-5,
                )
            )
            self.projector = MlpProjector(
                MlpProjectorConfig(input_dim=2048, n_embed=1280, projector_type="linear")
            )
        self.image_newline = nn.Parameter(torch.empty(1280))
        self.view_seperator = nn.Parameter(torch.empty(1280))
        self.image_size = pipeline.config.models.image_size
        self.transform = ImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
        config = pipeline.config.models
        index_path = hf_hub_download(
            config.encoder_id, "model.safetensors.index.json", revision=config.encoder_revision
        )
        weight_map = json.loads(Path(index_path).read_text())["weight_map"]
        prefixes = (
            "model.sam_model.",
            "model.vision_model.",
            "model.projector.",
            "model.image_newline",
            "model.view_seperator",
        )
        names = [name for name in weight_map if name.startswith(prefixes)]
        if len(names) != config.encoder_tensor_count:
            raise ValueError(
                f"expected {config.encoder_tensor_count} encoder tensors, got {len(names)}"
            )
        weights = {}
        for filename in sorted({weight_map[name] for name in names}):
            path = hf_hub_download(config.encoder_id, filename, revision=config.encoder_revision)
            with safe_open(path, framework="pt", device="cpu") as checkpoint:
                for name in names:
                    if weight_map[name] == filename:
                        weights[name.removeprefix("model.")] = checkpoint.get_tensor(name)
        self.load_state_dict(weights, strict=True)
        self.to(device=device, dtype=torch.bfloat16).requires_grad_(False).eval()

    @torch.no_grad()
    def forward(self, images: list) -> torch.Tensor:
        from vllm.config import set_current_vllm_config

        device = self.image_newline.device
        pixels = torch.stack(
            [self.transform(image.resize((self.image_size, self.image_size))) for image in images]
        ).to(device=device, dtype=torch.bfloat16)
        with set_current_vllm_config(self.vllm_config):
            sam = self.sam_model(pixels)
            clip = self.vision_model(pixels, sam)
            features = self.projector(torch.cat([clip[:, 1:], sam.flatten(2).transpose(1, 2)], -1))
        batch, count, width = features.shape
        side = int(count**0.5)
        rows = features.reshape(batch, side, side, width)
        newline = self.image_newline.reshape(1, 1, 1, width).expand(batch, side, 1, width)
        sequence = torch.cat([rows, newline], dim=2).reshape(batch, -1, width)
        separator = self.view_seperator.reshape(1, 1, width).expand(batch, 1, width)
        result = torch.cat([sequence, separator], dim=1)
        if result.shape[1:] != (111, 1280) or result.dtype != torch.bfloat16:
            raise RuntimeError(f"unexpected DeepSeek embeddings: {result.shape}, {result.dtype}")
        return result

    def close(self) -> None:
        from vllm.distributed import cleanup_dist_env_and_memory

        cleanup_dist_env_and_memory()
        self.rendezvous.cleanup()
