from __future__ import annotations

import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

from optical_adaptor.renderer import render_pages


class NativeQwen:
    def __init__(self, pipeline, qwen):
        self.pipeline, self.qwen = pipeline, qwen
        config = pipeline.config.models
        full_config = AutoConfig.from_pretrained(config.qwen_id, revision=config.qwen_revision)
        self.processor = AutoProcessor.from_pretrained(
            config.qwen_id, revision=config.qwen_revision
        )
        # Reuse the existing frozen text weights and load only the native vision tower.
        with torch.device("meta"):
            self.model = Qwen3_5ForConditionalGeneration(full_config)
        self.model.model.language_model = qwen.model.model
        self.model.lm_head = qwen.model.lm_head
        self.model.model.visual = Qwen3_5VisionModel(full_config.vision_config)
        index_path = hf_hub_download(
            config.qwen_id, "model.safetensors.index.json", revision=config.qwen_revision
        )
        weight_map = json.loads(Path(index_path).read_text())["weight_map"]
        names = [name for name in weight_map if name.startswith("model.visual.")]
        weights = {}
        for filename in sorted({weight_map[name] for name in names}):
            path = hf_hub_download(config.qwen_id, filename, revision=config.qwen_revision)
            with safe_open(path, framework="pt", device="cpu") as checkpoint:
                for name in names:
                    if weight_map[name] == filename:
                        weights[name.removeprefix("model.visual.")] = checkpoint.get_tensor(name)
        self.model.model.visual.load_state_dict(weights, strict=True)
        self.model.model.visual.to(device=qwen.device, dtype=torch.bfloat16)
        self.model.requires_grad_(False).eval()
        if any(value.is_meta for value in self.model.parameters()):
            raise RuntimeError("unloaded native Qwen parameters")

    def inputs(self, record, task, *, targets: bool):
        image = render_pages(record["visual"], config=self.pipeline.render)[0][0]
        processed = self.processor(
            text="<|vision_start|><|image_pad|><|vision_end|>", images=image, return_tensors="pt"
        ).to(self.qwen.device)
        image_ids = processed["input_ids"][0].tolist()
        if image_ids[0] != self.qwen.vision_start or image_ids[-1] != self.qwen.vision_end:
            raise ValueError("unexpected native image placeholder template")
        if task == "continuation":
            ids = record["prefix_ids"] + image_ids
            target = record["continuation_ids"]
        else:
            ids = self.qwen.reconstruction_before + image_ids[1:-1] + self.qwen.reconstruction_after
            target = record["visual_ids"] + [self.qwen.assistant_end]
        if targets:
            ids += target[:-1]
        ids = torch.tensor([ids], device=self.qwen.device)
        values = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "pixel_values": processed["pixel_values"],
            "image_grid_thw": processed["image_grid_thw"],
            "mm_token_type_ids": (ids == self.model.config.image_token_id).long(),
        }
        return values, target, len(image_ids) - 2

    @torch.no_grad()
    def hidden(self, record, task):
        values, target, visual_tokens = self.inputs(record, task, targets=True)
        self.model.eval()
        hidden = self.model.model(**values, use_cache=False).last_hidden_state[0, -len(target) :]
        mask = [True] * len(target)
        if task == "reconstruction":
            mask[-1] = False
        return (
            hidden,
            torch.tensor(target, device=self.qwen.device),
            torch.tensor(mask, device=self.qwen.device),
            visual_tokens,
        )

    @torch.no_grad()
    def generate(self, record):
        values, _, _ = self.inputs(record, "reconstruction", targets=False)
        prefix_length = values["input_ids"].shape[1]
        self.model.eval()
        output = self.model.generate(
            **values,
            do_sample=False,
            use_cache=True,
            max_new_tokens=self.pipeline.config.evaluation.max_new_tokens,
            eos_token_id=self.qwen.assistant_end,
            pad_token_id=self.qwen.tokenizer.pad_token_id,
        )[0, prefix_length:].tolist()
        stopped = bool(output and output[-1] == self.qwen.assistant_end)
        if stopped:
            output = output[:-1]
        return self.qwen.tokenizer.decode(
            output, skip_special_tokens=False, clean_up_tokenization_spaces=False
        ), not stopped
