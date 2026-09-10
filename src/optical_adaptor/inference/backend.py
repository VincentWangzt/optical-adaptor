"""Live image inference behind a task-independent chat request interface."""

from __future__ import annotations

import base64
import io
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from PIL import Image

from optical_adaptor.inference.messages import chat_ids
from optical_adaptor.training.config import file_sha256


@dataclass(frozen=True)
class ChatRequest:
    messages: list[dict]
    max_tokens: int
    seed: int


@dataclass(frozen=True)
class ChatResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    image_tokens: int
    images: int
    finish_reason: str
    elapsed_seconds: float

    def to_dict(self):
        return asdict(self)


def open_image(url: str) -> Image.Image:
    if url.startswith("data:image/"):
        header, data = url.split(",", 1)
        if not header.endswith(";base64"):
            raise ValueError("image data URLs must use base64")
        source = io.BytesIO(base64.b64decode(data, validate=True))
    elif url.startswith("file://"):
        parsed = urlparse(url)
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError("file URLs must be local")
        source = Path(unquote(parsed.path))
    else:
        raise ValueError("provide a base64 data:image URL or an absolute file:// URL")
    with Image.open(source) as image:
        return image.convert("RGB")


def normalize_messages(messages: list[dict]) -> tuple[list[dict], list[Image.Image]]:
    normalized, images = [], []
    for message in messages:
        if message["role"] not in {"system", "user", "assistant"}:
            raise ValueError("supported roles are system, user, assistant")
        content = message["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        parts = []
        for part in content:
            if part["type"] == "text":
                parts.append({"type": "text", "text": part["text"]})
            elif part["type"] == "image_url":
                images.append(open_image(part["image_url"]["url"]))
                parts.append({"type": "image"})
            else:
                raise ValueError(f"unsupported content type: {part['type']}")
        normalized.append({"role": message["role"], "content": parts})
    if not normalized:
        raise ValueError("messages must not be empty")
    return normalized, images


def expand_image_tokens(ids: list[int], image_token: int, lengths: list[int]) -> list[int]:
    if ids.count(image_token) != len(lengths):
        raise ValueError("chat template image placeholders do not match supplied images")
    expanded, index = [], 0
    for token in ids:
        if token == image_token:
            if lengths[index] < 1:
                raise ValueError("each image needs at least one embedding")
            expanded.extend([image_token] * lengths[index])
            index += 1
        else:
            expanded.append(token)
    return expanded


class VllmBackend:
    """Qwen3.5 native images, or live DeepSeek + adapter prompt embeddings.

    The full pinned Qwen checkpoint supplies the shared language weights. In the
    adapter path its native image tower is bypassed, and vLLM consumes the complete
    embedding sequence with ordinary sequential positions, as used in training.
    No training manifest, targets, or frozen feature caches enter this class.
    """

    def __init__(self, pipeline, settings, *, checkpoint: Path | None, seed: int):
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM

        self.settings = settings
        self.torch = torch
        model = pipeline.config.models
        self.tokenizer = AutoTokenizer.from_pretrained(model.qwen_id, revision=model.qwen_revision)
        self.image_token = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.end_token = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.checkpoint = checkpoint
        self.vision = None
        self.llm = LLM(
            model=model.qwen_id,
            revision=model.qwen_revision,
            tokenizer_revision=model.qwen_revision,
            dtype="bfloat16",
            seed=seed,
            tensor_parallel_size=1,
            max_model_len=settings.max_model_len,
            max_num_seqs=settings.max_num_seqs,
            max_num_batched_tokens=settings.max_num_batched_tokens,
            gpu_memory_utilization=settings.gpu_memory_utilization,
            enforce_eager=settings.enforce_eager,
            enable_prefix_caching=False,
            enable_prompt_embeds=True,
            mm_processor_cache_gb=0,
            limit_mm_per_prompt={"image": settings.max_images},
            skip_mm_profiling=True,
            mm_processor_kwargs={"cap_pixels_per_frame": False},
        )
        self.identity = {
            "backend": "vllm-live-adapter" if checkpoint else "vllm-native",
            "qwen_revision": model.qwen_revision,
            "encoder_revision": model.encoder_revision if checkpoint else None,
            "checkpoint_sha256": file_sha256(checkpoint / "adapter.safetensors")
            if checkpoint
            else None,
            "settings": settings.model_dump(),
            "vision_cache": False,
            "temperature": 0.0,
            "enable_thinking": False,
        }
        if checkpoint:
            self._load_adapter(pipeline, checkpoint)

    def _load_adapter(self, pipeline, checkpoint):
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        from safetensors.torch import load_file

        from optical_adaptor.training.models import DeepSeekVision, build_adapter

        torch = self.torch
        metadata = json.loads((checkpoint / "adapter.json").read_text())
        if metadata != {"kind": "mlp", "config": pipeline.config.adapter.model_dump()}:
            raise ValueError("checkpoint must be the configured MLP architecture")
        state = json.loads((checkpoint / "state.json").read_text())
        self.identity["checkpoint_state_sha256"] = file_sha256(checkpoint / "state.json")
        self.identity["checkpoint_state_step"] = state["step"]
        self.adapter = build_adapter("mlp", pipeline.config.adapter)
        self.adapter.load_state_dict(load_file(checkpoint / "adapter.safetensors"), strict=True)
        # Match training/evaluation: FP32 adapter parameters with BF16 autocast.
        self.adapter.to(device="cuda:0").requires_grad_(False).eval()
        self.vision = DeepSeekVision(pipeline, torch.device("cuda:0"))
        model = pipeline.config.models
        index = hf_hub_download(
            model.qwen_id, "model.safetensors.index.json", revision=model.qwen_revision
        )
        weights = json.loads(Path(index).read_text())["weight_map"]
        name = "model.language_model.embed_tokens.weight"
        path = hf_hub_download(model.qwen_id, weights[name], revision=model.qwen_revision)
        with safe_open(path, framework="pt", device="cpu") as source:
            self.embedding_table = source.get_tensor(name).to(torch.bfloat16)

    def _adapted_prompt(self, ids, images):
        torch = self.torch
        with torch.inference_mode():
            features = []
            for offset in range(0, len(images), self.settings.vision_batch_size):
                pixels = images[offset : offset + self.settings.vision_batch_size]
                encoded = self.vision(pixels)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    features.extend(self.adapter(encoded).cpu().unbind())
            expanded = expand_image_tokens(ids, self.image_token, [len(f) for f in features])
            embeds = self.embedding_table[expanded].clone()
            positions = [i for i, token in enumerate(expanded) if token == self.image_token]
            if features:
                embeds[positions] = torch.cat(features)
        # IDs without this explicit mask make vLLM ignore supplied prompt embeddings.
        return {
            "prompt_embeds": embeds,
            "prompt_token_ids": expanded,
            "prompt_is_token_ids": [token != self.image_token for token in expanded],
        }, len(positions)

    def generate(self, requests: list[ChatRequest]) -> list[ChatResponse]:
        from vllm import SamplingParams

        started = time.perf_counter()
        prompts, parameters, counts, visual_counts = [], [], [], []
        for request in requests:
            if request.max_tokens < 1:
                raise ValueError("max_tokens must be positive")
            messages, images = normalize_messages(request.messages)
            if len(images) > self.settings.max_images:
                raise ValueError("request exceeds configured maximum image count")
            ids = chat_ids(self.tokenizer, messages)
            if self.checkpoint and images:
                prompt, visual = self._adapted_prompt(ids, images)
            else:
                prompt, visual = {"prompt_token_ids": ids}, 0
                if images:
                    prompt["multi_modal_data"] = {"image": images}
                    # New UUIDs force live vision even when questions reuse repository pages.
                    prompt["multi_modal_uuids"] = {"image": [uuid.uuid4().hex for _ in images]}
            prompts.append(prompt)
            counts.append(len(images))
            visual_counts.append(visual)
            parameters.append(
                SamplingParams(
                    temperature=0.0,
                    max_tokens=request.max_tokens,
                    seed=request.seed,
                    stop_token_ids=[self.end_token],
                    skip_special_tokens=False,
                )
            )
        outputs = self.llm.generate(prompts, parameters, use_tqdm=False)
        elapsed = time.perf_counter() - started
        responses = []
        for output, count, visual in zip(outputs, counts, visual_counts, strict=True):
            choice = output.outputs[0]
            tokens = output.prompt_token_ids
            responses.append(
                ChatResponse(
                    text=choice.text,
                    prompt_tokens=len(tokens),
                    completion_tokens=len(choice.token_ids),
                    image_tokens=visual or tokens.count(self.image_token),
                    images=count,
                    finish_reason=choice.finish_reason,
                    elapsed_seconds=elapsed / len(outputs),
                )
            )
        return responses

    def close(self):
        from optical_adaptor.infer_ocr import _shutdown_vllm

        _shutdown_vllm(self.llm)
        if self.vision:
            self.vision.close()


BACKENDS = {"vllm-native": False, "vllm-adapter": True}


def build_backend(name, pipeline, settings, *, checkpoint: Path, seed: int):
    if name not in BACKENDS:
        raise ValueError(f"unknown inference backend: {name}")
    return VllmBackend(
        pipeline, settings, checkpoint=checkpoint if BACKENDS[name] else None, seed=seed
    )
