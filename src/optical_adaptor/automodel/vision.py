"""DeepSeek's standalone torch encoder; no decoder, vLLM, or process groups."""

from __future__ import annotations

import importlib.util
import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from easydict import EasyDict
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors import safe_open
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel


def image_pixels(images: list[Image.Image], image_size: int) -> torch.Tensor:
    """CPU preprocessing shared by the training processor and live inference."""
    arrays = [
        np.asarray(image.convert("RGB").resize((image_size, image_size))).copy() for image in images
    ]
    # The official ToTensor path produces contiguous CHW pixels. Under BF16
    # autocast a different convolution layout changes the encoder's rounding.
    return (
        torch.from_numpy(np.stack(arrays))
        .permute(0, 3, 1, 2)
        .contiguous()
        .float()
        .div_(127.5)
        .sub_(1)
    )


class DeepSeekOCRVision(nn.Module):
    def __init__(
        self,
        model_id: str,
        revision: str,
        image_size: int,
        output_dim: int,
        tokens_per_image: int,
        expected_tensors: int,
        sdpa_backend: Literal["math", "auto"],
    ):
        super().__init__()
        if sdpa_backend not in {"math", "auto"}:
            raise ValueError(f"Unsupported vision SDPA backend: {sdpa_backend}")
        self.sdpa_backend = sdpa_backend
        # deepencoder.py at this pinned revision imports only torch, einops and
        # easydict. Importing modeling_deepseekocr would load the obsolete HF
        # decoder API, which is incompatible with Transformers 5.
        if len(revision) != 40:
            raise ValueError("DeepSeek executable encoder code requires a full commit SHA")
        path = hf_hub_download(model_id, "deepencoder.py", revision=revision)
        spec = importlib.util.spec_from_file_location("optical_deepseek_encoder", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import DeepSeek encoder from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Preserve the official scripted activation. Under official autocast,
        # TorchScript promotes QuickGELU to FP32; an eager replacement does not.
        self.sam_model = module.build_sam_vit_b()
        self.vision_model = module.build_clip_l()
        # The native module persists this deterministic arange buffer, but the
        # released checkpoint (and vLLM port) intentionally omit it.
        self.vision_model.embeddings._non_persistent_buffers_set.add("position_ids")
        self.projector = module.MlpProjector(
            EasyDict(projector_type="linear", input_dim=2048, n_embed=output_dim)
        )
        self.image_newline = nn.Parameter(torch.empty(output_dim))
        # Spelling is part of the upstream checkpoint contract.
        self.view_seperator = nn.Parameter(torch.empty(output_dim))
        self.image_size = image_size
        self.output_dim = output_dim
        self.tokens_per_image = tokens_per_image
        index = hf_hub_download(model_id, "model.safetensors.index.json", revision=revision)
        weight_map = json.loads(Path(index).read_text())["weight_map"]
        keys = {"model." + name for name in self.state_dict()}
        if len(keys) != expected_tensors or not keys <= weight_map.keys():
            raise ValueError("DeepSeek encoder tensor names/count differ from the pinned contract")
        state = {}
        for filename in sorted({weight_map[key] for key in keys}):
            weights = hf_hub_download(model_id, filename, revision=revision)
            with safe_open(weights, framework="pt", device="cpu") as checkpoint:
                for key in sorted(keys):
                    if weight_map[key] == filename:
                        state[key.removeprefix("model.")] = checkpoint.get_tensor(key)
        self.load_state_dict(state, strict=True, assign=True)
        self.requires_grad_(False).eval()

    @torch.no_grad()
    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode frozen image features in the pinned checkpoint's token order.

        Args:
            pixels: Normalized images [images, channels, height, width].

        Returns:
            Features [images, tokens_per_image, output_dim], including row
            newlines and the final separator, in the encoder's dtype/device.
        """
        pixels = pixels.to(self.image_newline).contiguous()
        # Scope the chosen numerical reference to vision: the native text tower
        # still needs AutoModel's CP-compatible attention backend.
        context = sdpa_kernel(SDPBackend.MATH) if self.sdpa_backend == "math" else nullcontext()
        # Official infer wraps the full vision path, including the projector,
        # in BF16 autocast. Keep that contract independent of recipe wrapping.
        # An explicitly FP32 encoder remains FP32 for numerical diagnostics.
        with (
            context,
            torch.autocast("cuda", dtype=torch.bfloat16, enabled=pixels.dtype == torch.bfloat16),
        ):
            sam = self.sam_model(pixels)
            clip = self.vision_model(pixels, sam)
            features = self.projector(torch.cat([clip[:, 1:], sam.flatten(2).transpose(1, 2)], -1))
        batch, count, width = features.shape
        side = math.isqrt(count)
        if side * side != count:
            raise ValueError("DeepSeek returned a non-square patch grid")
        rows = features.reshape(batch, side, side, width)
        newline = self.image_newline.reshape(1, 1, 1, width).expand(batch, side, 1, width)
        sequence = torch.cat([rows, newline], dim=2).reshape(batch, -1, width)
        separator = self.view_seperator.reshape(1, 1, width).expand(batch, 1, width)
        result = torch.cat([sequence, separator], dim=1)
        if result.shape[1:] != (self.tokens_per_image, self.output_dim):
            raise ValueError(f"Unexpected DeepSeek feature shape: {tuple(result.shape)}")
        return result
