"""Opt-in GPU comparison with the pinned official OCR infer/vision forward bodies.

The legacy text decoder is replaced by a boundary that returns inputs_embeds.
Official preprocessing, vision construction, forward, and masked image insertion
run unchanged. No OCR decoding/quality claim is made by this diagnostic.
Run with torchvision installed and one explicitly selected idle GPU.
"""

import __future__

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import sys
from abc import ABC
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from easydict import EasyDict
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from optical_adaptor.automodel.vision import DeepSeekOCRVision, image_pixels, quick_gelu
from optical_adaptor.renderer import load_render_config, render_pages


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class DecoderBoundary(nn.Module):
    """Stop the official model exactly where its text decoder receives embeddings."""

    def __init__(self, config):
        super().__init__()

    def get_input_embeddings(self):
        return lambda ids: torch.zeros(
            *ids.shape,
            self.image_newline.numel(),
            device=ids.device,
            dtype=self.image_newline.dtype,
        )

    def forward(self, *, inputs_embeds, **kwargs):
        return inputs_embeds


class CaptureInputs(Exception):
    """Intentional stop before official generation, carrying its prepared inputs."""

    def __init__(self, input_ids, kwargs):
        self.input_ids, self.kwargs = input_ids, kwargs


class InferBoundary:
    def disable_torch_init(self):
        # The official method globally disables initialization; unnecessary here.
        pass

    def generate(self, input_ids, **kwargs):
        raise CaptureInputs(input_ids, kwargs)


def official_reference(config):
    """Compile complete original methods, without importing the legacy HF decoder."""
    from torchvision import transforms

    paths = {
        name: Path(hf_hub_download(config["model_id"], name, revision=config["revision"]))
        for name in ("modeling_deepseekocr.py", "deepencoder.py", "conversation.py")
    }
    encoder = import_file("ocr_reference_deepencoder", paths["deepencoder.py"])
    conversation = import_file("ocr_reference_conversation", paths["conversation.py"])
    namespace = {
        "__name__": "ocr_reference",
        "torch": torch,
        "nn": nn,
        "math": math,
        "os": os,
        "Image": Image,
        "ImageOps": ImageOps,
        "ABC": ABC,
        "transforms": transforms,
        "Dict": EasyDict,
        "DeepseekV2Model": DecoderBoundary,
        "DeepseekOCRConfig": SimpleNamespace,
        "build_sam_vit_b": encoder.build_sam_vit_b,
        "build_clip_l": encoder.build_clip_l,
        "MlpProjector": encoder.MlpProjector,
        "get_conv_template": conversation.get_conv_template,
    }
    required = {
        "load_image",
        "load_pil_images",
        "find_closest_aspect_ratio",
        "dynamic_preprocess",
        "normalize_transform",
        "format_messages",
        "text_encode",
        "BaseTransform",
        "BasicImageTransform",
        "DeepseekOCRModel",
    }
    tree = ast.parse(paths["modeling_deepseekocr.py"].read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in required]
    assert {node.name for node in nodes} == required
    causal = next(
        node for node in tree.body if getattr(node, "name", None) == "DeepseekOCRForCausalLM"
    )
    infer = next(node for node in causal.body if getattr(node, "name", None) == "infer")
    # Compile the unchanged infer method at module scope for binding to our sink.
    nodes.append(infer)
    code = compile(
        ast.Module(body=nodes, type_ignores=[]),
        str(paths["modeling_deepseekocr.py"]),
        "exec",
        flags=__future__.annotations.compiler_flag,
    )
    exec(code, namespace)
    return (
        namespace,
        encoder,
        {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
    )


def official_inputs(namespace, image_path, output, size, *, crop_mode=False, base_size=None):
    sink = InferBoundary()
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [2] * len(text), eos_token_id=1)
    try:
        namespace["infer"](
            sink,
            tokenizer,
            prompt="<image>\nFree OCR.",
            image_file=str(image_path),
            output_path=str(output),
            base_size=size if base_size is None else base_size,
            image_size=size,
            crop_mode=crop_mode,
            eval_mode=True,
        )
    except CaptureInputs as captured:
        return captured.input_ids, captured.kwargs
    raise AssertionError("Official infer did not reach generation")


def difference(actual, expected):
    delta = actual.float() - expected.float()
    return {
        "equal": torch.equal(actual, expected),
        "shape": list(actual.shape),
        "max_abs": delta.abs().max().item(),
        "relative_l2": (delta.norm() / expected.float().norm().clamp_min(1e-12)).item(),
    }


def encode_stages(model, pixels, chunk_size):
    """Capture the first stage affected by changing only image batch size."""
    captured = {name: [] for name in ("sam_model", "vision_model", "projector")}
    handles = [
        getattr(model, name).register_forward_hook(
            lambda module, inputs, output, name=name: captured[name].append(output.detach().clone())
        )
        for name in captured
    ]
    try:
        features = torch.cat([model(chunk) for chunk in pixels.split(chunk_size)])
    finally:
        for handle in handles:
            handle.remove()
    return {**{name: torch.cat(values) for name, values in captured.items()}, "features": features}


def fixtures(config):
    y, x = np.indices((193, 257))
    ramp = np.stack([x % 256, y % 256, (x + y) % 256], axis=-1).astype(np.uint8)
    rendered = {}
    for name, lines in (("rendered_training_page", 50), ("rendered_short_page", 8)):
        pages, _, truncated = render_pages(
            "\n".join(
                f"def function_{i}(value): return value + {i}  # OCR parity" for i in range(lines)
            ),
            config=load_render_config(config["optical"]["render_config"]),
        )
        assert not truncated and len(pages) == 1
        rendered[name] = pages[0]
    # Six inputs create 4/2 image chunks, exercising rendered pages in a real
    # batch and a non-singleton partial chunk.
    return {
        **rendered,
        "color_landscape": Image.fromarray(ramp),
        "color_portrait": Image.fromarray(ramp).transpose(Image.Transpose.ROTATE_90),
        "grayscale": Image.fromarray(ramp[:, :, 0]),
        "rgba": Image.fromarray(ramp).convert("RGBA"),
    }


@torch.no_grad()
def compare(config_path, output):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(config_path.read_text())
    vision_config = config["optical"]["vision"]
    namespace, official_encoder, source_hashes = official_reference(vision_config)
    actual = DeepSeekOCRVision(**vision_config).cuda().bfloat16().eval()
    reference = namespace["DeepseekOCRModel"](SimpleNamespace())
    # This deterministic buffer is registered persistently upstream but absent
    # from the released checkpoint. Keep the original constructor's value.
    reference_state = dict(actual.state_dict())
    reference_state["vision_model.embeddings.position_ids"] = (
        reference.vision_model.embeddings.position_ids
    )
    reference.load_state_dict(reference_state, strict=True, assign=True)
    reference = reference.cuda().bfloat16().requires_grad_(False).eval()
    report = {
        "revision": vision_config["revision"],
        "source_sha256": source_hashes,
        "reference_boundary": "unchanged official infer + OCRModel.forward, before text decoder",
        "checkpoint_tensors": len(actual.state_dict()),
        "official_extra_buffer": "vision_model.embeddings.position_ids",
        "images": {},
    }

    def reference_forward(ids, prepared):
        result = reference(
            input_ids=ids,
            images=prepared["images"],
            images_seq_mask=prepared["images_seq_mask"],
            images_spatial_crop=prepared["images_spatial_crop"],
        )
        return result[prepared["images_seq_mask"]].unsqueeze(0)

    prepared_images = []
    expected_features = []
    for name, image in fixtures(config).items():
        image_path = output / f"{name}.png"
        image.save(image_path)
        ids, prepared = official_inputs(namespace, image_path, output / name, actual.image_size)
        expected_pixels = prepared["images"][0][1]
        pixels = image_pixels([image], actual.image_size).cuda().bfloat16()
        pixel_diff = difference(pixels, expected_pixels)
        assert pixel_diff["equal"], (name, pixel_diff)
        with sdpa_kernel(SDPBackend.MATH):
            if not report["images"]:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    stock = [reference_forward(ids, prepared) for _ in range(3)]
                report["stock_scripted_repeats"] = [difference(value, stock[0]) for value in stock]
            # Isolate our documented activation change from encoding-path parity.
            official_encoder.quick_gelu = quick_gelu
            with torch.autocast("cuda", dtype=torch.bfloat16):
                expected = reference_forward(ids, prepared)
            features = actual(pixels)
            # The official infer method wraps generation (and thus the model
            # prefill) in BF16 autocast. Check that execution contract too.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                official_autocast = reference_forward(ids, prepared)
                actual_autocast = actual(pixels)
        feature_diff = difference(features, expected)
        assert feature_diff["equal"], (name, feature_diff)
        side = math.isqrt(actual.tokens_per_image - 1)
        torch.testing.assert_close(
            features[:, side :: side + 1], actual.image_newline.expand(1, side, -1), rtol=0, atol=0
        )
        torch.testing.assert_close(features[:, -1], actual.view_seperator[None], rtol=0, atol=0)
        report["images"][name] = {
            "original_size": list(image.size),
            "pixels": pixel_diff,
            "features": feature_diff,
            "official_autocast_vs_optical": difference(features, official_autocast),
            "matched_autocast_paths": difference(actual_autocast, official_autocast),
            "official_image_tokens": int(prepared["images_seq_mask"].sum()),
        }
        prepared_images.append(pixels)
        expected_features.append(expected)
    pixels = torch.cat(prepared_images)
    expected = torch.cat(expected_features)
    # Match the real image microbatch boundary and compare each image's position.
    chunk_size = config["optical"]["processing"]["image_microbatch_size"]
    single_stages = encode_stages(actual, pixels, 1)
    batch_stages = encode_stages(actual, pixels, chunk_size)
    report["bf16_batch_vs_single"] = {
        name: difference(batch_stages[name], single_stages[name]) for name in single_stages
    }
    report["bf16_batch_per_image"] = {
        name: difference(batch_stages["features"][index], single_stages["features"][index])
        for index, name in enumerate(report["images"])
    }
    torch.testing.assert_close(single_stages["features"], expected, rtol=0, atol=0)
    repeated = encode_stages(actual, pixels, chunk_size)
    report["bf16_batched_repeat"] = difference(repeated["features"], batch_stages["features"])
    torch.testing.assert_close(repeated["features"], batch_stages["features"], rtol=0, atol=0)
    # Reverse within each chunk, keeping each image's batch size fixed, to check
    # image ownership/order separately from batch-shape numerical sensitivity.
    permuted = torch.cat([actual(chunk.flip(0)).flip(0) for chunk in pixels.split(chunk_size)])
    report["bf16_batched_permutation"] = difference(permuted, batch_stages["features"])
    torch.testing.assert_close(permuted, batch_stages["features"], rtol=0, atol=0)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    # The official dynamic mode is a separate contract: verify its larger token
    # count and encoding boundary, without pretending our fixed path implements it.
    image_path = output / "rendered_training_page.png"
    ids, prepared = official_inputs(
        namespace, image_path, output / "dynamic", 640, crop_mode=True, base_size=1024
    )
    with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", dtype=torch.bfloat16):
        dynamic = reference_forward(ids, prepared)
    report["official_dynamic_mode"] = {
        "features": list(dynamic.shape),
        "crop_grid": prepared["images_spatial_crop"].tolist(),
        "local_images": int(prepared["images"][0][0].shape[0]),
        "image_tokens": int(prepared["images_seq_mask"].sum()),
        "supported_by_fixed_optical_path": False,
    }

    # Use identical, already BF16-quantized pixels and weights, changing only
    # arithmetic precision. This measures numerical error, not a new image path.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    actual.float()
    fp32_single = encode_stages(actual, pixels.float(), 1)
    fp32_batch = encode_stages(actual, pixels.float(), chunk_size)
    report["fp32_batch_vs_single"] = {
        name: difference(fp32_batch[name], fp32_single[name]) for name in fp32_single
    }
    report["bf16_vs_fp32"] = {
        "single": difference(single_stages["features"], fp32_single["features"]),
        "batch": difference(batch_stages["features"], fp32_single["features"]),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    assert report["fp32_batch_vs_single"]["features"]["relative_l2"] < 1e-4
    assert all(item["official_autocast_vs_optical"]["equal"] for item in report["images"].values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    compare(args.config, args.output)
