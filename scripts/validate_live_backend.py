"""Check live features against the training cache and vLLM against Transformers.

This is a one-record diagnostic; benchmark inference never reads feature caches.
Run on a free GPU with CUDA_VISIBLE_DEVICES set explicitly.
"""

import gc
import json
from pathlib import Path

import torch

from optical_adaptor.benchmark.config import load_benchmark
from optical_adaptor.benchmark.prepare import save_image
from optical_adaptor.inference.backend import ChatRequest, build_backend, normalize_messages
from optical_adaptor.inference.messages import chat_ids
from optical_adaptor.training.cache import TensorCache
from optical_adaptor.training.config import load_credentials, write_json
from optical_adaptor.training.data import load_manifest
from optical_adaptor.training.models import FrozenQwen


def main():
    config, pipeline, directory = load_benchmark(Path("configs/benchmark.yaml"))
    load_credentials(pipeline, wandb=False)
    records = load_manifest(pipeline)
    record = next(r for r in records if r["split"] == "reconstruction")
    metadata = save_image(record["visual"], directory, pipeline)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": pipeline.config.evaluation.instruction},
                {
                    "type": "image_url",
                    "image_url": {"url": (directory / metadata["path"]).as_uri()},
                },
            ],
        }
    ]
    backend = build_backend(
        "vllm-adapter",
        pipeline,
        config.backend,
        checkpoint=pipeline.repo / config.checkpoint,
        seed=config.seed,
    )
    normalized, images = normalize_messages(messages)
    ids = chat_ids(backend.tokenizer, normalized)
    prompt, image_tokens = backend._adapted_prompt(ids, images)
    embeddings = prompt["prompt_embeds"]
    response = backend.generate([ChatRequest(messages, 64, config.seed)])[0]
    # Match the training forward convention before testing the language backend.
    cache = TensorCache(pipeline, records)
    cached = cache.batch([record], "encoder", torch.device("cuda:0"))[0]
    live = backend.vision(images)[0]
    difference = (live.float() - cached.float()).abs()
    report = {
        "image_tokens": image_tokens,
        "live_cache_max_abs": difference.max().item(),
        "live_cache_mean_abs": difference.mean().item(),
        "vllm": response.to_dict(),
    }
    backend.vision.cpu()
    backend.adapter.cpu()
    backend.close()
    del backend, live, cached
    gc.collect()
    torch.cuda.empty_cache()
    qwen = FrozenQwen(pipeline, torch.device("cuda:0"))
    qwen.model.gradient_checkpointing_disable()
    with torch.inference_mode():
        generated = qwen.model.generate(
            inputs_embeds=embeddings.to("cuda:0").unsqueeze(0),
            attention_mask=torch.ones((1, len(embeddings)), device="cuda:0", dtype=torch.long),
            max_new_tokens=64,
            do_sample=False,
            use_cache=True,
            eos_token_id=qwen.assistant_end,
            pad_token_id=qwen.tokenizer.pad_token_id,
        )[0].tolist()
    if qwen.assistant_end in generated:
        generated = generated[: generated.index(qwen.assistant_end)]
    text = qwen.tokenizer.decode(
        generated, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    report["transformers_text"] = text
    report["greedy_text_equal"] = text == response.text
    report["expected_prefix"] = record["visual"][:300]
    write_json(directory / "validation" / "live-backend.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if difference.max().item() > 0.1 or not report["greedy_text_equal"]:
        raise RuntimeError("backend parity check requires investigation; see diagnostic JSON")


if __name__ == "__main__":
    main()
