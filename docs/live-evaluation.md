# Live multi-image and LongCodeQA evaluation

The canonical settings are in `configs/benchmark.yaml`. The evaluated adapter is
`outputs/adapter-v1/runs/mlp-10epochs/step-002500`, with the pinned DeepSeek-OCR
encoder and Qwen3.5-4B language model from `configs/training.yaml`.

## Prepare data

On the GPU server, after pulling the committed code:

```bash
uv run prepare-live-benchmark --config configs/benchmark.yaml
```

Preparation reuses the verified training manifest and the already downloaded
`bigcode/the-stack-smol` source snapshot. It does not download another source
corpus or read cached vision embeddings.

- **2, 4, 8 images:** deterministically ordered, disjoint groups within each
  image-count suite, drawn from the held-out reconstruction split. Groups at
  larger image counts combine adjacent smaller groups. Each page keeps the
  original cached visual span; a reference joins page transcriptions with one
  newline. These are independent snippets, possibly from different languages,
  rather than consecutive pages from the same source file. The suites overlap
  with each other and must not be treated as independent samples when comparing
  image counts.
- **80 original source lines:** take the first 80 lines of each eligible held-out
  original source file. Reject shorter files; do not pad or repeat lines. Keep
  the training canonicalization, fonts, and wrapping. A source line longer than
  100 columns wraps, so these images can have more than 80 displayed lines.
- **LongCodeQA:** download `LongCodeQA.zip` from the pinned
  [Steefano/LCB snapshot](https://huggingface.co/datasets/Steefano/LCB/tree/989d5eff750d65a72c522e47f8f745ef2e22906b).
  Inspect every JSON shard and retain unique examples whose **full text prompt,
  including Qwen chat framing and generation prefix, is strictly below 32,768
  Qwen tokens**. The official prompt is preserved for the text control. For the
  image condition, replace only `repo_text` with pages of up to 80 original
  source lines; preserve all instructions, questions and answer choices.

Any held-out snippet with an exact visual-text match in training is excluded from
every reconstruction case containing it. The audit records exclusions, source
identities, dropped remainders, pinned revisions and checksums. This catches exact
overlap with this adapter's training manifest; it cannot establish absence from
Qwen pretraining or rule out semantic duplicates.

Images and their dimensions, source/display line counts, original references,
source provenance and hashes are saved under `outputs/benchmarks/live-2500`.
LongCodeQA image canonicalization expands tabs and escapes unsupported characters;
each case records whether rendering changed the repository text. PNG reuse is
allowed; model features are always computed live.

## Run inference

First verify GPUs 8 and 9 are idle with `nvidia-smi`, then start separate jobs:

```bash
CUDA_VISIBLE_DEVICES=8 uv run evaluate-live-benchmark --backend vllm-adapter
CUDA_VISIBLE_DEVICES=9 uv run evaluate-live-benchmark --backend vllm-native
```

The adapter runs image reconstruction and image QA. Native Qwen runs the same
images for both tasks, plus the official full-text QA prompt and a question-only
QA control with repository text removed. The latter retains the same question,
answer choices and instructions, and tests knowledge without supplied code.

All conditions use greedy decoding with thinking disabled. This differs from the
sampled generations logged during training. Reconstruction has a 20,000-token
output budget, checked against every reference during preparation; QA has 32
tokens. Generation-limit stops are reported rather than hidden. The 32K filter
defines the common **text-prompt cohort**; native image inputs can require more
tokens and have a separate 262,144-token context capacity. Inputs are never
silently truncated. The native processor uses its original image geometry;
DeepSeek resizes each image to the training 640-by-640 input. Token usage records
this unequal visual budget explicitly.

Results are written atomically per example under `results/<backend>/<mode>`.
Re-running the same command resumes missing examples after checking provenance.
`summary.json` reports completion counts, exact-match accuracy, corpus character
and word error rates, QA accuracy, invalid answers, visual token counts and
generation-limit stops. QA accepts a single A-D letter with optional terminal
`.` or `)`; commentary and ambiguous answers count as incorrect. Partial results
are explicitly labeled. Per-response time is an amortized batch wall time,
including preprocessing and live vision, rather than an individual latency.

For a small end-to-end check, use a separate output directory:

```bash
CUDA_VISIBLE_DEVICES=8 uv run evaluate-live-benchmark --backend vllm-adapter \
  --limit-per-suite 1 --output outputs/benchmarks/smoke-adapter
```

Use `--suites` and `--modes` to select conditions. `--report-only` aggregates saved
results without loading a model.

## Standalone inference interface

`optical_adaptor.inference.backend` accepts `ChatRequest(messages, max_tokens,
seed)` and returns text, usage and stop metadata. It has no dependency on benchmark
records, targets or feature caches. `build_backend` selects `vllm-adapter` or
`vllm-native`; both use the installed vLLM inference engine.

The adapter performs live DeepSeek vision inference, applies the MLP with the
training FP32-parameter/BF16-autocast convention, and replaces each image
placeholder with 111 embeddings. Every image retains its own vision-start/end
tokens. Adjacent image parts in a single message produce adjacent vision blocks,
exactly as verified against the pinned Qwen chat template. The adapter uses the
sequential positions used in training; native Qwen uses its own vision encoder and
spatial rotary positions, so identical chat framing does not imply identical
internal computation.

vLLM prefix and multimodal processor caches are disabled. Native image requests
receive fresh image identifiers to avoid encoder reuse across questions; the
adapter runs its vision tower for every request. Model weights and compiled kernels
are reused. No custom model architecture or serialized feature input is required:
the adapter backend supplies vLLM's supported `prompt_embeds` input.

An optional localhost endpoint accepts ordinary OpenAI-style message content:

```bash
CUDA_VISIBLE_DEVICES=8 uv run serve-optical --backend vllm-adapter --port 8008
curl http://127.0.0.1:8008/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"vllm-adapter","messages":[{"role":"user","content":[{"type":"text","text":"Transcribe exactly."},{"type":"image_url","image_url":{"url":"file:///workspace/optical-adaptor/outputs/example.png"}}]}],"max_tokens":4096,"temperature":0}'
```

Use base64 `data:image/...` URLs for client-provided images, or absolute `file://`
URLs for server-local files. Multiple images and multiple chat messages are
supported. The small endpoint serves one request at a time, supports greedy
non-streaming completions, and exposes `/health` and `/v1/models`. It is intended
for local integration; the batch evaluator calls the same backend directly.

## Validation

Run focused checks on the server:

```bash
uv run pytest tests/test_live_benchmark.py
CUDA_VISIBLE_DEVICES=8 uv run python scripts/validate_live_backend.py
```

The diagnostic compares one live feature tensor to its original training cache
and compares 64 greedy output tokens between vLLM and Transformers using the same
adapted prompt. Only this diagnostic reads the feature cache; evaluation does not.
