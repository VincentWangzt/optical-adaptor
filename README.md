# Optical Adaptor

Train an optical MLP adapter from DeepSeek-OCR's frozen visual encoder into
Qwen3.5-4B, using online CE + teacher KL in NeMo AutoModel. The project also provides:

- `render-code` turns a source file into configurable, paginated Pillow images.
- `ocr-infer` runs a configured Qwen3.5 or DeepSeek OCR model on a document image.
- `ocr-edit-distance` compares reference and OCR text at character or word level.
- `ocr-evaluate` selects source text, renders it, runs OCR, and writes token/accuracy metrics.

The standalone OCR inference commands use vLLM's native multimodal implementations.

## Set up

Install the locked Linux training environment on the GPU server:

```bash
uv sync --locked --group dev
```

AutoModel is installed from the pinned editable checkout in `third_party/Automodel`, with Transformers 5.15.1, PyTorch 2.13/CUDA 13,
and FLA 0.5.2. For the standalone OCR inference commands, install the optional
`inference` extra with `uv sync --locked --extra inference` and retain that extra
when running them (`uv run --extra inference ...`).

## Train the optical adapter

Edit the variables at the top of [scripts/launch_optical_training.sh](scripts/launch_optical_training.sh)
and the full [configs/automodel.yaml](configs/automodel.yaml), then run on the server:

```bash
bash scripts/launch_optical_training.sh --smoke  # Three steps on two idle GPUs
bash scripts/launch_optical_training.sh          # Configured training schedule
```

The launcher prepares conversation slices, renders/encodes images online, starts
AutoModel DDP, logs to W&B, evaluates by slice, and writes resumable adapter checkpoints.
See [docs/training.md](docs/training.md) for data, masks, losses, resume, and limitations.
Measured results and validation scope are in
[docs/automodel-validation.md](docs/automodel-validation.md).

## Render code into images

```bash
uv run render-code examples/sample.py \
  --config configs/render.yaml \
  --output-dir outputs/rendered
```

Edit [`configs/render.yaml`](configs/render.yaml) to control:

- page resolution and DPI metadata;
- page margins, background, page numbering, and maximum pages;
- font/font size, line height, line width, tab width, wrapping, and line numbers;
- output format and filename pattern.

`text.font: null` uses Pillow's embedded font. Set it to a TTF/OTF path for a specific
monospaced face. Relative font paths are resolved from the YAML directory. `line_width` is a
maximum number of code-point columns, while `line_height` and `font_size` are pixels.

The retained default configuration renders 1280-by-1280 pages with bundled JetBrains Mono NL,
a 20-pixel font, 28-pixel line height, 100-column wrapping, and 1% margins. The renderer also
accepts `pages.resolution: auto`, pixel margins, percentage margins, or per-side combinations if
you need to customize this file later. Bundled font provenance and licenses are recorded in
[`assets/fonts/README.md`](assets/fonts/README.md).

## Run OCR

```bash
uv run --extra inference ocr-infer \
  outputs/rendered/sample-page-001.png \
  --config configs/inference.qwen3.5-4b.yaml \
  --gpu 0 \
  --output-dir outputs/inference
```

There is also one convenience script per model. Each supplies its model-specific configuration:

```bash
uv run --extra inference python scripts/infer_qwen3_5.py outputs/rendered/sample-page-001.png
uv run --extra inference python scripts/infer_deepseek_ocr.py outputs/rendered/sample-page-001.png
uv run --extra inference python scripts/infer_deepseek_ocr_2.py outputs/rendered/sample-page-001.png
```

Override the prompt, output budget, or vLLM execution mode:

```bash
uv run --extra inference ocr-infer image.png \
  --config configs/inference.qwen3.5-4b.yaml \
  --max-new-tokens 4096 \
  --enforce-eager \
  --prompt "Transcribe the image exactly."
```

Qwen3.5 thinks by default. The project passes `enable_thinking: false` for clean transcription;
use `--thinking` to enable it or `--no-thinking` to explicitly disable it.

Each model writes `<image>.<model>.md`; the command also writes a JSON summary containing the
model settings, prompt/image/output token counts, finish reason, timings, and peak allocated GPU
memory. DeepSeek's `image_tokens` count includes its projected local/global tokens and learned
newline/view-separator tokens exactly as passed into the language model. The adapter synchronizes
vLLM's native model constants with `base_size`, `image_size`, and `crop_mode` so processor
overrides are also honored by placeholder sizing and the vision encoder.

Available model-specific configs are:

- [`configs/inference.qwen3.5-4b.yaml`](configs/inference.qwen3.5-4b.yaml)
- [`configs/inference.deepseek-ocr.yaml`](configs/inference.deepseek-ocr.yaml)
- [`configs/inference.deepseek-ocr-2.yaml`](configs/inference.deepseek-ocr-2.yaml)

[`configs/inference.yaml`](configs/inference.yaml) remains the backward-compatible Qwen default.

## Evaluate OCR and token compression

Evaluate a whole file with Qwen:

```bash
uv run --extra inference ocr-evaluate examples/sample.py \
  --inference-config configs/inference.qwen3.5-4b.yaml \
  --output-dir outputs/evaluation-qwen
```

Or select inclusive line numbers and cap the rendered source at a tokenizer-exact budget:

```bash
uv run --extra inference ocr-evaluate examples/sample.py \
  --inference-config configs/inference.deepseek-ocr-2.yaml \
  --start-line 10 --end-line 80 \
  --max-source-tokens 512 \
  --output-dir outputs/evaluation-deepseek-2
```

The JSON report contains source token counts before and after truncation, total visual image
tokens, OCR output tokens, per-page measurements, and character/word edit distance. `image_tokens`
means the model's post-merge visual embeddings (not raw pixels or vision patches).

To compare any two existing text files independently:

```bash
uv run ocr-edit-distance reference.py outputs/result.md
uv run ocr-edit-distance reference.py outputs/result.md --unit word --json
```

The reusable helpers are in [`src/optical_adaptor/token_utils.py`](src/optical_adaptor/token_utils.py):
`count_tokens`, `truncate_to_tokens`, and `count_and_truncate` accept a Hugging Face tokenizer;
`load_tokenizer` loads one without loading model weights.

### Runtime note

DeepSeek's Hugging Face remote code still targets legacy Transformers, but it is not imported for
inference here. vLLM 0.28.0 loads the same checkpoint weights into its native DeepSeek-OCR and
DeepSeek-OCR-2 implementations. Qwen3.5 also uses its native vLLM implementation. The CUDA runtime
comes from vLLM's pinned PyTorch wheel and is independent of the system `nvcc`.

## Verify

```bash
uv run --no-sync pytest tests/test_automodel_processing.py tests/test_automodel_losses.py tests/test_automodel_data.py tests/test_automodel_mesh.py
uvx ruff check src/optical_adaptor/automodel tests/test_automodel_*.py
```

Upstream references:

- <https://huggingface.co/Qwen/Qwen3.5-4B>
- <https://huggingface.co/deepseek-ai/DeepSeek-OCR>
- <https://huggingface.co/deepseek-ai/DeepSeek-OCR-2>
- <https://docs.vllm.ai/en/stable/models/supported_models/>

Live multi-image reconstruction, 80-source-line reconstruction, LongCodeQA and the
standalone vLLM chat interface are documented in [docs/live-evaluation.md](docs/live-evaluation.md).
