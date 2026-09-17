# Official DeepSeek-OCR encoding validation

Date: September 18, 2026 (Asia/Shanghai). Production correction: `cb60a114`.
Reference: `deepseek-ai/DeepSeek-OCR` at
`9f30c71f441d010e5429c532364a86705536c53a`. All execution used the server,
`uv`, and explicitly selected idle A6000 devices. No OCR decoder was trained or
evaluated for transcription quality.

## Result and correction

The configured 640×640 non-crop path has the same image preprocessing, SAM/CLIP
feature concatenation, projector, patch ordering, row newlines and final separator
as the official Small mode. It produces 100 patch features, 10 newline features
and one separator: `[image, 111, 1280]` at the adapter input.

The audit found two execution details missing from our implementation:

- The official `ToTensor` path produces contiguous CHW pixels. Our NHWC-to-NCHW
  view had channels-last strides. Equal pixel values did not imply identical
  convolution results under BF16 autocast.
- Official `infer` wraps generation, including vision and its projector, in BF16
  autocast. Our standalone vision forward omitted that scope. Merely storing
  weights and pixels in BF16 does not reproduce the same operator precision.

`image_pixels` now returns contiguous NCHW tensors, and `DeepSeekOCRVision.forward`
normalizes input layout and owns its BF16 autocast scope, including the projector.
An explicitly FP32 encoder stays FP32 for diagnostics. These changes affect
training and live optical inference through their shared encoder. They change the
numerical baseline; historical losses before this commit are not expected to match.

The existing eager QuickGELU and math SDPA policies remain. With those same
policies applied to the official reference, all six singleton fixtures match
**bitwise**, including official masked insertion into decoder input embeddings.
The comparison also passes with an enclosing BF16 autocast context.

| Fixture | Original size | BF16 pixels | 111 feature rows |
| --- | --- | --- | --- |
| Rendered 50-line training page | 1280×1430 | Exact | Exact |
| Rendered 8-line page | 1280×230 | Exact | Exact |
| RGB landscape | 257×193 | Exact | Exact |
| RGB portrait | 193×257 | Exact | Exact |
| Grayscale | 257×193 | Exact | Exact |
| RGBA | 257×193 | Exact | Exact |

The official non-crop mode directly stretches images at sizes up to 640, so our
stretching at this size is consistent with that mode. This conclusion does not
extend to larger official modes, which use aspect-preserving padding, or to
dynamic crops. The actual official dynamic path was also executed on the 50-line
page: a 1024 global view plus four 640 crops produced **693 tokens** (273 global,
420 local). Our fixed 111-token path does not implement that mode.

## Numerical differences that remain

Exact matching above controls for our existing eager activation change. The
**unmodified scripted activation** under official BF16 autocast produces different
features: relative L2 differences versus our eager version are 0.514% and 0.672%
for the two rendered pages, and 1.409–2.542% for the synthetic fixtures. Three
official singleton repetitions under autocast were themselves identical in this
check. The earlier warm-up diagnosis was made without the official autocast
contract; it does not establish that eager QuickGELU is necessary under the
corrected contract. We retain the existing policy rather than silently treating
the two numerical implementations as interchangeable.

Batch size also changes BF16 results. Six images were encoded as singletons and
as chunks of 4/2, with the rendered pages in the full chunk:

| Comparison | Relative L2 feature difference |
| --- | ---: |
| BF16 4/2 chunks versus singleton, all six images | 6.062% |
| BF16 batching, 50-line rendered page | 4.016% |
| BF16 batching, 8-line rendered page | 2.599% |
| FP32 4/2 chunks versus singleton, same quantized inputs/weights | 0.000926% |
| Repeating the same BF16 batches | 0, bitwise equal |
| Reversing images within each batch, then restoring order | 0, bitwise equal |

The first batch-size difference appears in SAM and is amplified by CLIP. Maximum
final absolute difference is 0.298828 in BF16, versus 0.000038564 in FP32. Relative
L2 here is `norm(actual - reference) / norm(reference)`, not an OCR error rate.
The FP32 control uses the same BF16-quantized checkpoint weights and pixels, then
changes arithmetic precision with TF32 disabled. It isolates arithmetic behavior;
it is not a claim about full-precision training accuracy.

These controls support a numerical-sensitivity explanation, while the exact
singleton comparison checks the encoding implementation. They do not prove that
the BF16 variation is harmless for optimization. Image batching, activation
implementation, attention backend and precision must be held fixed for controlled
comparisons. Even a fixed maximum image batch size can produce different final
chunk sizes. No convergence or OCR-quality claim follows from these tests.

## Reference boundary and reproduction

`tests/test_automodel_ocr_parity.py` parses the pinned source and executes the
original `infer` and `DeepseekOCRModel.forward` bodies. It uses the original
image-loading, transform, conversation and crop helpers. Generation is intercepted
after preprocessing, and the legacy text decoder base is replaced by a sink that
returns `inputs_embeds`. The actual vision code, projector, newline/separator
assembly and masked image insertion execute unchanged. The reference uses the
same 476 released checkpoint tensors; its extra persistent `position_ids` buffer
retains the official constructor's deterministic value. The eager-control run
changes only the module's QuickGELU binding; stock activation comparisons are
recorded separately.

This avoids importing the legacy HF text decoder into Transformers 5. It does
not run that decoder or compare generated OCR text. The six fixtures cover the
rendered-image contract; they do not audit arbitrary EXIF-bearing file inputs.

```bash
# Test-only dependency, outside the project environment/lockfile.
uv pip install --target outputs/automodel/ocr-parity-deps --no-deps torchvision==0.28.0
nvidia-smi
CUDA_VISIBLE_DEVICES=8 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 \
  PYTHONPATH=outputs/automodel/ocr-parity-deps \
  uv run --locked python tests/test_automodel_ocr_parity.py \
  configs/automodel.yaml outputs/automodel/ocr-parity-new-run
```

Use an unused output directory and currently idle device. Final report:
`/workspace/optical-adaptor/outputs/automodel/ocr-parity-complete/report.json`.
Earlier `ocr-parity-v1`, `ocr-parity-v2`, `ocr-parity-final`, and
`ocr-parity-autocast` artifacts preserve the diagnostic progression and failures;
they are not the final result. The final job is
`20260917-163600-ocr-aligned-full-validat-33a1a9`.

Pinned source SHA-256:

- `modeling_deepseekocr.py`: `5835e0b9e1942fe36df9123009d1d80a2b78ccbe6a2d535668a6561e3d4068b1`
- `deepencoder.py`: `0ae2fb6d1e5ae8cf100fc32f854830acd08c821a0a1f23a94a76588c222ddcf2`
- `conversation.py`: `ec7b6ce89bcda643de1f43269ffa66a7b2e65dc3ed30e427958f776546b4ba03`

Sources: [pinned official model](https://huggingface.co/deepseek-ai/DeepSeek-OCR/blob/9f30c71f441d010e5429c532364a86705536c53a/modeling_deepseekocr.py),
[official mode descriptions](https://github.com/deepseek-ai/DeepSeek-OCR#support-modes).
