# Official DeepSeek-OCR encoding validation

Date: September 18, 2026 (Asia/Shanghai). Pixel/autocast correction: `cb60a114`;
official activation restored at `c7341327`.
Reference: `deepseek-ai/DeepSeek-OCR` at
`9f30c71f441d010e5429c532364a86705536c53a`. All execution used the server,
`uv`, and explicitly selected idle A6000 devices. No OCR decoder was trained or
evaluated for transcription quality.

## Current official precision contract

The eager QuickGELU override has been removed. The optical encoder now imports
the pinned official scripted activation unchanged and retains official BF16
autocast. This preserves the official mixed-precision policy: QuickGELU runs in
FP32, while eligible linear/convolution operations use BF16. It does not convert
the entire encoder to FP32. Encoder optimizations must preserve this official
precision behavior rather than substituting a mathematically identical formula
with different intermediate dtypes.
Checkpoint replay comparisons use the same code version on both sides. Loading
an older eager-activation checkpoint under the restored official path changes
future computations; exact continuation across that code change is not claimed.

The reason for the earlier disagreement is now directly established. Under CUDA
autocast, the TorchScript graph inserts `aten::_autocast_to_full_precision` around
QuickGELU; the eager replacement stays BF16. On 65,536 BF16 values in [-8, 8], the
official result was FP32 and matched the FP32 formula exactly, while eager output
was BF16 and differed by 0.0673% relative L2. That precision difference propagates
through CLIP. The original warm-up workaround was based on a standalone path
missing official autocast and was not a reason to keep changing activation
precision after autocast was corrected.

The updated regression executes the official reference without changing its
activation binding. All six fixtures below match **unmodified official features
bitwise** under the same math SDPA backend. It checks the actual tensors entering
each CLIP feed-forward second linear layer: every QuickGELU output is FP32.
Cold singleton repeats, full/partial image-batch repeats and within-batch
permutations also match exactly. Current feature report:
`/workspace/optical-adaptor/outputs/automodel/ocr-official-scripted/report.json`;
job `20260917-171806-official-scripted-ocr-pa-0973a7` completed successfully.

Batch-size sensitivity remains in the official mixed-precision encoder: 4/2
image chunks versus singletons differ by 6.282% relative L2 across all six
fixtures, and 4.001% / 2.582% on the two rendered pages. The FP32 diagnostic control
differs by 0.000926%. This is separate from the removed activation override;
holding the batch shapes fixed is repeatable. Full-run replay evidence is recorded
in the [parallel validation report](automodel-parallel-validation.md).

## Image path and original correction

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

All six singleton fixtures match the official scripted path **bitwise**, including
official masked insertion into decoder input embeddings, under the same math
SDPA backend. The comparison also passes with an enclosing BF16 autocast context.

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

## Historical measurements with the removed eager override

Before `c7341327`, exact matching controlled for our eager activation change. The
**unmodified scripted activation** under official BF16 autocast produces different
features: relative L2 differences versus our eager version are 0.514% and 0.672%
for the two rendered pages, and 1.409–2.542% for the synthetic fixtures. Three
official singleton repetitions under autocast were themselves identical in this
check. The earlier warm-up diagnosis was made without the official autocast
contract; it does not establish that eager QuickGELU is necessary under the
corrected contract. The override has now been removed; these historical
comparisons explain why it was not numerically interchangeable with the official
implementation.

With the former eager activation, six images were encoded as singletons and
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
retains the official constructor's deterministic value. The current test never
replaces the reference activation and verifies FP32 activation output in the
actual CLIP blocks. Earlier eager-control reports are retained as history.

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

Use an unused output directory and currently idle device. Current report:
`/workspace/optical-adaptor/outputs/automodel/ocr-official-scripted/report.json`.
Earlier `ocr-parity-v1`, `ocr-parity-v2`, `ocr-parity-final`, and
`ocr-parity-autocast` artifacts preserve the diagnostic progression and failures;
they are not the final result. `ocr-parity-complete` and job
`20260917-163600-ocr-aligned-full-validat-33a1a9` record the intermediate correction
that still used eager QuickGELU.

Pinned source SHA-256:

- `modeling_deepseekocr.py`: `5835e0b9e1942fe36df9123009d1d80a2b78ccbe6a2d535668a6561e3d4068b1`
- `deepencoder.py`: `0ae2fb6d1e5ae8cf100fc32f854830acd08c821a0a1f23a94a76588c222ddcf2`
- `conversation.py`: `ec7b6ce89bcda643de1f43269ffa66a7b2e65dc3ed30e427958f776546b4ba03`

Sources: [pinned official model](https://huggingface.co/deepseek-ai/DeepSeek-OCR/blob/9f30c71f441d010e5429c532364a86705536c53a/modeling_deepseekocr.py),
[official mode descriptions](https://github.com/deepseek-ai/DeepSeek-OCR#support-modes).
