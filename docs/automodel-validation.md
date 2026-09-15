# AutoModel pipeline validation

Validated on 2026-09-16 (Asia/Shanghai), using two NVIDIA RTX A6000 GPUs, physical
IDs 8 and 9. Other devices were left alone. The main smoke and accumulation runs
used `95fc9a0`; the distributed preflight follow-up used `d8c2caa`.
Python execution and tests ran on the Linux server after GitHub synchronization.

## Environment and encoder

- NeMo AutoModel: `2c0df17efff1fc9d5128a9df1a2ed716d0f2c54a`.
- Transformers 5.15.1, PyTorch 2.13.0/CUDA 13, FLA 0.5.2; `uv sync --locked` passed.
- Native DeepSeek-OCR encoder: all 476 released tensors loaded. A real image
  produced finite BF16 features of shape `[1, 111, 1280]`.
- Qwen3.5-4B text backbone: 4,205,751,296 frozen parameters.
- MLP: 9,838,080 trainable parameters. Runtime assertions prohibit encoder/backbone
  parameter gradients. The optimizer and model checkpoint contain only the MLP.
- NCCL P2P was disabled because the server's GPU 8/9 P2P path hung at the first
  collective. Shared-memory communication completed the distributed runs.

## Data and focused checks

The final smoke preparation read 64 original records from each of the three
sources and wrote 4,649 derived SFT records. All 128 selected SWE trajectories
have a complete `full` record, and all original source rows are retained.
Repository overlap between train and eval was zero.

Dataset fingerprint:
`9c0d5f66a2c5db055a272c14bed16df2c8ee91f89296bd575a308e81e614e79f`.

The smoke's maximum of four images and 4,096 tokens per branch left 1,690 training
records across 32 slices. Evaluation selected one eligible example from each of
30 slices, totaling 19,120 supervised tokens. Preflight rejected 860 train and
86 eval records for length after the image filter. These were whole-example
rejections; stored conversations were not shortened.

All **21 focused tests passed**:

```bash
uv run --no-sync pytest tests/test_automodel_processing.py tests/test_automodel_losses.py tests/test_automodel_data.py tests/test_training_data.py
```

They cover whitespace/Unicode target identity, causal positions, all/last assistant
masks, historical reasoning, tool argument normalization, complete trajectories,
short-file reconstruction, whole-example rejection, dense-versus-chunked KL/CE
values and gradients at temperature 1.7, unequal-microbatch token normalization,
preparation identity, and data-loader recovery on both ranks with zero/two workers.
Real-data snippets also verified full/window view filters and exactly-one-image
selection. Ruff check and format check passed for all 18 changed Python files.

## End-to-end launcher

```bash
bash scripts/launch_optical_training.sh --smoke
```

The launcher installed the locked environment, prepared data, trained for three
optimizer steps, ran teacher-forced and generative evaluation, logged to W&B,
and exported an AutoModel checkpoint plus consolidated adapter safetensors.

| Optimizer step (zero-based) | CE | KL | Gradient norm before clipping |
| --- | ---: | ---: | ---: |
| 0 | 0.793155 | 0.071537 | 0.792136 |
| 1 | 1.018087 | 1.013772 | 2.000436 |
| 2 | 2.161528 | 1.377362 | 9.087356 |

The final three-step evaluation reported CE **1.069183**, KL **0.602130**, and
teacher CE **0.492583**. The highest logged training allocation was 10.01 GiB on
the logging rank; this does not measure the full 32K-context configuration.

Ten greedy reconstruction generations were scored. CER was **0.970622**, LER
**0.998626**, and **30%** reached the 64-token smoke generation cap. Reconstruction
quality is still poor after these few updates. The cap also makes these numbers
unsuitable for a full-document quality comparison.

[W&B smoke run](https://wandb.ai/2162681069-peking-university/optical-adaptor/runs/awhpz2s9).
Server artifacts are under `/workspace/optical-adaptor/outputs/automodel/smoke/`.
Source text and generated outputs remain in the server artifacts, not W&B.

## Resume and gradient accumulation

Loading `LATEST` from the three-step smoke checkpoint restored the adapter,
optimizer, LR scheduler, RNG, and per-rank data-loader state. The next optimizer
step was 3, and both sampler cursors advanced from 3 to 4. Consolidated adapter
weights stayed finite and changed by an L2 norm of **0.003110** during that step.
The existing W&B run resumed. A new checkpoint was saved at `epoch_0_step_3`.
Its evaluation CE/KL were **1.066194 / 0.598832**.

The continuation preserved the original three-step LR decay horizon, leaving the
additional step at the minimum LR, **2e-6**. Raising `max_steps` extends the step
budget, not the checkpoint's learning-rate schedule.

A separate run used global batch size 4, local batch size 1, two GPUs, and two
microbatches per optimizer step. It completed one optimizer step, evaluation,
and checkpoint export with finite CE **0.889079**, KL **0.467867**, and gradient
norm **1.129724**. The accumulation denominator included **3,105** supervised tokens
across the four examples.

[W&B accumulation run](https://wandb.ai/2162681069-peking-university/optical-adaptor/runs/hicr36w3).
Its effective config and artifacts are under
`/workspace/optical-adaptor/outputs/automodel/accum-smoke/`.

After splitting CPU preflight across ranks and rejecting overlength inputs earlier,
the complete data report matched the previous serial scan exactly: selected rows,
lengths, rejection counts, and run-contract fingerprint were identical. The run
restored again, completed step 4, evaluated, and exported `epoch_0_step_4`.
The affected processing/data checks were rerun successfully after that change.

## Scope

This validates execution, masking/loss arithmetic, and checkpoint mechanics, not
convergence, exact bitwise GPU replay, or downstream SWE success. The smoke profile
does not train on the long full trajectories, although it stores them intact.
The default 32K context limits and larger image bins still need memory/throughput
measurement before a long production run. Current distribution is shared-setup
DDP; TP/CP/PP and separate teacher/student meshes are explicitly unsupported.
