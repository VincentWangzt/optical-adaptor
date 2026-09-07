# v1 training results

Both three-epoch experiments completed successfully. MLP learned substantially
more than Transformer-plus-MLP under the specified budget, but neither final
checkpoint achieves reliable exact transcription. The native-Qwen reference is
currently running and will be added to this report.

The primary results use the final checkpoints, without best-checkpoint selection.
Each continuation evaluation set contains 500 records and 64,000 target tokens.
The reconstruction set contains 500 records and 244,902 source-text tokens.
See [training.md](training.md) for commands and
[training-validation.md](training-validation.md) for implementation validation.

## Why there are 939 updates

The global batch contains 32 continuation–reconstruction pairs, or 64 task
examples. Each epoch visits all 10,000 training records once under each task:

```text
312 full updates × 32 pairs + 1 partial update × 16 pairs = 10,000 pairs
313 updates/epoch × 3 epochs = 939 optimizer updates
10,000 pairs/epoch × 3 epochs = 30,000 pairs = 60,000 task examples
```

With two ranks and microbatch 2 per task per rank, a full update accumulates eight
microbatches for each task on each rank. Thus 939 is an optimizer-update count,
not a count of examples or forward passes. Both complete histories and checkpoint
states were audited; neither run stopped early.

This verifies the requested schedule, not the adequacy of three epochs. MLP was
still improving at the end. Transformer reconstruction stayed near a plateau,
so extra steps alone are not established as the remedy.

## Final continuation results

| Model | Evaluation set | KL to text teacher | CE | Perplexity | Teacher top-1 agreement |
| --- | --- | ---: | ---: | ---: | ---: |
| Pure-text teacher | Front | 0 | 0.840671 | 2.317922 | 100% |
| MLP | Front | 0.571395 | 1.362635 | 3.906475 | 79.90% |
| Transformer + MLP | Front | 0.783443 | 1.573605 | 4.824009 | 76.74% |
| Pure-text teacher | Middle | 0 | 0.778418 | 2.178025 | 100% |
| MLP | Middle | 0.414094 | 1.137306 | 3.118355 | 83.81% |
| Transformer + MLP | Middle | 0.529402 | 1.246065 | 3.476635 | 81.65% |

Middle examples include up to 256 preceding text tokens. Their numbers should
not be interpreted as a controlled comparison with front examples.

## Final reconstruction results

Teacher-forced source-text metrics exclude the chat template, padding, and
assistant end token. The training objective also supervises that end token.

| Model | Initialization CE | Final CE | Final perplexity | Final token accuracy |
| --- | ---: | ---: | ---: | ---: |
| MLP | 1.168034 | 0.482754 | 1.620532 | 88.03% |
| Transformer + MLP | 1.162642 | 1.064037 | 2.898047 | 76.57% |

Both initializations already achieved about 75% teacher-forced token accuracy:
Qwen receives the correct preceding output tokens, so this metric alone is not
evidence of effective image reading.

Greedy generation uses all 500 reconstruction records at completion, with a
4,096-token limit and assistant-end-token stopping.

| Model | CER | WER | Mean character edit distance | Mean word edit distance | Exact match | Truncated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP | 2.078972 | 1.797981 | 3,582.306 | 310.666 | 0% | 26.2% |
| Transformer + MLP | 7.504275 | 6.276226 | 12,930.722 | 1,084.444 | 0% | 84.4% |

CER and WER divide total edit counts by total reference units. Excess output
adds insertions, so these ratios can exceed 1. These generation results are a
substantial failure of exact transcription despite the teacher-forced gains.

The fixed 50-record subset permits comparison at identical record membership:

| Model | Step | CER | WER | Exact match | Truncated |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP | 500 | 7.909719 | 6.708722 | 0% | 84% |
| MLP | 939 | 1.884738 | 1.606265 | 0% | 22% |
| Transformer + MLP | 500 | 10.528546 | 8.077150 | 0% | 98% |
| Transformer + MLP | 939 | 8.208506 | 7.081572 | 0% | 86% |

Loss and generation metrics are stratified by language, aspect ratio, logical
lines, and wrapped display lines. Complete aggregate breakdowns are in W&B and
server-side JSON. Source and predictions remain on the server.

## Learning curves

| Optimizer update | MLP reconstruction CE | Transformer reconstruction CE | MLP front KL | Transformer front KL |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.1680 | 1.1626 | 1.2327 | 1.1606 |
| 100 | 1.0865 | 1.0951 | 0.8489 | 0.8720 |
| 300 | 1.0752 | 1.0854 | 0.7974 | 0.8291 |
| 500 | 1.0656 | 1.0741 | 0.7808 | 0.8038 |
| 600 | 0.9003 | 1.0730 | 0.7378 | 0.7968 |
| 700 | 0.5794 | 1.0709 | 0.6204 | 0.7930 |
| 800 | 0.5178 | 1.0658 | 0.5923 | 0.7855 |
| 939 | 0.4828 | 1.0640 | 0.5714 | 0.7834 |

MLP's sharp reconstruction improvement began around updates 500–700, late in a
939-update run. Its later improvements motivate a longer controlled continuation
experiment. Transformer did not show that transition. Its larger capacity alone
did not improve learning under these shared hyperparameters.

## Image-dependence diagnostic

After training, each final checkpoint was evaluated with correct, mismatched,
and zero encoder embeddings on the same 50 reconstruction records used for the
generation subset. No weights were changed. Mismatching cyclically shifts images
within each split after sorting by visual token length, preserving input shapes.
The probe also includes 50 front and 50 middle continuation records.

| Model | Reconstruction CE, correct image | Mismatched image | Zero embeddings | Increase from mismatch |
| --- | ---: | ---: | ---: | ---: |
| MLP | 0.355422 | 1.154662 | 1.093680 | 0.799240 |
| Transformer + MLP | 0.978195 | 1.059336 | 1.040173 | 0.081141 |

MLP strongly depends on the correct image in this subset. Transformer also uses
some image information, but the reconstruction effect is much smaller. This
supports investigating its optimization and preservation of useful visual
features before assuming that additional steps will solve the problem. The
probe does not identify the causal reason for the plateau, and its 50-record
scores are not substitutes for the full 500-record results.

Detailed aggregates and selection fingerprints are in
`diagnostics/image-ablation-{mlp,transformer}.json` on the server.

## Compression and execution

Both adapters use 111 visual tokens per image. Mean source-text-to-adapted-token
ratios are 4.562 for front continuation, 4.956 for middle continuation, and 4.413
for reconstruction. The pure-text reference ratio is 1. Native visual-token
counts will be reported from the official processor.

Both runs used two A6000 GPUs through Accelerate DDP with BF16 computation,
FP32 adapter and AdamW states, and microbatch 2 per task per rank.

| Model | Trainable parameters | Job start, UTC | Job completion, UTC |
| --- | ---: | --- | --- |
| MLP | 9,838,080 | 2026-09-05 19:07:04 | 2026-09-05 22:31:51 |
| Transformer + MLP | 49,335,040 | 2026-09-05 22:32:24 | 2026-09-06 02:10:08 |

Job durations include initialization, periodic evaluations, and generation.
The runs used identical data, record schedules, configurations, and topology.

Completion audits verified for both experiments:

- 939 optimizer updates and 30,000 total pairs across three epochs.
- Exactly 16 pairs at updates 313, 626, and 939; 32 pairs otherwise.
- 29 warmup updates followed by constant learning rate. Logged applied learning
  rate is `2e-4 * min((update - 1) / 29, 1)`; scheduler state advances 939 times.
- Initialization, nine periodic evaluations, and final evaluation, each with
  exactly 500 records per set and consistent stratum counts.
- 500 unique final generation records, split 250 per rank.
- FP32 adapter/optimizer state, both rank RNG files, and final cursor
  `epoch=3, update=0, step=939`.
- Periodic checkpoints 250, 500, and 750, plus the final checkpoint.

Data fingerprint:
`1222980789809750122b186971c40be006bc2da62c398ae82dcfc96bcedafcad`.

Training-record fingerprint:
`dfdd55db097cdfc46e1b927534fd714e66234e601117d5f8dec868116ad82de2`.

Runs: [MLP](https://wandb.ai/2162681069-peking-university/optical-adaptor/runs/jl52d63v),
[Transformer + MLP](https://wandb.ai/2162681069-peking-university/optical-adaptor/runs/u6hvj4kj),
[pure-text teacher](https://wandb.ai/2162681069-peking-university/optical-adaptor/runs/ds7jol7t).

Artifacts are under `/workspace/optical-adaptor/outputs/adapter-v1/` on the
server. Each run's `completion-audit.json` and the shared
`comparison-input-audit.json` record the checks above. MLP generation strata
were added from saved predictions without repeating generation.

## Interpretation limits

The v1 renderer preserves unmarked wrapping ambiguity in the reconstruction
target, and DeepSeek's official no-crop preprocessing distorts the image aspect
ratio. Both are intentional constraints. These runs use one seed and repeatedly
inspected evaluation sets. They do not establish robustness across seeds or
an untouched final test result. The native reference is still needed to quantify
how much of the difficulty comes from the rendering/task versus the learned
compression and adapters.
