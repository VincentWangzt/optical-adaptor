# v1 experiment results

MLP training and its final evaluation are complete. Transformer-plus-MLP training
is running; the native-Qwen reference remains pending. This report will be
completed with both experiments and the shared references.

The primary result is each adapter's final checkpoint after three epochs. Each
continuation evaluation set contains 500 records and 64,000 supervised tokens.
The reconstruction set contains 500 records and 244,902 source-text tokens.
See [training.md](training.md) for commands and
[training-validation.md](training-validation.md) for preparation and validation.

## Continuation

| Model | Evaluation set | KL to text teacher | CE | Perplexity | Teacher top-1 agreement |
| --- | --- | ---: | ---: | ---: | ---: |
| Pure-text teacher | Front | 0 | 0.840671 | 2.317922 | 100% |
| MLP, final | Front | 0.571395 | 1.362635 | 3.906475 | 79.90% |
| Pure-text teacher | Middle | 0 | 0.778418 | 2.178025 | 100% |
| MLP, final | Middle | 0.414094 | 1.137306 | 3.118355 | 83.81% |

Middle examples include up to 256 preceding text tokens. Their numbers should
not be interpreted as a controlled comparison with front examples.

## Reconstruction

Teacher-forced source-text metrics exclude the chat template, padding, and
assistant end token. The training objective also supervises that end token.

| Model | CE | Perplexity | Token accuracy |
| --- | ---: | ---: | ---: |
| MLP, initialization | 1.168034 | 3.215665 | 75.07% |
| MLP, final | 0.482754 | 1.620532 | 88.03% |

Greedy generation uses the entire 500-record reconstruction set at completion,
with a 4,096-token limit and assistant-end-token stopping.

| Model | Records | CER | WER | Exact match | Truncated |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP, final | 500 | 2.078972 | 1.797981 | 0% | 26.2% |

MLP's mean character edit distance is 3,582.306 and its mean word edit distance
is 310.666. CER and WER are ratios of total edit counts to total reference units;
insertions can make either ratio exceed 1. The final MLP does not yet provide
reliable exact transcription, despite its improved teacher-forced loss.

The fixed 50-record subset allows a comparison at the same record membership:

| MLP step | Records | CER | WER | Exact match | Truncated |
| --- | ---: | ---: | ---: | ---: | ---: |
| 500 | 50 | 7.909719 | 6.708722 | 0% | 84% |
| 939, same subset | 50 | 1.884738 | 1.606265 | 0% | 22% |

Both loss metrics and generation metrics are stratified by language, aspect
ratio, logical lines, and wrapped display lines. Complete aggregate breakdowns
are available in W&B and server-side result JSON. Source and predictions remain
on the server.

## Compression and execution

MLP uses 111 adapted visual tokens per image. Mean source-text-to-adapted-token
ratios are 4.562 for front continuation, 4.956 for middle continuation, and 4.413
for reconstruction. The pure-text reference ratio is 1. Native visual-token
counts will be reported from the official processor.

The MLP run used two A6000 GPUs through Accelerate DDP with BF16 computation,
FP32 adapter and AdamW states, and microbatch 2 per task per rank. Its complete
job, including initialization and generation evaluations, ran from
2026-09-05 19:07:04 to 22:31:51 UTC.

The completion audit verified:

- 939 optimizer updates and 30,000 total pairs across three epochs.
- Exactly 16 pairs at updates 313, 626, and 939; 32 pairs otherwise.
- 29 warmup updates followed by constant learning rate. The logged learning
  rate is the rate applied to the optimizer: `2e-4 * min((update - 1) / 29, 1)`.
  The scheduler state advances exactly 939 times.
- Initialization, nine periodic evaluations, and final evaluation, each with
  exactly 500 records in every set and consistent stratum counts.
- 250 final generation records per rank, with 500 unique records in total.
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
server. MLP's `runs/mlp/completion-audit.json` records the checks above.
Generation stratification was added to the completed MLP predictions without
repeating generation; the numerical training and generation paths were unchanged.

## Interpretation limits

The v1 renderer preserves unmarked wrapping ambiguity in the reconstruction
target, and DeepSeek's official no-crop preprocessing distorts the image aspect
ratio. Both are intentional design constraints. These runs use one seed and
repeatedly inspected evaluation sets; they do not establish robustness across
seeds or an untouched final test set.

