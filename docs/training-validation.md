# v1 training validation

Validation was run on the GPU server on 2026-09-06 (Asia/Shanghai). Commands and
configuration are documented in [training.md](training.md). This record contains
aggregate results only; source, rendered images, predictions, and caches stay on
the server.

## Data and frozen models

- The manifest contains 10,000 training records and 500 records in each of the
  front-continuation, middle-continuation, and reconstruction evaluation sets.
  All 30 dataset languages are represented.
- All 11,500 records passed token round-trip and record-hash validation. Repository
  assignments are disjoint, and each source file appears at most once.
- The audit found no complete-file exact-content overlap between splits. Two
  visual-span hash groups occur across splits and remain in the dataset under the
  agreed no-filtering policy. Details are in `data/eligibility.json` on the server.
- DeepSeek extraction loaded exactly 476 vision tensors and returned BF16
  `[111,1280]` embeddings. The language decoder was excluded.
- All 23 encoder shards and 22 teacher shards passed fingerprint, checksum,
  shape, and dtype verification. They cover 11,500 encoder records and 11,000
  continuation teacher records respectively.
- An eight-record cache-parity check included short, maximum-length, front, and
  middle examples. Online FLA-backed Qwen predictions versus the cached teacher
  had batch-mean full-vocabulary KL at most `0.000736` and CE difference at most
  `0.000775`. The predefined BF16 limits were `0.005` and `0.02` respectively.
  Teacher shards were produced using the PyTorch reference attention path; the
  parity check verifies their use with the pinned FLA student kernels.

## Objectives, distribution, and replay

The remote test suite passed all 24 tests. Focused checks cover canonicalization,
geometry, offsets, configuration, split integrity, cache corruption, first/last
prediction alignment, full-vocabulary FP32 KL, metric-only continuation CE, chunked
gradients, independent task permutations, and partial-update coverage.

A subsequent focused generation-reporting test also passed. It verifies summed
edit-count weighting across ranks and strata, duplicate detection, and complete
reaggregation from saved prediction files without repeating generation.

The two-process CPU validation compares accumulated gradients directly with a
single-process reference before AdamW normalization, using unequal reconstruction
lengths. It also verifies exact checkpoint replay and RNG restoration.

Real Qwen probes verified both adapters and both objectives, with gradients only
on FP32 adapter parameters. A two-GPU Accelerate replay check restored weights,
AdamW state, scheduler, sample order, and RNG. The subsequent weights and loss
totals matched the uninterrupted result exactly. Per-rank details are stored in
`outputs/validation/gpu-resume-deterministic/` on the server.

Two runtime issues were resolved before full training:

1. GPU 8/9 peer-to-peer NCCL transport hung at the first collective. Launches use
   `NCCL_P2P_DISABLE=1`, with verified shared-memory transport.
2. Repeated GPU backward passes initially differed despite identical forward
   losses. Deterministic algorithms and the configured cuBLAS workspace eliminated
   the difference. These settings are explicit in `configs/training.yaml`.

The locked runtime retains Torch `2.13.0`, Transformers `5.16.1`, and vLLM `0.28.0`.
FLA `0.5.2` supplies the verified linear-attention kernels. Checkpoint finalization
waits for every rank's files, and resume restores RNG after tracker initialization.

## Throughput selection

Every candidate used the same 32-record workload sampled across length quantiles,
including the shortest and longest reconstructions. Measurements exclude two
warmup updates and average three timed updates, taking the slower rank's time and
the larger rank's peak reserved memory.

| Per-rank examples per task | MLP seconds/update | MLP GiB | Transformer seconds/update | Transformer GiB |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 11.818 | 9.336 | 11.900 | 10.066 |
| **2** | **10.576** | **10.902** | **10.840** | **11.512** |
| 4 | 11.577 | 13.400 | 11.682 | 13.488 |
| 8 | 13.266 | 19.023 | 13.372 | 20.221 |

Both adapters select microbatch 2 and eight accumulation steps per rank, preserving
32 global continuation–reconstruction pairs per full update. Every candidate is
below the 40 GiB limit. A separate MLP stress run using only the 32 longest records
also passed at 9.535 GiB with microbatch 1.

Generation batch 16 passed a native-Qwen probe using the 16 tallest reconstruction
images at 20.287 GiB peak reserved memory. Generation groups the selected records
by target length; evaluation membership is unchanged.

## Overfit gates

Each adapter trained for 100 updates on the same eight records. Reconstruction
loss in this table includes the supervised assistant end token, matching the
optimized objective.

| Adapter | Initial KL | Final KL | Initial reconstruction loss | Final reconstruction loss |
| --- | ---: | ---: | ---: | ---: |
| MLP | 1.278802 | 0.018473 | 1.125569 | 0.089897 |
| Transformer + MLP | 1.189630 | 0.072533 | 1.121051 | 0.323755 |

Both gates passed. MLP was deliberately stopped at update 5 and resumed at update
6, retaining its W&B run ID. Both per-rank RNG files were verified in its final
checkpoint.

Runs: [MLP overfit](https://wandb.ai/2162681069-peking-university/optical-adaptor/runs/b3rt5lr5)
and [Transformer overfit](https://wandb.ai/2162681069-peking-university/optical-adaptor/runs/s2j2wl0w).

## Pure-text teacher reference

| Evaluation set | Records | Target tokens | CE | Perplexity |
| --- | ---: | ---: | ---: | ---: |
| Front continuation | 500 | 64,000 | 0.840671 | 2.317922 |
| Middle continuation | 500 | 64,000 | 0.778418 | 2.178025 |

The pure-text reference has compression ratio 1. Its aggregate results are cached
for reuse across both experiments in `references/teacher/complete.json` and in
[W&B](https://wandb.ai/2162681069-peking-university/optical-adaptor/runs/ds7jol7t).
