# AutoModel migration validation

Date: 2026-09-17. Branch: `codex/automodel-optical-kd`.

Follow-up: [parallel validation](automodel-parallel-validation.md) records the
fixes, native FSDP2/TP/CP integration, and exact repeated-update and FSDP resume
checks. The [review](automodel-code-review.md) now explains the TorchScript
QuickGELU warm-up issue that the initial attention experiments did not isolate.
The original results below remain the historical migration smoke-run record;
their unresolved-reproducibility statement does not describe the current code.
The later [official OCR audit](ocr-encoding-validation.md) also fixes pixel layout
and vision autocast, and distinguishes repeatability from agreement with the
official scripted activation and from invariance to image batch size.

The confirmed decisions in `design-decisions-draft.md`, including inherited,
independent generation budgets (P11), are implemented and synchronized through
GitHub. Server validation completed: **94 distinct focused tests passed and nine
GPU optimizer updates completed**. Checkpoint restoration and continuation work,
but **numerically exact GPU continuation is not validated**: both resumed and
fresh repeated GPU runs diverged. The cause remains unresolved.

GPU runs used the implementation through `c149b64b`; the final configuration
instantiation adjustment and affected upstream regression tests passed at
`675da4f0`. Subsequent report edits do not change executable code. Historical
results in `automodel-validation.md` describe the earlier implementation.

## Implemented scope

- Pinned, tracked, editable AutoModel checkout; ordinary CLI/YAML launch and
  official VLM/KD recipe, accumulation, bridge, optimizer and checkpoint flow.
- Unified CPU processor and GPU optical model, with independently chunked frozen
  vision execution. Teacher/student select supervised hidden positions before
  full-vocabulary projection. Their boundary carries logits; position-axis
  projection/loss chunking is disabled.
- Schema-2 inline visual annotations, logical-source/task/primary-bin leaves,
  fixed per-leaf sample holdouts, inherited relative weights and evaluation
  quotas, filtering counts/percentages, and checkpointed consumed-sample totals.
- Independent inherited generation counts and output limits; core overall,
  source and task metrics, auxiliary leaf metrics, teacher CE/agreement/accuracy,
  reconstruction CER/LER, and batch workload/timing measurements.
- Adapter exports and retained inference/benchmark config migration; obsolete
  training/cache paths retired. DDP supports shared or separate placement;
  unsupported optical sharding strategies fail explicitly. Packing and adaptive
  grouping remain deferred as agreed.

Execution uncovered and fixed editable lock metadata, a retained benchmark
import, addon recipe worker launch, quoted YAML scalar coercion, and inference
module cleanup. Quoted YAML identifiers now retain their types during both
configuration loading and object instantiation; CLI/env value parsing remains
separate.

## Focused validation

All Python execution took place on the server using `uv`. Local checks were
`uvx ruff check src tests`, Ruff on changed AutoModel Python files, and
`git diff --check`; all passed. `uv sync --locked --group dev` installed the
editable checkout successfully.

| Coverage | Distinct passing tests |
| --- | ---: |
| Optical processing, data, losses and unequal teacher/student meshes | 23 |
| Standard CLI configuration/override regression | 1 |
| Exact CPU adapter/Adam checkpoint continuation | 1 |
| Retained benchmark, inference config and token utilities | 11 |
| Affected upstream launcher and config tests | 58 |
| **Total** | **94** |

The data checks exercise inherited controls, fixed holdouts and loader
continuation with and without worker prefetch. CPU Gloo bridge checks cover both
2-student/1-teacher and 1-student/2-teacher layouts, unequal target counts, and two
accumulation microbatches against a serial gradient reference. Loss checks cover
selected-logit alignment and gradients against dense projection. The CPU
checkpoint test reloads a trained MLP and AdamW through AutoModel's checkpointer
and obtains bitwise-identical next-step loss, parameters and optimizer state.

Representative commands (the upstream suite additionally needs pytest-timeout):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
  uv run --locked pytest tests/test_automodel_processing.py \
  tests/test_automodel_data.py tests/test_automodel_losses.py \
  tests/test_automodel_mesh.py tests/test_automodel_checkpoint.py \
  tests/test_live_benchmark.py tests/test_infer_config.py tests/test_token_utils.py -q

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 uv run --locked --with pytest-timeout \
  pytest -c pyproject.toml \
  third_party/Automodel/tests/unit_tests/launcher/test_interactive_launcher.py \
  third_party/Automodel/tests/unit_tests/config/test_loader.py -q
```

These commands consolidate the focused test invocations used during validation;
the CLI regression was also rerun alongside the upstream tests.

## GPU optimization

Server workspace: `/workspace/optical-adaptor`. GPUs 8 and 9 were checked idle
before use and selected explicitly. Runs used two A6000 GPUs, pinned Qwen3.5-4B
and DeepSeek-OCR models, FP32 trainable adapter, BF16 frozen language models,
AdamW at `2e-4`, CE/KD weights 0.5/0.5, image microbatch size 4, two loader workers,
and `NCCL_P2P_DISABLE=1`. W&B was offline.

Fresh curation replayed 192 raw originals (64 per configured source), producing
4,649 curated samples with 471 fixed holdouts. Capacity filtering left 550
eligible training rows in 16 leaves for separate placement and 321 rows in 13
leaves for shared placement. The difference comes from the smoke caps, not a
change to the underlying held-out split.

| Run | Placement | Updates | Local/global batch | Accumulation | Input caps | Loss by update |
| --- | --- | ---: | --- | ---: | --- | --- |
| Separate | Student DP 1 on GPU 8; teacher DP 1 on GPU 9 | 3 | 1 / 2 | 2 | 2,048 tokens; 2 images | 0.406585, 0.721320, 2.432194 |
| Replay | Restore separate checkpoint after update 2 | 1 | 1 / 2 | 2 | Same | 2.398705 |
| Repeat | Fresh run with the separate run's seed/settings | 3 | 1 / 2 | 2 | Same | 0.406585, 0.718077, 2.716426 |
| Shared | Student DP 2; separate logical teacher on each GPU | 2 | 2 / 8 | 2 | 1,024 tokens; 2 images | 1.248106, 1.192134 |

All runs exited successfully with finite losses, changed adapter weights,
evaluation results, adapter exports and framework checkpoints. Consumed-sample
totals were 2/4/6 for the separate run and 8/16 for the shared run. Replay restored
the total of 4 and advanced it to 6; its next workload and per-leaf consumed
counts matched the original third update. Ratios normalized correctly and
prefetched records did not inflate consumed counts.

The separate run's adapter changed by a maximum absolute value of `0.0004037023`
between its first and third exports. Pre-clipping gradient norms reached 410.87
in that run and 452.44 during replay (configured clipping maximum 1). These are
functional smoke runs on changing batches, not evidence of convergence or
training quality.

| Measurement | Separate run | Shared run |
| --- | --- | --- |
| Step wall times, including loader wait | 5.490, 1.548, 1.113 s | 5.562, 2.123 s |
| Maximum reported main-rank allocator peak | 14.28 GiB | 25.96 GiB |
| Sampled GPU 8 / GPU 9 device-memory peaks | 20,779 / 9,881 MiB | 30,547 / 31,915 MiB |
| Teacher-forced evaluation samples | 15 | 10 |
| Overall student CE / KL | 1.136555 / 1.004569 | 1.172244 / 1.087596 |
| Teacher CE / agreement / target accuracy | 0.157566 / 0.780651 / 0.764489 | 0.102801 / 0.756186 / 0.749547 |

Device memory was sampled with `nvidia-smi` once per second and may miss brief
peaks; it includes allocations outside the PyTorch allocator. Timings include
cold-start effects and differing workloads and are not a steady-state speed
comparison. Seven reconstruction samples in the separate run exercised
generation with a deliberately small 32-token cap: CER 0.955947, LER 0.996364,
with one of seven generations reaching the cap. Shared-run generation was
disabled. Evaluation emitted aggregate and leaf metrics without token-count
curves.

## Checkpoint and numerical reproducibility finding

The GPU replay restored adapter tensors exactly, loaded nonzero Adam moments
and step 2, resumed the correct data position and counters, and saved step 3.
Its third-update loss was 2.398705 versus the uninterrupted run's 2.432194;
final adapter tensors differed by at most `0.0003148564`.

A fresh same-seed repeat also diverged. Its first forward loss matched exactly,
but its first gradient norm was 2.182348 versus 2.188737 and its first adapter
export differed by up to `0.0003999099`. By update 3 the loss was 2.716426 and
maximum adapter difference was `0.0011110548`; evaluation CE was 1.833209 versus
1.136555. Thus the observed discrepancy is not limited to checkpoint restore,
and its amplification is material. These checks do not isolate the cause or
establish whether it lies in GPU numerics or another training-path component.

Exact CPU checkpoint continuation passed, but it does not establish exact GPU
replay. GPU numerical reproducibility remains an open investigation before
relying on tightly controlled experiment comparisons.

## Artifacts and reproduction

Server artifact root:
`/workspace/optical-adaptor/outputs/automodel/migration-validation`.

- `validation-summary.json`: per-step loss, workload, timing and evaluation
  summaries plus sampled device-memory peaks.
- `gpu-memory.csv`: timestamped GPU memory/utilization samples.
- `data/`: fresh originals, prepared records and curation summary.
- `{separate,replay,repeat,shared}.yaml`: effective run configurations.
- Corresponding run directories: metrics, data reports, checkpoints,
  `exports/step-NNNNNN/`, and offline W&B records.

Example launch, after checking that devices 8/9 are still idle:

```bash
CUDA_VISIBLE_DEVICES=8,9 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=4 \
  TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 uv run --locked automodel \
  outputs/automodel/migration-validation/separate.yaml --nproc-per-node 2
```

Use fresh output directories when repeating these runs. Replay's configuration
names the saved checkpoint after the separate run's second update.

| Run | Remote job ID |
| --- | --- |
| Separate | `20260917-071238-migration-separate-gpu-586a3a` |
| Replay | `20260917-071426-migration-replay-7fa098` |
| Repeat | `20260917-072005-migration-repeat-ed89af` |
| Shared | `20260917-072222-migration-shared-gpu-24b48c` |

All training jobs finished, the memory monitor was stopped, and GPUs 8/9 returned
to idle. Full 32K sequences, unequal GPU mesh sizes, online W&B resume, and actual
vLLM live inference were not exercised. Unequal meshes were validated on CPU.
FSDP/TP/CP/PP are explicitly unsupported by this optical integration. No speed,
convergence or quality improvement is claimed from this migration.
