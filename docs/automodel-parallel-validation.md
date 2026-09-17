# Native optical parallelism and numerical validation

Date: 2026-09-17 UTC / September 18 Asia/Shanghai. Branch:
`codex/automodel-optical-kd`. Server:
`/workspace/optical-adaptor`, A6000 GPUs 8/9, PyTorch 2.13.0+cu130,
Transformers 5.15.1, FLA 0.5.2. All executable changes were committed locally,
pushed through GitHub, then pulled before server execution.

## Implementation

The canonical configuration now uses FSDP2. `OpticalParallelizationStrategy`
delegates language sharding to AutoModel's existing
`Qwen3_5ParallelizationStrategy`, preserving its TP plan, mixed-dtype FSDP,
context-parallel GatedDeltaNet and activation checkpointing. The optical strategy
owns only the adapter/vision wrapping and the model-specific target mapping.
The existing KD recipe still owns accumulation, global target normalization,
gradient synchronization, clipping, AdamW, the student/teacher mesh bridge and
checkpoint lifecycle.

The native text model retains the pinned Qwen checkpoint's 426 text tensors and
tied embedding/head storage. Its checkpoint adapter maps the VLM text prefix and
FP32 recurrent gates; auxiliary MTP is disabled. Embedding lookup runs inside the
native FSDP forward, so tied parameters are materialized correctly. Under CP,
each branch shards its different input sequence, redistributes selected hidden
rows to the common target owners, and projects only those rows. TP shards the
full-vocabulary head. No position-chunk loop was introduced.

The other fixes are:

- Ordinary unpaired shared CP forwards sharded inputs to both models.
- Single-worker GPU initialization retains NCCL; DDP placement keeps original
  FP32 adapter values.
- New adapter parameters are synchronized over the existing DP/CP and TP groups
  before FSDP sharding. The framework's ranked construction seed otherwise leaves
  replicated TP adapters different; FSDP does not perform DDP's initial broadcast.
- Empty CP target shards keep the gradient graph and use a consistent FP32 loss
  scalar. Singleton DP preflight stays local inside larger TP/CP meshes.
- Exports gather adapter DTensors collectively. Generation keeps sharded peers
  in lockstep across quotas and EOS. Timing reduces over the complete student
  mesh; data counts reduce over DP only.

PP/EP, Megatron FSDP and sequence parallelism remain explicit unsupported cases.
The pinned VLM KD recipe has no optical pipeline schedule; Qwen's current TP plan
keeps the recurrent core replicated. HSDP and combinations of TP+CP+DP retain the
framework mesh construction but have not been exercised on four or more GPUs.

## Numerical findings

Independent AdamW instances receiving the captured gradients reproduce parameters
and moments exactly. Frozen parameters have no gradients or version changes, and
only FP32 adapter parameters enter the optimizer. The original DDP update route
passed synchronization checks; the new FSDP/TP route required an additional
initialization correction described below.

TP checkpoint replay exposed an initialization bug that the first tiny-model
test missed because it started every rank with identical weights. The official
recipe uses ranked RNG, and the newly initialized adapter was replicated over TP
without an initial broadcast. Checkpoint deduplication then mixed whole
parameters from different replicas: one saved adapter weight differed from the
rank-zero export by 0.03951. This was not BF16 rounding or a loader failure.
The strategy now synchronizes the adapter over the existing mesh groups before
sharding. The regression deliberately perturbs each rank's initial adapter and
requires the replicas to agree after actual optimizer steps.

The earlier attention-only explanation was incomplete: the diagnostic always
ran math-attention trials later, after warm-up. Isolated vision calls with math
attention showed identical SAM outputs but a maximum CLIP feature difference of
0.09375 and final feature difference of 0.041015625. DeepSeek's scripted
QuickGELU switches its BF16 rounding after profiling. Replacing it with the same
eager formula made three repeated vision calls bitwise identical, with either
math or automatic SDPA selection in that isolated check.

The canonical run uses eager QuickGELU, strict PyTorch determinism and math SDPA
scoped to vision. Text retains its native attention path for CP.

| Full-model repeated setting | Loss in both trials | Gradient norm | Gradient/update maximum difference |
| --- | --- | --- | --- |
| Canonical vision, default text SDPA | 0.40904158 | 1.62170672 | 0 / 0 |
| Canonical vision, math text SDPA | 0.40962106 | 1.55563146 | 0 / 0 |

Both pairs also matched independent Adam parameters and moments exactly.
Across the two text backends, gradients differed by 29.6% in relative L2, despite
small loss differences. This is sensitivity to BF16 computation, not failure of
repeatability within one backend. These short tests do not establish convergence.

The full pinned native/HF comparison found all 426 loaded tensors equal and
selected logits exactly equal for two short, right-padded rows. Input gradients
differed by 1.87% relative L2 under BF16 native kernels. CPU FP32 attention/target
selection has a separate forward and input-gradient parity check; hybrid
GatedDeltaNet uses the GPU checks because the installed FLA kernels require CUDA.

## Training and checkpoint evidence

Full-model smoke configurations use global batch 2, local batch 1, a 2,048-token
cap, at most two images per conversation, activation checkpointing, and two
optimizer updates. Evaluation selects fixed per-leaf samples with four-token
generation caps; these caps test control flow and are not quality evaluations.

FSDP DP2 and separate FSDP student1/teacher1 completed optimization, teacher-forced
and generation evaluation, and adapter checkpoint exports. A fresh FSDP DP2
process resumed the first checkpoint and reproduced the uninterrupted second
update exactly: loss 0.65468198, gradient norm 2.38788910, every adapter tensor,
and all 100 leaves of the saved optimizer/scheduler state. This validates one
resumed update on the same layout, not arbitrary topology changes. This replay
preceded the final initialization-sync correction. Earlier full TP2 and CP2 runs
also completed two updates, evaluation and exports, but the TP replay failure
described above invalidates treating those runs as final correctness evidence.

After the final initialization-sync fix (`8cd162b7`), the deliberately mismatched
initialization tests passed for DDP2, FSDP DP2, TP2 and CP2, with two updates each
and bitwise-equal adapter replicas. The latest full-model FSDP DP2 run completed
two updates (loss 0.40904158 / 0.64375138), evaluation and both exports. The user
then requested stopping numerical-matching work and reporting the current state.
The remaining full-model TP/CP reruns and replay matrix were stopped; their
completion is not claimed. Training quality/convergence and performance neutrality
of the encoder changes have not been established by these smoke runs.

The focused tiny hybrid test uses tied embeddings, unequal target counts,
two-microbatch accumulation followed by a partial accumulation window, clipping,
AdamW, and an empty-target CP rank. It compares raw gradients, global gradient
norm and updated parameters against a serial reference, and requires bitwise
equality of reconstructed adapter replicas across all ranks. Distributed BF16
TP/CP gradient differences are about 1%; they are not silently treated as exact
serial equivalence.

Thirty optical CPU tests and 94 selected upstream tests passed. Six unrelated
bitsandbytes loader tests were excluded because that optional dependency is absent.
The final synchronization change also passed the seven focused loss/regression
tests before its GPU matrix. Ruff and formatting checks passed on changed Python
files. Relevant jobs are `20260917-155415-optical-final-mesh-and-p-6ad79c`,
`20260917-154728-optical-upstream-focused-d2c769`, and the user-stopped final job
`20260917-160542-optical-synchronized-mes-aac780`.

## Reproduction and artifacts

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 uv run pytest tests/test_automodel_*.py -q

# Check these devices are idle immediately before launching.
CUDA_VISIBLE_DEVICES=8,9 NCCL_P2P_DISABLE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  uv run torchrun --standalone --nproc_per_node=2 \
  tests/test_automodel_parallel.py --axis cp --checkpointing

CUDA_VISIBLE_DEVICES=8 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  uv run torchrun --standalone --nproc_per_node=1 \
  tests/test_automodel_update_route.py \
  outputs/automodel/parallel-validation/single.yaml \
  outputs/automodel/parallel-validation/repeat-check --require-repeatable
```

Server artifacts under `outputs/automodel/parallel-validation/` include
`native-parity.json`, `eager-update-route/update-route.json`, runtime YAMLs,
`dp-eager/`, `dp-resume/`, `dp-synced/`, `separate-fsdp/`, and TP/CP run directories. The original
`update-route/` and `stable-update-route/` reports preserve the pre-QuickGELU-fix
failures. No 32K capacity run, long training run, PP, Megatron FSDP, or multi-node
test was performed. Generation currently recomputes vision and the prefix each
token, so generation throughput is not optimized.

The DeepSeek investigation isolated scripted-activation execution. It did not
audit image resizing/padding/normalization, patch and newline ordering, or the
adapter input against the official full OCR encoding path. That remains a separate
check; numerical repeatability alone does not establish correct optical encoding.
