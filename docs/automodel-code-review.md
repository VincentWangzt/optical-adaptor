# AutoModel integration and weight-update review

## Resolution after implementation

The follow-up implementation fixes the three findings below and replaces the
DDP-only optical route with AutoModel's native Qwen strategy for FSDP2, TP and CP.
See [parallel validation](automodel-parallel-validation.md) for the current
implementation, numerical checks and remaining supported-layout limits.

The new TP integration also required an initial adapter broadcast: the recipe
uses rank-dependent seeds, and FSDP does not synchronize new replicated weights.
TP checkpoint replay exposed this issue after the initial identical-weight
microtests passed. The strategy now synchronizes initialization using the existing
mesh groups, and the regression starts from deliberately different replicas.
Small DDP/FSDP/TP/CP tests and two full FSDP updates passed after this fix; the
remaining full TP/CP reruns were stopped when the user requested a status report.

The earlier attention-only diagnosis was incomplete. Its fixed trial order did
not isolate DeepSeek's TorchScript QuickGELU warm-up. An isolated vision test
found identical SAM outputs but changing CLIP features between first and later
calls, even with math attention. Replacing scripted QuickGELU with the same eager
formula made the isolated calls and full repeated optimizer updates bitwise
identical. Fresh-process FSDP resume also reproduced weights, optimizer moments,
scheduler, loss and gradient norm exactly for the tested next update.

These results supersede the historical reproducibility conclusions below.
Different BF16 backends and mesh layouts still need not produce identical
gradients; repeatability within one execution contract is a different claim.

## Historical review before fixes

Reviewed 2026-09-17. Production code: `75ca63fe`; review diagnostics through
`655ebf69`. This review adds diagnostics and extends existing mesh tests; it does
not change production behavior or fix the findings below.

## Conclusions

Existing AutoModel FSDP/TP/CP infrastructure can be reused. Our optical wrapper
currently bypasses it, and there is an additional shared-mesh CP regression in
our modified KD recipe. The unsupported strategies are an integration gap, not
a reason to build another distributed training framework.

The inspected DDP optimizer route passes focused update and synchronization
checks. Fixed-input GPU experiments reproduce numerical differences before the
optimizer, first visible at the frozen vision encoder output. AdamW produces
exactly the expected parameters and state when given those gradients. Math SDPA
or strict deterministic execution separately eliminates the repeated-update
differences in this diagnostic. This substantially narrows the earlier
reproducibility issue, but does not identify the exact kernel or establish
full-run, cross-process checkpoint reproducibility.

## Actionable findings

### P1: shared-mesh CP teacher inputs bypass sequence sharding

In `nemo_automodel/recipes/vlm/kd.py`, `_forward_backward_step` copies
`teacher_inputs` before `ContextParallelSharder.shard`, then uses that original
copy for the shared-mesh teacher forward. The student uses the sharded batch.
The pinned upstream implementation forwarded the teacher from the sharded batch.
This changes the ordinary, unpaired VLM path as well as the optical path.

A CPU control-flow reproducer replaced only the sharder with a deterministic
half-sequence shard and called the real KD forward/loss method. The teacher
received `[1, 4, 3]`, the student `[1, 2, 3]`, and KD failed because a length-2
label mask indexed length-4 teacher logits. This is a focused routing reproducer,
not a full distributed CP run.

Preserve the original sharded-teacher path for ordinary shared batches. Optical
paired inputs require branch-specific CP preprocessing and target-position
mapping; bypassing the DDP guard alone cannot provide that support.

### P1: ordinary single-worker GPU launch moves the student to CPU

`initialize_distributed` forces a single-worker run to use Gloo, while retaining
a CUDA device in `DistInfo`. `DDPManager._setup_distributed` chooses CPU for a
non-NCCL backend, and `parallelize` moves the student there. The recipe still
moves its input tensors to CUDA.

The full-model diagnostic through normal one-worker setup reproduced the device
mismatch before an optimizer update. For subsequent GPU diagnostics, the test
explicitly initializes NCCL before recipe setup. That is an isolation measure
inside the diagnostic, not a production fix.

The configured computation device and the DDP manager must agree; a CPU-capable
process-group backend should not implicitly relocate an explicitly GPU model.

### P2: DP-size-dependent adapter initialization loses FP32 precision

For a single-rank NCCL student group, `DDPManager.parallelize` casts the entire
model to BF16. `OpticalModel.from_pretrained` subsequently calls
`unwrapped.adapter.float()`. This restores the dtype but cannot restore the
original FP32 values. The multi-rank DDP branch does not perform that cast.

A GPU reproducer using `MLPAdapter(3, 5, 7)` changed 62 of its 68 parameter values
with maximum absolute error `0.0018306375` after this BF16-to-FP32 round trip.
Changing DP layout therefore also changes initialization. Preserve the adapter's
FP32 values through placement/wrapping. This does not explain repeat divergence
when the DP layout is unchanged.

## What can be reused for other meshes

`OpticalModel.from_pretrained` rejects every strategy except DDP. Its language
loader receives no `distributed_setup`, uses `force_hf=True`, and the resulting
composite is passed explicitly to `DDPManager`.

AutoModel already contains:

- Student/teacher distributed setup construction and mesh-to-mesh logit routing.
- FSDP sharding, mixed-precision handling, TP plans and replica-gradient reduction.
- `Qwen3_5ParallelizationStrategy`, including mixed-dtype FSDP handling and CP
  integration with `CPAwareGatedDeltaNet`.
- VLM pre-embedding and context-parallel sharding hooks.

The optical composite needs to expose its language/vision/adapter structure to
those existing mechanisms and preserve their wrapper/hook boundaries when
calling embeddings, decoder blocks and the selected-position LM head. Using the
HF loader alone does not automatically apply native Qwen CP machinery.

CP additionally needs explicit mappings between each branch's full input
sequence and the common compact target axis. For example, 2,000 input positions
and 200 targets cannot be sharded as though they were the same sequence. Image
replacement must respect the chosen pre-embedding/sharding order. Generation
must also keep sharded replicas on compatible collective schedules. PP remains
unsupported by the pinned upstream VLM KD recipe itself.

## Optimizer and synchronization audit

The current DDP route is:

1. Build the optimizer from trainable parameters; these are the adapter objects.
2. Sum valid targets over the student process group and accumulation window.
3. Normalize CE/KD by that global count; multiply backward loss by student DP
   size to compensate for DDP's gradient averaging.
4. Suppress DDP synchronization on intermediate microbatches, synchronize on the
   final backward, then clip and perform one AdamW update.
5. Clear gradients and advance the LR scheduler.

There is no periodic teacher/student weight-copy requirement: the teacher and
student language models and the vision encoder are frozen. Each student DDP
replica updates its own adapter using the synchronized gradient and matching
optimizer state.

`tests/test_automodel_mesh.py` now also runs two complete production KD optimizer
steps, including clipping and AdamW state, for 2-student/1-teacher and
1-student/2-teacher CPU process groups. Parameters and optimizer state matched a
serial reference; student replica parameters were bitwise identical after each
update. Both extended tests passed. These tests exercise the actual framework
optimizer boundary, beyond the earlier accumulated-gradient comparison.

The full-model GPU diagnostic verifies that optimizer parameters are exactly the
adapter parameters, each is FP32, frozen resources have no gradients or parameter
version changes, and gradients are cleared. A separate AdamW instance receives
the captured clipped gradients and pre-step optimizer state; every trial matched
its resulting parameters and optimizer state exactly.

## Fixed-input GPU isolation results

One A6000 (GPU 8), actual Qwen/DeepSeek/adapter models, two fixed processed
microbatches per trial. Each trial restores adapter weights, optimizer state,
scheduler state, and CPU/CUDA RNG state. No new data is fetched between trials.
The diagnostic explicitly initializes NCCL to avoid the single-worker placement
finding. `CUBLAS_WORKSPACE_CONFIG=:4096:8` was set for all trials.

| Repeated setting | First/second loss | Relative L2 gradient difference | Maximum updated-parameter difference |
| --- | --- | --- | --- |
| Default attention, determinism off | 0.41393048 / 0.41311425 | 0.12392002 | 0.000399991 |
| Math attention only | 0.41274482 / 0.41274482 | 0 | 0 |
| Strict determinism, default attention selection | 0.41311425 / 0.41311425 | 0 | 0 |
| Strict determinism and math attention | 0.41274482 / 0.41274482 | 0 | 0 |

The default pair had 448,585 gradient sign changes. Teacher logit hashes matched;
vision feature hashes already differed, followed by adapter outputs, language
hidden states and student logits. All of those hashes matched within each
controlled pair. Strict-math execution with activation checkpointing disabled
also matched the checkpointed strict-math gradients and updates exactly.

Different attention backends produced different numerical answers, so agreement
within each deterministic setting does not establish which backend is the best
numerical reference. The observed first differing stage narrows the search to
vision computation or its runtime backend behavior. SDPA selection affects the
whole model; this test does not isolate an individual attention/convolution
kernel. It also does not establish equivalence to the earlier two-process replay
run. It does show that the optimizer is consuming already-different gradients.

The default pair's approximately `0.0004` maximum first-update difference is
consistent with opposite-sign Adam first updates at learning rate `0.0002`.
The sign differences are observed; this explanation does not require a missing
weight broadcast or a double optimizer step.

## Reproduction and artifacts

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 uv run --locked pytest \
  tests/test_automodel_mesh.py -q

# Check GPU 8 is idle immediately before use.
CUDA_VISIBLE_DEVICES=8 NCCL_P2P_DISABLE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
  uv run --locked torchrun --standalone --nproc-per-node=1 \
  tests/test_automodel_update_route.py \
  outputs/automodel/migration-validation/separate.yaml \
  outputs/automodel/update-route-review-isolation
```

Server artifacts are under `/workspace/optical-adaptor/outputs/automodel/`:

- `update-route-review/update-route.json`: ordinary single-worker placement failure.
- `update-route-review-nccl/update-route.json`: initial five-trial GPU diagnostic.
- `update-route-review-isolation/update-route.json`: nine-trial isolation, including
  stage hashes, gradient comparisons and independent Adam checks.

Remote jobs: `20260917-132107-mesh-optimizer-review-31f63f`,
`20260917-132107-optical-update-route-rev-e7f5bb`,
`20260917-132347-optical-update-route-ncc-4ff0ac`, and
`20260917-132621-optical-kernel-isolation-e8055f`. All finished. GPUs 8/9 were idle
after validation. Ruff passed on changed tests; no production code was modified.
