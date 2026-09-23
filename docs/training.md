# Optical training

The canonical configuration is `configs/automodel.yaml`. Run Python, tests and
optimization only on the GPU server. Commit local changes, push to GitHub, then
pull on the server before execution.

## Setup and launch

`third_party/Automodel` is the editable research checkout pinned at upstream
`2c0df17efff1fc9d5128a9df1a2ed716d0f2c54a`; see its `OPTICAL_FORK.md` for the patch
boundary. The optical recipe inherits AutoModel's component setup, accumulation,
optimizer loop, mesh bridge and checkpoint engine.

```bash
uv sync --locked --group dev
uv run python -m optical_adaptor.automodel.prepare --config configs/automodel.yaml
nvidia-smi
CUDA_VISIBLE_DEVICES=8,9 NCCL_P2P_DISABLE=1 uv run automodel configs/automodel.yaml \
  --nproc-per-node 2 --step_scheduler.max_steps 3
```

Choose devices that are idle immediately before launch. GPU IDs above are an
example, not a reservation. `scripts/launch_optical_training.sh --smoke` builds a
small effective configuration; normal CLI dotted overrides work without a custom
training parser. The server needs `NCCL_P2P_DISABLE=1` for its GPU 8/9 pair.
YAML scalars retain their declared types: write numeric settings as numbers and
quote strings such as bin identifiers. The fork does not coerce quoted numeric
strings during object instantiation.

## Data and weighting

Prepared records contain readable `<visual_AREA_start>...<visual_AREA_end>` tags
inside user/tool content. These storage annotations compile to internal spans;
model-visible vision boundaries remain configured separately. Assistant targets
retain exact whitespace, reasoning and tool syntax. Both branches supervise the
same target IDs using their own causal prediction positions. Whole examples are
rejected on capacity overflow; trajectories are never truncated or replaced.

Leaves are `<logical_source>/<task>/<size_bin>`. Stack front/middle and SWE
window/full are distinct sources. Reconstruction and continuation use image bins;
next-action uses turn bins. Boundaries are 1, 2, 3-4, 5-8, 9-16 and 17+. Raw image
and turn counts remain metadata. There is no view level or cross-product metric.

```yaml
optical:
  data:
    weights:
      "": 1.0
      stack_front: 3.0
      stack_front/reconstruction/images-1: 5.0
    eval_samples:
      "": 16
      stack_front: 8
  evaluation:
    generation_samples:
      "": 2
      stack_front/reconstruction: 4
    max_new_tokens:
      "": 4096
      stack_front/reconstruction/images-1: 1024
```

A leaf inherits its nearest configured ancestor's value. Weights are copied to
leaves, then normalized globally across nonempty eligible leaves. They are not
multiplied or divided as parent quotas. Rows are sampled uniformly within a leaf,
with replacement. Root weight 1 gives equal leaves. Zero weights disable draws;
unknown paths, invalid values and zero eligible total weight fail explicitly.

Curation sorts sample IDs by a seeded hash independently per leaf and extracts
`prepare.eval_fraction`. Counts round to nearest integer with ties up, capped to
leave at least one training row; singleton leaves remain in training. The fixed
holdout IDs are written to `summary.json`. Training setup chooses a seeded fixed
subset using inherited absolute counts and writes its IDs to `data-report.json`.
Unused holdout samples stay held out. Related samples from the same repository or
trajectory can cross splits: this is sample-level validation.

Filters/preflight report per-leaf before/after counts, absolute removals and
percentages of stage input, with rejection reasons. Empty training leaves are
omitted and surviving weights normalized. Evaluation uses available rows and
reports requested/selected counts and percentage shortfalls. An entirely empty
evaluation selection disables validation; no eligible training rows is an error.

## Model, batching and objective

The CPU processor renders images, preprocesses pixels, tokenizes conversations
and creates paired padded tensors. The student model executes the frozen vision
encoder in `processing.image_microbatch_size` groups, then the trainable MLP and
frozen Qwen backbone. Generation uses the same embedding and vision paths.

The teacher and student are separate logical model instances. Each selects
supervised hidden positions before a single full-vocabulary LM-head projection.
The teacher returns detached logits shaped `[batch, padded_targets, vocabulary]`;
labels mask target padding with -100. Hidden states never cross the KD bridge.
`kd_loss_fn.chunk_size: 0` is required. There is no optical position-chunk loop or
loss-block checkpointing. Language activation checkpointing and image batching
remain independent controls.

Local batch size counts conversations per student replica per forward. Global
batch size counts conversations per optimizer update. AutoModel accumulates
ordinary microbatches and normalizes CE/KD by the global valid-target count.
Sampling epochs round up to complete global batches. Packing and adaptive
regrouping are deferred. Unchunked logits can be large; set sequence/image limits
and batch sizes to fit the chosen placement.

The canonical configuration currently uses DDP for inline generation evaluation.
For FSDP2 training and teacher-forced evaluation, the optical strategy delegates
language parallelization to AutoModel's existing
`Qwen3_5ParallelizationStrategy`; it adds adapter precision ownership and the
mapping between input positions and compact targets. Configure
`distributed.tp_size` or `distributed.cp_size` to use tensor or context
parallelism with FSDP2. The remaining ranks form the data-parallel mesh.
Activation checkpointing belongs to `distributed.activation_checkpointing`.

For independent placement, configure disjoint student and teacher meshes, for
example one FSDP2 worker each:

```yaml
separate_meshes: true
distributed:
  strategy: fsdp2
  dp_size: 1
teacher_distributed:
  strategy: fsdp2
  dp_size: 1
```

Each mesh uses `DP * TP * CP` workers; the two mesh sizes must sum to the launcher
process count. The official bridge routes paired requests and selected logits,
including unequal DP sizes. Each branch shards its own input sequence; CP shards
supervision on the common target axis, and TP shards the vocabulary projection.
The framework handles gradient reduction, normalization, clipping and AdamW.
The adapter keeps FP32 master parameters and uses BF16 compute. Its newly
initialized weights are synchronized over the existing student mesh groups before
FSDP sharding, so rank-dependent construction seeds cannot split TP replicas.
Different adaptive teacher/student microbatch partitions remain deferred.
For an FSDP2 run, set `optical.evaluation.generation_samples: {"": 0}` to retain
teacher-forced inline validation. Nonzero inline generation quotas require DDP;
evaluate FSDP2 adapter exports in a separate inference job.

PP and EP must be 1, and `sequence_parallel` must be false. There is no optical
pipeline schedule or expert model, and the existing Qwen TP plan keeps the
recurrent core replicated. Megatron FSDP is not integrated. These limits are
checked explicitly; TP and CP use the existing FSDP2 strategy.

`optical.deterministic: true` enables strict PyTorch determinism. The encoder uses
contiguous NCHW pixels and owns the official BF16 autocast scope, including its
projector. The official scripted QuickGELU is imported unchanged: under autocast
it computes in FP32. The eager override was removed because it changed activation
precision to BF16. Preserve the official encoder precision policy. The
[official encoding audit](ocr-encoding-validation.md) verifies FP32 activation
outputs in the actual CLIP blocks and exact features against the unmodified
official reference under the same attention backend. The
canonical `optical.vision.sdpa_backend: math` fixes the vision attention reference;
`auto` allows fused selection. Math attention is scoped to vision, leaving the
language model's CP attention available. These settings are part of the exported
model and resume contract. Exact repeatability is demonstrated for the canonical
settings, not equivalence across different backends or mesh layouts. Changing
image batch size also changes BF16 vision features (about 2.6–4.0% relative L2
on the two audited rendered pages). A fixed maximum batch size still permits
different partial chunks. The current 640×640 encoder implements official Small
non-crop processing, not the variable-token dynamic-crop mode.

Inline generation uses a temporary, full Hugging Face Qwen decoder with a KV
cache. The live vision encoder and adapter supply its image embeddings. This
path requires DDP and adds a full decoder copy to each participating GPU during
evaluation. The native FSDP2/TP/CP model still handles teacher-forced metrics;
its earlier lockstep generation loop is no longer used.

## Evaluation, metrics and recovery

Teacher-forced metrics are loss, student CE/KL, teacher CE, argmax agreement and
student target-token accuracy. Overall, source and task aggregates live under
`eval-core`; leaf metrics live under `eval-aux`. Generation adds CER/LER and
sample/edit/limit diagnostics. Evaluation does not publish token-count curves.
Generation sample counts and output limits use their own inherited defaults.
Teacher-forced and generation subsets are selected independently from the same
eligible holdout pool; reducing one quota does not reduce the other.

Every consumed optimizer update reports `batch/size`, teacher/student input-token
totals, loss-token totals and image totals. Input counts exclude padding and
student counts include visual positions. `data/ratio/<leaf>` is intended draw
probability; observed ratios emit zero when absent. Cumulative training samples
and equivalent epochs are checkpointed. Prefetched records do not increment them.

GPU substeps use CUDA events collected at one update boundary and report the
maximum rank duration. Teacher timing includes mesh request/transfer time when
meshes are separate. Student forward includes vision/adapter time, also reported
separately; overlapping/nested timings should not be summed. Backward includes
checkpoint recomputation and gradient communication. Step wall time and input /
target throughput include waiting for the next batch and the optimizer update,
excluding curation, evaluation and checkpoint writing. `timing/data_wait` exposes
the non-overlapped loader wait. CPU preprocessing may overlap through
data-loader workers; no per-example normalized workload curves are emitted.

Checkpoints persist the registered adapter submodule, optimizer, RNG, loader,
scheduler, data contract and consumed counters through AutoModel. Frozen modules
are reconstructed from pinned model revisions. Incompatible contracts fail.
Set `checkpoint.restore_from` to `LATEST` or a checkpoint directory to resume.
Each save also writes `exports/step-NNNNNN/{adapter.safetensors,optical-model.json}`
for retained live inference/benchmark entry points. They now use the canonical
optical config and fresh prepared records. Historical training, tensor caches,
old checkpoint schemas and replay tests have been retired; historical reports
remain documentation only.

The current [parallel validation report](automodel-parallel-validation.md) records
full FSDP DP2, TP2 and CP2 runs after restoring the official scripted activation,
with two updates followed by a fresh-process replay of the second update for each
layout. Adapter weights,
optimizer/scheduler state, losses, sample counters and loader state matched
exactly, and checkpoints matched exports. Thirty focused CPU tests also passed.
This is bounded replay evidence, not a long-run convergence, 32K capacity or
cross-layout equivalence claim. The initial migration results remain historical.
