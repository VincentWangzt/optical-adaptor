# AutoModel optical research checkout

Upstream: https://github.com/NVIDIA-NeMo/Automodel
Pinned base: `2c0df17efff1fc9d5128a9df1a2ed716d0f2c54a`.

This directory is an editable research checkout tracked in the optical-adaptor
repository. `uv sync --locked` installs it as a local editable dependency. Changes
travel with optical-adaptor commits through GitHub, so the server never patches
site-packages or silently follows upstream HEAD. NVIDIA's license and notices are
retained. The package version is fixed to `0.7.0+optical.2c0df17e`; the enclosing
optical-adaptor Git commit identifies the exact research changes.

Local extensions:

- The VLM recipe exposes data, run-state, W&B setup, and checkpoint-module hooks.
- The KD recipe accepts paired `student` / `teacher` tensor dictionaries with a
  common compact label axis; ordinary batches retain their existing behavior.
- The official bridge accepts disjoint DDP replica groups, including unequal DP
  sizes. DDP construction and recipe reductions use the student subgroup.
- Timing and aligned-logit metric hooks run inside the official forward/backward
  and optimizer flow. The optical addon owns their implementation.
- The optical model factory is accepted by the normal VLM model builder.
- Interactive launcher workers invoke the CLI module so addon recipe modules
  need not live inside the AutoModel checkout.
- YAML scalar types, including quoted numeric identifiers, are preserved through
  loading and instantiation; explicit CLI/env parsing is handled separately.
- Ordinary shared-batch CP forwards the teacher's sharded inputs; paired optical
  branches use their own model-owned input-to-target mappings.
- Single-worker CUDA launch retains NCCL, and DDP placement preserves parameter
  precision. Empty CP supervision retains its backward graph and FP32 loss scalar.
- Explicit native configs are consumed once; the Qwen causal-convolution fallback
  accepts the Transformers 5.15 argument convention.

The optical addon registers a native Qwen text architecture and an optical
strategy through existing registration hooks. Its strategy delegates to
`Qwen3_5ParallelizationStrategy`, synchronizes new adapter initialization over
existing mesh groups, adds FP32 adapter ownership, and supplies the
compact-target CP sharder. It uses the existing FSDP2 manager, mesh bridge,
accumulation, reductions, optimizer and checkpoint engine. DDP remains available.
PP, EP, sequence parallelism and Megatron FSDP are explicit unsupported cases for
this optical model. Packing and adaptive grouping remain deferred.

`tests/test_automodel_mesh.py` in the parent repository passed with both
2-student/1-teacher and 1-student/2-teacher CPU process layouts, unequal target
counts, and two accumulation microbatches against a serial gradient reference.
Affected upstream launcher/config tests also passed. The parent
[`docs/automodel-migration-validation.md`](../../docs/automodel-migration-validation.md)
records the initial migration. The current
[`parallel validation`](../../docs/automodel-parallel-validation.md) records
FSDP/TP/CP optimization, weight/gradient checks and the reproducibility fix.
Unrelated upstream suites were not run.
