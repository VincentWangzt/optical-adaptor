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

The optical wrapper currently validates DDP only. It rejects FSDP/TP/CP/PP rather
than claiming untested sharding support. The original upstream mesh-backed routes
remain available for other supported models. Packing and adaptive grouping are
explicitly deferred.

`tests/test_automodel_mesh.py` in the parent repository passed with both
2-student/1-teacher and 1-student/2-teacher CPU process layouts, unequal target
counts, and two accumulation microbatches against a serial gradient reference.
Affected upstream launcher/config tests also passed. The parent
[`docs/automodel-migration-validation.md`](../../docs/automodel-migration-validation.md)
records GPU optimization, checkpoint continuation and its unresolved numerical
reproducibility limit. Unrelated upstream suites were not run.
