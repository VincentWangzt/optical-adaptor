# AutoModel migration validation status

Date: 2026-09-17. Implementation branch: `codex/automodel-optical-kd`.

The migration is committed locally. `uvx ruff check src tests` and Ruff checks on
the changed AutoModel files pass. `git diff --check` passes. Python execution and
GPU optimization for this migration have **not run**. Historical results in
`automodel-validation.md` belong to the earlier implementation.

GitHub push protection blocks the imported upstream source, misclassifying the
public class name `Mistral3ForConditionalGeneration` as a Mistral API key. The
flagged import/class references were inspected; they are not credentials. The
false-positive bypass API returned HTTP 500, and browser access was unavailable.
The user has been asked to allow this false positive through GitHub's unblock
page. No direct local-to-server copy was attempted: repository policy requires a
local commit, GitHub push, and server pull before execution.

## Remaining server checks

After the push is allowed, pull the branch and synchronize the editable dependency
with `uv sync --locked --group dev`. Resolve any lock/build issues before testing.

1. Run focused processing, data, objective, and retained benchmark tests with
   `uv run pytest`. The data tests cover inherited weights, fixed holdouts, primary
   bins, and loader continuation with/without worker prefetch.
2. Run `CUDA_VISIBLE_DEVICES='' uv run pytest tests/test_automodel_mesh.py` for the
   official bridge with 2-student/1-teacher and 1-student/2-teacher DDP groups,
   differing target counts, and accumulated gradients versus a serial reference.
3. Prepare fresh schema-2 smoke data. Existing raw originals can be reused via
   `uv run python -m optical_adaptor.automodel.prepare --originals-dir ...` when
   their source limits match the smoke configuration.
4. Inspect `nvidia-smi`, explicitly select idle GPUs, and run several optimizer
   updates through the ordinary `automodel` CLI. Exercise shared and separate
   teacher/student placement, accumulation, evaluation, checkpoint/export, and
   continuation from a saved checkpoint. Compare adapter changes, finite losses,
   consumed counts, and restored sampler state. Record actual timings and memory.

Initial optical parallel support is DDP. Full 32K sequences, FSDP/TP/CP/PP and
retained vLLM live inference are not validated by static inspection. Unchunked
full-vocabulary logits require measured memory limits; no performance or quality
improvement is claimed from the migration alone.
