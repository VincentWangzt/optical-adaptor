# Online optical-adaptor training

The default experiment trains a token-wise LayerNorm → Linear → GELU → Linear MLP
from DeepSeek-OCR's 1280-wide visual features to Qwen3.5-4B's 2560-wide text
embeddings. DeepSeek and Qwen stay frozen. Only the MLP receives parameter gradients
and optimizer updates.

## Launch

Run Python and GPU work on the Linux server. Commit local edits, push them to
GitHub, then pull into `~/optical-adaptor` before executing them.

```bash
git pull --ff-only
bash scripts/launch_optical_training.sh --smoke
```

The smoke profile reads 64 original records per source, prepares derived samples,
and runs three optimizer steps on two GPUs, followed by sliced evaluation and a
checkpoint. This tests execution, not convergence or downstream task success.

Smoke output directories are fixed in the script's `--smoke` block. For another
fresh smoke run, change those directories in that block; to continue an existing
run, explicitly set its resume path and a larger step budget.

For training, edit the hardcoded variables at the top of
[`scripts/launch_optical_training.sh`](../scripts/launch_optical_training.sh), then:

```bash
bash scripts/launch_optical_training.sh
```

The complete canonical configuration is
[`configs/automodel.yaml`](../configs/automodel.yaml). The launcher saves the fully
resolved configuration under `RUN_DIR/config.yaml`. Change GPU IDs, output/data
directories, steps, batch sizes, context limits, image-count filter, assistant loss
scope, generation limits, and resume path in the script. Change source paths,
weights, task prompts, bins, model revisions, loss mixture, and rendering in YAML.
The launcher checks selected GPUs immediately before training and refuses devices
with more than 1 GiB already occupied.
The launcher also disables NCCL peer-to-peer transport for this server's known
GPU 8/9 issue; DDP communicates through the working shared-memory path instead.

`uv sync --locked --group dev` installs the pinned Linux environment. vLLM is an
optional `inference` extra; training imports neither vLLM nor DeepSeek's obsolete
Transformers language decoder. The standalone DeepSeek loader imports only
`deepencoder.py` from a pinned model commit and checks all 476 checkpoint tensors.
Its deterministic CLIP position-ID buffer is reconstructed locally, as it is absent
from the released weights.

Store `HF_TOKEN` and `WANDB_API_KEY` in the server's ignored `.env`. Online W&B is
the default and missing credentials are an error. Explicitly setting
`WANDB_MODE=offline` in the launcher is available for isolated tests. W&B receives
configuration and aggregate metrics; source text, images, and generations stay on
the server.

## Data contract

Three configurable sources are pinned:

| Source | Tasks | Reasoning |
| --- | --- | --- |
| `bigcode/the-stack-smol` | Exact reconstruction and continuation, front/middle excerpts | Disabled |
| `nvidia/Open-SWE-Traces`, `minisweagent/qwen38_27b` | Next action; observation reconstruction/continuation | Retained for next action |
| `nvidia/Open-SWE-Traces`, `sweagent/qwen35_122b` | Next action; observation reconstruction/continuation | Disabled |

The Stack uses round-robin language-shard reads, so a bounded preparation run
does not accidentally use only the first language. Model provenance follows the
dataset's documented source directories; it is not inferred from generated text.

Every derived record stores ordinary SFT `messages` and `tools`. Assistant turns
contain the real target transcription, continuation, or action. A sidecar
`visual_areas` list addresses `(message, start, end)` text spans. The source text
does not contain an executable image markup language; batch processing inserts
configured boundary markers from these offsets. Tool-call argument JSON strings
are normalized to objects for the native Qwen tool template.

For naive tasks, seeded variation selects prompt wording, system/user instruction
placement, and user/tool image placement. Tool placement includes a real preceding
`read_document` call. That scaffolding call is context only; the final answer is
the reconstruction/continuation target. No reasoning is requested for naive tasks.

Visual text uses the existing newline, tab, trailing-whitespace, and font-coverage
canonicalization. Unsupported characters become visible escapes in both the
teacher's text and the rendered image. Images use the existing 1280px-wide,
automatic-height rendering defaults, then DeepSeek's direct resize to 640×640.
Each image contains at most `lines_per_image` display rows. Long observations
paginate without ellipses. One observation can therefore produce several images.

### Trajectories and slices

Each SWE trajectory produces:

- A `full` record retaining the complete conversation, including reasoning and
  any trailing observations. Training consumes through the final assistant target.
- Selected next-action records with configurable 1/2/4/8 observation-turn histories,
  plus the initial instructions. A window includes the assistant call that caused
  its first observation; it does not create orphan tool responses.
- Simple reconstruction/continuation tasks from sufficiently long observations.

`turn_count` counts assistant actions whose observations have visual areas; multiple
tool responses from one action count once. `image_count` counts actual rendered
areas. Both have independent, configurable bins. Thus a one-turn example can
belong to the 3–4-image bin. Filter image count to 1 for strictly single-image data.

Preparation writes:

```text
DATA_DIR/
  originals/<source>.jsonl           # Complete source records, including long trajectories
  slices/<source>/<task>/<view>/images-<bin>/turns-<bin>/<train|eval>.jsonl
  manifest.parquet                  # Polars index, offsets, hashes, split and slice metadata
  summary.json                     # Source counts and dataset fingerprint
```

Training verifies the preparation fingerprint before reusing a data directory.
Source limits, revisions, prompts, pagination, seed, and rendering must match the
effective configuration. Changing only sampling weights does not require new data.
Use `views: [full]`, `[window]`, `[front]`, `[middle]`, or `[observation]` to filter
view families; image and turn bins provide the separate size filters.

Repository-hash assignment is shared across sources and derived tasks. Variants
from a repository cannot cross train/eval. This is repository separation, not a
claim that copied code in unrelated repositories has been eliminated. The eval
sets are used throughout training and should not be described as a held-out test.

Preflight checks token lengths and produces `checkpoints/data-report.json` with
per-slice eligibility, rejection reasons, and teacher/student/target lengths.
Ranks share the CPU preflight scan and merge their results in sample order.
`overlength: drop` rejects a whole example; `error` stops instead. Nothing is
token-truncated. All full records remain in the prepared dataset. To train on long
full trajectories, select `views: [full]` and raise both context limits to fit the
intended examples and available memory. A context limit is not a guarantee that
the corresponding batch fits two A6000s.

Training draws from a configurable source/task mixture. `balance_slices: true`
also balances the available bins within each source/task, using replacement.
`samples_per_epoch` controls the number of draws; an epoch is not necessarily a
single visit to every row. A global seeded draw list is split across DP ranks.

## Paired inputs and losses

1. Read one conversation and its visual offsets.
2. Serialize the native Qwen tool/chat format, preserving exact content whitespace
   and all historical reasoning. The inference template's trimming and thought
   removal are explicitly disabled for SFT.
3. The teacher sees text inside the configured visual boundary markers. The
   student replaces that text with 111 adapted feature vectors per image.
4. Construct two prediction-position lists. For each supervised token at `p`,
   the corresponding prediction comes from `p - 1` in that branch's own sequence.
   The target token IDs must match exactly across both branches.
5. Gather only these hidden states, project them into vocabulary logits in small
   chunks, and compute ground-truth CE and AutoModel's full-vocabulary forward KL.

The objective is:

```text
loss = (1 - kd_ratio) * CE + kd_ratio * T² KL(teacher_T || student_T)
```

Image slots, padding, user/system/tool messages, and injected assistant prefixes
never contribute target labels. With `assistant_loss: all`, all genuine SWE
assistant targets contribute, including reasoning and tool calls. With `last`, only
the last action contributes; earlier assistant turns remain conditioning context.
The empty no-thinking prefix is supplied as context, not learned as transcription.
The first assistant action may precede all images: its KL is then zero in exact
arithmetic and its CE cannot improve the adapter. Select `last` when this dilution
is undesirable for a next-action experiment.

Both branches use the **original frozen Qwen3.5-4B** as the online teacher/backbone.
The 27B/122B trajectories supply SFT labels; those large generators are not loaded
as distillation teachers. One Qwen instance per rank runs the teacher pass under
`no_grad`, then the student pass with activation checkpointing. Hidden states and
image features live only for the current batch. There are no embedding or teacher
state caches.

Loss normalization uses the number of supervised tokens across all ranks and
accumulation microbatches, with the matching DDP reduction factor. This avoids
giving short and long microbatches equal weight accidentally. The AutoModel VLM
KD subclass inherits the optimizer-step loop, scheduling, gradient synchronization,
clipping, checkpoint engine, and main training loop. Its forward hook handles the
different sequence positions and its evaluation hook handles slice aggregation.
The current strategy is shared-setup DDP; TP/CP/PP and separate teacher meshes are
rejected explicitly.

## Evaluation

Teacher-forced evaluation visits a deterministic per-slice subset without padding
or repeating examples across ranks:

- `eval-core/{loss,ce,kl,teacher_ce,agreement,token_accuracy,tokens}`: token-weighted
  totals over the selected evaluation mixture.
- `eval-aux/<source>/<task>/<view>/images-<bin>/turns-<bin>/...`: the same metrics
  for every populated slice.
- `cer`, `ler`, and `generated_samples`: greedy, no-thinking reconstruction
  generation, with character/line Levenshtein distances divided by reference size.
- `character_edits`, `line_edits`, and `generation_limit_fraction`: raw edit totals
  and the fraction of generations that reached the token cap before EOS.

Core metrics depend on the selected mixture and token lengths; compare runs with
the same data fingerprint and evaluation selection. Separate slice metrics are
needed to see whether progress comes only from easy OCR examples.

Generations are stored in per-rank JSONL files, never sent to W&B. The smoke profile
has a deliberately small 64-token generation limit, so its edit distances are an
execution check, not a meaningful reconstruction-quality measurement.

## Checkpoints and resume

AutoModel checkpoints contain the MLP, optimizer, scheduler, RNG, data-loader
position, effective configuration, W&B run ID, and a strict run-contract fingerprint.
Frozen model weights are reloaded from the pinned upstream models. Checkpoints
from the old cached training pipeline are not resume-compatible.

To resume, set `RESUME_FROM` in the launcher to `LATEST` or a concrete checkpoint
directory and keep `RUN_DIR` and `DATA_DIR` unchanged. Incompatible model, loss,
data, or distributed contracts fail before resuming. Reusing an occupied checkpoint
directory without an explicit resume setting is an error. A longer maximum-step
budget can be used for a continuation. AutoModel restores the checkpoint's LR
schedule, including its original decay horizon; extending the step budget does
not restart the learning-rate curve. Changing the configured LR policy, batch
sizes, or gradient clipping is rejected by the run contract.

## Verification and extension points

```bash
uv run --no-sync pytest tests/test_automodel_processing.py tests/test_automodel_losses.py tests/test_automodel_data.py tests/test_training_data.py
```

The focused checks cover exact whitespace/Unicode targets, varying image lengths,
causal prediction positions, multi-turn all/last masking, historical reasoning,
full-trajectory preservation, whole-example length rejection, exact chunked KD
values/gradients, global-token normalization, data identity, and loader recovery
with both zero and two data workers. Run the launcher
for model loading, frozen-gradient, distributed update, evaluation, and checkpoint
validation.

`optical.llm` configures the model ID, revision, text-subconfiguration key, attention
backend, and gradient checkpointing. `optical.vision._target_` selects an importable
encoder factory with `pixels()` and `forward()` methods; feature dimensions and
tokens per image are explicit. `model`/`optical.adapter` share one YAML anchor.
The default compiler validates the Qwen3.5 template rather than assuming every
model uses the same chat syntax. Supporting a different template requires adapting
that compiler and checking its masks, in addition to changing the model ID.

Upstream contracts:

- [AutoModel VLM KD recipe, pinned commit](https://github.com/NVIDIA-NeMo/Automodel/blob/2c0df17efff1fc9d5128a9df1a2ed716d0f2c54a/nemo_automodel/recipes/vlm/kd.py)
- [Open-SWE-Traces, pinned revision](https://huggingface.co/datasets/nvidia/Open-SWE-Traces/tree/f967cba3312573981a47fd7a7b80029b53909b5f)
- [DeepSeek-OCR vision code](https://huggingface.co/deepseek-ai/DeepSeek-OCR/blob/9f30c71f441d010e5429c532364a86705536c53a/deepencoder.py)
