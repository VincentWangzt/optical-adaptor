# Optical adaptor training

The v1 pipeline freezes DeepSeek-OCR and the post-trained Qwen3.5-4B text model, and
trains either a token-wise MLP or a two-layer bidirectional Transformer followed by
the MLP. `configs/training.yaml` is the canonical configuration. Unknown fields,
incompatible shapes, stale manifests/caches, and mismatched resume configurations
are errors.

## Data and caches

Run Python only on the Linux server with `uv`. Commit and push local changes before
pulling them on the server. The server's `.env` supplies `HF_TOKEN` and
`WANDB_API_KEY`; the explicit W&B destination is in the training configuration.
Real `.env` files, data, caches, predictions, and checkpoints are ignored by Git.

```bash
uv sync --locked
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false uv run --locked prepare-adapter-data --workers 8
nvidia-smi
# Select currently idle GPUs before executing these examples.
CUDA_VISIBLE_DEVICES=8 uv run --locked extract-deepseek-embeddings --batch-size 8 --workers 8
CUDA_VISIBLE_DEVICES=9 uv run --locked cache-qwen-teacher --batch-size 4
```

Preparation uses the pinned revision of The Stack Smol, preserving all 30 languages.
Repository-hash assignment makes splits repository-disjoint. Seeded file ordering
and ordered consumption of CPU-worker results make selection independent of worker
completion order. Each source file supplies at most one record. Selection resumes
from partial checkpoints and fails with an eligibility report if exact quotas
cannot be filled.

The manifest contains 10,000 training records and three 500-record evaluation sets:
front continuation, middle continuation, and reconstruction. Each span contains
40–60 logical lines, no more than 80 display lines or 2,048 Qwen tokens, and exactly
128 continuation tokens. Middle spans begin halfway through a file's logical lines
and include up to 256 preceding tokens. An exact-hash overlap audit is written to
`outputs/adapter-v1/data/eligibility.json`; repository separation alone does not
eliminate copied files or spans.

Rendering uses 1280px width, automatic height, 20px JetBrains Mono with Liberation
Serif fallback, 28px line spacing, 100-column wrapping, and 1% margins. Unsupported
characters and control characters become visible escapes. Newlines, four-column
tabs, and trailing horizontal whitespace are canonicalized. Reconstruction targets
retain logical source lines rather than inserted display wraps. An invisible final
visual newline is excluded, while the following continuation retains its boundary
newline. Inspection markup is generated from offsets with escaped source content.

DeepSeek uses its official direct square resize to 640px and only its 476 vision
checkpoint tensors. Encoder shards store `[N,111,1280]` BF16 tensors. Teacher shards
store `[N,128,2560]` BF16 final normalized hidden states at the positions predicting
the continuation. Only the 11,000 continuation-capable records require teacher
states. Both cache types have fingerprints, exact shapes/dtypes, and file checksums;
only verified completed shards are reused. Image rendering is prefetched by CPU
workers; full images remain ephemeral apart from configured debug samples.

## Training and profiling

Each rank processes both continuation and reconstruction. Independent task
permutations cover every training record once per task per epoch. The global batch
is **32 pairs**, or 64 examples; each rank contributes 16 pairs on two GPUs. The
final update contains 16 global pairs. There are 313 updates per epoch and 939 over
three epochs. AdamW warms up for 29 updates and then holds learning rate at `2e-4`.
Weight decay remains `0.01`.

The objective is half token-mean full-vocabulary teacher-to-student KL and half
token-mean reconstruction CE. Normalization spans every rank and microbatch in the
entire update, including its partial final batch. Continuation CE is metric-only.
Reconstruction optimizes the assistant end token but excludes it from reported text
CE and token accuracy. Losses use FP32 log probabilities and reductions; the frozen
model computation uses BF16. Chunked vocabulary projection and non-reentrant
checkpointing bound memory. Only the adapter is wrapped in DDP and checkpointed as
a trainable model. One gradient synchronization and optimizer step occur per update.
The pinned FLA package supplies Qwen's linear-attention kernels; model loading fails
if Transformers selects its slow PyTorch fallback. Torch, Transformers, and vLLM
versions remain unchanged. Kernel versions participate in profile and resume identities.
Deterministic GPU algorithms and the configured cuBLAS workspace make repeated
backward passes reproducible, including checkpoint replay.

Profile each candidate separately so an out-of-memory failure cannot leave another
candidate with a damaged CUDA/DDP process. Use the same commands for
`train_transformer_adapter.py`. Check `nvidia-smi` before each job.

```bash
# This server's GPU 8/9 peer-to-peer path hangs during the first NCCL collective.
# Shared-memory transport works; apply this only to our launch environment.
export NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=8,9 uv run --locked accelerate launch \
  --multi_gpu --num_processes 2 --mixed_precision bf16 \
  --num_cpu_threads_per_process 4 --main_process_port 29571 \
  scripts/train_mlp_adapter.py --mode profile --microbatch-size 1
# Repeat with --microbatch-size 2, 4, and 8.
```

Profiles include the longest training reconstructions, two warmup updates and three
timed updates. The trainer chooses the lowest measured time among passing profiles
below 40 GiB peak reserved memory. Results are locked per adapter and configuration.

```bash
CUDA_VISIBLE_DEVICES=8,9 uv run --locked accelerate launch \
  --multi_gpu --num_processes 2 --mixed_precision bf16 \
  --num_cpu_threads_per_process 4 --main_process_port 29571 \
  scripts/train_mlp_adapter.py --mode overfit --max-updates 100

CUDA_VISIBLE_DEVICES=8,9 uv run --locked accelerate launch \
  --multi_gpu --num_processes 2 --mixed_precision bf16 \
  --num_cpu_threads_per_process 4 --main_process_port 29571 \
  scripts/train_mlp_adapter.py --mode train
```

Full training requires a passing overfit gate for both losses. Run the two adapter
experiments separately and sequentially. Use the remote MCP's durable job tools for
long runs and inspect their exit codes and logs.

Checkpoints include adapter safetensors/configuration, optimizer, scheduler,
per-rank RNG, next epoch/update, dataset/configuration fingerprints, runtime versions,
and W&B ID. Frozen model weights are excluded. To continue a saved run, append
`--resume outputs/adapter-v1/runs/mlp/step-000250`. Existing runs require explicit
resume. `--stop-after N` saves a checkpoint for controlled interruption tests.
The final checkpoint and latest three periodic checkpoints are retained.

## Evaluation

Teacher-forced evaluation runs at initialization, every 100 updates, and completion.
Greedy reconstruction uses a fixed 50-record subset every 500 updates and all 500
records at completion, with a 4,096-token generation cap. Teacher-forced evaluation
batches four records; generation batches 16 records per rank, grouped by target
length after selecting the fixed evaluation subset. Metrics are token-weighted and stratified
by language, aspect ratio, logical lines, and display lines. Generation reports
character/word edit distance, CER/WER, exact match, and truncation rate. Edit distance
is exact unit-cost Levenshtein, using RapidFuzz through the existing metric helper.

```bash
CUDA_VISIBLE_DEVICES=8 uv run --locked evaluate-adapter --reference teacher
CUDA_VISIBLE_DEVICES=8 uv run --locked evaluate-adapter --reference native --generate
CUDA_VISIBLE_DEVICES=8 uv run --locked evaluate-adapter \
  --checkpoint outputs/adapter-v1/runs/mlp/final --generate
```

References are fingerprinted, evaluated once, and reused across experiments. Native
Qwen uses its official image processor and spatial positions on the same rendered
images; its actual grid and visual-token count are recorded under
`references/native/inputs/` and used for compression metrics. Adapter
embeddings use sequential pseudo-language positions instead of a native spatial grid.
The final checkpoint is the primary result; evaluation metrics do not select a best
checkpoint. These repeatedly inspected sets are evaluation sets, not held-out tests.

W&B receives aggregate metrics, hashes, configuration, throughput, and system
statistics. Source, images, targets, predictions, manifests, and tensors stay on the
server. Authentication errors never silently switch to offline logging.

## Validation

```bash
uv run --locked pytest
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run --locked python -m torch.distributed.run \
  --standalone --nproc_per_node=2 tests/test_training_distributed.py \
  --output outputs/validation/ddp

CUDA_VISIBLE_DEVICES=8,9 uv run --locked accelerate launch \
  --multi_gpu --num_processes 2 --mixed_precision bf16 \
  --num_cpu_threads_per_process 4 --main_process_port 29571 \
  tests/test_training_gpu_resume.py --output outputs/validation/gpu-resume
```

Focused tests cover canonicalization, render geometry, configuration, repository
assignment, task coverage, final partial batches, exact KL and metric detachment,
chunked gradients, target alignment, cache corruption, and checkpoint replay.
The distributed test compares unequal-length accumulated losses with an unbatched
reference and checks exact CPU replay. Real GPU probes additionally validate both
frozen model boundaries, adapter-only gradients, and native/pseudo-image generation.
