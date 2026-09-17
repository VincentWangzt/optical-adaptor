#!/usr/bin/env bash
set -euo pipefail

# Edit these variables for an experiment. Run this script on the GPU server.
GPU_IDS="8,9"
NUM_GPUS=2
NCCL_P2P_DISABLE=1                # This server's GPU 8/9 P2P transport hangs.
CONFIG="configs/automodel.yaml"
RUN_DIR="outputs/automodel/100k-recon-cont-ddp"
DATA_DIR="outputs/automodel/data-100k-5pct"
PREPARE_DATA=1
INSTALL_ENVIRONMENT=1
SOURCE_LIMIT=""                  # Empty: use each source's limit in the YAML.
MAX_STEPS=""                     # Empty: use the YAML's complete epoch schedule.
GLOBAL_BATCH_SIZE=64
LOCAL_BATCH_SIZE=1
MAX_TEACHER_TOKENS=3072
MAX_STUDENT_TOKENS=2048
MAX_IMAGES=2
ASSISTANT_LOSS="all"              # all | last
EVAL_SAMPLES_PER_SLICE=16
GENERATION_SAMPLES_PER_SLICE=2
MAX_NEW_TOKENS=4096
WANDB_MODE="online"
RESUME_FROM=""                   # Explicit checkpoint directory or LATEST.

if [[ "${1:-}" == "--smoke" ]]; then
    RUN_DIR="outputs/automodel/migration-smoke"
    DATA_DIR="outputs/automodel/migration-smoke-data"
    SOURCE_LIMIT=64
    MAX_STEPS=3
    GLOBAL_BATCH_SIZE=2
    MAX_TEACHER_TOKENS=4096
    MAX_STUDENT_TOKENS=4096
    MAX_IMAGES=4
    EVAL_SAMPLES_PER_SLICE=1
    GENERATION_SAMPLES_PER_SLICE=1
    MAX_NEW_TOKENS=64
elif [[ $# -gt 0 ]]; then
    echo "Usage: bash scripts/launch_optical_training.sh [--smoke]" >&2
    exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export NCCL_P2P_DISABLE
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

if [[ "$INSTALL_ENVIRONMENT" == 1 ]]; then
    uv sync --locked --group dev
fi
mkdir -p "$RUN_DIR"
EFFECTIVE_CONFIG="$RUN_DIR/config.yaml"
uv run --no-sync python - "$CONFIG" "$EFFECTIVE_CONFIG" "$RUN_DIR" "$DATA_DIR" \
    "$MAX_STEPS" "$GLOBAL_BATCH_SIZE" "$LOCAL_BATCH_SIZE" "$MAX_TEACHER_TOKENS" \
    "$MAX_STUDENT_TOKENS" "$MAX_IMAGES" "$ASSISTANT_LOSS" "$WANDB_MODE" "$RESUME_FROM" \
    "$EVAL_SAMPLES_PER_SLICE" "$GENERATION_SAMPLES_PER_SLICE" "$MAX_NEW_TOKENS" "$SOURCE_LIMIT" <<'PY'
import sys
from pathlib import Path
import yaml
from optical_adaptor.automodel.config import OpticalConfig

(source, output, run, data, steps, global_batch, local_batch, teacher_tokens,
 student_tokens, max_images, assistant_loss, mode, resume, eval_per_slice,
 generation_per_slice, max_new_tokens, source_limit) = sys.argv[1:]
config = yaml.safe_load(Path(source).read_text())
config['checkpoint']['checkpoint_dir'] = str(Path(run) / 'checkpoints')
config['checkpoint']['restore_from'] = resume or None
config['step_scheduler']['global_batch_size'] = int(global_batch)
config['step_scheduler']['local_batch_size'] = int(local_batch)
if steps:
    config['step_scheduler']['max_steps'] = int(steps)
    config['lr_scheduler']['lr_warmup_steps'] = min(
        config['lr_scheduler']['lr_warmup_steps'], max(0, int(steps) - 1)
    )
config['wandb']['mode'] = mode
config['wandb']['name'] = Path(run).name
optical = config['optical']
optical['prepare']['output_dir'] = data
if source_limit:
    for source in optical['prepare']['sources']:
        source['max_records'] = int(source_limit)
optical['processing'].update(assistant_loss=assistant_loss,
    max_teacher_tokens=int(teacher_tokens), max_student_tokens=int(student_tokens))
for split in ('train_filter', 'eval_filter'):
    optical['data'][split]['max_images'] = int(max_images) if max_images else None
optical['data']['eval_samples'] = {'': int(eval_per_slice)}
optical['evaluation']['generation_samples'] = {'': int(generation_per_slice)}
optical['evaluation']['max_new_tokens'] = {'': int(max_new_tokens)}
OpticalConfig.model_validate(optical)
Path(output).write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
PY

if [[ "$PREPARE_DATA" == 1 && ! -f "$DATA_DIR/manifest.parquet" ]]; then
    uv run --no-sync python -m optical_adaptor.automodel.prepare --config "$EFFECTIVE_CONFIG" \
        2>&1 | tee "$RUN_DIR/prepare.log"
fi

# Check the chosen devices immediately before training, after data preparation.
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
IFS=',' read -ra DEVICES <<< "$GPU_IDS"
if [[ "${#DEVICES[@]}" != "$NUM_GPUS" ]]; then
    echo "GPU_IDS and NUM_GPUS disagree" >&2
    exit 1
fi
for GPU in "${DEVICES[@]}"; do
    USED=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
    if (( USED > 1024 )); then
        echo "GPU $GPU is occupied ($USED MiB); edit GPU_IDS to select idle devices." >&2
        exit 1
    fi
done
git rev-parse HEAD > "$RUN_DIR/git-commit.txt"
uv run --no-sync automodel "$EFFECTIVE_CONFIG" --nproc-per-node "$NUM_GPUS" \
    2>&1 | tee -a "$RUN_DIR/train.log"
