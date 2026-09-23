set -o pipefail
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
uv run --no-sync automodel \
    configs/automodel.yaml --nproc-per-node 8 \
    2>&1 | tee outputs/automodel/reconstruct-continuation-3000-steps/train.log
