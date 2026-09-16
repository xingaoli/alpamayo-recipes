#!/bin/bash
# CoC Stage 1: Train VLM with CoC reasoning + trajectory tokens (Alpamayo 1.5)
#
# Requires ALPAMAYO_COC_JSONL and ALPAMAYO_DATA_DIR (see .env).
# Usage: [NUM_GPUS=4] ./scripts/train_coc_stage1.sh [hydra overrides...]

RECIPE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$RECIPE_DIR"

# Load environment variables (ALPAMAYO_COC_JSONL, ALPAMAYO_DATA_DIR, ALPAMAYO_WORKER_NUM_THREADS, ...)
set -a
source "$RECIPE_DIR/.env" 2>/dev/null || true
set +a

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# Limit per-worker threads and malloc arenas to prevent memory fragmentation.
# These must be set BEFORE process start; os.environ in Python is too late for glibc.
export ALPAMAYO_WORKER_NUM_THREADS=${ALPAMAYO_WORKER_NUM_THREADS:-2}
export OMP_NUM_THREADS=$ALPAMAYO_WORKER_NUM_THREADS
export MKL_NUM_THREADS=$ALPAMAYO_WORKER_NUM_THREADS
export OPENBLAS_NUM_THREADS=$ALPAMAYO_WORKER_NUM_THREADS
export MALLOC_ARENA_MAX=$ALPAMAYO_WORKER_NUM_THREADS

NUM_GPUS=${NUM_GPUS:-4}
CONFIG_NAME=${CONFIG_NAME:-sft_coc_stage1}

# DeepSpeed config must be an absolute path: Hydra changes the working directory
# at runtime, so the relative path in sft_coc_base.yaml may not resolve.
torchrun --nproc_per_node=${NUM_GPUS} \
    -m alpamayo1_5_sft.train_hf \
    --config-path pkg://alpamayo1_5_sft/configs \
    --config-name ${CONFIG_NAME} \
    trainer.deepspeed="${RECIPE_DIR}/configs/deepspeed/zero2.json" \
    "$@"
