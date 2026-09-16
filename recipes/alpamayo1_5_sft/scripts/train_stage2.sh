#!/bin/bash
# Stage 2: Freeze VLM, train trajectory expert (Alpamayo 1.5)
#
# Adapted from NVlabs/alpamayo scripts/train_stage2_2b.sh.
# Usage: [NUM_GPUS=4] [STAGE1_CKPT=...] ./scripts/train_stage2.sh [hydra overrides...]

RECIPE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$RECIPE_DIR"

# Load environment variables (ALPAMAYO_DATA_DIR etc.) if present. `set -a`
# exports them: a plain `source` sets shell-only vars that never reach the
# Python process.
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
CONFIG_NAME=${CONFIG_NAME:-sft_stage2}

# Stage-1 VLM output to load into the frozen VLM (overrides the pretrained one).
STAGE1_CKPT=${STAGE1_CKPT:-${RECIPE_DIR}/outputs/output_stage1/checkpoint-XXXX}

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    RESUME_ARGS+=(resume_from_checkpoint="${RESUME_FROM_CHECKPOINT}")
fi

# Stage 2 runs pure DDP (deepspeed: null in sft_stage2.yaml) — do NOT pass a
# trainer.deepspeed override here.
torchrun --nproc_per_node=${NUM_GPUS} \
    -m alpamayo1_5_sft.train_hf \
    --config-path pkg://alpamayo1_5_sft/configs \
    --config-name ${CONFIG_NAME} \
    model.stage1_vlm_checkpoint_path="${STAGE1_CKPT}" \
    "${RESUME_ARGS[@]}" \
    "$@"
