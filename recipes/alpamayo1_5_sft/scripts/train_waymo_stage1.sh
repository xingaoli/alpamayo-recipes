#!/bin/bash
# Waymo Stage 1: Train VLM with discrete trajectory tokens (Alpamayo 1.5)
#
# Waymo rds_hq dataset (PAI-contract samples via alpamayo.data.waymo_rds).
# Requires ALPAMAYO_WAYMO_DIR (see .env) pointing at the rds_hq root.
#
# Configs are templates: tune from this script via env vars (defaults below).
# Extra hydra overrides may be appended on the command line — they come last,
# so they win over both the config and the env vars below.
#
# Usage:
#   ./scripts/train_waymo_stage1.sh                                    # defaults
#   RESUME_FROM_CHECKPOINT=outputs/output_waymo_stage1/checkpoint-03000 \
#     ./scripts/train_waymo_stage1.sh                                  # resume
#   # smoke (subset, step cap, no eval/save):
#   MAX_TRAIN_CLIPS=4 MAX_VAL_CLIPS=2 MAX_STEPS=4 NUM_WORKERS=2 \
#     ./scripts/train_waymo_stage1.sh trainer.eval_strategy=no +trainer.save_strategy=no

RECIPE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$RECIPE_DIR"

# Load environment variables (ALPAMAYO_WAYMO_DIR etc.) if present. `set -a`
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

# ------------------------- tunables (env-var overridable) -------------------------
NUM_GPUS=${NUM_GPUS:-4}
CONFIG_NAME=${CONFIG_NAME:-sft_waymo_stage1}
# Base model for stage 1 — must be the A1-format conversion (see README).
CKPT=${CKPT:-ckpts/Alpamayo-1.5-10B-A1-format}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/output_waymo_stage1}
BATCH_SIZE=${BATCH_SIZE:-4}          # per-device train batch size
GRAD_ACCUM=${GRAD_ACCUM:-4}          # effective batch = bs * accum * NUM_GPUS
LR=${LR:-1e-5}
NUM_EPOCHS=${NUM_EPOCHS:-3}
SAVE_STEPS=${SAVE_STEPS:-300}
EVAL_STEPS=${EVAL_STEPS:--1}
MAX_EVAL_STEPS=${MAX_EVAL_STEPS:-50} # per-rank eval batches per eval round; -1 = full val set
NUM_WORKERS=${NUM_WORKERS:-16}
# Optional (empty = no override): clip subset / step cap — smoke tests.
MAX_TRAIN_CLIPS=${MAX_TRAIN_CLIPS:-}
MAX_VAL_CLIPS=${MAX_VAL_CLIPS:-}
MAX_STEPS=${MAX_STEPS:-}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
# ----------------------------------------------------------------------------------

if [[ ! -d "$CKPT" ]]; then
    echo "ERROR: CKPT does not exist: $CKPT" >&2
    echo "       Set CKPT=<path/to/Alpamayo-1.5-10B-A1-format> (A1-format required)." >&2
    exit 1
fi

OVERRIDES=(
    model.checkpoint_path="${CKPT}"
    paths.output_dir="${OUTPUT_DIR}"
    trainer.per_device_train_batch_size="${BATCH_SIZE}"
    trainer.gradient_accumulation_steps="${GRAD_ACCUM}"
    trainer.learning_rate="${LR}"
    trainer.num_train_epochs="${NUM_EPOCHS}"
    trainer.save_steps="${SAVE_STEPS}"
    trainer.eval_steps="${EVAL_STEPS}"
    evaluate.max_eval_steps="${MAX_EVAL_STEPS}"
    trainer.dataloader_num_workers="${NUM_WORKERS}"
    # DeepSpeed config must be an absolute path: Hydra changes the working
    # directory at runtime, so the relative path in sft_base_waymo.yaml may
    # not resolve.
    trainer.deepspeed="${RECIPE_DIR}/configs/deepspeed/zero2.json"
)
[[ -n "$MAX_TRAIN_CLIPS" ]] && OVERRIDES+=("data.train_dataset.max_clips=${MAX_TRAIN_CLIPS}")
[[ -n "$MAX_VAL_CLIPS" ]] && OVERRIDES+=("data.val_dataset.max_clips=${MAX_VAL_CLIPS}")
[[ -n "$MAX_STEPS" ]] && OVERRIDES+=("+trainer.max_steps=${MAX_STEPS}")
[[ -n "$RESUME_FROM_CHECKPOINT" ]] && OVERRIDES+=("resume_from_checkpoint=${RESUME_FROM_CHECKPOINT}")

torchrun --nproc_per_node=${NUM_GPUS} \
    -m alpamayo1_5_sft.train_hf \
    --config-path pkg://alpamayo1_5_sft/configs \
    --config-name ${CONFIG_NAME} \
    "${OVERRIDES[@]}" \
    "$@"
