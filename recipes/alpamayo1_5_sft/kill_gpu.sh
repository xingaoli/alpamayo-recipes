#!/bin/bash
# kill_gpu.sh — one-shot cleanup of every process holding GPU memory.
#
# Container-safe: GPU processes are found by scanning /proc/*/fd for an
# open /dev/nvidia-uvm device, i.e. namespace-local PIDs that ps/kill can
# act on. Do NOT target via `nvidia-smi --query-compute-apps`: inside a
# container it reports HOST pids, which do not exist in /proc here.
# Holding /dev/nvidia-uvm is the CUDA-process marker: every real compute
# process opens it, while NVML tools (nvidia-smi, health checks like
# cudaCheck) only open /dev/nvidiaN briefly and never touch uvm — so
# transient monitors are not misdetected as GPU jobs.
#
# Sends SIGTERM, waits GRACE seconds, then SIGKILLs survivors. Also reaps
# leftover torchrun agents so they cannot respawn ranks.
# By default only kills processes owned by the current user; run as root
# or set KILL_ALL=1 to sweep every user's processes.
#
# Usage:
#   ./kill_gpu.sh                     # kill my GPU processes
#   DRY_RUN=1 ./kill_gpu.sh           # show targets, kill nothing
#   MIN_AGE=30 ./kill_gpu.sh           # only kill processes older than 30s
#   GRACE=30 ./kill_gpu.sh            # longer SIGTERM grace period
#   KILL_ALL=1 sudo ./kill_gpu.sh     # kill every user's (needs root)

set -u

GRACE=${GRACE:-10}
KILL_ALL=${KILL_ALL:-0}
MIN_AGE=${MIN_AGE:-60} # skip CUDA procs younger than this (platform health checks do cuInit and live for seconds)
DRY_RUN=${DRY_RUN:-0}

# Compute process = holds /dev/nvidia-uvm open. (Holders of bare
# /dev/nvidiaN include short-lived NVML monitors; see header.)
gpu_pids() {
    find /proc/[0-9]*/fd -lname '/dev/nvidia-uvm*' 2>/dev/null \
        | sed -E 's#^/proc/([0-9]+)/fd/.*#\1#' | sort -un
}

targets=()
for pid in $(gpu_pids); do
    [[ "$pid" == "$$" ]] && continue
    ps -o user= -p "$pid" >/dev/null 2>&1 || continue  # exited meanwhile
    age="$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')"
    if [[ -n "$age" && "$age" -lt "$MIN_AGE" ]]; then
        echo "SKIP pid $pid (age ${age}s < MIN_AGE=${MIN_AGE}s — transient monitor?)"
        continue
    fi
    user="$(ps -o user= -p "$pid")"
    if [[ "$KILL_ALL" != 1 && "$(id -u)" != 0 && "$user" != "$(whoami)" ]]; then
        echo "SKIP pid $pid (user $user) — KILL_ALL=1 to include"
        continue
    fi
    targets+=("$pid")
done

if [[ ${#targets[@]} -eq 0 ]]; then
    echo "No GPU compute processes found."
else
    echo "GPU compute processes:"
    for pid in "${targets[@]}"; do
        ps -o pid=,user=,etime=,args= -p "$pid" | cut -c1-140
    done
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "DRY_RUN=1 — nothing killed."
        exit 0
    fi

    kill -TERM "${targets[@]}" 2>/dev/null || true
    alive=("${targets[@]}")
    for _ in $(seq 1 "$GRACE"); do
        still=()
        for pid in "${alive[@]}"; do
            kill -0 "$pid" 2>/dev/null && still+=("$pid")
        done
        alive=("${still[@]}")
        [[ ${#alive[@]} -eq 0 ]] && break
        sleep 1
    done
    if [[ ${#alive[@]} -gt 0 ]]; then
        echo "SIGKILL survivors: ${alive[*]}"
        kill -9 "${alive[@]}" 2>/dev/null || true
        sleep 1
    fi
fi

# Orphaned torchrun / elastic agents would respawn or hang ranks; reap them.
mapfile -t AGENTS < <(pgrep -u "$(id -u)" -f 'torchrun|torch\.distributed\.run' || true)
if [[ ${#AGENTS[@]} -gt 0 && "$DRY_RUN" != 1 ]]; then
    echo "Killing torchrun agents: ${AGENTS[*]}"
    kill -9 "${AGENTS[@]}" 2>/dev/null || true
fi

# Re-scan to confirm. nvidia-smi per-process output is host-pid based here,
leftover="$(gpu_pids)"
old_leftover=()
for pid in $leftover; do
    age="$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')"
    [[ -n "$age" && "$age" -ge "$MIN_AGE" ]] && old_leftover+=("$pid")
done
if [[ ${#old_leftover[@]} -eq 0 ]]; then
    echo "OK: GPUs clean."
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
else
    echo "WARNING: GPU process(es) still alive:" >&2
    for pid in "${old_leftover[@]}"; do
        ps -o pid=,user=,args= -p "$pid" 2>/dev/null | cut -c1-140 >&2
    done
    exit 1
fi
