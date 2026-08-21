#!/bin/bash
# Sealed launcher for one autoresearch trial.
# Full PATH is required: scrubbing it breaks Triton linker discovery ("cannot find ld").
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

WORKDIR="$1"
GPU="${2:-2}"
HARD_TIMEOUT="${3:-900}"

cd "$WORKDIR" || exit 97
export CUDA_VISIBLE_DEVICES="$GPU"
# Per-GPU inductor cache so concurrent tenants never collide on compile artifacts.
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/inductor_gpu${GPU}"

timeout --signal=KILL "$HARD_TIMEOUT" "$WORKDIR/.venv/bin/python" train.py > run.log 2>&1
echo "EXIT_CODE=$?" >> run.log
