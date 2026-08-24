#!/bin/bash
# supervise.sh — keep the search running for the full campaign window.
#
# Third layer of the same lesson. The harness now falls back to a replicate instead of
# exiting on a rendezvous timeout, and the restocker keeps the queue from emptying in
# the first place; this catches everything else — an unhandled exception, an OOM kill,
# a transient CUDA fault. The journal is persisted after every trial, so a restart
# resumes with nothing lost.
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"

BASE="$HOME/ai_scientist_v2_baseline"
PY="$BASE/task/.venv/bin/python"
GPU="${1:-0}"
HOURS="${2:-24}"

END=$("$PY" -c "import time; print(time.time() + $HOURS*3600)")
N=0

while :; do
  NOW=$("$PY" -c 'import time; print(time.time())')
  if [ "$("$PY" -c "print(1 if $NOW >= $END else 0)")" = "1" ]; then
    echo "[supervise] ${HOURS}h window complete after $N launches"
    break
  fi
  N=$((N+1))
  REMAIN=$("$PY" -c "print(max(0.05, ($END - $NOW)/3600))")
  echo "[supervise] launch #$N on GPU $GPU, ${REMAIN}h remaining"

  # --min-improvement restores the guard upstream's LLM selector provides (2 sigma of the
  # measured pooled noise floor); --num-seeds is the port of multi_seed_eval. Both are
  # load-bearing: without them the search promotes noise and never measures seed variance.
  cd "$BASE" && "$PY" ai_scientist_ar/run_bfts.py \
      --gpu "$GPU" --num-drafts 3 --rendezvous-timeout 1800 \
      --min-improvement 0.00036 --num-seeds 3 \
      --max-hours "$REMAIN" 2>&1 | tee -a "$BASE/campaign/harness.out"

  echo "[supervise] harness exited (code ${PIPESTATUS[0]}); restarting in 20s"
  sleep 20
done
