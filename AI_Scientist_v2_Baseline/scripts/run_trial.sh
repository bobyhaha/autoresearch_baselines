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

# --- contention gate -------------------------------------------------------
# The budget is wall clock, so a co-tenant computing on the same device steals SMs and
# depresses step count directly. This was measured, not assumed: a byte-identical
# replicate of the incumbent scored 0.981862 against its n=9 band of 0.969766 +/- 0.000126
# — 96 sigma — with steps down 1314 -> 1109 and MFU 44.6% -> 37.6%. Data collected under
# contention is not comparable to the rest of the campaign, so it is worse than no data.
#
# With no trial of ours running, any utilisation on this device belongs to someone else.
# Wait for it to clear rather than record a confounded trial.
GATE_MAX="${GATE_MAX:-1800}"      # give up waiting after this long and run anyway
GATE_UTIL="${GATE_UTIL:-25}"      # foreign utilisation percent considered contended
GATE_FREE="${GATE_FREE:-75000}"    # need this much free MB (config peaks ~67GB)
GATE_GROWTH="${GATE_GROWTH:-4000}"# abort if foreign memory GROWS by this much mid-run
#                                 # (was an absolute foreign-memory bar; that refused a GPU
#                                 # holding 49GB while computing nothing, which is
#                                 # harmless. Growth means a NEW tenant arrived.
gate_start=$(date +%s)
GATE_MSG=""
FOREIGN_BASE=0
while :; do
  u=0
  for _ in 1 2 3; do
    s=$(nvidia-smi --id="$GPU" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')
    u=$(( u + ${s:-0} )); sleep 1
  done
  u=$(( u / 3 ))
  fm=$(nvidia-smi --id="$GPU" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
       | awk -F', ' 'BEGIN{s=0}{s+=$2}END{print s}')
  fm=${fm:-0}
  read -r used total <<<"$(nvidia-smi --id="$GPU" --query-gpu=memory.used,memory.total \
                            --format=csv,noheader,nounits | tr -d ',')"
  freemem=$(( total - used ))
  waited=$(( $(date +%s) - gate_start ))
  # Two independent conditions, because they guard different failures:
  #   free memory  -> OOM risk (our config peaks ~67GB)
  #   utilisation  -> compute contention, which costs step count
  if [ "$u" -le "$GATE_UTIL" ] && [ "$freemem" -ge "$GATE_FREE" ]; then
    [ "$waited" -gt 0 ] && FOREIGN_BASE=$fm
    GATE_MSG="CONTENTION_GATE: clear after ${waited}s (util ${u}%, free ${freemem}MB, foreign base ${fm}MB)"
    break
  fi
  # Before giving up, look for a device that IS clear. Which GPU runs a trial is an
  # implementation detail, not part of the experiment, so insisting on the assigned one
  # turns a transient conflict into a 30-minute stall while an idle GPU sits next to it.
  ALT=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
          --format=csv,noheader,nounits 2>/dev/null \
        | awk -F', ' -v need="$GATE_FREE" -v maxu="$GATE_UTIL" \
            '{ free=$3-$2; if (free>=need && $4<=maxu) { print $1; exit } }')
  if [ -n "${ALT:-}" ] && [ "$ALT" != "$GPU" ]; then
    GPU="$ALT"
    export CUDA_VISIBLE_DEVICES="$GPU"
    export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/inductor_gpu${GPU}"
    FOREIGN_BASE=$(nvidia-smi --id="$GPU" --query-compute-apps=pid,used_memory \
                     --format=csv,noheader,nounits 2>/dev/null \
                   | awk -F', ' 'BEGIN{s=0}{s+=$2}END{print s+0}')
    GATE_MSG="CONTENTION_GATE: assigned GPU busy after ${waited}s; switched to GPU $ALT (foreign base ${FOREIGN_BASE}MB)"
    break
  fi
  if [ "$waited" -ge "$GATE_MAX" ]; then
    # Proceeding under contention wastes the trial: the audit rejects it anyway, eight
    # minutes later. Skip and let the harness retry on the next iteration.
    echo "CONTENTION_SKIP: no clear GPU after ${waited}s (assigned GPU $GPU at util ${u}%)" >> run.log
    echo "EXIT_CODE=99" >> run.log
    exit 99
  fi
  sleep 30
done

# Sample foreign occupancy FOR THE DURATION of the trial, not just before it.
# A start-only gate cannot catch a co-tenant that arrives mid-run, which is exactly what
# happened: two trials passed the gate and then landed at 1109 and 1115 steps against a
# clean 1314, because someone started computing after we did. Peak foreign memory during
# the run is recorded so the audit can reject the trial rather than score it.
( peak=0
  while :; do
    mine=$(pgrep -f "$WORKDIR" | tr '\n' '|'); mine="${mine%|}"
    tot=$(nvidia-smi --id="$GPU" --query-compute-apps=pid,used_memory \
            --format=csv,noheader,nounits 2>/dev/null \
          | awk -F', ' -v m="${mine:-none}" 'BEGIN{s=0} { p=$1; gsub(/ /,"",p);
              split(m,a,"|"); own=0; for(i in a) if(a[i]==p) own=1; if(!own) s+=$2 } END{print s}')
    [ -n "${tot:-}" ] && [ "$tot" -gt "$peak" ] && peak=$tot
    echo "$peak" > "$WORKDIR/.foreign_peak"
    if [ $(( peak - FOREIGN_BASE )) -gt "$GATE_GROWTH" ]; then
      echo "CONTENTION_ABORT: foreign memory grew ${FOREIGN_BASE}MB -> ${peak}MB mid-run (new tenant)" >> "$WORKDIR/run.log"
      pkill -f "$WORKDIR/.venv/bin/python" 2>/dev/null
      break
    fi
    sleep 20
  done ) &
SAMPLER=$!

: > run.log                     # truncate ONCE, so gate/sampler lines survive
[ -n "$GATE_MSG" ] && echo "$GATE_MSG" >> run.log
timeout --signal=KILL "$HARD_TIMEOUT" "$WORKDIR/.venv/bin/python" train.py >> run.log 2>&1
code=$?
kill "$SAMPLER" 2>/dev/null
fp=$(cat "$WORKDIR/.foreign_peak" 2>/dev/null || echo 0)
echo "RAN_ON_GPU=$GPU" >> run.log
echo "FOREIGN_PEAK_MB=$fp" >> run.log
echo "FOREIGN_GROWTH_MB=$(( fp - FOREIGN_BASE ))" >> run.log
echo "EXIT_CODE=$code" >> run.log
