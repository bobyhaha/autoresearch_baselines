#!/bin/bash
# gpu_watch.sh — sample the pinned GPU on a fixed cadence for the length of the campaign.
#
# Timing uses python's time.time() rather than a bare `sleep 1800`. The difference
# matters over a 24-hour run: each iteration spends real time shelling out to
# nvidia-smi and reading the journal, so a plain sleep loop drifts later and later —
# roughly a full sample lost by the end. Here the next wake instant is computed from an
# absolute epoch deadline, so samples land on the intended cadence no matter how long
# the work in between takes.
#
# What it records, per sample: GPU utilisation and memory for the pinned device, whether
# *our own* trial process is alive (owner-filtered — a busy GPU on a shared box may be
# someone else's work), the campaign's current node and best val_bpb, and an explicit
# IDLE verdict when the GPU is quiet while the harness is supposed to be training.
#
#   ./gpu_watch.sh [--gpu 2] [--interval 1800] [--duration 86400] [--log PATH]

set -uo pipefail
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"

BASE="$HOME/ai_scientist_v2_baseline"
PY="$BASE/task/.venv/bin/python"
GPU=2
INTERVAL=1800          # 30 minutes
DURATION=86400         # 24 hours
LOG="$BASE/campaign/gpu_usage.log"

while [ $# -gt 0 ]; do
  case "$1" in
    --gpu)      GPU="$2";      shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --log)      LOG="$2";      shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

now() { "$PY" -c 'import time; print(repr(time.time()))'; }

START=$(now)
END=$("$PY" -c "print($START + $DURATION)")

mkdir -p "$(dirname "$LOG")"
if [ ! -s "$LOG" ]; then
  echo "epoch,iso,elapsed_h,gpu,util_pct,mem_used_mb,mem_total_mb,our_trial_procs,foreign_mem_mb,verdict,journal_nodes,best_val_bpb,current_node" >> "$LOG"
fi

echo "gpu_watch: GPU $GPU, every ${INTERVAL}s for ${DURATION}s, logging to $LOG" >&2

while :; do
  T=$(now)
  # Absolute deadline check — never relies on accumulated sleeps.
  if [ "$("$PY" -c "print(1 if $T >= $END else 0)")" = "1" ]; then
    echo "gpu_watch: 24h window complete" >&2
    break
  fi

  ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  ELAPSED=$("$PY" -c "print(f'{($T - $START)/3600:.2f}')")

  READING=$(nvidia-smi --id="$GPU" --query-gpu=utilization.gpu,memory.used,memory.total \
              --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  # Memory on this device held by processes that are not ours.
  FOREIGN=$("$PY" - "$BASE" "$GPU" << 'FEOF' 2>/dev/null || echo 0
import subprocess, sys
base, gpu = sys.argv[1], sys.argv[2]
try:
    mine = set(subprocess.run(["pgrep","-f",base+"/trials/"],capture_output=True,text=True).stdout.split())
    rows = subprocess.run(["nvidia-smi","--id="+gpu,"--query-compute-apps=pid,used_memory",
                           "--format=csv,noheader,nounits"],capture_output=True,text=True).stdout
    print(sum(int(m) for p,m in (r.split(",") for r in rows.strip().splitlines()) if p.strip() not in mine))
except Exception:
    print(0)
FEOF
)
  UTIL=$(echo "$READING" | cut -d, -f1)
  MEM=$(echo   "$READING" | cut -d, -f2)
  MEMTOT=$(echo "$READING" | cut -d, -f3)
  [ -z "$UTIL" ] && { UTIL=-1; MEM=-1; MEMTOT=-1; }

  # Owner-filtered: only processes running out of OUR trials directory count as ours.
  # pgrep -fc prints "0" AND exits 1 when nothing matches, so a `|| echo 0` fallback
  # appends a SECOND zero and injects a newline into the CSV row. Take the output as-is.
  OURS=$(pgrep -fc "$BASE/trials/" 2>/dev/null | tr -dc '0-9')
  OURS=${OURS:-0}

  STATE=$("$PY" - "$BASE" << 'PYEOF' 2>/dev/null || echo "0,,"
import json, sys
from pathlib import Path
base = Path(sys.argv[1])
try:
    nodes = json.loads((base / "campaign" / "journal.json").read_text())["nodes"]
except Exception:
    print("0,,"); raise SystemExit
scored = [n for n in nodes if (n.get("metric") or {}).get("value") is not None]
best = min((n["metric"]["value"] for n in scored), default="")
try:
    cur = json.loads((base / "campaign" / "status.json").read_text()).get("last_node", "")
except Exception:
    cur = ""
print(f"{len(nodes)},{best},{cur}")
PYEOF
)
  NODES=$(echo "$STATE" | cut -d, -f1)
  BEST=$(echo  "$STATE" | cut -d, -f2)
  CUR=$(echo   "$STATE" | cut -d, -f3)

  # A quiet GPU with no trial of ours running means budget is being burned for nothing —
  # usually the harness blocked waiting for code. That is the signal worth surfacing.
  if [ "${OURS:-0}" -gt 0 ] && [ "${UTIL:-0}" -ge 50 ]; then VERDICT=TRAINING
  elif [ "${OURS:-0}" -gt 0 ];                          then VERDICT=STARTUP_OR_EVAL
  else                                                       VERDICT=IDLE
  fi

  echo "$T,$ISO,$ELAPSED,$GPU,$UTIL,$MEM,$MEMTOT,$OURS,${FOREIGN:-0},$VERDICT,$NODES,$BEST,$CUR" >> "$LOG"
  echo "[gpu_watch ${ELAPSED}h] gpu$GPU util=${UTIL}% mem=${MEM}MiB ours=${OURS} foreign=${FOREIGN:-0}MiB $VERDICT nodes=$NODES best=$BEST" >&2

  # Drift-corrected: next wake is an absolute instant, not "now plus interval".
  NEXT=$("$PY" -c "print($T + $INTERVAL)")
  SLEEP=$("$PY" -c "import time; print(max(1.0, $NEXT - time.time()))")
  sleep "$SLEEP"
done
