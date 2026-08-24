#!/bin/bash
# supervisor.sh — keep the campaign off contended GPUs.
#
# Pauses the controller when a foreign job lands on our GPU, and relaunches it on a
# genuinely free GPU once one appears. "Genuinely free" means low memory AND low
# utilisation on two consecutive checks a minute apart, so we do not land on a GPU
# that is merely between kernels.
#
#   ./supervisor.sh                 # 5-minute checks, runs until killed
#   INTERVAL=120 ./supervisor.sh

set -uo pipefail

ROOT="${ROOT:?set ROOT=<remote fengheguai root>}"
CAMPAIGN="${CAMPAIGN:-$ROOT/campaigns/h200-claude}"
PIDFILE="$CAMPAIGN-controller.pid"
LOG="${LOG:-$ROOT/logs/supervisor.log}"
INTERVAL="${INTERVAL:-300}"
FREE_MEM="${FREE_MEM:-5000}"        # MiB below which a GPU counts as free
FREE_UTIL="${FREE_UTIL:-20}"        # percent below which a GPU counts as idle
FOREIGN_MEM="${FOREIGN_MEM:-10000}" # MiB of foreign use that counts as contention
MFU_PAUSE="${MFU_PAUSE:-30}"      # our own measured occupancy; below this the device is bad
BADGPU_TTL="${BADGPU_TTL:-900}"   # seconds to avoid a device after it starved a trial
BADGPU="$ROOT/logs/bad_gpus"
FOREIGN_PROC_MIB="${FOREIGN_PROC_MIB:-200}" # a foreign compute process above this counts as
# contention regardless of its memory. GPU 4 held a 1166MiB tenant burning ~20% of the SMs:
# invisible to the memory threshold, but it slowed our trials from 105ms to 145ms per step and
# the throughput guard aborted two runs (t0451, t0453) at ~7.2 steps/s. Memory is a proxy for
# occupancy and a bad one; a co-tenant is a co-tenant.
TAG="${TAG:-h200-claude}"           # marks a process as belonging to this campaign
PIN="${PIN:-}"                      # if set, only ever run on this GPU
# Empirically slow devices. t0408 and t0409 both ran on gpu 2 at ~7.1 steps/s against a
# campaign norm of 8.9-9.3, with no throttling reported, full clock and a cool die. Two
# trials at a 23% deficit is enough to stop spending measurements there: a slow trial that
# COMPLETES is voided by the step-count rule but still burns its coordinate permanently.
EXCLUDE="${EXCLUDE:-2}"
AGENT_ERROR_LIMIT="${AGENT_ERROR_LIMIT:-3}"  # consecutive agent errors before pausing

mkdir -p "$(dirname "$LOG")"
say() { echo "$(python3 -c 'import time,datetime as d; print(f"{d.datetime.fromtimestamp(time.time(),d.timezone.utc):%Y-%m-%dT%H:%M:%SZ}")') $*" | tee -a "$LOG"; }

running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; }

our_gpu() {
  running || return 1
  tr '\0' '\n' < "/proc/$(cat "$PIDFILE")/environ" 2>/dev/null |
    sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1
}

# MiB held on GPU $1 by processes that are not this campaign.
#
# Owner is the wrong discriminator here: several campaigns run under the same Unix
# account on this host, and one of them was found holding 77GB on our GPU. What
# identifies our work is the campaign path in the process command line.
foreign_on() {
  local idx="$1" uuid mib=0
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$idx" '$1==i{print $2}')
  [ -z "$uuid" ] && { echo 0; return; }
  while IFS=, read -r u pid used; do
    u="${u// /}"; pid="${pid// /}"; used="${used// /}"
    [ "$u" = "$uuid" ] || continue
    if tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "$TAG"; then continue; fi
    mib=$((mib + ${used%% *}))
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null)
  echo "$mib"
}

# Our own measured occupancy, from the newest trial log. This is the direct signal: it does
# not care whether a co-tenant is large or small, only whether we are actually getting the
# device. t0463 ran at 8.09 steps/s -- above the old step-rate floor -- at 18.1% MFU.
our_mfu() {
  local rl age
  rl=$(ls -t "$CAMPAIGN"/evidence/*/run.log 2>/dev/null | head -1)
  [ -z "$rl" ] && { echo ""; return; }
  age=$(( $(date +%s) - $(stat -c %Y "$rl" 2>/dev/null || echo 0) ))
  [ "$age" -gt 90 ] && { echo ""; return; }
  tr '\r' '\n' < "$rl" | sed -n 's/.*mfu: \([0-9.]*\)%.*/\1/p' | tail -1
}

mark_bad_gpu() { printf '%s %s\n' "$1" "$(date +%s)" >> "$BADGPU"; }

gpu_is_bad() {
  [ -f "$BADGPU" ] || return 1
  local now; now=$(date +%s)
  awk -v g="$1" -v now="$now" -v ttl="$BADGPU_TTL" '$1==g && (now-$2)<ttl {found=1} END{exit !found}' "$BADGPU"
}

# Count foreign compute processes on a GPU, ignoring trivially small ones. Memory-based
# contention detection misses a small-footprint, high-occupancy tenant entirely.
foreign_procs_on() {
  local idx="$1" uuid n=0
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$idx" '$1==i{print $2}')
  [ -z "$uuid" ] && { echo 0; return; }
  while IFS=, read -r u pid used; do
    u="${u// /}"; pid="${pid// /}"; used="${used// /}"
    [ "$u" = "$uuid" ] || continue
    if tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "$TAG"; then continue; fi
    [ "${used%% *}" -ge "$FOREIGN_PROC_MIB" ] && n=$((n + 1))
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null)
  echo "$n"
}

# First GPU that is quiet on two checks a minute apart.
find_free() {
  local a b
  a=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
      awk -F', ' -v m="$FREE_MEM" -v u="$FREE_UTIL" -v p="$PIN" -v x=",$EXCLUDE," '$2<m && $3<u && (p=="" || $1==p) && index(x, ","$1",")==0 {print $1}' | tr '\n' ' ')
  # Drop candidates that already host a foreign compute process. Without this the selector
  # picks a device the pause check will reject one cycle later, and the supervisor thrashes:
  # resume gpu 3, detect the tenant, pause, resume gpu 3 again.
  # Prefer devices with no foreign compute process; fall back to the rest rather than
  # returning nothing, since on a fully occupied box "nothing" means the campaign stops.
  a=$(for g in $a; do gpu_is_bad "$g" || printf '%s ' "$g"; done)
  clean=$(for g in $a; do [ "$(foreign_procs_on "$g")" -eq 0 ] && printf '%s ' "$g"; done)
  [ -n "$clean" ] && a="$clean"
  [ -z "$a" ] && return 1
  sleep 60
  b=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
      awk -F', ' -v m="$FREE_MEM" -v u="$FREE_UTIL" -v p="$PIN" -v x=",$EXCLUDE," '$2<m && $3<u && (p=="" || $1==p) && index(x, ","$1",")==0 {print $1}' | tr '\n' ' ')
  cleanb=$(for g in $b; do [ "$(foreign_procs_on "$g")" -eq 0 ] && printf '%s ' "$g"; done)
  [ -n "$cleanb" ] && b="$cleanb"
  for g in $a; do case " $b " in *" $g "*) echo "$g"; return 0;; esac; done
  return 1
}

start_on() {
  local gpu="$1"
  cd "$ROOT" || return 1
  setsid nohup env PYTHONPATH=. PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$gpu" \
    python3 -m fengheguai run --campaign "$CAMPAIGN" --forever \
    < /dev/null >> "$CAMPAIGN-controller.log" 2>&1 &
  echo $! > "$PIDFILE"
  sleep 5
  running && say "resumed on gpu $gpu (pid $(cat "$PIDFILE"))" || say "resume on gpu $gpu FAILED"
}

# Killing the controller leaves its training subprocess running: it holds ~38 GB, its result
# can never be recorded because nothing is listening, and it counts against the free-GPU
# check that decides when we may resume -- so the campaign can block itself. Match on the
# campaign's own node path, never on the Unix owner: several campaigns share this account.
reap_orphan_trials() {
  local nodes="$CAMPAIGN/nodes/" pid cmd n=0
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null) || continue
    case "$cmd" in
      *"$nodes"*) kill "$pid" 2>/dev/null; n=$((n+1)) ;;
    esac
  done
  [ "$n" -gt 0 ] && { sleep 3; say "reaped $n orphaned trial process(es)"; }
  return 0
}

stop_controller() {
  local pid; pid=$(cat "$PIDFILE" 2>/dev/null) || return 0
  kill "$pid" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
  pkill -u "$USER" -f "rendezvous_agent.py" 2>/dev/null
  reap_orphan_trials
  say "paused (pid $pid stopped)"
}

# Consecutive trailing agent_error records. When the operator agent is unreachable
# every rendezvous times out and the controller immediately opens another, which once
# burned 224 slots in a row. Pausing stops the bleed; resuming is deliberate.
agent_error_streak() {
  python3 - "$CAMPAIGN/ledger.jsonl" <<'PYEOF' 2>/dev/null || echo 0
import json, sys
try:
    rows = [json.loads(l) for l in open(sys.argv[1])]
except Exception:
    print(0); raise SystemExit
n = 0
for e in reversed(rows):
    if e.get("kind") != "trial_completed":
        continue
    if e.get("payload", {}).get("status") == "agent_error":
        n += 1
    else:
        break
print(n)
PYEOF
}

base_streak=$(agent_error_streak); base_streak=${base_streak:-0}
say "supervisor starting: ignoring $base_streak pre-existing agent errors; interval ${INTERVAL}s, free<${FREE_MEM}MiB & <${FREE_UTIL}%, foreign>${FOREIGN_MEM}MiB"
trap 'say "supervisor exiting"; exit 0' INT TERM

while :; do
  if running; then
    st=$(agent_error_streak)
    st=${st:-0}
    # Only new failures count. The ledger carries 224 agent errors from an earlier
    # connectivity outage; a breaker that trips on history would pause every restart.
    [ "$st" -lt "$base_streak" ] && base_streak="$st"
    if [ $((st - base_streak)) -ge "$AGENT_ERROR_LIMIT" ]; then
      say "agent unreachable: $((st - base_streak)) new agent_error trials — pausing (resume is manual)"
      stop_controller
      touch "$ROOT/logs/PAUSED_AGENT_UNREACHABLE"
      sleep "$INTERVAL"; continue
    fi
    gpu=$(our_gpu)
    if [ -n "$gpu" ]; then
      f=$(foreign_on "$gpu"); fp=$(foreign_procs_on "$gpu")
      # Pause only on HEAVY contention. Every GPU on this box now hosts a tenant, so a
      # zero-tolerance rule stalls the campaign outright -- it did, for 9 minutes, with no
      # device left to move to. Measurement integrity no longer depends on this check: the
      # throughput guard aborts any trial running below its MFU floor whatever the cause, so
      # a contaminated run cannot be scored. This check now exists to avoid wasting time on a
      # device that is obviously hopeless, not to guarantee correctness.
      if [ "$f" -gt "$FOREIGN_MEM" ]; then
        say "contention on gpu $gpu: ${f}MiB foreign, ${fp} foreign process(es) — pausing"
        stop_controller
      else
        # Low MFU alone is NOT starvation -- a narrow model has intrinsically worse occupancy.
        # 21 layers at width 384 ran a healthy 32.9% and this check paused it anyway, then
        # blacklisted a genuinely clean gpu 5 for 900s. The guard learned this an hour earlier;
        # I had duplicated the heuristic here and fixed only one copy. Require a co-tenant.
        m=$(our_mfu)
        if [ -n "$m" ] && [ "${fp:-0}" -gt 0 ] && awk -v m="$m" -v t="$MFU_PAUSE" 'BEGIN{exit !(m<t)}'; then
          say "starved on gpu $gpu: our MFU ${m}% below ${MFU_PAUSE}% WITH ${fp} co-tenant(s) — pausing and avoiding it for ${BADGPU_TTL}s"
          mark_bad_gpu "$gpu"
          stop_controller
        fi
      fi
    fi
  else
    if [ -f "$ROOT/logs/PAUSED_AGENT_UNREACHABLE" ]; then
      sleep "$INTERVAL"; continue
    fi
    if gpu=$(find_free); then
      say "gpu $gpu free on two checks — resuming"
      start_on "$gpu"
    fi
  fi
  sleep "$INTERVAL"
done
