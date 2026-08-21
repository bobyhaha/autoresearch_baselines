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
TAG="${TAG:-h200-claude}"           # marks a process as belonging to this campaign

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

# First GPU that is quiet on two checks a minute apart.
find_free() {
  local a b
  a=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
      awk -F', ' -v m="$FREE_MEM" -v u="$FREE_UTIL" '$2<m && $3<u {print $1}' | tr '\n' ' ')
  [ -z "$a" ] && return 1
  sleep 60
  b=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
      awk -F', ' -v m="$FREE_MEM" -v u="$FREE_UTIL" '$2<m && $3<u {print $1}' | tr '\n' ' ')
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

stop_controller() {
  local pid; pid=$(cat "$PIDFILE" 2>/dev/null) || return 0
  kill "$pid" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
  pkill -u "$USER" -f "rendezvous_agent.py" 2>/dev/null
  say "paused (pid $pid stopped)"
}

say "supervisor starting: interval ${INTERVAL}s, free<${FREE_MEM}MiB & <${FREE_UTIL}%, foreign>${FOREIGN_MEM}MiB"
trap 'say "supervisor exiting"; exit 0' INT TERM

while :; do
  if running; then
    gpu=$(our_gpu)
    if [ -n "$gpu" ]; then
      f=$(foreign_on "$gpu")
      if [ "$f" -gt "$FOREIGN_MEM" ]; then
        say "contention on gpu $gpu: ${f}MiB foreign — pausing"
        stop_controller
      fi
    fi
  else
    if gpu=$(find_free); then
      say "gpu $gpu free on two checks — resuming"
      start_on "$gpu"
    fi
  fi
  sleep "$INTERVAL"
done
