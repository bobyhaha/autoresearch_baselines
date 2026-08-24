#!/bin/bash
# v2: also record CPU pressure, which v1 missed entirely.
#
# v1 watched GPU compute apps only. t0363 lost 75 steps (23 sd of the 0.11% machine-noise
# floor) with an init-only change while the GPU tenant sat idle -- the cost was other
# tenants' CPU jobs starving our Python data path. GPU-only monitoring cannot see that.
#
# Env: GPU, CTRL_PID, OUT (all required), INTERVAL (default 20)
set -u
: "${OUT:?set OUT}"
# GPU is resolved per-sample too, not fixed at launch. The supervisor is unpinned and
# relaunches the controller on whichever device frees first, so a watcher tied to one GPU
# ends up reporting a device we left -- quiet, and meaningless. Empty GPU means "no
# controller running", which is a state worth recording rather than an error.
# Resolve the controller from its pidfile on every sample rather than pinning it at launch:
# the supervisor restarts the controller on each pause/resume, and a stale CTRL_PID makes
# this watcher report our own trial as foreign load.
CTRL_PIDFILE="${CTRL_PIDFILE:-/data3/zhubaiyu/fengheguai/campaigns/h200-claude-controller.pid}"
current_ctrl() { head -1 "$CTRL_PIDFILE" 2>/dev/null; }
INTERVAL="${INTERVAL:-20}"
ME=$(id -un)

descends_from_ctrl() {
  local pid="$1" hop=0
  while [ -n "$pid" ] && [ "$pid" != "0" ] && [ "$pid" != "1" ] && [ $hop -lt 8 ]; do
    [ "$pid" = "$CTRL" ] && return 0
    pid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
    hop=$((hop+1))
  done
  return 1
}

[ -s "$OUT" ] || printf 'epoch\tiso\tgpu\tutil\tmem_used\tours_mib\tforeign_mib\tload1\tcpu_ours\tcpu_foreign\tsm_mhz\tpower_w\ttemp_c\tbox_power_w\ttop_foreign\n' > "$OUT"

while :; do
  read -r EPOCH ISO <<< "$(python3 -c 'import time,datetime as d; t=time.time(); print(f"{t:.3f}", d.datetime.fromtimestamp(t,d.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))')"
  UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU")
  MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
  LOAD1=$(cut -d' ' -f1 /proc/loadavg)
  # This box power-caps: a neighbour was seen at 693W of a 700W limit, downclocked to
  # 1650MHz against a 1980 max, while our throughput varies ~4% between trials with no
  # code reason. Record clocks rather than speculating about them afterwards.
  read -r SM PW TC <<< "$(nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader,nounits -i "$GPU" | tr -d ',')"
  BOXPW=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | awk '{s+=$1} END {printf "%.0f", s}')

  CTRL=$(current_ctrl)
  GPU=$( [ -n "$CTRL" ] && tr '\0' '\n' < "/proc/$CTRL/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' )
  if [ -z "$GPU" ]; then sleep "$INTERVAL"; continue; fi
  ours=0; foreign=0
  while IFS=, read -r pid mib; do
    pid=$(echo "$pid" | tr -d ' '); mib=$(echo "$mib" | tr -d ' MiB')
    [ -z "$pid" ] && continue
    if descends_from_ctrl "$pid"; then ours=$((ours + mib)); else foreign=$((foreign + mib)); fi
  done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$GPU")

  # CPU pressure split by account: ours vs everyone else, plus the single worst offender
  read -r CPU_OURS CPU_FOREIGN TOP <<< "$(ps -eo pcpu,user,comm --no-headers | awk -v me="$ME" '
    { if ($2 == me) o += $1; else { f += $1; if ($1 > mx) { mx = $1; who = $2 ":" $3 } } }
    END { printf "%.0f %.0f %s", o, f, (who == "" ? "-" : who) }')"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$EPOCH" "$ISO" "$GPU" "$UTIL" "$MEM" "$ours" "$foreign" "$LOAD1" "$CPU_OURS" "$CPU_FOREIGN" \
    "$SM" "$PW" "$TC" "$BOXPW" "$TOP" >> "$OUT"
  sleep "$INTERVAL"
done
