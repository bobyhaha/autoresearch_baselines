#!/bin/bash
# Abort a trial whose step rate has collapsed, so the coordinate survives.
#
# The supervisor pauses on foreign GPU memory, which is a proxy. t0408 showed the proxy can
# read clean while throughput collapses anyway: 6.82 steps/s against a campaign range of
# 8.87-9.33, with foreign memory at 1166 MiB, our clock at a full 1980 MHz and the host at
# load 40 on 192 cores. Cause unidentified. It completed, and completing is what destroyed
# the coordinate -- the audit refuses duplicate sources, so a contaminated *completion*
# closes that question forever while an abort merely costs a slot.
#
# So this watches the actual measured quantity rather than a proxy. A trial below the floor
# is measuring the environment, not the change, and is better killed than recorded.
#
# Env: CAMPAIGN (required), FLOOR (steps/s, default 8.0), GRACE (steps before judging, 300)
set -uo pipefail
CAMPAIGN="${CAMPAIGN:?set CAMPAIGN}"
FLOOR="${FLOOR:-8.0}"
MFU_FLOOR="${MFU_FLOOR:-34.0}"
# ...but low MFU ALONE is not contention. A narrow model has intrinsically worse occupancy:
# 21 layers at width 384 ran a healthy 32.9% because small GEMMs use the tensor cores badly,
# and the guard aborted it. Nor does a within-run DROP work -- t0451 and t0453 were contended
# from step one and show 0.0% and 0.3% decline. The discriminator is low MFU *together with a
# co-tenant on our device*: 28.8% beside a foreign process is contention, 32.9% alone is a
# small model. Measured populations: contended 28.8/29.1, healthy-narrow 32.6, healthy 39-41.   # the real contention signal.
# steps/s alone cannot tell "the GPU is shared" from "this model does more work per step".
# t0457 ran depth 13 at 6.56 steps/s -- below the old floor, aborted -- while reporting MFU
# 41.4%, HIGHER than the champion's healthy 39.8%. It was a legitimately slower model running
# at full efficiency, and the guard burned its coordinate. Genuinely contended runs look
# different: t0451 and t0453 sat at 28.8% MFU. Occupancy separates them; step rate does not.
# Healthy trials measured across t0393/t0395/t0396: 12.50 steps/s at step 50, 10.00 at 100,
# 9.52 at 200, never below 9.30 thereafter. A slow trial (t0411) sat at 4.55 and 4.00 at the
# same points. So judging from step 100 against a floor of 8.0 leaves a 20% margin on healthy
# runs and catches a bad one five minutes earlier than GRACE=300 did.
GRACE="${GRACE:-100}"
INTERVAL="${INTERVAL:-30}"
LOG="${LOG:-$(dirname "$CAMPAIGN")/logs/throughput_guard.log}"
# Which GPU are we on, and is anyone else on it? Read from the controller's own environment
# so this cannot drift from what the campaign is actually using.
our_gpu() {
  local c
  c=$(head -1 "$(dirname "$CAMPAIGN")/h200-claude-controller.pid" 2>/dev/null)
  [ -n "$c" ] && [ -d "/proc/$c" ] || { echo ""; return; }
  tr '\0' '\n' < "/proc/$c/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1
}

cotenant_on() {
  local idx="$1" uuid n=0
  [ -z "$idx" ] && { echo 0; return; }
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$idx" '$1==i{print $2}')
  [ -z "$uuid" ] && { echo 0; return; }
  while IFS=, read -r u pid used; do
    u="${u// /}"; pid="${pid// /}"
    [ "$u" = "$uuid" ] || continue
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "h200-claude" && continue
    n=$((n + 1))
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null)
  echo "$n"
}

say() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG"; }

say "throughput guard starting: MFU floor ${MFU_FLOOR}%, grace ${GRACE} steps"
while :; do
  # newest evidence run.log = the trial in flight
  rl=$(ls -t "$CAMPAIGN"/evidence/*/run.log 2>/dev/null | head -1)
  if [ -n "$rl" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$rl" 2>/dev/null || echo 0) ))
    if [ "$age" -gt "${FRESH:-90}" ]; then sleep "$INTERVAL"; continue; fi
    verdict=$(CAMPAIGN="$CAMPAIGN" FLOOR="$FLOOR" MFU_FLOOR="$MFU_FLOOR" GRACE="$GRACE" python3 - "$rl" <<'PYEOF'
import os, re, sys
txt = open(sys.argv[1], errors="replace").read()
if "training_seconds:" in txt:            # finished; nothing to guard
    raise SystemExit
m = re.findall(r"step (\d+) \([^)]*\).*?remaining: (\d+)s", txt)   # log is \r-delimited
if not m:
    raise SystemExit
step, rem = int(m[-1][0]), int(m[-1][1])
el = 300 - rem
if el <= 5 or step <= int(os.environ["GRACE"]):
    raise SystemExit
rate = step / el
# Occupancy is the contention signal, not step rate. A model that legitimately does more work
# per step is slow AND efficient; a contended one is slow AND inefficient.
# MFU is an INSTANTANEOUS per-step figure and it is noisy; the step rate it replaced was
# cumulative and therefore smooth. Judging on the last reading alone aborted t0473 on a single
# transient dip to 31.3% while the trial was averaging 9.63 steps/s at 43.5% MFU -- faster than
# the champion. Take the median of the recent window instead, so a spike cannot kill a healthy
# run but a genuine collapse (t0451/t0453 sat at ~29% for hundreds of steps) still does.
mfu = [float(x) for x in re.findall(r"mfu: ([0-9.]+)%", txt)]
if len(mfu) < 20:
    raise SystemExit                       # too few readings to judge
window = sorted(mfu[-20:])
mfu_med = window[len(window)//2]
if mfu_med < float(os.environ["MFU_FLOOR"]):
    print(f"{step} {rate:.2f} {mfu_med:.1f}")
PYEOF
)
    if [ -n "$verdict" ]; then
      set -- $verdict
      g=$(our_gpu); ct=$(cotenant_on "$g")
      if [ "${ct:-0}" -eq 0 ]; then
        say "low MFU $3% on gpu ${g:-?} but no co-tenant — treating as an intrinsically slower model, not aborting"
        sleep "$INTERVAL"; continue
      fi
      for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        case "$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)" in
          *"$CAMPAIGN/nodes/"*) kill "$pid" 2>/dev/null
            say "aborted trial at step $1: median MFU $3% below $MFU_FLOOR% WITH $ct co-tenant(s) on gpu $g (${2} steps/s) (killed $pid)" ;;
        esac
      done
    fi
  fi
  sleep "$INTERVAL"
done
