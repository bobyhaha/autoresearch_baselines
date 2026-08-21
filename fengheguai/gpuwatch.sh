#!/bin/bash
# gpuwatch.sh — sample GPU usage on a fixed interval for a fixed duration.
#
# Timestamps come from Python's time.time(), so every row carries an unambiguous
# epoch value that survives timezone and DST changes; the ISO column is derived
# from the same call rather than from a second clock read.
#
#   ./gpuwatch.sh                      # 30-minute samples for 24 hours
#   INTERVAL=600 DURATION=7200 ./gpuwatch.sh
#   OUT=/tmp/gpu.tsv ./gpuwatch.sh
#
# One TSV row per GPU per sample, plus a one-line summary per sample on stdout.

set -uo pipefail

INTERVAL="${INTERVAL:-1800}"          # seconds between samples
DURATION="${DURATION:-86400}"         # total seconds to run
CAMPAIGN="${CAMPAIGN:?set CAMPAIGN=<remote>/campaigns/h200-claude}"
PIDFILE="${PIDFILE:-${CAMPAIGN}-controller.pid}"
OUT="${OUT:?set OUT=<remote>/logs/gpu_usage.tsv}"
OURS="${OURS:-7}"                     # the GPU this campaign is pinned to

mkdir -p "$(dirname "$OUT")" || exit 1

# time.time() as the single clock: one call, both representations.
now() {
  python3 -c 'import time,datetime as d; t=time.time(); print(f"{t:.3f}\t{d.datetime.fromtimestamp(t,d.timezone.utc):%Y-%m-%dT%H:%M:%SZ}")'
}

if [ ! -s "$OUT" ]; then
  printf 'epoch\tiso_utc\telapsed_s\tgpu\tname\tutil_pct\tmem_used_mb\tmem_total_mb\tours\tctrl_alive\ttrials\tchampion\n' > "$OUT"
fi

read -r START _ < <(now)
finish() { echo "gpuwatch: stopping after $(printf '%.0f' "$elapsed")s, $samples samples -> $OUT"; exit 0; }
trap finish INT TERM

samples=0
elapsed=0

while :; do
  read -r EPOCH ISO < <(now)
  elapsed=$(python3 -c "print(f'{$EPOCH - $START:.0f}')")
  [ "$elapsed" -ge "$DURATION" ] && finish

  # Controller liveness, from our own pid file only — never a name match, which
  # would count other tenants' python processes as ours.
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
    CTRL=alive
  else
    CTRL=down
  fi

  TRIALS=$(( $(wc -l < "$CAMPAIGN/reports/results.tsv" 2>/dev/null || echo 1) - 1 ))
  CHAMP=$(grep -m1 '^- Best' "$CAMPAIGN/reports/STATUS.md" 2>/dev/null | tr -d '`' | awk '{print $NF}')
  CHAMP="${CHAMP:-?}"

  ours_util=0; ours_mem=0; busy=0
  while IFS=, read -r idx name util used total; do
    idx="${idx// /}"; util="${util// /}"; used="${used// /}"; total="${total// /}"
    name="$(echo "$name" | sed 's/^ *//;s/ *$//')"
    if [ "$idx" = "$OURS" ]; then mine=yes; ours_util="$util"; ours_mem="$used"; else mine=no; fi
    [ "$util" -ge 50 ] 2>/dev/null && busy=$((busy+1))
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$EPOCH" "$ISO" "$elapsed" "$idx" "$name" "$util" "$used" "$total" "$mine" "$CTRL" "$TRIALS" "$CHAMP" >> "$OUT"
  done < <(nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
             --format=csv,noheader,nounits 2>/dev/null)

  samples=$((samples+1))
  echo "$ISO  gpu${OURS} ${ours_util}% ${ours_mem}MiB | ${busy}/8 busy | ctrl ${CTRL} | ${TRIALS} trials | best ${CHAMP}"

  sleep "$INTERVAL"
done
