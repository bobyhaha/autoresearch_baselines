#!/bin/bash
# Sample foreign compute tenants on our pinned GPU, at a resolution fine enough to
# annotate a 300-second trial. The supervisor's pause rule keys on memory (>10 GiB),
# which a compute-heavy but memory-light job slips under while still taking SMs. This
# only records; it never pauses. Ownership is decided by walking the parent chain to our
# controller pid, never by Unix owner -- several campaigns share this account.
#
# Env: GPU (required), CTRL_PID (required), OUT (required), INTERVAL (default 20)
set -u
: "${GPU:?set GPU}"; : "${CTRL_PID:?set CTRL_PID}"; : "${OUT:?set OUT}"
INTERVAL="${INTERVAL:-20}"

descends_from_ctrl() {
  local pid="$1" hop=0
  while [ -n "$pid" ] && [ "$pid" != "0" ] && [ "$pid" != "1" ] && [ $hop -lt 8 ]; do
    [ "$pid" = "$CTRL_PID" ] && return 0
    pid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
    hop=$((hop+1))
  done
  return 1
}

[ -s "$OUT" ] || printf 'epoch\tiso\tgpu\tutil\tmem_used\tours_mib\tforeign_mib\tforeign_n\tforeign_cmd\n' > "$OUT"

while :; do
  read -r EPOCH ISO <<< "$(python3 -c 'import time,datetime as d; t=time.time(); print(f"{t:.3f}", d.datetime.fromtimestamp(t,d.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))')"
  UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU")
  MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
  ours=0; foreign=0; fn=0; fcmd="-"
  while IFS=, read -r pid mib; do
    pid=$(echo "$pid" | tr -d ' '); mib=$(echo "$mib" | tr -d ' MiB')
    [ -z "$pid" ] && continue
    if descends_from_ctrl "$pid"; then
      ours=$((ours + mib))
    else
      foreign=$((foreign + mib)); fn=$((fn+1))
      [ "$fcmd" = "-" ] && fcmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-70)
    fi
  done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$GPU")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$EPOCH" "$ISO" "$GPU" "$UTIL" "$MEM" "$ours" "$foreign" "$fn" "$fcmd" >> "$OUT"
  sleep "$INTERVAL"
done
