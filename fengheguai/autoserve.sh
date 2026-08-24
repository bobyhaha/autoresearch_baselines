#!/bin/bash
# Deliver pre-authored bets to the rendezvous within its 840s deadline.
#
# I remain the research agent: every edit script and every hypothesis in queue/ is written
# by me. This only removes the requirement that I be *present* at the moment a request
# opens, which is what cost t0423 and t0425-t0427 to agent_error.
#
# A queue entry is a directory queue/NNN-name/ containing:
#   edit.py     the train.py edit (receives <node_id> plus any words in args)
#   result.json the four-field AgentResult
#   args        optional, one line, extra argv for edit.py
#
# Entries are served in lexical order and moved to queue/served/ on success. An empty
# queue serves nothing and says so -- it never invents a hypothesis.

P="$(cd "$(dirname "$0")" && pwd)"
Q="$P/queue"; LOG="$P/logs/autoserve.log"; mkdir -p "$P/logs"
R=/data3/zhubaiyu/fengheguai
. "$P/ssh_target.sh"
SSH=(ssh "${FENGHEGUAI_SSH_ARGS[@]}")
INTERVAL="${INTERVAL:-20}"
say(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

last=""; ticks=0; last_unreach=0
say "autoserve starting (interval ${INTERVAL}s, queue $(ls -d "$Q"/[0-9]* 2>/dev/null | wc -l | tr -d ' ') entries)"
while :; do
  # Distinguish "no pending request" from "cannot reach the box". Both leave $pend empty,
  # and treating them the same made a 27-minute SSH outage invisible: t0470 opened and expired
  # inside it with no log line, and the courier looked idle rather than blind.
  pend=$("${SSH[@]}" "cat $R/campaigns/h200-claude-rendezvous/PENDING.json 2>/dev/null" 2>/dev/null)
  ssh_rc=$?
  # 255 is ssh's own failure code. A plain 1 is the remote `cat` failing because there is no
  # pending request, which is the normal idle case -- conflating them turned every quiet poll
  # into a false outage report the moment I deployed this.
  if [ "$ssh_rc" -eq 255 ]; then
    now=$(date +%s)
    if [ $(( now - ${last_unreach:-0} )) -gt 300 ]; then
      say "SSH unreachable (rc=$ssh_rc) — cannot see pending requests; they will expire"
      last_unreach=$now
    fi
    sleep "$INTERVAL"; continue
  fi
  tid=$(printf '%s' "$pend" | sed -n 's/.*"trial_id": *"\([^"]*\)".*/\1/p')
  dl=$(printf '%s' "$pend" | sed -n 's/.*"deadline": *\([0-9]*\).*/\1/p')

  if [ -n "$tid" ] && [ "$tid" != "$last" ]; then
    now=$(date +%s)
    if [ -n "$dl" ] && [ "$now" -ge "$dl" ]; then
      say "$tid: deadline already passed ($((now-dl))s ago); not serving a stale request"
      last="$tid"
    else
      entry=$(ls -d "$Q"/[0-9]* 2>/dev/null | head -1)
      if [ -z "$entry" ]; then
        say "$tid: QUEUE EMPTY -- not serving. Refill $Q before the ${dl:+$((dl-now))}s deadline."
        last="$tid"
      else
        # Try entries in order until one applies. An edit's guards can legitimately refuse a
        # node -- a debug trial inherits its parent's source, so a bet that refuses to stack on
        # an existing change will correctly fail there. Giving up after the first refusal would
        # hand that request an agent_error, so a blocked entry steps aside and the next is tried.
        served=0
        for entry in $(ls -d "$Q"/[0-9]* 2>/dev/null); do
          eargs=""; [ -f "$entry/args" ] && eargs=$(cat "$entry/args")
          say "$tid: serving $(basename "$entry") ${eargs:+[$eargs]}"
          if "$P/serve_rendezvous.sh" "$tid" "$entry/edit.py" "$entry/result.json" $eargs >>"$LOG" 2>&1; then
            mv "$entry" "$Q/served/$(basename "$entry")-$tid"
            say "$tid: served ok, $(ls -d "$Q"/[0-9]* 2>/dev/null | wc -l | tr -d ' ') entries left"
            served=1; break
          fi
          # A refusal is node-specific, not permanent: a debug trial inherits its parent's
          # source, so a bet that will not stack there may apply perfectly to the next
          # champion-parented node. Skip it for this trial, but leave it in the queue.
          say "$tid: $(basename "$entry") does not apply to this node — skipping, trying next"
        done
        [ "$served" -eq 0 ] && say "$tid: NO QUEUE ENTRY APPLIED — refill $Q"
        last="$tid"
      fi
    fi
  fi
  # Recover bets whose trial died before producing a record. autoserve retires an entry on
  # successful *delivery*, but a trial can be killed by contention before it completes, and
  # the ledger then holds no source hash -- so the coordinate is still free and the bet is
  # recoverable rather than merely lost. Checked every ~5 minutes.
  ticks=$((ticks + 1))
  if [ $((ticks % 15)) -eq 0 ]; then
    r=$(python3 "$P/requeue_lost.py" 2>&1 | grep -c requeued)
    [ "${r:-0}" -gt 0 ] && say "reaper: recovered $r lost bet(s)"
  fi
  sleep "$INTERVAL"
done
