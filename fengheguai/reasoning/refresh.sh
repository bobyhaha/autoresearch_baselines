#!/bin/bash
# Sync the ledger and regenerate the 30-minute reasoning windows.
# Emits one line per run so the caller can see it happened.
cd /Users/baiyu/Desktop/OPHIS/fengheguai || exit 1
REMOTE_HOST="${OPHIS_REMOTE_HOST:?set OPHIS_REMOTE_HOST=user@host}"
REMOTE_PORT="${OPHIS_REMOTE_PORT:-22}"
REMOTE_ROOT="${OPHIS_REMOTE_ROOT:?set OPHIS_REMOTE_ROOT=/path/to/fengheguai}"
SSH_OPTS=(-i "$HOME/.ssh/id_ed25519" -o IdentitiesOnly=yes -o ConnectTimeout=20
          -o ControlMaster=auto -o ControlPath=/tmp/fgh2-%r@%h:%p -o ControlPersist=600)
while true; do
  if scp -q "${SSH_OPTS[@]}" -P "$REMOTE_PORT" \
      "$REMOTE_HOST:$REMOTE_ROOT/campaigns/h200-claude/ledger.jsonl" \
      campaigns/h200-claude-ledger.jsonl 2>/dev/null; then
    out=$(python3 reasoning/build_log.py 2>&1 | tail -1)
    n=$(python3 -c "
import json
c=[json.loads(l)['payload'] for l in open('campaigns/h200-claude-ledger.jsonl')]
c=[p for p in c if p.get('trial_id') and p.get('status')]
best=min([p['metric'] for p in c if p.get('metric')], default=None)
print(f\"{len(c)} trials, best {best:.6f}\")" 2>/dev/null)
    echo "$(date -u +%H:%MZ) reasoning log refreshed — $out — $n"
  else
    echo "$(date -u +%H:%MZ) reasoning refresh skipped — ledger sync failed"
  fi
  sleep 1800
done
