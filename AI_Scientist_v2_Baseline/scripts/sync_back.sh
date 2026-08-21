#!/bin/bash
# Pull campaign state (journal, results, logs) from the remote box into the local mirror.
# Trial workspaces are left remote; only the run.log of each is worth mirroring.
set -euo pipefail
REMOTE="${OPHIS_REMOTE_HOST:?set OPHIS_REMOTE_HOST=user@host}"
REMOTE_PORT="${OPHIS_REMOTE_PORT:-22}"
SSH="ssh -i $HOME/.ssh/id_ed25519 -o IdentitiesOnly=yes -p $REMOTE_PORT"
LOCAL="$(cd "$(dirname "$0")/.." && pwd)"

rsync -az -e "$SSH" "$REMOTE:~/ai_scientist_v2_baseline/campaign/" "$LOCAL/campaign/"
echo "synced campaign state -> $LOCAL/campaign/"
if [ -f "$LOCAL/campaign/results.tsv" ]; then column -t -s "$(printf "\t")" "$LOCAL/campaign/results.tsv"; fi
