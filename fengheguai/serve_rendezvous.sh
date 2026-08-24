#!/bin/bash
# Serve one rendezvous atomically: apply the edit, verify it changed the file, and only
# then publish the result. The result file is what unblocks the controller, so publishing
# it after a failed edit hands the engine an unmodified train.py -- which is how t0392 was
# lost to a duplicate-source rejection.
#
# Usage: serve_rendezvous.sh <node_id> <edit_script.py> <result.json> [edit args...]
set -euo pipefail
NODE="$1"; EDIT="$2"; RESULT="$3"; shift 3
. "$(cd "$(dirname "$0")" && pwd)/ssh_target.sh"
SSH=(ssh "${FENGHEGUAI_SSH_ARGS[@]}")
REMOTE="/data3/zhubaiyu/fengheguai/campaigns/h200-claude/nodes/$NODE"

before=$("${SSH[@]}" "sha256sum $REMOTE/train.py | cut -d' ' -f1")
"${SSH[@]}" "python3 - $NODE $*" < "$EDIT"
after=$("${SSH[@]}" "sha256sum $REMOTE/train.py | cut -d' ' -f1")

if [ "$before" = "$after" ]; then
  echo "ABORT: train.py unchanged after the edit; result NOT written" >&2
  exit 1
fi
"${SSH[@]}" "python3 -c 'import ast,sys; ast.parse(open(sys.argv[1]).read())' $REMOTE/train.py"
"${SSH[@]}" "cat > $REMOTE/.fengheguai-agent-result.json" < "$RESULT"
echo "$NODE served: train.py $before -> $after, result published"
