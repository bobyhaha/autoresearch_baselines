#!/bin/bash
# Resolve the campaign host without hardcoding it. Precedence: environment, then
# ssh_target.local beside this script. Exits with a clear message if neither is set,
# rather than silently trying to reach an empty host.
_d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$_d/ssh_target.local" ] && . "$_d/ssh_target.local"
FENGHEGUAI_SSH_USER="${FENGHEGUAI_SSH_USER:?set FENGHEGUAI_SSH_USER or create ssh_target.local}"
FENGHEGUAI_SSH_HOST="${FENGHEGUAI_SSH_HOST:?set FENGHEGUAI_SSH_HOST or create ssh_target.local}"
FENGHEGUAI_SSH_PORT="${FENGHEGUAI_SSH_PORT:-22}"
FENGHEGUAI_SSH_KEY="${FENGHEGUAI_SSH_KEY:-$HOME/.ssh/id_ed25519}"
FENGHEGUAI_SSH_ARGS=(-i "$FENGHEGUAI_SSH_KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no \
                     -o ConnectTimeout=15 -p "$FENGHEGUAI_SSH_PORT" \
                     "$FENGHEGUAI_SSH_USER@$FENGHEGUAI_SSH_HOST")
