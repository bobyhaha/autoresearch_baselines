"""Resolve the campaign host without hardcoding it.

Precedence: environment variables, then ssh_target.local beside this file. Raises with a
clear message if neither supplies a host, rather than silently trying to reach an empty one.
"""
import os, pathlib

def _load_local():
    f = pathlib.Path(__file__).resolve().parent / "ssh_target.local"
    if not f.exists():
        return {}
    out = {}
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = os.path.expandvars(v.strip())
    return out

_local = _load_local()

def _get(name, default=None):
    return os.environ.get(name) or _local.get(name) or default

USER = _get("FENGHEGUAI_SSH_USER")
HOST = _get("FENGHEGUAI_SSH_HOST")
PORT = _get("FENGHEGUAI_SSH_PORT", "22")
KEY  = _get("FENGHEGUAI_SSH_KEY", os.path.expanduser("~/.ssh/id_ed25519"))
if not USER or not HOST:
    raise SystemExit("set FENGHEGUAI_SSH_USER and FENGHEGUAI_SSH_HOST, or create ssh_target.local")

TARGET = f"{USER}@{HOST}"
SSH_CMD = ["ssh", "-i", KEY, "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=no",
           "-o", "ConnectTimeout=15", "-p", PORT, TARGET]
