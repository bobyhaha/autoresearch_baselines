#!/usr/bin/env python3
"""Locate nodes by plan substring across the journal and the archive.

Written after repeatedly mangling inline `python -c` through ssh: nested quotes inside an
f-string inside a shell heredoc inside an ssh argument fail in a different way each time.
Shipping a file is the reliable path and takes the same effort once.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BASE = Path.home() / "ai_scientist_v2_baseline"
needle = " ".join(sys.argv[1:]) or "SEED-PAIRED"

for fname, label in (("journal.json", "journal"), ("env_failures.json", "archived")):
    path = BASE / "campaign" / fname
    if not path.exists():
        continue
    for n in json.loads(path.read_text())["nodes"]:
        plan = n.get("plan") or ""
        if needle.lower() not in plan.lower():
            continue
        m = (n.get("metric") or {}).get("value")
        s = n.get("summary") or {}
        val = ("%.6f" % m) if m else "BUGGY "
        steps = int(s.get("num_steps") or 0)
        foreign = int(s.get("foreign_peak_mb") or 0)
        print("  %-9s %s %s steps=%-5d foreign=%dMB" % (label, n["id"], val, steps, foreign))
        analysis = (n.get("analysis") or "").strip()
        if analysis:
            print("            " + analysis[:110])
