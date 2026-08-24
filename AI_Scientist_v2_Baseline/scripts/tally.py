#!/usr/bin/env python3
"""Exact experiment count, separating what ran from what counts."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

BASE = Path.home() / "ai_scientist_v2_baseline"

trial_dirs = [d for d in (BASE / "trials").iterdir() if d.is_dir()]
launched = [d for d in trial_dirs if (d / "run.log").exists()]

journal = json.loads((BASE / "campaign" / "journal.json").read_text())["nodes"]
arch_path = BASE / "campaign" / "env_failures.json"
archived = json.loads(arch_path.read_text())["nodes"] if arch_path.exists() else []

scored = [n for n in journal if (n.get("metric") or {}).get("value") is not None]
buggy = [n for n in journal if n.get("is_buggy")]

def bucket(n):
    p = (n.get("plan") or "")
    for key, label in (("baseline (pristine", "baseline"), ("ABLATION", "ablation"),
                       ("SEED-PAIRED", "seed-paired baseline"), ("SEED VARIANCE", "seed variance"),
                       ("SEED EVAL", "seed variance"), ("REPLICATE", "replicate"),
                       ("RETEST", "interaction retest"), ("CONSTANT-CAPACITY", "shape allocation")):
        if key in p:
            return label
    return "lever experiment"

print(f"trial directories on disk        : {len(trial_dirs)}")
print(f"  of which actually launched     : {len(launched)}")
print()
print(f"journal nodes (retained)         : {len(journal)}")
print(f"  scored                         : {len(scored)}")
print(f"  buggy (in-journal)             : {len(buggy)}")
print(f"archived environmental failures  : {len(archived)}")
print()
print(f"TOTAL trials executed            : {len(journal) + len(archived)}")
print()
print("retained scored nodes by kind:")
for k, v in Counter(bucket(n) for n in scored).most_common():
    print(f"  {k:<24} {v}")
print()
reasons = Counter()
for n in archived:
    a = (n.get("analysis") or "")
    reasons["contention" if "contended" in a else
            "OOM" if "OutOfMemory" in a else
            "skipped (no clear GPU)" if "skipped" in a else "other"] += 1
print("archived failures by cause:")
for k, v in reasons.most_common():
    print(f"  {k:<24} {v}")
