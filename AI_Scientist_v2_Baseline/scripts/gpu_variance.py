#!/usr/bin/env python3
"""Did GPU auto-switching introduce between-device variance into the replicate group?

The runner switches to whichever device is clear, which is right for throughput but means
the incumbent's replicates are no longer all from one GPU. If devices differ even slightly
in clocks or thermals, that variance lands in the noise floor every significance claim is
judged against — a confound introduced by my own fix.
"""
from __future__ import annotations

import hashlib
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path.home() / "ai_scientist_v2_baseline"
sys.path.insert(0, str(BASE))
from ai_scientist_ar.journal import Journal  # noqa: E402

j = Journal.load(BASE / "campaign" / "journal.json")
best = j.get_best_node(min_improvement=0.00036)
h = hashlib.sha256(best.code.encode()).hexdigest()

by_gpu: dict[str, list[float]] = defaultdict(list)
rows = []
for n in j.nodes:
    if n.is_buggy or not (n.metric and n.metric.value):
        continue
    if hashlib.sha256(n.code.encode()).hexdigest() != h:
        continue
    log = Path(n.trial_dir or "") / "run.log"
    gpu = "?"
    if log.exists():
        txt = log.read_text(errors="replace")[:4000]
        m = re.search(r"switched to GPU (\d+)", txt)
        if not m:
            m = re.search(r"CUDA_VISIBLE_DEVICES=(\d+)", txt)
        gpu = m.group(1) if m else "assigned"
    by_gpu[gpu].append(n.metric.value)
    rows.append((n.id, n.metric.value, int((n.summary or {}).get("num_steps") or 0), gpu))

print(f"incumbent replicate group: n={len(rows)}")
for nid, v, steps, gpu in rows:
    print(f"  {nid} {v:.6f} steps={steps:<5} gpu={gpu}")

vals = [r[1] for r in rows]
print(f"\noverall: mean={st.mean(vals):.6f} sd={st.stdev(vals):.6f}")
print("\nby device group:")
for gpu, vs in sorted(by_gpu.items()):
    sd = f"{st.stdev(vs):.6f}" if len(vs) > 1 else "n/a"
    print(f"  gpu={gpu:<9} n={len(vs):<3} mean={st.mean(vs):.6f} sd={sd}")

# Split the group in half chronologically: if variance grew after auto-switching landed,
# the later half should be visibly wider.
if len(vals) >= 8:
    half = len(vals) // 2
    a, b = vals[:half], vals[half:]
    print(f"\nfirst half  (n={len(a)}): mean={st.mean(a):.6f} sd={st.stdev(a):.6f}")
    print(f"second half (n={len(b)}): mean={st.mean(b):.6f} sd={st.stdev(b):.6f}")
