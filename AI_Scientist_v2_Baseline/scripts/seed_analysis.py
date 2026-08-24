#!/usr/bin/env python3
"""Seed-paired analysis: does the campaign's improvement generalise beyond seed 42?

The whole campaign selected on seed-42 val_bpb, so the final configuration is to some
degree tuned to seed 42's initialisation. Measuring the baseline AND the final config at
the same set of seeds is the only way to separate a real recipe improvement from that
selection effect.
"""

from __future__ import annotations

import hashlib
import re
import statistics as st
import sys
from pathlib import Path

BASE = Path.home() / "ai_scientist_v2_baseline"
sys.path.insert(0, str(BASE))
from ai_scientist_ar.journal import Journal  # noqa: E402

j = Journal.load(BASE / "campaign" / "journal.json")
best = j.get_best_node(min_improvement=0.00036)
h = hashlib.sha256(best.code.encode()).hexdigest()

base, fin = {42: [0.991192]}, {42: []}
for n in j.nodes:
    if not (n.metric and n.metric.value) or n.is_buggy:
        continue
    plan = n.plan or ""
    if hashlib.sha256(n.code.encode()).hexdigest() == h:
        fin[42].append(n.metric.value)
        continue
    m = re.search(r"seed (\d+)", plan)
    if not m:
        continue
    s = int(m.group(1))
    if "SEED-PAIRED BASELINE" in plan:
        base.setdefault(s, []).append(n.metric.value)
    elif "SEED VARIANCE" in plan or "SEED EVAL" in plan:
        fin.setdefault(s, []).append(n.metric.value)

print(f"{'seed':>6} {'baseline':>10} {'final':>10} {'improvement':>12} {'%':>7}")
print("-" * 50)
imps = []
for s in sorted(set(base) | set(fin)):
    b = st.mean(base[s]) if base.get(s) else None
    f = st.mean(fin[s]) if fin.get(s) else None
    if b and f:
        imp = b - f
        imps.append(imp)
        print(f"{s:>6} {b:>10.6f} {f:>10.6f} {imp:>12.6f} {imp/b*100:>6.2f}%")
    else:
        print(f"{s:>6} {b if b else 0:>10.6f} {f if f else 0:>10.6f} {'pending':>12}")

if len(imps) > 1:
    print("-" * 50)
    print(f"  mean improvement across {len(imps)} seeds: {st.mean(imps):.6f} "
          f"({st.mean(imps)/0.9902*100:.2f}%)   sd {st.stdev(imps):.6f}")
    held_out = imps[1:]
    if held_out:
        print(f"  seed 42 (optimised on): {imps[0]:.6f}")
        print(f"  held-out seeds mean   : {st.mean(held_out):.6f}")
        print(f"  selection effect      : {imps[0]-st.mean(held_out):.6f} "
              f"({(imps[0]-st.mean(held_out))/imps[0]*100:.0f}% of the seed-42 gain)")
