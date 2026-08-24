#!/usr/bin/env python3
"""One-command campaign status.

Written because the same status query kept being re-derived as an inline `python -c`
through ssh, where nested quoting fails in ways that waste a round trip each time.
"""

from __future__ import annotations

import hashlib
import json
import statistics as st
import sys
from pathlib import Path

BASE = Path.home() / "ai_scientist_v2_baseline"
sys.path.insert(0, str(BASE))

from ai_scientist_ar.journal import Journal  # noqa: E402

GUARD = 0.00036   # 2 sigma of the pooled noise floor; must match the harness
BASELINE = 0.991192

j = Journal.load(BASE / "campaign" / "journal.json")
best = j.get_best_node(min_improvement=GUARD)

print(f"nodes: {len(j)}  buggy: {sum(1 for n in j.nodes if n.is_buggy)}  "
      f"seed-eval: {sum(1 for n in j.nodes if getattr(n, 'is_seed_eval', False))}")

if best is not None:
    h = hashlib.sha256(best.code.encode()).hexdigest()
    grp = [n.metric.value for n in j.nodes
           if hashlib.sha256(n.code.encode()).hexdigest() == h and n.metric and n.metric.value]
    mean = st.mean(grp)
    sd = st.stdev(grp) if len(grp) > 1 else float("nan")
    print(f"incumbent: {best.id}  n={len(grp)}  min={min(grp):.6f}  mean={mean:.6f}  sd={sd:.6f}")
    print(f"  baseline {BASELINE:.6f} -> improvement {BASELINE-mean:.6f} "
          f"({(BASELINE-mean)/BASELINE*100:.2f}%)  [mean, not the lucky minimum]")

print("\nlast 6 trials:")
for n in j.nodes[-6:]:
    s = n.summary or {}
    val = "BUGGY  " if n.is_buggy else f"{n.metric.value:.6f}"
    print(f"  {n.id} {val} steps={int(s.get('num_steps') or 0):>5} "
          f"foreign={int(s.get('foreign_peak_mb') or 0):>6}MB | {(n.plan or '')[:44]}")

q = BASE / "rendezvous" / "queue"
items = []
for f in q.glob("*.json"):
    try:
        items.append(json.loads(f.read_text()))
    except (OSError, json.JSONDecodeError):
        pass
print(f"\nqueue: {len(items)}  ablations: {sum(1 for d in items if 'ABLATION' in d.get('plan',''))}"
      f"  seed-baselines: {sum(1 for d in items if 'SEED-PAIRED' in d.get('plan',''))}")

# Throughput invariant: an ablation that changes only a constant cannot alter step count,
# so a deviation from the clean band is environmental, not the lever.
CONST_ONLY = ("RoPE base", "EMBEDDING_LR", "UNEMBEDDING_LR", "WEIGHT_DECAY", "FINAL_LR_FRAC")
sus = [n for n in j.nodes
       if n.metric and n.metric.value and "ABLATION" in (n.plan or "")
       and any(k in (n.plan or "") for k in CONST_ONLY)
       and (n.summary or {}).get("num_steps", 1314) < 1290]
if sus:
    print("\nWARNING - constant-only ablations with depressed step count (contended):")
    for n in sus:
        print(f"  {n.id} steps={int(n.summary['num_steps'])} — expected ~1314")
