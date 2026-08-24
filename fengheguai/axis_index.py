#!/usr/bin/env python3
"""Which coordinates has this campaign actually visited?

Keyword greps over change_summary prose lie in both directions: they match trials
that only mention a knob, and miss trials that moved it without naming it. The
source is authoritative, so read the constants out of every node's train.py and
join them to that node's honest metric.

Usage:  axis_index.py [CAMPAIGN_DIR] [--const NAME] [--champion TRIAL]
"""
import json, re, sys, pathlib
from collections import defaultdict

args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = {a.split("=")[0]: (a.split("=")[1] if "=" in a else True) for a in sys.argv[1:] if a.startswith("--")}
root = pathlib.Path(args[0] if args else "/data3/zhubaiyu/fengheguai/campaigns/h200-claude")

metric = {}
for line in (root / "ledger.jsonl").open():
    e = json.loads(line)
    if e["kind"] == "trial_completed":
        p = e["payload"]
        if p.get("metric"):
            metric[p["trial_id"]] = (float(p["metric"]), p.get("status"))

CONST = re.compile(r"^([A-Z][A-Z0-9_]{2,})\s*=\s*([^#\n]+?)\s*(?:#.*)?$", re.M)
axes = defaultdict(lambda: defaultdict(list))
for node in sorted((root / "nodes").iterdir()):
    f = node / "train.py"
    if not f.is_file() or node.name not in metric:
        continue
    m, status = metric[node.name]
    for name, val in CONST.findall(f.read_text()):
        axes[name][val.strip()].append((m, node.name, status))

champ = flags.get("--champion", "t0096")
champ_vals = {}
cf = root / "nodes" / champ / "train.py"
if cf.is_file():
    champ_vals = {n: v.strip() for n, v in CONST.findall(cf.read_text())}

only = flags.get("--const")
print(f"# axis index over {len(metric)} scored trials; champion {champ}\n")
for name in sorted(axes):
    if only and only.upper() not in name:
        continue
    vals = axes[name]
    if len(vals) < 2 and not only:
        continue  # never varied: nothing to learn
    best = {v: min(rows) for v, rows in vals.items()}
    ranked = sorted(best.items(), key=lambda kv: kv[1][0])
    print(f"{name}   ({len(vals)} distinct values tried)")
    for v, (m, tid, status) in ranked:
        mark = " <-- champion" if champ_vals.get(name) == v else ""
        print(f"    {v:<28} best={m:.6f}  n={len(vals[v]):<3d} {tid} {status}{mark}")
    print()
