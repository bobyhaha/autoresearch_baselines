#!/usr/bin/env python3
"""Regenerate campaign_chart.html from the live ledger.

The chart's headline stats and its data array must not be able to drift apart, so every
derived number -- trial count, valid/crashed split, promotions, champion, improvement --
is recomputed from the ledger rather than hand-edited. Milestone labels are hand-authored
and are carried forward by trial id.

Usage: refresh_chart.py [--label "t0353 gated attention"] [--set tXXXX="short label"]
"""
import json, re, subprocess, sys, pathlib

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from ssh_target import TARGET as HOST
SSH = ["ssh", "-i", str(pathlib.Path.home()/".ssh/id_ed25519"), "-o", "IdentitiesOnly=yes",
       "-o", "StrictHostKeyChecking=no", "-p", "50002", HOST]
CAMP = "/data3/zhubaiyu/fengheguai/campaigns/h200-claude"
HTML = pathlib.Path(__file__).resolve().parent / "campaigns/campaign_chart.html"

DUMP = f'''
import json, pathlib
out = []
for l in (pathlib.Path("{CAMP}")/"ledger.jsonl").open():
    e = json.loads(l)
    if e["kind"] != "trial_completed": continue
    p = e["payload"]
    out.append({{"id": p.get("trial_id"), "metric": p.get("metric"), "status": p.get("status"),
                "stage": p.get("stage"),
                "steps": (p.get("measurements") or [{{}}])[0].get("num_steps")}})
print(json.dumps(out))
'''

new_labels = dict(a.split("=", 1) for a in sys.argv[1:] if a.startswith("t") and "=" in a)

trials = json.loads(subprocess.run(SSH + ["python3 -"], input=DUMP,
                                   capture_output=True, text=True).stdout)
trials = [t for t in trials if t["status"] != "agent_error"]

html = HTML.read_text()
old = json.loads(re.search(r"^const D = (\[.*\]);$", html, re.M).group(1))
labels = {r["id"]: r["label"] for r in old if r.get("label")}
labels.update(new_labels)

D, best = [], None
for i, t in enumerate(trials):
    v = t["metric"]
    if v is not None and (best is None or v < best): best = v
    D.append({"i": i, "id": t["id"], "stage": t["stage"], "status": t["status"], "v": v,
              "promoted": t["status"] in ("keep", "baseline"), "steps": t["steps"],
              "label": labels.get(t["id"], ""), "best": best})

n, nv = len(D), sum(1 for r in D if r["v"] is not None)
nc, np_ = n - nv, sum(1 for r in D if r["promoted"])
base = D[0]["v"]
champ = min((r for r in D if r["v"] is not None), key=lambda r: r["v"])
delta = base - champ["v"]; pct = delta / base * 100

payload = "const D = " + json.dumps(D, separators=(",", ":")) + ";"
html, ok = re.subn(r"^const D = \[.*\];$", lambda m: payload, html, count=1, flags=re.M)
subs = [
 (r"<h1>Minimizing val_bpb, \d+ measured trials</h1>", f"<h1>Minimizing val_bpb, {n} measured trials</h1>"),
 (r'(<span class="k">Champion</span><span class="v accent">)[\d.]+(</span><span class="n">)t\d+(</span>)',
  f'\\g<1>{champ["v"]:.6f}\\g<2>{champ["id"]}\\g<3>'),
 (r'(<span class="k">Improvement</span><span class="v accent">&minus;)[\d.]+(%</span><span class="n">&minus;)[\d.]+( bpb</span>)',
  f'\\g<1>{pct:.2f}\\g<2>{delta:.6f}\\g<3>'),
 (r'(<span class="k">Trials</span><span class="v">)\d+(</span><span class="n">)\d+ valid(\s*(?:&middot;|·)\s*)\d+( crashed</span>)',
  f'\\g<1>{n}\\g<2>{nv} valid\\g<3>{nc}\\g<4>'),
 (r'(<span class="k">Promotions</span><span class="v">)\d+(</span>)', f'\\g<1>{np_}\\g<2>'),
 (r'aria-label="Scatter of val_bpb for \d+ trials with a descending best-so-far step line from [\d.]+ to [\d.]+\."',
  f'aria-label="Scatter of val_bpb for {n} trials with a descending best-so-far step line from {base:.6f} to {champ["v"]:.6f}."'),
 (r'(style="fill:var\(--accent\)">)0\.\d+(</text>)', f'\\g<1>{champ["v"]:.6f}\\g<2>'),
]
# The engine promotes on median-of-two, and a re-measurement of an existing champion can win
# by less than the same-source spread (0.000481 at n=4). t0506 did exactly that: its non-comment
# diff against t0500 is empty. Annotate the caption when the best trial is a re-measurement, so
# the published figure does not imply an improvement that was not made.
import subprocess as _sp
def _norm(text):
    return "\n".join(l.rstrip() for l in text.splitlines()
                      if l.strip() and not l.strip().startswith("#"))

def _same_model(a, b):
    """Compare two nodes' train.py ignoring comments. Fetch and compare locally so no shell
    quoting is involved -- the remote-diff version failed silently on argv quoting."""
    root = "/data3/zhubaiyu/fengheguai/campaigns/h200-claude/nodes"
    try:
        ta = _sp.run(SSH + [f"cat {root}/{a}/train.py"], capture_output=True, text=True, timeout=60)
        tb = _sp.run(SSH + [f"cat {root}/{b}/train.py"], capture_output=True, text=True, timeout=60)
        if ta.returncode or tb.returncode or not ta.stdout or not tb.stdout:
            return False
        return _norm(ta.stdout) == _norm(tb.stdout)
    except Exception:
        return False

_promoted = [r["id"] for r in D if r.get("promoted")]
_note = ""
if len(_promoted) >= 2 and _same_model(_promoted[-1], _promoted[-2]):
    _note = (f' <strong>Note:</strong> {_promoted[-1]} is byte-identical to {_promoted[-2]} apart '
             f'from comments \u2014 it is the same model re-measured, promoted by a margin smaller '
             f'than the measured same-source spread. The real result is {_promoted[-2]}.')
if _note:
    html = re.sub(r'(<figcaption>)', r'\1' + _note.replace("\\", "\\\\"), html, count=1)

missing = [p for p, r in subs if not re.search(p, html)]
for p, r in subs: html = re.sub(p, r, html)
# The x-axis ticks are generated in-page from D.length. Guard against the stale-hardcoded
# form ever returning: it silently stopped labelling the right-hand side of the plot.
import sys as _sys
if re.search(r"\[0,10,20,30,40,50,60,70,80,\d+\]\.forEach", html):
    print("ERROR: chart has hardcoded x ticks again; axis will go stale", file=_sys.stderr)
    _sys.exit(1)
HTML.write_text(html)
print(f"trials={n} valid={nv} crashed={nc} promotions={np_}")
print(f"champion={champ['id']} {champ['v']:.6f}  improvement=-{pct:.2f}%")
print(f"data array: {'ok' if ok else 'MISS'};  unmatched patterns: {len(missing)}")
