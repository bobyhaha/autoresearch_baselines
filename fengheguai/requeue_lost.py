#!/usr/bin/env python3
"""Return served queue entries to the queue when their trial never produced a record.

autoserve retires an entry the moment it is delivered, but a trial can die to contention
before completing (t0431 did). Those bets are then silently dropped despite never having
been measured -- and because the ledger holds no completion, the coordinate is still free,
so the bet is not merely lost but recoverable.

An entry is recovered only when the controller has demonstrably moved past its trial:
there is a later trial_started and no completion for its own. That excludes the in-flight
trial, which legitimately has no completion yet.
"""
import json, subprocess, shutil, sys, re
from pathlib import Path

P = Path(__file__).resolve().parent
Q, S = P / "queue", P / "queue" / "served"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ssh_target import SSH_CMD as SSH
PROBE = r'''
import json
R="/data3/zhubaiyu/fengheguai/campaigns/h200-claude/ledger.jsonl"
done=set(); started=[]
for l in open(R):
    try: r=json.loads(l)
    except Exception: continue
    p=r.get("payload") or {}; t=p.get("trial_id")
    if not t: continue
    if r["kind"]=="trial_started": started.append(t)
    elif r["kind"] in ("trial_completed","trial_failed"): done.add(t)
print(json.dumps({"done":sorted(done),"max_started":max(started) if started else ""}))
'''
out = subprocess.run(SSH + ["python3 -"], input=PROBE, capture_output=True, text=True, timeout=90)
if out.returncode != 0:
    print("probe failed:", out.stderr.strip()[:200]); sys.exit(1)
info = json.loads(out.stdout.strip().splitlines()[-1])
done, max_started = set(info["done"]), info["max_started"]

recovered = 0
for d in sorted(S.glob("*-t[0-9]*")):
    m = re.match(r"^(.*)-(t\d+)$", d.name)
    if not m: continue
    name, tid = m.groups()
    if tid in done or tid >= max_started:
        continue                      # measured, or still the controller's current trial
    dest = Q / name
    if dest.exists():
        print(f"{tid}: {name} already back in queue"); continue
    shutil.move(str(d), str(dest))
    print(f"{tid}: never completed -> requeued {name}")
    recovered += 1
print(f"recovered {recovered}; queue now: {sorted(p.name for p in Q.glob('[0-9]*'))}")
