#!/usr/bin/env python3
"""Restocker daemon — guarantees the candidate queue never runs dry.

This exists because the previous campaign died exactly this way: the queue emptied,
the harness blocked on the rendezvous waiting for the agent, the agent did not answer,
and after the timeout the harness exited and left a shared H200 idle for 55 hours.

The queue was deliberately kept shallow so that a miss would wake the agent to look at
results. That optimises for freshness of ideas at the cost of survivability, and the
cost turned out to be the entire campaign. This daemon keeps the mechanical part
autonomous — the agent still contributes ideas on check-in, but its absence no longer
stops the search.

Policy, in order:
  1. Below the low-water mark, enqueue untried levers from the library against whatever
     node is currently best. "Untried" is judged by hashing the candidate source and
     comparing against every node the journal has already run — so a lever is skipped
     when it would reproduce an existing trial, not merely when its name looks familiar.
  2. If the library is exhausted, enqueue a replicate of the current best. Always
     scientifically useful (it tightens the noise estimate) and it keeps the GPU busy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path.home() / "ai_scientist_v2_baseline"
# Must equal the harness's --min-improvement, or the two disagree on "best".
MIN_IMPROVEMENT = float(os.environ.get("RESTOCK_MIN_IMPROVEMENT", "0.00036"))
LIB = BASE / "scripts" / "levers.json"
PY = BASE / "task/.venv/bin/python"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_nodes() -> list[dict]:
    p = BASE / "campaign" / "journal.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))["nodes"]
    except (OSError, json.JSONDecodeError, KeyError):
        return []


def best_node(nodes: list[dict], min_improvement: float = 0.0) -> dict | None:
    """Current incumbent, using the SAME rule as the harness.

    This must match `Journal.get_best_node` exactly. When the harness gained a
    significance guard and this did not, the two disagreed about which node was best —
    the restocker queued against one parent while the harness requested another, and
    since the queue is keyed by parent they could never meet. That is a deadlock: the
    harness blocks forever with a full queue beside it. Any change to the selection rule
    has to be made in both places.
    """
    scored = [
        n for n in nodes
        if (n.get("metric") or {}).get("value") is not None
        and not n.get("is_buggy") and not n.get("is_seed_eval")
    ]
    if not scored:
        return None
    lo = min(n["metric"]["value"] for n in scored)
    if min_improvement <= 0:
        return min(scored, key=lambda n: n["metric"]["value"])
    band = [n for n in scored if n["metric"]["value"] <= lo + min_improvement]
    return min(band, key=lambda n: (n.get("step") if n.get("step") is not None else 0,
                                    n.get("ctime") or 0))


def apply_lever(src_text: str, spec: dict) -> str | None:
    """Apply a lever: one or more whole-line replacements matched by unique prefixes.

    A lever is `{prefix, line}` for the single-line case, or `{subs: [[prefix, line], ...]}`
    when it must touch several lines at once — the MLP ratio, for instance, is only
    coherent if `c_fc` and `c_proj` change together. Returns None if any prefix does not
    match exactly once, or if the result is identical to the source (a no-op lever, which
    would enqueue a "change" that changes nothing).
    """
    subs = spec.get("subs") or [[spec["prefix"], spec["line"]]]
    lines = src_text.splitlines(keepends=True)
    changed = False
    for prefix, new_line in subs:
        hits = [i for i, ln in enumerate(lines) if ln.startswith(prefix)]
        if len(hits) != 1:
            return None
        if lines[hits[0]].rstrip("\n") != new_line:
            lines[hits[0]] = new_line + "\n"
            changed = True
    if not changed:
        return None
    text = "".join(lines)
    try:
        compile(text, "<candidate>", "exec")
    except SyntaxError:
        return None
    return text


def enqueue(op: str, parent: str, plan: str, path: Path) -> bool:
    r = subprocess.run(
        [str(PY), str(BASE / "scripts/rv.py"), "enqueue", "--op", op,
         "--parent", parent, "--plan", plan, "--code-file", str(path)],
        capture_output=True, text=True,
    )
    return r.returncode == 0


def queue_depth() -> int:
    q = BASE / "rendezvous" / "queue"
    return len(list(q.glob("*.json"))) if q.exists() else 0


def tick(low: int, target: int, log) -> None:
    """One restock pass.

    Ordering matters and has been got wrong three times in this campaign, always the
    same way: new filtering logic placed *after* the `depth >= low` early return, so it
    only ran when the queue was empty — never when it was full of the wrong things.
    Both filters therefore run BEFORE depth is measured, and depth is measured exactly
    once, afterwards, on what is left.

    Filter 1 (unreachable): entries parented on a node that is no longer best can never
    be popped, so counting them as depth would starve the harness.
    Filter 2 (closed): entries whose lever has since been closed are known losses;
    leaving them queued spends trials re-confirming what the log already records.
    """
    nodes = load_nodes()
    best = best_node(nodes, MIN_IMPROVEMENT)
    if best is None:
        log("no scored node yet; nothing to restock against")
        return

    lib = json.loads(LIB.read_text(encoding="utf-8"))["levers"]
    closed_plans = {v["plan"] for v in lib.values() if v.get("closed")}

    seen = {sha(n.get("code") or "") for n in nodes}
    qdir, stale = BASE / "rendezvous" / "queue", BASE / "rendezvous" / "stale"
    stale.mkdir(parents=True, exist_ok=True)

    swept = evicted = reparented = 0
    for f in sorted(qdir.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if d.get("parent_id") != best["id"]:
            if d.get("reparent_ok"):
                # Parent-independent measurement (seed sweep, paired baseline): its source
                # does not derive from any parent, so sweeping it as "unreachable" silently
                # discards a measurement. Retarget onto the new best instead.
                d["parent_id"] = best["id"]
                tmp = f.with_suffix(".tmp")
                tmp.write_text(json.dumps(d, indent=2), encoding="utf-8")
                tmp.replace(f)
                reparented += 1
                seen.add(sha(d.get("code", "")))
                continue
            f.rename(stale / f.name); swept += 1
        elif d.get("plan") in closed_plans:
            f.rename(stale / f.name); evicted += 1
        else:
            seen.add(sha(d.get("code", "")))
    if swept:
        log(f"swept {swept} unreachable (parent != {best['id']})")
    if evicted:
        log(f"evicted {evicted} queued candidates whose lever is now closed")
    if reparented:
        log(f"re-parented {reparented} measurement entries onto {best['id']}")

    if queue_depth() >= low:
        return

    src = BASE / "trials" / best["id"] / "train.py"
    if not src.exists():
        log(f"best {best['id']} has no train.py on disk; skipping tick")
        return
    src_text = src.read_text(encoding="utf-8")

    # Closed axes are never restocked. As the winning config absorbs levers they become
    # no-ops on it, so the applicable pool shrinks; replicates are the better fallback
    # because they measure the noise floor every comparison depends on.
    added = 0
    for name, spec in [(k, v) for k, v in lib.items() if not v.get("closed")]:
        if queue_depth() >= target:
            break
        text = apply_lever(src_text, spec)
        if text is None or sha(text) in seen:
            continue
        dst = BASE / "candidates" / f"R_{name}.py"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding="utf-8")
        if enqueue("improve", best["id"], spec["plan"], dst):
            seen.add(sha(text)); added += 1
            log(f"queued lever {name} against {best['id']}")

    while queue_depth() < low:
        dst = BASE / "candidates" / f"R_replicate_{best['id']}.py"
        dst.write_text(src_text, encoding="utf-8")
        plan = (f"AUTO-REPLICATE of {best['id']} (no applicable untried lever). "
                f"Adds a sample to the run-to-run noise estimate.")
        if not enqueue("improve", best["id"], plan, dst):
            break
        added += 1
        log(f"queued replicate of {best['id']}")

    if added:
        log(f"restocked {added}; queue depth now {queue_depth()}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--low", type=int, default=4, help="restock when depth falls below this")
    ap.add_argument("--target", type=int, default=10, help="restock up to this depth")
    ap.add_argument("--interval", type=float, default=120.0)
    ap.add_argument("--duration", type=float, default=90000.0)
    args = ap.parse_args()

    end = time.time() + args.duration

    def log(msg: str) -> None:
        print(f"[restock {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"started: low={args.low} target={args.target} every {args.interval}s")
    while time.time() < end:
        try:
            tick(args.low, args.target, log)
        except Exception as exc:  # a restocker crash must not stop the campaign
            log(f"tick failed: {exc!r}")
        time.sleep(args.interval)
    log("duration elapsed, exiting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
