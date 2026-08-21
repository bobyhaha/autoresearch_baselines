#!/usr/bin/env python3
"""Queue levers from the library against the *current* best node.

Solves a churn problem. Candidates are queued keyed by parent node id, but every time
a queued candidate wins, the best node moves and every other queued candidate becomes
unreachable — the search can never select their parent again. Re-deriving them by hand
each time is slow and error-prone.

Levers here are stored as *whole-line replacements matched by a unique prefix*, not as
exact old/new string pairs. That way a lever stays valid as the winning train.py
evolves: `MATRIX_LR` can be queued regardless of what its current value happens to be.

Usage:
  queue_levers.py --list
  queue_levers.py head64 span4 mlr03      # queue these against the current best
  queue_levers.py --all-untested          # queue everything not yet tried anywhere
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

BASE = Path.home() / "ai_scientist_v2_baseline"
LIB = Path(__file__).with_name("levers.json")


def load_journal() -> list[dict]:
    p = BASE / "campaign" / "journal.json"
    return json.loads(p.read_text(encoding="utf-8"))["nodes"] if p.exists() else []


def best_node(nodes: list[dict]) -> dict | None:
    scored = [n for n in nodes if (n.get("metric") or {}).get("value") is not None and not n.get("is_buggy")]
    return min(scored, key=lambda n: n["metric"]["value"]) if scored else None


def apply_lever(src: Path, dst: Path, prefix: str, new_line: str) -> str | None:
    """Replace the single line starting with `prefix`. Returns the old line, or None."""
    lines = src.read_text(encoding="utf-8").splitlines(keepends=True)
    hits = [i for i, ln in enumerate(lines) if ln.startswith(prefix)]
    if len(hits) != 1:
        return None
    old = lines[hits[0]].rstrip("\n")
    if old == new_line:
        return None  # already at this value; queueing it would be a no-op experiment
    lines[hits[0]] = new_line + "\n"
    text = "".join(lines)
    try:
        compile(text, str(dst), "exec")
    except SyntaxError as exc:
        print(f"  SYNTAX ERROR after lever: {exc}", file=sys.stderr)
        return None
    dst.write_text(text, encoding="utf-8")
    return old


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("levers", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all-untested", action="store_true")
    ap.add_argument("--clear-stale", action="store_true", default=True)
    args = ap.parse_args()

    lib = json.loads(LIB.read_text(encoding="utf-8"))["levers"]
    if args.list:
        for k, v in lib.items():
            print(f"{k:<12} {v['line'][:70]}")
        return 0

    nodes = load_journal()
    best = best_node(nodes)
    if best is None:
        print("no scored node yet", file=sys.stderr)
        return 1
    print(f"current best: {best['id']}  val_bpb={best['metric']['value']:.6f}")

    # Drop queue entries whose parent is no longer the best; they can never be popped.
    qdir = BASE / "rendezvous" / "queue"
    stale_dir = BASE / "rendezvous" / "stale"
    stale_dir.mkdir(parents=True, exist_ok=True)
    if args.clear_stale:
        moved = 0
        for f in qdir.glob("*.json"):
            try:
                if json.loads(f.read_text(encoding="utf-8")).get("parent_id") != best["id"]:
                    f.rename(stale_dir / f.name)
                    moved += 1
            except (OSError, json.JSONDecodeError):
                continue
        if moved:
            print(f"cleared {moved} stale queue entries")

    tried = {(n.get("plan") or "").split(".")[0] for n in nodes}
    names = args.levers
    if args.all_untested:
        names = [k for k, v in lib.items() if v["plan"].split(".")[0] not in tried]

    src = BASE / "trials" / best["id"] / "train.py"
    queued = 0
    for name in names:
        if name not in lib:
            print(f"  unknown lever: {name}", file=sys.stderr)
            continue
        spec = lib[name]
        dst = BASE / "candidates" / f"L_{name}.py"
        old = apply_lever(src, dst, spec["prefix"], spec["line"])
        if old is None:
            print(f"  skip {name}: no unique match, or already at this value")
            continue
        subprocess.run(
            [str(BASE / "task/.venv/bin/python"), str(BASE / "scripts/rv.py"), "enqueue",
             "--op", "improve", "--parent", best["id"],
             "--plan", spec["plan"], "--code-file", str(dst)],
            check=True, stdout=subprocess.DEVNULL,
        )
        print(f"  queued {name}: {old.strip()[:46]}  ->  {spec['line'].strip()[:46]}")
        queued += 1

    print(f"queued {queued}; queue depth now {len(list(qdir.glob('*.json')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
