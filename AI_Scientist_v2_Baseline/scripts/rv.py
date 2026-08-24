#!/usr/bin/env python3
"""Rendezvous CLI — the agent's side of the LLM backend.

The campaign harness blocks on `rendezvous/PENDING` when it needs code written.
This tool lets the agent inspect that request, answer it, or pre-author candidates
into the queue so the GPU never idles waiting for the next idea.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

BASE = Path.home() / "ai_scientist_v2_baseline"
RV = BASE / "rendezvous"


def _compact_request(req: dict, history_limit: int) -> dict:
    """Drop the bulky fields so a request fits comfortably in an agent's context."""
    out = {k: v for k, v in req.items() if k not in ("task_memo", "history", "traceback")}
    hist = req.get("history", [])
    out["history_tail"] = hist[-history_limit:]
    out["history_len"] = len(hist)
    if req.get("traceback"):
        out["traceback"] = req["traceback"][-2500:]
    return out


def cmd_pending(args) -> int:
    pending = RV / "PENDING"
    if not pending.exists():
        print(json.dumps({"pending": False}))
        return 0
    req_id = pending.read_text(encoding="utf-8").strip()
    path = RV / "requests" / f"{req_id}.json"
    if not path.exists():
        print(json.dumps({"pending": False, "note": f"stale PENDING marker {req_id}"}))
        return 0
    req = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps({"pending": True, **_compact_request(req, args.history)}, indent=2))
    return 0


def _write(target_dir: Path, payload: dict, name: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / name
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)  # atomic: the harness must never read a half-written response
    return path


def cmd_respond(args) -> int:
    code = Path(args.code_file).read_text(encoding="utf-8")
    if not code.strip():
        raise SystemExit("refusing to submit empty code")
    payload = {"plan": args.plan, "code": code, "answered": time.time()}
    path = _write(RV / "responses", payload, f"{args.request_id}.json")
    print(f"responded to {args.request_id} -> {path} ({len(code)} bytes)")
    return 0


def cmd_enqueue(args) -> int:
    code = Path(args.code_file).read_text(encoding="utf-8")
    if not code.strip():
        raise SystemExit("refusing to enqueue empty code")
    payload = {
        "op": args.op,
        "parent_id": args.parent or None,
        "plan": args.plan,
        "code": code,
        "queued": time.time(),
        # Measurement trials (seed sweeps, paired baselines) carry fixed source that does
        # not derive from any parent. The queue is keyed by parent, so when the frontier
        # advances they would be swept as unreachable and the measurement silently lost —
        # which is exactly what happened to the seed-paired baselines. Marked entries are
        # re-parented onto the new best instead of discarded.
        "reparent_ok": bool(args.measurement),
    }
    name = f"{int(time.time())}_{uuid.uuid4().hex[:6]}.json"
    path = _write(RV / "queue", payload, name)
    print(f"queued op={args.op} parent={args.parent or '-'} -> {path.name} ({len(code)} bytes)")
    return 0


def cmd_status(args) -> int:
    campaign = BASE / "campaign"
    out: dict = {}
    status_path = campaign / "status.json"
    if status_path.exists():
        out["status"] = json.loads(status_path.read_text(encoding="utf-8"))
    journal_path = campaign / "journal.json"
    if journal_path.exists():
        nodes = json.loads(journal_path.read_text(encoding="utf-8")).get("nodes", [])
        out["nodes"] = [
            {
                "step": n.get("step"),
                "id": n.get("id"),
                "parent": n.get("parent_id"),
                "op": n.get("stage_name"),
                "val_bpb": (n.get("metric") or {}).get("value"),
                "buggy": n.get("is_buggy"),
                "steps": (n.get("summary") or {}).get("num_steps"),
                "mfu": (n.get("summary") or {}).get("mfu_percent"),
                "plan": (n.get("plan") or "")[:90],
            }
            for n in nodes[-args.limit :]
        ]
        scored = [n for n in nodes if (n.get("metric") or {}).get("value") is not None]
        scored.sort(key=lambda n: n["metric"]["value"])
        out["leaderboard"] = [
            {"id": n["id"], "val_bpb": n["metric"]["value"], "plan": (n.get("plan") or "")[:90]}
            for n in scored[:8]
        ]
    out["queue_depth"] = len(list((RV / "queue").glob("*.json"))) if (RV / "queue").exists() else 0
    out["pending"] = (RV / "PENDING").exists()
    print(json.dumps(out, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pending", help="show the open rendezvous request")
    p.add_argument("--history", type=int, default=12)
    p.set_defaults(func=cmd_pending)

    p = sub.add_parser("respond", help="answer the open request")
    p.add_argument("request_id")
    p.add_argument("--plan", required=True)
    p.add_argument("--code-file", required=True)
    p.set_defaults(func=cmd_respond)

    p = sub.add_parser("enqueue", help="pre-author a candidate for a given parent")
    p.add_argument("--op", required=True, choices=["draft", "improve", "debug"])
    p.add_argument("--parent", default=None)
    p.add_argument("--plan", required=True)
    p.add_argument("--code-file", required=True)
    p.add_argument("--measurement", action="store_true",
                   help="source is parent-independent; re-parent rather than sweep")
    p.set_defaults(func=cmd_enqueue)

    p = sub.add_parser("status", help="campaign leaderboard and recent nodes")
    p.add_argument("--limit", type=int, default=15)
    p.set_defaults(func=cmd_status)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
