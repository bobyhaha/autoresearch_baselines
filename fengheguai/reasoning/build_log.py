"""Generate the 30-minute reasoning log from the campaign ledger.

Each node's hypothesis, change summary, expected effect and risk were written by
the researcher *before* the measurement and are recorded verbatim in the
append-only ledger. This script slices those records into 30-minute windows so the
reasoning chain can be read in time order rather than by trial id.

Entries produced by this script are RECONSTRUCTED from the ledger. Contemporaneous
commentary added while the session was live is kept in NOTES-*.md alongside them
and is never overwritten.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

LEDGER = Path("campaigns/h200-claude-ledger.jsonl")
OUT = Path("reasoning")
WINDOW = dt.timedelta(minutes=30)


def load():
    events = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    return events


def fmt(v, nd=6):
    return "—" if v is None else f"{v:.{nd}f}"


def main() -> int:
    events = load()
    start = dt.datetime.fromisoformat(events[0]["timestamp"]).replace(second=0, microsecond=0)
    start -= dt.timedelta(minutes=start.minute % 30)
    end = dt.datetime.fromisoformat(events[-1]["timestamp"])

    completed = [e for e in events if e["kind"] == "trial_completed"]
    best_at = {}
    best = None
    for e in completed:
        m = e["payload"].get("metric")
        if m is not None and (best is None or m < best):
            best = m
        best_at[e["payload"]["trial_id"]] = best

    written = 0
    w = start
    idx = 0
    while w < end:
        nxt = w + WINDOW
        inwin = [e for e in completed
                 if w <= dt.datetime.fromisoformat(e["timestamp"]) < nxt]
        idx += 1
        path = OUT / f"{w:%Y%m%d-%H%M}Z.md"
        if not inwin:
            w = nxt
            continue
        lines = [
            f"# Window {idx:02d} · {w:%Y-%m-%d %H:%M}–{nxt:%H:%M} UTC",
            "",
            "> Reconstructed from `ledger.jsonl`. Every hypothesis and risk below was",
            "> written before its measurement and is quoted verbatim from the record.",
            "",
            f"Trials completed in window: **{len(inwin)}**",
            "",
        ]
        for e in inwin:
            p = e["payload"]
            prop = p.get("proposal") or {}
            tid = p["trial_id"]
            lines += [
                f"## {tid} · {p.get('stage','')} · {p.get('status','')} · val_bpb {fmt(p.get('metric'))}",
                "",
                f"- **best after:** {fmt(best_at.get(tid))}"
                + ("  ← new champion" if p.get("promoted") else ""),
                "",
                "**Hypothesis (pre-registered).** " + (prop.get("hypothesis") or "—"),
                "",
                "**Change.** " + (prop.get("change_summary") or "—"),
                "",
                "**Predicted effect.** " + (prop.get("expected_val_bpb_effect") or "—"),
                "",
                "**Stated risk.** " + (prop.get("risk") or "—"),
                "",
            ]
        path.write_text("\n".join(lines), encoding="utf-8")
        written += 1
        w = nxt

    print(f"wrote {written} window files to {OUT}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
