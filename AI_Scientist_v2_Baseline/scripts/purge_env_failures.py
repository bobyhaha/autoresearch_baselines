#!/usr/bin/env python3
"""Remove trials that failed for environmental reasons rather than on their merits.

A foreign tenant took 104GB of the pinned GPU mid-campaign and two trials OOM'd. Those
are not results: the levers were never actually evaluated. Leaving them in the journal
does active harm, because the search's debug branch would try to *fix* them — and the
only way to fix an OOM is to shrink the model, which would then be attributed to the
lever under test and silently corrupt the record.

Removed nodes are archived, not discarded, so the incident stays auditable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path.home() / "ai_scientist_v2_baseline"
JOURNAL = BASE / "campaign" / "journal.json"
ARCHIVE = BASE / "campaign" / "env_failures.json"

victims = set(sys.argv[1:])
if not victims:
    raise SystemExit("usage: purge_env_failures.py <node_id> [<node_id> ...]")

data = json.loads(JOURNAL.read_text(encoding="utf-8"))
nodes = data["nodes"]

removed = [n for n in nodes if n["id"] in victims]
if len(removed) != len(victims):
    found = {n["id"] for n in removed}
    raise SystemExit(f"not found in journal: {sorted(victims - found)}")
for n in removed:
    if n.get("children"):
        raise SystemExit(f"refusing to remove {n['id']}: it has children {n['children']}")

kept = [n for n in nodes if n["id"] not in victims]
# Drop the removed ids from any surviving parent's children list.
for n in kept:
    if n.get("children"):
        n["children"] = [c for c in n["children"] if c not in victims]
# Renumber so step indices stay contiguous.
for i, n in enumerate(kept):
    n["step"] = i

archive = json.loads(ARCHIVE.read_text(encoding="utf-8")) if ARCHIVE.exists() else {"nodes": []}
archive["nodes"].extend(removed)
ARCHIVE.write_text(json.dumps(archive, indent=2), encoding="utf-8")

data["nodes"] = kept
tmp = JOURNAL.with_suffix(".json.tmp")
tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
tmp.replace(JOURNAL)

for n in removed:
    print(f"archived {n['id']}: {(n.get('plan') or '')[:70]}")
print(f"journal now {len(kept)} nodes (was {len(nodes)})")
