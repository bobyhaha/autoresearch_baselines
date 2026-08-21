#!/usr/bin/env python3
"""Seed the journal with the already-measured pristine baseline as root node 0.

The task's protocol says the first run establishes the baseline. That run is done, so
rather than spend another 6 minutes reproducing it, its result is imported as the first
draft node — the root every later improvement descends from.
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE = Path.home() / "ai_scientist_v2_baseline"
sys.path.insert(0, str(BASE))

from ai_scientist_ar.interpreter import parse_run_log  # noqa: E402
from ai_scientist_ar.journal import Journal, Node  # noqa: E402
from ai_scientist_ar.metric import MetricValue, WorstMetricValue  # noqa: E402

trial = BASE / "trials" / "baseline"
journal_path = BASE / "campaign" / "journal.json"
journal_path.parent.mkdir(parents=True, exist_ok=True)

journal = Journal.load(journal_path)
if any(n.plan.startswith("baseline") for n in journal.nodes):
    print("baseline already seeded; nothing to do")
    raise SystemExit(0)

text = (trial / "run.log").read_text(encoding="utf-8", errors="replace")
summary, exit_code, exc_type = parse_run_log(text)
if "val_bpb" not in summary:
    raise SystemExit("baseline run.log has no val_bpb — refusing to seed")

node = Node(
    plan="baseline (pristine karpathy/autoresearch train.py, unmodified)",
    code=(trial / "train.py").read_text(encoding="utf-8"),
    parent=None,
)
node.trial_dir = str(trial)
node._term_out = text
node.exec_time = summary.get("total_seconds", 0.0)
node.exit_code = exit_code
node.exc_type = exc_type
node.summary = summary
node.is_buggy = False
node.metric = MetricValue(summary["val_bpb"], maximize=False, name="val_bpb")
node.analysis = (
    f"val_bpb={summary['val_bpb']:.6f} steps={summary.get('num_steps')} "
    f"mfu={summary.get('mfu_percent')}% vram={summary.get('peak_vram_mb', 0)/1024:.1f}GB "
    f"params={summary.get('num_params_M')}M"
)

journal.append(node)
journal.save(journal_path)
print(f"seeded baseline as node {node.id} step {node.step}: {node.analysis}")
