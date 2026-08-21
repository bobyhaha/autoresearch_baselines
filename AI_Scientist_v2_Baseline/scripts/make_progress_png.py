#!/usr/bin/env python3
"""Render the campaign in the style of karpathy/autoresearch's analysis.ipynb.

The plotting conventions here are lifted from that notebook's progress cell so this
campaign's chart is directly comparable to the reference progress.png: grey dots for
discarded runs, green dots for kept ones, a green `where="post"` step line for the
running best, each kept point annotated with its description at 30 degrees, and
y-limits set from the best/baseline span.

Deviation, deliberate: the notebook clips the scatter to val_bpb <= baseline + 0.0005.
That hides runs that came in worse than baseline. Those are kept here as open markers
pinned to the top edge with a count, because in this campaign the large losses are what
closed off whole axes and they should not silently vanish from the record.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = Path.home() / "ai_scientist_v2_baseline"

df = pd.read_csv(BASE / "campaign" / "results.tsv", sep="\t")
df["val_bpb"] = pd.to_numeric(df["val_bpb"], errors="coerce")
df["memory_gb"] = pd.to_numeric(df["memory_gb"], errors="coerce")
df["status"] = df["status"].str.strip().str.upper()

fig, ax = plt.subplots(figsize=(16, 8))

valid = df[df["status"] != "CRASH"].copy().reset_index(drop=True)
baseline_bpb = valid.loc[0, "val_bpb"]

cutoff = baseline_bpb + 0.0005
below = valid[valid["val_bpb"] <= cutoff]
above = valid[valid["val_bpb"] > cutoff]

disc = below[below["status"] == "DISCARD"]
ax.scatter(disc.index, disc["val_bpb"], c="#cccccc", s=12, alpha=0.5, zorder=2, label="Discarded")

kept_v = below[below["status"] == "KEEP"]
ax.scatter(kept_v.index, kept_v["val_bpb"], c="#2ecc71", s=50, zorder=4,
           label="Kept", edgecolors="black", linewidths=0.5)

kept_mask = valid["status"] == "KEEP"
kept_idx = valid.index[kept_mask]
kept_bpb = valid.loc[kept_mask, "val_bpb"]
running_min = kept_bpb.cummin()
ax.step(kept_idx, running_min, where="post", color="#27ae60", linewidth=2,
        alpha=0.7, zorder=3, label="Running best")

for idx, bpb in zip(kept_idx, kept_bpb):
    desc = str(valid.loc[idx, "description"]).strip()
    if len(desc) > 45:
        desc = desc[:42] + "..."
    ax.annotate(desc, (idx, bpb), textcoords="offset points", xytext=(6, 6),
                fontsize=8.0, color="#1a7a3a", alpha=0.9, rotation=30, ha="left", va="bottom")

best_bpb = kept_bpb.min()
margin = (baseline_bpb - best_bpb) * 0.15
ax.set_ylim(best_bpb - margin, baseline_bpb + margin)

# Losses worse than baseline, pinned to the top edge rather than dropped.
if len(above):
    top = baseline_bpb + margin
    ax.scatter(above.index, [top - margin * 0.06] * len(above), marker="v", s=34,
               facecolors="none", edgecolors="#b0b0b0", linewidths=1.0, zorder=2,
               label=f"Worse than baseline ({len(above)}, off scale)")

n_total, n_kept = len(df), int((df["status"] == "KEEP").sum())
ax.set_xlabel("Experiment #", fontsize=12)
ax.set_ylabel("Validation BPB (lower is better)", fontsize=12)
ax.set_title(f"Autoresearch Progress: {n_total} Experiments, {n_kept} Kept Improvements", fontsize=14)
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.2)

plt.tight_layout()
out = BASE / "campaign" / "progress.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
print(f"baseline {baseline_bpb:.6f}  best {best_bpb:.6f}  "
      f"improvement {baseline_bpb - best_bpb:.6f} ({(baseline_bpb-best_bpb)/baseline_bpb*100:.2f}%)")
