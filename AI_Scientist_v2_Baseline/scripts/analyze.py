#!/usr/bin/env python3
"""Campaign analysis: separate real gains from step-count effects and run-to-run noise.

## Why this is not a single regression

The first version of this tool fitted `val_bpb ~ a + b*ln(num_steps)` across *all*
scored trials. That broke as soon as a capacity-changing trial landed: DEPTH 8->6 sat
0.033 off the line and dragged the fitted slope positive, i.e. "more steps make things
worse", which is backwards. The model was misspecified — step count is only the
dominant explanatory variable when model capacity is held fixed.

So the fit is now restricted to a single capacity group (trials sharing a parameter
count), and cross-capacity trials are reported separately rather than pooled.

## Noise estimation

The residual spread of a regression across *different interventions* is not a noise
estimate — it is mostly real between-intervention variation. The only honest estimate
of run-to-run noise is repeated runs of identical code, so replicates are detected by
hashing each node's source and reported separately when they exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

BASE = Path.home() / "ai_scientist_v2_baseline"


def fit_log_law(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Ordinary least squares of y on log(x). Returns (intercept, slope)."""
    lx = [math.log(x) for x in xs]
    n = len(lx)
    mx = sum(lx) / n
    my = sum(ys) / n
    sxx = sum((v - mx) ** 2 for v in lx)
    if sxx == 0:
        return my, 0.0
    sxy = sum((lx[i] - mx) * (ys[i] - my) for i in range(n))
    slope = sxy / sxx
    return my - slope * mx, slope


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", default=str(BASE / "campaign" / "journal.json"))
    args = ap.parse_args()

    nodes = json.loads(Path(args.journal).read_text(encoding="utf-8"))["nodes"]
    scored = [
        n
        for n in nodes
        if (n.get("metric") or {}).get("value") is not None
        and (n.get("summary") or {}).get("num_steps")
    ]
    if not scored:
        print("no scored trials yet")
        return 0

    for n in scored:
        n["_bpb"] = n["metric"]["value"]
        n["_steps"] = n["summary"]["num_steps"]
        n["_params"] = round(n["summary"].get("num_params_M") or 0, 1)
        n["_hash"] = hashlib.sha256((n.get("code") or "").encode()).hexdigest()[:12]

    # ---- replicate-based noise estimate (the only trustworthy one) ----
    by_code: dict[str, list] = defaultdict(list)
    for n in scored:
        by_code[n["_hash"]].append(n)
    reps = {h: v for h, v in by_code.items() if len(v) > 1}
    print("=" * 74)
    if reps:
        # Pool across groups. A single group's sd is a 1-2 dof estimate and swings wildly
        # (observed range here: 0.000019 to 0.000502, a 26x spread). Quoting whichever
        # group happens to be at hand understates or overstates the floor depending on
        # which one it is. The pooled estimate is sqrt(sum(SS) / sum(dof)) and is what
        # every significance claim in the log should be judged against.
        ss = dof = 0.0
        for group in reps.values():
            vals = [g["_bpb"] for g in group]
            mu = sum(vals) / len(vals)
            ss += sum((v - mu) ** 2 for v in vals)
            dof += len(vals) - 1
        pooled = math.sqrt(ss / dof) if dof else float("nan")
        print(f"POOLED NOISE ESTIMATE over {len(reps)} groups, {int(dof)} dof: "
              f"sd = {pooled:.6f}")
        print(f"  => 2-sigma = {2*pooled:.5f}; treat smaller differences as unresolved at n=1")
        print(f"  => a 0.001 effect is {0.001/pooled:.1f} sigma\n")
        print("PER-GROUP (each is a 1-2 dof estimate; do not quote these individually)")
        for h, group in reps.items():
            vals = [g["_bpb"] for g in group]
            steps = [g["_steps"] for g in group]
            mean = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1))
            print(f"  {h}  n={len(vals)}  mean={mean:.6f}  sd={sd:.6f}")
            print(f"      val_bpb: {', '.join(f'{v:.6f}' for v in vals)}")
            print(f"      steps:   {', '.join(f'{s:.0f}' for s in steps)}")
            print(f"      => differences below ~{2 * sd:.5f} bpb are not resolvable at n=1")
    else:
        print("REPLICATES: none yet. Run the current best 2-3x to measure run-to-run")
        print("  noise directly. Until then every 'win' smaller than a few thousandths")
        print("  of a bpb is unverified.")
    print("=" * 74)

    # ---- step law, fitted within one capacity group only ----
    groups: dict[float, list] = defaultdict(list)
    for n in scored:
        groups[n["_params"]].append(n)
    main_params, main_group = max(groups.items(), key=lambda kv: len(kv[1]))

    print(f"\ncapacity groups (num_params_M): "
          f"{', '.join(f'{p}M x{len(g)}' for p, g in sorted(groups.items()))}")
    print(f"fitting the step law within the {main_params}M group only "
          f"({len(main_group)} trials); capacity changes are NOT comparable this way.\n")

    if len(main_group) >= 3:
        a, b = fit_log_law([n["_steps"] for n in main_group], [n["_bpb"] for n in main_group])
        print(f"step law ({main_params}M): val_bpb = {a:.6f} + ({b:.6f}) * ln(num_steps)")
        print(f"  an e-fold more steps is worth {b:+.5f} bpb")
        if b > 0:
            print("  WARNING: slope is positive, which is backwards. The group still mixes")
            print("  intervention kinds (batch size changes optimizer dynamics, not just")
            print("  step count). Treat this fit as uninformative.")
        resids = [(n["_bpb"] - (a + b * math.log(n["_steps"])), n) for n in main_group]
        rs = math.sqrt(sum(r * r for r, _ in resids) / max(1, len(resids) - 2))
        print(f"  residual spread: {rs:.6f}  (between-intervention variation, NOT noise)\n")
    else:
        print(f"only {len(main_group)} trials in the main group; need >=3 to fit\n")

    # ---- leaderboard ----
    print(f"{'id':<10}{'params':>8}{'steps':>7}{'mfu':>7}{'val_bpb':>11}  plan")
    for n in sorted(scored, key=lambda n: n["_bpb"]):
        print(
            f"{n['id']:<10}{n['_params']:>7.1f}M{n['_steps']:>7.0f}"
            f"{(n['summary'].get('mfu_percent') or 0):>7.1f}{n['_bpb']:>11.6f}  "
            f"{(n.get('plan') or '')[:48]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
