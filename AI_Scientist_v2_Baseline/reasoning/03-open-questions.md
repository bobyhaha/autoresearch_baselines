# 03 — Open questions and known weaknesses

Things that could invalidate conclusions, kept visible rather than buried.

## 1. n=1 per configuration; the search will chase noise

Every configuration is measured once. The residual spread around the step law is the
only noise estimate available, and early on it reads σ ≈ 0.0012 — meaning differences
below roughly 0.0024 bpb are not resolvable.

The gains so far (−0.0096 cumulative) are comfortably clear of that. But the search is
greedy on the single best node, so once the large levers are exhausted it will start
promoting noise into the tree and building on it. Two mitigations worth spending
trials on later:

- **Replicate the current best** a few times to measure run-to-run σ directly, rather
  than inferring it from a regression residual across heterogeneous changes.
- **Prefer structural levers with large expected effects** over knob tweaks whose
  expected effect is below the noise floor. A knob whose plausible effect is 0.001 is
  not measurable here at n=1 and the trial is better spent elsewhere.

## 2. Step count varies run to run even for identical code

The budget is wall clock, so timing jitter changes how many steps fit. Two identical
configurations can differ in `num_steps`, which feeds straight into `val_bpb`. This is
a real variance source, not measurement error, and it is part of why n=1 comparisons
are shaky.

## 3. The box is shared

Other tenants are on other GPUs. Contention can move throughput, and throughput moves
step count, which moves the metric. GPU 2 is pinned and exclusive to this campaign,
which limits but does not eliminate the exposure (host-level contention on CPU, PCIe
and the input pipeline is still shared). A throughput "win" measured while neighbours
happened to be idle would not reproduce.

## 4. The step law is not yet a valid control variate

See `01-regime-analysis.md`. Fitted across interventions of different kinds; treat the
slope as indicative only until there are enough same-kind points to fit it cleanly.

## 5. Greedy best-first may be the wrong search for this budget

The policy always improves the single global best. With ~240 trials and a noise floor
this close to the effect sizes, a policy that spent some budget re-measuring the top
few candidates might end up at a better final answer than one that always pushes the
frontier. This is a deliberate fidelity choice — the policy is AI-Scientist-v2's and
was kept unchanged — but it is worth naming as a limitation rather than a given.
