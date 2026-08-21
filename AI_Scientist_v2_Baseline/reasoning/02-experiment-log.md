# 02 — Experiment log

Hypothesis → result → interpretation, appended per check-in. Machine-readable results
live in `campaign/results.tsv` and `campaign/journal.json`; this file is the *why*.

Cadence: an entry every ~30 minutes for the duration of the 24-hour session.

---

## Entry 1 — 2026-08-21 12:24 CST — Baseline established

**Action.** Ran the pristine `train.py` unmodified on GPU 2, per the task protocol that
the first run establishes the baseline.

**Result.** `val_bpb = 0.991192` — 1017 steps, 42.52% MFU, 44.0GB peak, 50.3M params,
300.0s training / 348.8s wall.

**Read.** ~49s of startup and compile sits outside the measured budget, so a trial
costs ~350s of wall clock but only 300s counts. Cycle time, not budget, is what limits
how many experiments fit in 24 hours.

Seeded as root node `6d0ccd43` rather than re-run, since the measurement already
existed and a repeat would have cost 6 minutes for nothing.

---

## Entry 2 — 2026-08-21 12:34 CST — Choosing the three drafts

AI-Scientist-v2 builds `num_drafts` independent roots before it starts improving, so
the three drafts should be *orthogonal directions*, not variations on one idea.
Given the compute-bound single-epoch reading in `01-regime-analysis.md`, I picked one
per axis:

- **A — attention throughput.** Halve the SSSL short-attention span, seq/2 → seq/4.
- **B — optimizer cadence.** Halve `TOTAL_BATCH_SIZE`, 2¹⁹ → 2¹⁸.
- **C — model shape.** Held in reserve; the search reached the improve phase first.

**Disclosure of prior.** A and B are not blind guesses. Both directions had been
kept in an earlier campaign on this same baseline. I am using that as a prior over
*which levers to try first*, not as evidence — every claim in this log rests on the
measurement taken here.

---

## Entry 3 — 2026-08-21 12:57 CST — First four trials

| node | change | val_bpb | steps |
|---|---|---|---|
| `6d0ccd43` | baseline | 0.991192 | 1017 |
| `fbec943f` | short-attn span seq/2 → seq/4 | 0.988966 | 1059 |
| `d05761a3` | `TOTAL_BATCH_SIZE` 2¹⁹ → 2¹⁸ | 0.983545 | 2020 |
| `c89462a1` | both stacked | **0.981576** | 2099 |

**Interpretation.**

*A bought steps, not quality.* The span cut moved step count +4.1% and `val_bpb`
−0.00223. Against a step law of roughly −0.06 bpb per e-fold of steps, +4.1% predicts
about −0.0025 — essentially the whole observed gain. So the mechanism looks like pure
throughput: cheaper attention, more steps, no change in per-step learning. That is
still a win by the task's metric, but it predicts the lever saturates once attention
stops being the bottleneck.

*B is the big lever and it is not merely step count.* Halving the batch doubled
optimizer steps at constant token throughput and bought −0.00765. Same tokens, same
data, twice the parameter updates: this is optimizer cadence, not throughput.

*They stack.* Combined gives −0.0096, close to additive, consistent with the two
acting through different mechanisms.

**Decision.** Push the axis that paid most. Next trial halves the batch again
(2¹⁸ → 2¹⁷, with `DEVICE_BATCH_SIZE` 128 → 64 to keep `TOTAL_BATCH_SIZE %
tokens_per_fwdbwd == 0`). This is the informative experiment because it should find
the floor: below the critical batch size, gradient noise starts costing more than the
extra updates buy. Finding where B turns over matters more than another small win.

**Queued behind it** (so the GPU does not idle): span seq/8, `max-autotune-no-cudagraphs`
compile, and `WINDOW_PATTERN` SSSL → SSSSSSSL.

---

## Entry 4 — 2026-08-21 13:12 CST — The batch axis has a floor; and the first noise-level "win"

| node | change | val_bpb | steps | vs parent |
|---|---|---|---|---|
| `edd151d9` | batch 2¹⁸ → 2¹⁷ (`DEVICE_BATCH_SIZE` 128→64) | 0.987011 | 4046 | **+0.00544 worse** |
| `c002e4a2` | span seq/4 → seq/8 | 0.981310 | 2151 | −0.000266 |

### Finding 1: the batch lever turns over between 2¹⁸ and 2¹⁷ — a clean negative

Doubling the update count again (2099 → 4046 steps) made things *worse* by 0.0054,
which is well outside the noise floor. This is the predicted critical-batch-size
effect: at 131K tokens per update the gradient is too noisy for the extra updates to
pay for themselves.

This is the most valuable result so far even though it is a loss, because it *bounds*
the axis that produced the campaign's biggest win. 2¹⁸ is at or near the optimum, so
there is no point spending further trials pushing that direction — and the earlier
2¹⁹→2¹⁸ gain is now understood as landing near a peak rather than riding a trend.

Recorded so a later entry cannot mistake this for "smaller batch is always better".

### Finding 2: seq/8 is not a real improvement, and the search promoted it anyway

−0.000266 against a noise floor of roughly ±0.0024. This is indistinguishable from
zero. The change bought +52 steps (+2.5%), and the metric moved by about what step
count alone would explain.

The search nonetheless made `c002e4a2` the new best node, because best-first compares
raw metric values and has no notion of a confidence interval. This is exactly the
failure mode flagged in `03-open-questions.md` §1, now observed rather than predicted.

**Why I am not "fixing" it.** The search policy is AI-Scientist-v2's and keeping it
unmodified is the point of the port; bolting on significance testing would make this a
different algorithm. The honest handling is to keep the policy and *record* that the
frontier is now advancing partly on noise, so the final result is reported with that
caveat rather than as 6 clean wins. The practical cost is small here: `c002e4a2` and
`c89462a1` are near-identical in quality, so building on either is fine.

### Decision: stop tuning knobs, change the shape

Both throughput knobs (span) and cadence knobs (batch) are now at or past their useful
range. The remaining large lever is model shape.

Arithmetic that motivates it: the run sees ~564M tokens at 50.3M params ≈ **11 tokens
per parameter**. The compute-optimal ratio at fixed FLOPs is roughly 20. So the model
is over-parameterised for a 300s budget, and trading capacity for tokens should pay.

Next trial: `DEPTH` 8 → 6. Note `ASPECT_RATIO` rounding keeps `model_dim` at 512 for
depths 6–8, so this changes depth only, not width — a shallower model of the same
width, not a uniformly smaller one. Worth remembering when reading the result.

Queued behind it: `max-autotune-no-cudagraphs` compile (pure throughput, MFU has
drifted to 39.8%).

---

## Entry 5 — 2026-08-21 13:31 CST — My compute-optimal hypothesis was wrong

| node | change | val_bpb | steps | MFU | vs parent |
|---|---|---|---|---|---|
| `e28ccb85` | `DEPTH` 8 → 6 | 1.024066 | 2841 | 26.5% | **+0.0428 worse** |
| `c7c8d842` | `max-autotune-no-cudagraphs` | **0.980058** | 2204 | 40.8% | −0.00125 |

### Finding 3: the tokens-per-parameter argument does not hold here — falsified

Entry 4 reasoned that at ~11 tokens/param against a compute-optimal ~20, the model was
over-parameterised and trading capacity for tokens should pay. `DEPTH` 8→6 tested that
directly and lost by **0.043** — an order of magnitude larger than any effect the
campaign has produced in either direction, and far outside the noise floor.

The prediction was not merely unconfirmed; it was wrong in sign and enormous in
magnitude. Capacity is the binding constraint here, not token count.

Why the Chinchilla-style reasoning misfired is worth naming, since it will otherwise
tempt me again:

- The ratio is derived for a **fixed-FLOP** budget with a standard AdamW transformer.
  This setup is fixed-*wall-clock* with Muon, value embeddings, and windowed attention,
  and there is no reason the constant transfers.
- MFU **collapsed** 39.8% → 26.5%. A shallower model of unchanged width does less
  compute per kernel launch, so a large slice of the theoretical saving was eaten by
  overhead rather than converted into steps. Steps only rose 2151 → 2841 (+32%) for a
  25% capacity cut. The lever was much less efficient than the FLOP arithmetic implied.

**Correction to Entry 4:** the sentence "the model is over-parameterised for a 300s
budget" is retracted. The evidence says the opposite.

### Finding 4: max-autotune is a small, mechanistically coherent win

−0.00125 is below the ±0.0024 noise floor, so on its own it is not a distinguishable
result. But unlike the seq/8 case it comes with a consistent mechanism: MFU rose
39.84% → 40.83% and steps rose 2151 → 2204 (+2.5%), which is what a pure
kernel-selection improvement should look like. Model, data and hyperparameters are
untouched, so there is no quality channel for it to act through.

I am treating it as **probably real but individually unproven** — believable because
the mechanism is measured, not because the metric moved. Cost: compile time pushed the
iteration to 616s versus ~350s. VRAM rose to 48.5GB, still comfortable.

### Decision: bracket the shape axis in the other direction

Since shrinking hurt so much, the informative move is to grow. Next trial `DEPTH` 8 →
10, which via `ASPECT_RATIO` also widens `model_dim` 512 → 640 and `n_head` 4 → 5.

That confound is unavoidable through the `DEPTH` knob, so I queued
`ASPECT_RATIO` 64 → 80 behind it — width to 640 at unchanged depth 8. The pair
separates depth from width. It is queued on the *current* best rather than on the
depth-10 result, so that if depth-10 loses, the width probe runs automatically without
waiting on me.

---

## Entry 6 — 2026-08-21 13:35 CST — The analysis tool was wrong; two corrections

Scheduled 30-minute entry. No new trial finished (`fd76db94`, DEPTH 10, is running),
but re-running the analysis exposed two errors of mine that matter more than a result.

### Correction 1: the step law was misspecified and I had been quoting it

With 8 trials the fit returned a **positive** slope — "an e-fold more steps is worth
+0.0051 bpb", i.e. more training makes the model worse. That is backwards, and it is a
model-misspecification symptom, not a finding. `DEPTH` 8→6 sat 0.033 off the line and
dragged the whole fit.

The tool assumed step count is the dominant explanatory variable. That only holds with
**model capacity fixed**. Fixed now: the fit is restricted to a single parameter-count
group, and cross-capacity trials are reported separately instead of pooled. Within the
50.3M group the slope is −0.0046/e-fold, which is at least the right sign.

**What I had been quoting was wrong.** The "σ ≈ 0.0012 noise floor" in Entries 3–5 was
the residual spread of a regression across *different interventions*. That is mostly
real between-intervention variation, not noise. It was never a noise estimate, and I
should not have described it as one. The number happened to be small early on because
the first four trials were capacity-homogeneous, which made a broken method look fine.

### Correction 2: `DEPTH` 8→6 removed 48% of the parameters, not 25%

The capacity grouping surfaced it: 50.3M → **26.3M**. I had estimated ~25% from block
arithmetic and forgotten the value embeddings, which are allocated per layer and
dominate the count at this width. So Entry 5's "25% capacity cut" is wrong — the trial
removed nearly half the model.

That substantially changes how to read it. A +0.043 loss for halving the model is
unremarkable; it is not the dramatic refutation of the tokens-per-param argument that
Entry 5 implied. The argument is still not supported, but the evidence against it is
much weaker than I claimed, because the intervention was roughly twice as large as I
thought it was. It also means `DEPTH` 8→10 will add ~50% parameters, not ~25%.

### Action: buy a real noise estimate

Queued two **byte-identical replicates** of the current best. Identical source, so any
spread between them is pure run-to-run noise — and because the budget is wall clock,
step count genuinely varies between identical runs, so this variance is real rather
than measurement error.

Two trials (~12 minutes) out of a ~200-trial budget is cheap for the thing that decides
whether the last four "wins" (−0.00027, −0.00125) mean anything at all. Without it the
campaign cannot distinguish a result from a coin flip, and the greedy search will keep
promoting noise into the tree.

---

## Entry 7 — 2026-08-21 13:42 CST — Capacity is the dominant lever, and I had the sign backwards

`fd76db94` — `DEPTH` 10: **val_bpb 0.975832**, 85.9M params, 1325 steps, MFU 45.0%,
72.6GB peak. Against its parent (0.980058, 50.3M, 2204 steps, 40.8% MFU) that is
**−0.00423** — the largest single gain since the batch lever, and comfortably the
clearest structural result so far.

### Finding 5: on a wall-clock budget, bigger models get *more* effective compute

Line up the capacity ladder and the reason the Chinchilla-style argument failed becomes
obvious:

| params | MFU | steps | val_bpb |
|---|---|---|---|
| 26.3M (`DEPTH` 6) | 26.5% | 2841 | 1.024066 |
| 50.3M (`DEPTH` 8) | 40.8% | 2204 | 0.980058 |
| 85.9M (`DEPTH` 10) | **45.0%** | 1325 | **0.975832** |

**MFU rises monotonically with model size.** That breaks the premise the
tokens-per-parameter argument rests on. Chinchilla's ~20 tokens/param is the optimum
under a *fixed FLOP* budget. This budget is fixed **wall clock**, and wall clock buys
different amounts of compute depending on shape: a 26M model converts 26.5% of the
H200's peak into useful work, an 86M model converts 45%. Bigger models are not merely
spending a fixed FLOP budget differently — they are handed roughly 1.7x more actual
FLOPs for the same 300 seconds, because larger matmuls keep the GPU busy.

So the tokens/param heuristic was not just miscalibrated here, it was answering a
different question. At `DEPTH` 10 the run sees ~347M tokens at 85.9M params = **4.0
tokens/param**, five times further from Chinchilla than the baseline, and it wins.

**This supersedes Entries 4–6 on this subject.** Entry 4 predicted shrinking would pay;
that was wrong in sign. Entry 5 called `DEPTH` 6 a decisive refutation; Entry 6 rightly
weakened that because the intervention was twice as large as I thought. The correct
statement is the one here: capacity pays, and it pays partly through GPU utilization
rather than through statistics.

### Cost to watch: VRAM

44GB (baseline) → 72.6GB (`DEPTH` 10). The task treats VRAM as a soft constraint —
"some increase is acceptable for meaningful val_bpb gains, but it should not blow up
dramatically". −0.0042 is a meaningful gain and 72.6/143GB is comfortable, so this is
within bounds. `DEPTH` 12 will likely land near 100GB, which is where I would start
calling it a blow-up rather than an increase. If `DEPTH` 12 wins on metric but costs
~100GB I will record the tradeoff explicitly rather than bank the win silently.

Iteration time has also grown: 502s for `DEPTH` 10 versus ~350s at baseline, from
larger compiles. That cuts the 24h trial budget from roughly 240 to roughly 170.

### Next

`DEPTH` 12 (model_dim 768, n_head 6) — find where trading steps for capacity stops
paying. The two replicates stay queued behind it on the current best, so if `DEPTH` 12
loses they run immediately and finally deliver the noise estimate; if it wins I
re-target them. Either way the noise number arrives the moment the search stalls,
which is when it is actually needed.

---

## Entry 8 — 2026-08-21 14:04 CST — The noise floor, measured at last; capacity axis peaks

Scheduled 30-minute entry. Two results, one of which resets how the whole campaign
should be read.

| node | change | val_bpb | params | steps | MFU |
|---|---|---|---|---|---|
| `3c3fa271` | `DEPTH` 10 → 12 | 0.979841 | 135.3M | 870 | 47.5% |
| `c4cbd164` | **replicate** of `fd76db94` | 0.976128 | 85.9M | 1322 | 44.9% |

### Finding 6: run-to-run noise is ~0.0003, not ~0.0012

Two byte-identical runs: **0.975832 vs 0.976128**, a difference of **0.000296**
(step counts 1325 vs 1322). A second replicate is running now for n=3, but the
first pair already puts the noise scale an order of magnitude below what I had been
asserting.

This retroactively settles several open calls, and not all in my favour:

- **max-autotune (−0.00125) is real.** I called it "probably real but individually
  unproven" on mechanism grounds. At this noise scale it is roughly 4σ. The mechanism
  reasoning happened to reach the right answer, but it was doing work that a correct
  noise estimate would have done directly and better.
- **seq/8 (−0.000266) is still not established.** It sits at about 1σ — genuinely
  indistinguishable. Entry 4's scepticism stands.
- **The bigger structural results were never in doubt** and remain so.

The deeper lesson is about method, not about these particular trials. From Entry 3
onward I quoted a noise floor derived from a regression residual, and it was wrong by
4x in the conservative direction. A single cheap experiment — run the same code twice —
would have settled it at any point. I deferred it repeatedly in favour of "more
informative" trials while continuing to lean on a number I had not measured. That was
the wrong ordering: the measurement that calibrates every other measurement should have
come first.

### Finding 7: capacity peaks at DEPTH 10

`DEPTH` 12 lost by +0.004 despite the highest MFU yet (47.5%). The utilization story
from Entry 7 keeps holding — bigger really does use the GPU better — but at 135M params
only 870 steps fit, and that is too few. So the curve is genuinely single-peaked in this
budget:

| params | steps | val_bpb |
|---|---|---|
| 26.3M | 2841 | 1.024066 |
| 50.3M | 2204 | 0.980058 |
| **85.9M** | **1325** | **0.975832** |
| 135.3M | 870 | 0.979841 |

`DEPTH` 10 is the optimum along this knob. Note the knob is coarse — `ASPECT_RATIO`
rounding means `DEPTH` 11 would jump to width 768, so there is no fine adjustment
available without decoupling depth from width.

### Strategy shift: knob tuning is now worth trials

Earlier I argued (`03-open-questions.md` §1) for preferring structural levers because
knob-sized effects would be lost in the noise. With σ ≈ 0.0003 that argument is void —
a 0.001 effect is ~3σ and perfectly measurable. Knob tuning is back on the table, and
the levers that were tuned for the *old* shape are the obvious targets.

Queued on the DEPTH 10 winner: `MATRIX_LR` 0.04→0.05, `TOTAL_BATCH_SIZE` 2¹⁸→2¹⁹
(critical batch size grows with model size, so the old optimum may have moved),
`WARMDOWN_RATIO` 0.5→0.35, `EMBEDDING_LR` 0.6→0.8. All four were tuned at model_dim
512 and 2000+ steps; the winner is 640 wide with 1325 steps.

---

## Entry 9 — 2026-08-21 14:34 CST — Noise pinned at n=3; two hypotheses of mine falsified

Scheduled 30-minute entry.

### The noise floor, finalised

Three byte-identical runs: 0.975832 / 0.976128 / 0.975999 — **sd = 0.000148**, step
counts 1325 / 1322 / 1322. So **differences below ~0.0003 are not resolvable at n=1**,
and a 0.001 effect is ~7σ. This is the number every claim in this log should be read
against, and it replaces the incorrect ~0.0012 quoted in Entries 3–5.

The tight spread also explains itself: step count barely moves between identical runs
(1325 vs 1322), so the wall-clock budget introduces much less variance than I feared in
`03-open-questions.md` §2. That concern was overstated.

### Finding 8: critical batch size did *not* grow with model size — falsified

`5a90b3aa` — `TOTAL_BATCH_SIZE` 2¹⁸ → 2¹⁹ at DEPTH 10: **0.987102** (668 steps), worse
by **+0.0113**. I predicted the optimum would move *up* with model size, on the standard
argument that larger models tolerate larger batches. It did not move at all.

2¹⁸ is now confirmed optimal at both 50.3M and 85.9M params, bracketed on both sides at
each size (2¹⁷ lost at 50.3M, 2¹⁹ lost at both). That is a well-established result and
the axis can be closed.

Why the prediction failed is worth noting: at a *fixed wall-clock* budget, doubling the
batch halves the step count outright (1325 → 668). Whatever tolerance the larger model
gains for batch noise, it cannot pay for losing half its updates. The critical-batch
argument is about sample efficiency per token; here the binding constraint is updates
per second.

### Finding 9: `MATRIX_LR` 0.04 was already tuned

`b219d0f7` — 0.04 → 0.05: **0.976931**, worse by +0.0011 (~7σ). Clearly worse, not
noise. Since the miss is upward, 0.03 is queued to bracket from below. My expectation
is that 0.04 is simply correct and this axis closes too — the baseline's defaults were
tuned by someone, and I should stop assuming they are stale just because the shape moved.

### Standing back: what is actually left

Closed axes (bracketed on both sides, results ≫ noise): batch size, depth. Likely
closing: Muon LR. What remains untested is **shape at fixed depth** and **attention
structure**, so the queue is now:

- `ASPECT_RATIO` 48 and 72 — a three-point width sweep at constant `DEPTH` 10
  (512 / 640 / 768). The `DEPTH` knob confounds depth with width via rounding; this
  separates them, and answers whether 85.9M is a *parameter* optimum or 640 is a
  *width* optimum. Those are different claims and I have been conflating them.
- `HEAD_DIM` 128 → 64 — 10 heads instead of 5 at identical parameter count and matmul
  cost. Pure expressivity change with no throughput channel.
- `short_window` seq/8 → seq/4 — the span was tuned at the old shape, and seq/8 was
  never actually established over seq/4 (1σ).
- `EMBEDDING_LR` 0.6 → 0.8 (already queued).

Six deep, roughly 50 minutes of GPU, so the search runs unattended through the next
cycle.

---

## Entry 10 — 2026-08-21 14:44 CST — Best knob result yet, and a tooling fix

### Finding 10: `EMBEDDING_LR` is a live axis — 0.6 → 0.8 wins 0.0018

`502959c2`: **val_bpb 0.974019**, against 0.975832 for its parent. That is −0.00181 at
sd 0.000148, roughly **12σ** — the clearest knob result of the campaign and second only
to the batch and depth levers overall.

This partly walks back the caution in Entry 9, where I said I should "stop assuming the
defaults are stale just because the shape moved". That was right about `MATRIX_LR` and
`TOTAL_BATCH_SIZE`, and wrong here. The distinction that seems to matter: the defaults
governing *optimizer geometry* (Muon LR, batch size) were already at their optimum and
did not move with shape, whereas the *Adam group* LRs on embeddings appear genuinely
under-tuned for a run with 1325 steps instead of the ~1000 the baseline had.

Reasoning behind the original probe holds up: with far fewer optimizer steps than a
long run, embeddings accumulate less total movement, so a higher rate compensates.
Queued 1.0 to bracket, and `UNEMBEDDING_LR` 0.004 → 0.006 on the same argument since
it is the paired Adam group.

### Finding 11: `WARMDOWN_RATIO` 0.5 is already correct

`8bed6ff2`: 0.5 → 0.35 gave 0.978169, worse by +0.0023 (~15σ). Closed.

### Tooling: levers are now stored as line replacements, not string pairs

A recurring waste: candidates are queued keyed by parent node id, so every time a queued
candidate *wins*, the best node moves and all other queued candidates become
unreachable — the search can never select their parent again. Five queued experiments
were orphaned when `502959c2` won, having cost real time to author.

Root cause was that candidates were built with exact old/new string pairs, which bind
to one specific parent's text. The library now stores each lever as a **whole-line
replacement matched by a unique prefix**, so `MATRIX_LR` can be queued regardless of
its current value, and `scripts/queue_levers.py` re-derives any lever against whatever
node is currently best. It also sweeps orphaned queue entries automatically.

This is worth the detour with ~19 hours left: the alternative is hand-rebuilding
several candidates on every frontier advance, which is exactly the sort of repeated
manual step that eventually gets done wrong.

Re-queued against `502959c2`: `head64`, `span4`, `mlr03`, `aspect72`, `aspect48`,
`unemb006`.

---

## Entry 11 — 2026-08-21 15:05 CST — Two axes closed; a reporting error corrected

Scheduled 30-minute entry.

| node | change | val_bpb | vs best | verdict |
|---|---|---|---|---|
| `8282a51e` | `EMBEDDING_LR` 0.8 → 1.0 | 0.974545 | +0.00053 (~3.5σ) | overshoot |
| `9fee3cf3` | `HEAD_DIM` 128 → 64 | 0.980724 | +0.00670 (~45σ) | clear loss |

### Correction: I reported queueing a lever I had not queued

In the previous status message I wrote that `EMBEDDING_LR` 0.9 had been queued to refine
the axis. It had not — I described the intended next step as though it were done, and
the queue at the time held only the four earlier levers. Queued now.

Recording it because the whole value of this log rests on it matching what actually
ran. A plan stated as an action is the kind of drift that makes a research record
untrustworthy, and it is worth more to flag it than to quietly fix it.

### Finding 12: `EMBEDDING_LR` optimum is near 0.8

Bracketed: 0.6 → 0.975832, 0.8 → 0.974019, 1.0 → 0.974545. Single-peaked, and the
overshoot at 1.0 is ~3.5σ so it is a real turn rather than noise. 0.9 is queued to
locate the peak more precisely, though the remaining headroom between 0.8 and 0.9 is
plausibly at the noise scale, and I should be willing to call the axis closed rather
than chase it.

### Finding 13: `HEAD_DIM` 64 is a large loss — and it was not the clean test I claimed

`9fee3cf3`: 0.980724, worse by 0.0067. I queued this as "constant FLOPs, same parameter
count, more independent attention subspaces" — a pure expressivity probe with no
throughput channel.

That description was wrong. MFU fell 44.9% → 43.1% and steps fell 1323 → 1270. Halving
the head dimension doubles the head count, and the resulting attention kernels are
narrower and less efficient, so the change *did* act through throughput as well as
expressivity. The two effects are confounded in this result, and I cannot separate them
from one trial. What can be said is only the practical conclusion: at this shape,
`HEAD_DIM` 128 is better, by a wide margin.

I have now twice described an intervention as isolating one mechanism when it did not
(the `DEPTH` 6 parameter count in Entry 5, this here). The pattern is asserting a
change's mechanism from its intent rather than checking the measured side effects in
the summary block. The summary reports steps, MFU, params and VRAM on every run —
checking them before writing the interpretation is cheap and I should do it every time.

### State

Closed axes: batch size, depth, Muon LR (pending the 0.03 bracket), warmdown ratio,
head dim. Best remains `502959c2` at **0.974019**, −1.73% on baseline, 8 kept of 18.

Queue is 8 deep on the current best: width sweep (512 / 768 at fixed DEPTH 10),
`MATRIX_LR` 0.03, `UNEMBEDDING_LR` 0.006, `EMBEDDING_LR` 0.9, `SCALAR_LR` 0.7,
`WEIGHT_DECAY` 0.1, `FINAL_LR_FRAC` 0.1. That is roughly 70 minutes of unattended GPU.

---

## Entry 12 — 2026-08-21 15:36 CST — I closed an axis on a one-sided test

Scheduled 30-minute entry. Three results, and a methodological error they exposed.

| node | change | val_bpb | vs best | verdict |
|---|---|---|---|---|
| `e6d38dce` | `MATRIX_LR` 0.04 → 0.03 | 0.974240 | +0.00022 | within noise — tie |
| `b220b327` | span seq/8 → seq/4 | 0.975628 | +0.00161 (~11σ) | seq/8 confirmed better |
| `aaa966e6` | `ASPECT_RATIO` 72 (width 768) | 0.977611 | +0.00359 (~24σ) | clear loss |

### The error: premature closure of `WARMDOWN_RATIO`

In Entry 9 I tested `WARMDOWN_RATIO` 0.5 → 0.35, saw it lose by 0.0023, and wrote
"`WARMDOWN_RATIO` 0.5 is already correct. Closed."

That does not follow. I tested **one** direction. A single losing probe below the
default rules out *lower*; it says nothing about higher. I then repeated the claim in
two status reports as an established result.

Compare how I handled the axes I actually did bracket — batch size (2¹⁷ and 2¹⁹ both
lose), `MATRIX_LR` (0.03 and 0.05 both fail to beat 0.04), `EMBEDDING_LR` (0.6 and 1.0
both lose to 0.8), depth (6 and 12 both lose to 10). Those closures are sound. Warmdown
was the one I closed on half the evidence, and I did not notice the asymmetry because
the losing result felt conclusive on its own.

`WARMDOWN_RATIO` 0.7 is now queued. **Entry 9's Finding 11 is retracted pending it.**

### Finding 14: seq/8 is now established, and the earlier caution was right

In Entry 4 I flagged seq/8 as a −0.000266 change the search had promoted despite it
being ~1σ — not a real result. Testing the reverse at the current shape settles it:
going back to seq/4 costs +0.0016 (~11σ). So seq/8 genuinely is better, and the axis is
live enough that seq/16 is now queued to bracket it from below.

Worth noting what this vindicates: not the original trial, which really was
uninformative, but the decision to *record* it as uninformative rather than bank it. The
claim only became true when a properly powered test was run.

### Finding 15: 640 is a width optimum, not just a parameter optimum

`ASPECT_RATIO` 72 at fixed `DEPTH` 10 gives model_dim 768, 114.8M params, and loses by
0.0036 despite the highest MFU in the 85.9M-class runs (47.6%). With `DEPTH` 12 (768
wide, 135.3M) also losing, and the narrow probe (512) still queued, the 640-wide /
10-layer point is looking robust rather than lucky.

`MATRIX_LR` is now properly closed: 0.03 and 0.05 both fail to beat 0.04, and the 0.03
gap (+0.00022) is inside the noise floor, so the default stands. Under the task's
simplicity criterion a tie goes to the unchanged code.

### Disclosure: using the reference repo's own results as a prior

`task/progress.png` in this workspace is the reference repo's published example run. It
reports which levers were kept there: warmdown *up* to 0.7, 5% warmup, unembedding LR
0.008, an SSSSL window pattern, RoPE base-frequency increases, and x0_lambda init 0.05.

I am using that as a prior over **which levers to try next**, exactly as with the Entry 2
disclosure — it orders the search, it is not evidence. Everything still has to win here
to count. Notably the campaign independently reproduced four of its findings (batch
halving, short window 1/4, short window 1/8, embedding LR 0.6→0.8 — the same values),
which is some evidence the two setups behave alike.

It also independently flags the warmdown error above: that chart lists "warmdown
0.5→0.7 (more cooldown helps)", the exact direction I failed to test.

Queued: `WARMDOWN_RATIO` 0.7, RoPE base 50000, `UNEMBEDDING_LR` 0.008, x0_lambda 0.05,
span seq/16, 5% warmup. Queue 11 deep, ~90 minutes unattended.

---

## Entry 13 — 2026-08-21 15:50 CST — New best 0.972938; a pattern in which knobs pay

`c3a62a53` — `UNEMBEDDING_LR` 0.004 → 0.006: **val_bpb 0.972938**, −0.00108 on the
previous best (~7σ). Cumulative **−0.0183 (−1.84%)** from baseline.

### Finding 16: width is bracketed — 640 is the optimum

The sweep at fixed `DEPTH` 10 is now complete on both sides:

| model_dim | params | steps | MFU | val_bpb |
|---|---|---|---|---|
| 512 | 60.8M | 1808 | 41.7% | 0.976064 |
| **640** | **85.9M** | **1323** | **44.9%** | **0.974019** |
| 768 | 114.8M | 1017 | 47.6% | 0.977611 |

Single-peaked, and the losses on both sides are ≫ noise. Together with the depth sweep
this makes 10 layers × 640 a genuine two-dimensional optimum rather than an artifact of
the coarse `DEPTH` knob. Shape is closed.

Note again that MFU rises monotonically with width (41.7 → 47.6%) while quality peaks in
the middle — the same decoupling as Entry 7. Utilization is not a proxy for quality
here, and optimising it directly would have led straight to the worst of these three.

### Finding 17: the Adam groups were under-tuned; the Muon geometry was not

Every knob result so far sorts cleanly along one line:

| knob | family | outcome |
|---|---|---|
| `EMBEDDING_LR` 0.6 → 0.8 | Adam | **−0.0018 win** |
| `UNEMBEDDING_LR` 0.004 → 0.006 | Adam | **−0.0011 win** |
| `MATRIX_LR` 0.03 / 0.05 | Muon | both fail to beat 0.04 |
| `TOTAL_BATCH_SIZE` 2¹⁷ / 2¹⁹ | optimizer geometry | both lose badly |
| `HEAD_DIM` 64 | architecture | −0.0067 loss |
| width, depth | architecture | interior optimum, both bracketed |

Two of two Adam-group LR probes won; zero of two Muon/batch probes did. A plausible
reading: the baseline's Muon and batch settings were tuned at roughly this scale and
transfer, whereas the Adam-group LRs were set for a run with ~1000 steps and are simply
too low for one with ~1320.

I want to be careful not to over-read a 2-of-2. It is a hypothesis that predicts
`SCALAR_LR` (the third Adam group) should also pay — which is queued, and will test it.
`UNEMBEDDING_LR` 0.008 is running now to bracket the current win from above.

### Tooling note: the re-targeting fix paid for itself

This frontier advance orphaned ten queued candidates. `queue_levers.py` cleared all ten
and re-derived them against the new best in a single command. Before Entry 10's fix that
would have been ten hand-built substitutions — the exact repetitive step where mistakes
accumulate. Queue is 10 deep again, ~80 minutes unattended.

---

## Entry 14 — 2026-08-21 15:58 CST — VRAM correction, and a watchdog on the real budget

### Correction: the VRAM figures I have been reporting understate the device footprint

`gpu_watch.sh` reported GPU 2 holding **104 GB** while the running trial's
`peak_vram_mb` reads **67.8 GB**. Both are right; they measure different things.

- `peak_vram_mb` comes from `torch.cuda.max_memory_allocated()` — live tensor bytes.
- `nvidia-smi` reports the process's whole device footprint: the caching allocator's
  reserved pool, the CUDA context, and compile/autotune workspaces on top of that.

I have been quoting the first number throughout — "44 GB baseline", "72.6 GB at DEPTH
10" — and reasoning about the task's soft VRAM constraint against it. That is the wrong
number for that purpose. What a neighbouring tenant on a shared box actually sees is the
footprint, and at 104/143 GB this campaign is holding about **73% of the card**, not the
47% the allocated figure suggests.

It does not change any decision taken so far: GPU 2 is exclusively this campaign's, and
nothing has come near OOM. But the headroom I described for `DEPTH` 12 in Entry 7 was
overstated — I said ~100 GB "is where I would start calling it a blow-up", not realising
the *current* configuration was already there by the measure that matters. Both numbers
are logged per sample from here on.

Confirmed not a config change: the running trial differs from its parent only in
`WARMDOWN_RATIO` 0.5 → 0.7, which cannot move memory.

### The watchdog

`scripts/gpu_watch.sh` now samples GPU 2 every 30 minutes for 24 hours, appending to
`campaign/gpu_usage.log`. Two design points worth recording:

**Drift-corrected timing.** The loop computes each next wake as an absolute
`time.time()` deadline rather than sleeping a fixed 1800s. Each iteration spends real
time in `nvidia-smi` and reading the journal, so a naive sleep loop would fall
progressively behind — on the order of a whole sample lost across 48 iterations.

**Owner-filtered occupancy.** The `IDLE` verdict requires *our own* trial processes to
be absent, not merely low utilisation. On a shared box "the GPU is busy" and "our work
is running" are different claims, and conflating them is how instrumentation ends up
reporting healthy while the campaign is actually stalled. The verdict distinguishes
TRAINING, STARTUP_OR_EVAL (our process alive but the GPU quiet — compile or eval), and
IDLE (nothing of ours running at all, which on this setup means the harness is blocked
waiting for code and budget is being burned).

The alert monitor is also now wrapped in a reconnect loop; the previous one died
silently when its ssh tail dropped. The 30-minute reasoning heartbeat runs locally and
serves as the backstop, so a dead remote monitor costs at most one cycle of latency
rather than going unnoticed.

---

## Entry 15 — 2026-08-21 16:15 CST — A foreign tenant took the GPU; two retractions

Scheduled 30-minute entry, dominated by an infrastructure incident.

### What happened

At ~15:54 another user (`yushanbin`, PID 3259082) began holding **104 GB of GPU 2's
143 GB**. Our trials had ~38 GB left, and two OOM'd immediately:

- `7037fb62` — `WARMDOWN_RATIO` 0.7, the trial correcting Entry 12's premature closure
- `088ab583` — RoPE base 10000 → 50000, the first genuinely untested structural probe

Neither is a result. The levers were never evaluated; the process died before training.

### Retraction 1: Entry 14's VRAM explanation was wrong

Entry 14 saw `nvidia-smi` reporting 104 GB against `peak_vram_mb` of 67.8 GB and
explained the gap as PyTorch's caching-allocator reserve plus CUDA context. I wrote a
whole correction on that basis, including a claim that "this campaign is holding ~73% of
the card".

That was wrong. The 104 GB was **not ours at all** — it was the other tenant's process.
Our own footprint was ~38 GB. The OOM traceback stated this plainly ("Process 3259082
has 101.60 GiB memory in use... this process has 38.04 GiB"), and I had the reading
15 minutes before the traceback without investigating who owned it.

The failure mode is worth naming: I saw an anomalous number, produced a *plausible*
mechanism for it, and wrote that mechanism up as a correction with some confidence.
A plausible explanation is not a verified one. The check that settled it —
`nvidia-smi --query-compute-apps=pid,used_memory` — takes one command, and I ran it only
after crashes forced the issue. This is the same pattern as Entries 5 and 11: asserting
mechanism from inference instead of measuring it. Third occurrence.

### Retraction 2: "GPU 2 is exclusively mine"

I stated this several times as justification for not worrying about contention, and
`03-open-questions.md` §3 downplayed shared-box exposure on that basis. It was true when
checked at 12:24 and I never rechecked. Exclusivity on a shared cluster is a fact with a
timestamp, not a property.

### Actions taken

**Halted the harness before it could corrupt the record.** The urgent risk was not the
lost trials — it was that the search's debug branch (`debug_prob` 0.5) would select those
buggy leaves and try to *fix* them. The only fix for an OOM is a smaller model, and that
change would then have been attributed to the lever under test. The `_audit` guard does
not catch this: shrinking a model is a legitimate edit, not an invalid one. So a
plausible next few trials were "`WARMDOWN_RATIO` 0.7 with a reduced batch, scores worse,
warmdown closed" — a wrong conclusion with no visible defect.

**Archived rather than kept.** `scripts/purge_env_failures.py` moves the two nodes to
`campaign/env_failures.json` and unlinks them from their parent. They are out of the
search but auditable. Both levers are re-queued for a fair test.

**Moved to GPU 0** (free, 3.7 GB) with the operator's agreement, since the alternatives
were waiting on a tenant who might hold the card for the remaining 20 hours, or shrinking
our footprint — which would have changed throughput and made all 24 existing trials
non-comparable. Same H200 model, so results carry over. The journal resumed from disk
intact.

**Killed only our own processes**, matched against our trials directory. The foreign
process was left alone; it is not ours to reap.

### Cost and state

About 20 minutes of GPU time and two trials. Best is unchanged at **0.972938 (−1.84%)**,
24 valid nodes. Queue rebuilt to 10 unique entries — deduplicated by code hash, since
re-queueing after the move would otherwise have run six experiments twice.

Watchdog restarted against GPU 0 for the remaining 20 hours. Its `IDLE`/`TRAINING`
verdicts are owner-filtered, which is exactly the distinction that would have caught this
incident earlier had the sampling started sooner.

---

## Entry 16 — 2026-08-21 16:36 CST — Post-move results are not yet trustworthy

Scheduled 30-minute entry.

### The first post-move result is anomalous, and I am not attributing it to its lever

`d20fd8aa` — attention span seq/8 → seq/16 — scored **0.986212**, worse by 0.0133. Taken
at face value that closes the span axis: seq/8 is the optimum, bracketed by seq/4 above
and seq/16 below.

But the supporting numbers do not fit that story. A *smaller* attention window is
strictly less work per step, so it should produce **more** steps. It produced fewer:
1317 → 1153, with MFU down 44.7% → 38.4%. A cheaper model that runs slower is not a
coherent lever effect.

Two candidate explanations. Either seq/16 = 128 tokens falls off an efficient FA3 kernel
path and genuinely runs slower, or **GPU 0 is not delivering what GPU 2 delivered** and
the deficit has nothing to do with the lever. This is the first trial after the hardware
move, so both are live and the trial cannot distinguish them.

### The check that should have been the first thing I ran after moving

Queued at the front: a **byte-identical replicate of `c3a62a53` on GPU 0**. Its GPU 2
value is 0.972938 with sd 0.000148 measured from three replicates, so this is a
well-powered test of the hardware change itself.

- Reproduces within noise → the move is clean, pre- and post-move trials are comparable,
  and the seq/16 result stands as a real lever effect.
- Does not reproduce → every post-move result is confounded by hardware, the campaign
  needs re-baselining on GPU 0, and no cross-move comparison means anything.

I should have queued this *before* resuming the search, not after taking a result from
it. Moving hardware mid-campaign invalidates the comparability that every conclusion
rests on, and the fix costs one trial. Instead I restarted straight into the lever queue
and got a result I now cannot interpret — which is a wasted trial either way, and would
have been a wrong conclusion had the numbers looked less obviously odd.

Relevant prior, which I had and did not apply: contention artifacts on this cluster have
previously produced a clean-looking "win" that was purely an artifact of which GPU it ran
on. The lesson was recorded; I did not act on it at the moment it applied.

### Watchdog fixes

Two, both found by reading its own output rather than assuming it worked:

**Malformed CSV.** `pgrep -fc` prints `0` *and* exits non-zero when nothing matches, so
the `|| echo 0` fallback appended a second zero and broke every row across a newline.
The first sample after the move was already corrupt.

**No visibility of foreign occupancy.** The incident in Entry 15 turned on how much of
the card was held by processes that are not ours — a quantity the watchdog did not
record, which is why it could not have warned about it. It now logs `foreign_mem_mb`
per sample. GPU 0 currently reads 2166 MiB foreign (three small processes), against
104047 MiB on GPU 2 at the time of the OOMs.

---
