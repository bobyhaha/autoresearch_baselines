# 01 — What kind of problem is this, actually?

Written before proposing any lever, because the answer determines which levers are
worth spending trials on.

## Observation: the run does not repeat data

The baseline log still reads `epoch: 1` at step 1009 and only ticks to `epoch: 2` on
the final step. So a 300s run makes **roughly one pass** over the training shards and
essentially never sees a token twice.

## Why that matters

It rules a whole family of ideas out of contention. Anything whose mechanism is
"handle repeated data better" — dropout and other regularization, data ordering or
reshuffling tricks, anti-memorization measures — has almost nothing to act on here.
Spending trials on them would be spending them on a problem the task does not have.

It also means the run is **compute-bound**: quality is limited by how much useful
optimization fits in 300 seconds. That promotes two axes:

1. **Throughput** — cheaper steps buy more steps. Baseline MFU is 42.5% against the
   989.5 TFLOPS bf16 peak the script assumes, so there is real headroom.
2. **Optimizer cadence and quality** — more useful progress per step.

## The confound this creates, and why it needs handling

Because the budget is wall clock and not step count, *every* lever moves `num_steps`.
More steps lower `val_bpb` on their own. So a raw `val_bpb` comparison silently
conflates two different claims:

- "this change made the model better per step", and
- "this change made steps cheaper, so there were more of them".

Both are legitimate wins for this task — the metric is the metric. But they are not
the same finding, and they do not extrapolate the same way: a throughput win saturates
when the bottleneck moves, while a per-step quality win usually stacks.

`scripts/analyze.py` fits `val_bpb ~ a + b·ln(num_steps)` across all scored trials and
reports residuals, to tell the two apart. A change that only bought steps lands on the
line; one that improved the model lands below it.

## Caveat on the step law (recorded early, deliberately)

The law is fitted across interventions that differ in kind. Halving `TOTAL_BATCH_SIZE`
doubles step count *and* changes optimizer dynamics, so its point carries both effects
and the fitted slope absorbs them. Early fits will be unreliable. It is a diagnostic
for spotting which trials are step-driven, not yet a calibrated control variate.
