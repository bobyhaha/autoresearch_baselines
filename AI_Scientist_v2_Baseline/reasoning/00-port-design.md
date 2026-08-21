# 00 — Port design: AI-Scientist-v2 → autoresearch

## The gap between the two systems

AI-Scientist-v2 is built for ML research tasks where the agent writes a *whole
solution program* from scratch, runs it, and an LLM reads the stdout to work out how
well it did. Karpathy's autoresearch task is narrower and much more tightly
specified: there is one file to edit (`train.py`), one metric (`val_bpb`), one fixed
budget (300s of training wall clock), and a read-only harness (`prepare.py`) that
defines ground truth.

That mismatch drove every adaptation below.

## Decision 1 — keep the tree, replace the executor

**Kept:** `Node`/`Journal` (the solution forest), `MetricValue` (compares by *better*,
not larger), and the best-first search policy from `_select_parallel_nodes` — draft
`num_drafts` roots, then each iteration either debug a buggy leaf with probability
`debug_prob` (bounded by `max_debug_depth`) or improve the current best node.

**Replaced:** the interpreter. A node here *is* a complete `train.py`, executed in its
own trial workspace on a pinned GPU.

**Why:** the search policy is the valuable part of AI-Scientist-v2 and it is task
agnostic. The executor is entirely task-specific and had to go.

## Decision 2 — parse the metric, do not ask a model for it

Upstream calls an LLM to read program output and report the metric. Here `train.py`
always ends with a fixed summary block and `prepare.evaluate_bpb` is ground truth, so
the metric is pulled out with an anchored regex.

**Why:** an LLM parse step can hallucinate a score. When the score *is* the objective
being optimized, a hallucinated parse doesn't just add noise — it corrupts the search,
because a fabricated good score becomes the node the whole tree then builds on. Given
an exact contract exists, using it is strictly better. This also removes an LLM call
from every iteration, which matters when the LLM is a human-paced rendezvous.

## Decision 3 — audit runs before scoring them (the port's main addition)

`Agent._audit` marks a trial invalid, rather than scoring it, if either:

- `prepare.py` no longer matches the pristine checkout by sha256, or
- `training_seconds < 295`.

**Why:** the agent may rewrite `train.py` freely, and `train.py` both imports the
evaluation harness and controls the training loop. There are exactly two ways a
candidate could post a `val_bpb` that isn't comparable to the others: edit the harness,
or train for less than the fixed budget. Neither would look like a bug — both would
look like a win, and the greedy search would then build on it. The audit closes both.

This is a guard against accidental invalidity as much as anything adversarial: a
candidate that restructures the loop could plausibly break the time accounting by
mistake, and without the audit that would silently become the new best node.

## Decision 4 — one worker

The campaign owns one GPU, so `_select_node` returns a single node. Upstream's
`processed_trees` bookkeeping existed only to stop N parallel workers piling onto the
same tree in one step, so it drops out. The draft/debug/improve policy is untouched.

## Decision 5 — the LLM backend is a rendezvous, plus a queue

There is no API key here and no second interface: the Claude session driving the
campaign *is* the coding model. So `query()` publishes a request and blocks.

Blocking alone would idle the GPU for the entire time the agent spends thinking. At
~6 minutes per trial and ~240 trials in 24 hours, even 90 seconds of thinking per
request is ~6 hours of dead GPU. So candidates can be **pre-authored into a queue keyed
by parent node id** and popped without blocking.

The queue is deliberately kept shallow (a few per plausible parent). A queue miss is
not a failure mode — it is the pacing mechanism that wakes the agent to look at
results and think again. A very deep queue would run the campaign on stale ideas.

## Decision 6 — candidates are built by exact-string substitution, not by rewriting

`scripts/mkvar.py` derives a candidate from a parent by exact-string replacement, and
**errors if a pattern does not match exactly once**, then syntax-checks the result.

**Why:** two reasons. First, `train.py` is 26KB; moving it through the agent's context
on every one of ~240 iterations is wasteful and lossy. Second, a substitution that
silently matches zero times would enqueue a "change" identical to its parent, and the
search would then attribute a pure noise difference to a lever that was never applied.
Failing loudly is what keeps the experiment log meaningful.
