# 04 — The port vs. AI-Scientist-v2, and what the divergences cost

Written after re-reading the upstream source at `AI-Scientist-v2/ai_scientist/treesearch/`
with 95 trials of hindsight. `00-port-design.md` recorded the decisions when they were
made; this records how they turned out.

## What was ported faithfully

The search policy. `_select_parallel_nodes` → `Agent._select_node`: draft until
`num_drafts` roots exist, then with probability `debug_prob` debug a buggy leaf bounded by
`max_debug_depth`, else improve the best node. `Node`/`Journal` tree semantics —
`stage_name`, `debug_depth`, `is_leaf`, `good_nodes`/`buggy_nodes` — are unchanged.
Single worker instead of four is forced by one GPU and only drops the `processed_trees`
bookkeeping, which exists to keep parallel workers off the same tree.

## Divergence 1 — multi-seed evaluation, dropped. The most consequential omission.

Upstream has `_run_multi_seed_evaluation` (`parallel_agent.py:1261`), driven by
`multi_seed_eval.num_seeds: 3`. `agent_manager.py:743` calls it on the best node at the
end of **every stage**, then aggregates.

I dropped it as unnecessary. The entire campaign then fought the problem it solves:
Entry 4 caught the search promoting a 1σ result; Entry 8 added manual replicates *ad hoc*
after deferring them repeatedly; Entry 23 watched a 1.2σ "win" become the incumbent;
Entry 35 found a **byte-identical replicate** promoted as best on a 0.04σ difference.
Upstream ships a mechanism for exactly this and I did not port it.

Two qualifications, because "just port it" would have been wrong in two different ways:

**It would have silently done nothing here.** Upstream injects seeds by *prepending*
`torch.manual_seed(seed)` to the node's code. `train.py` sets `torch.manual_seed(42)` at
line 458, well after any prepended block — so all three "seeds" would have trained
identically and produced a spuriously tiny variance estimate. Porting it would have
required moving the injection to the existing seed line, not prepending.

**It measures a different quantity than my replicates, and the more relevant one.** My
replicates hold the seed at 42 and measure kernel nondeterminism plus step-count jitter:
pooled sd **0.000218** (Entry 35). Seed variance additionally covers initialisation and
data order, and is normally *larger*. So my noise floor probably **understates** the
variance that decides whether a lever generalises, and every σ figure in this log is an
upper bound on significance. That is a real caveat on the campaign's marginal results.

## Divergence 2 — `get_best_node`: upstream asks an LLM, I take the argmin

`journal.py:420` builds a prompt over candidate nodes and instructs the model to
"avoid relying too heavily on the validation loss alone". Pure `max(nodes, key=metric)`
is the `use_val_metric_only=True` branch — a non-default path.

My port is that non-default path, unconditionally. This is the direct cause of the
noise-chasing above: argmin has no notion of a confidence interval, so once effects fall
under the floor the frontier advances on chance.

I would still choose argmin for *this* task — `val_bpb` is exact, vocab-independent and
comparable across every candidate, which is precisely the situation upstream's warning is
not about, and an LLM judge adds hallucination risk and irreproducibility to a number
that is already ground truth. **But argmin then needs an explicit significance test, and
I never added one.** Taking the deterministic half of upstream's design without replacing
the guard it removed is the actual error, not the choice of argmin.

## Divergence 3 — the four-stage structure, collapsed into one loop

Upstream runs stage 1 (preliminary), 2 (hyperparameter tuning on the stage-1 best),
3 (research agenda), 4 (ablation), with `stage2_max_iters` etc. and multi-seed at each
boundary. `_select_parallel_nodes` branches on `stage_name` for stages 2 and 4.

I collapsed this to a single continuous improve loop, and then rediscovered the ordering
by hand and imperfectly: knobs were tuned at an operating point that later moved when
RoPE and the Adam-group LRs landed, which forced the interaction retests of Entries 31
and 33. Upstream's stage 2 is *literally* "tune hyperparameters on the best baseline" and
runs after stage 1 has settled the baseline — the discipline I ended up improvising.

## Divergence 4 — deterministic result parsing, and the hole it leaves

Upstream's `parse_exec_result` (`parallel_agent.py:683`) sends code and stdout to an LLM
which returns `is_buggy` and an analysis. I replaced it with a regex over the summary
block plus `_audit`.

I still think this is right — a hallucinated parse corrupts the objective itself, and the
task has an exact output contract. But it narrows "buggy" to what I anticipated, and I
anticipated two things: `prepare.py` modified (sha256) and training cut short
(`training_seconds < 295`).

**A concrete hole:** nothing stops a candidate calling
`make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "val")` and training on the
validation shard. `prepare.py` would hash clean, training would run the full 300s, and
`val_bpb` would come back spectacularly low. My audit passes it; upstream's LLM reviewer,
reading the diff and the output, would plausibly flag it.

Checked rather than assumed: **0 of 95 trials** deviate from the canonical
`make_dataloader(..., "train")` or `evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)`
calls. The record is clean. The guard still does not exist, and it should — the audit
should whitelist the dataloader split and the eval call, not just hash `prepare.py`.

## Divergence 5 — things dropped correctly

Plot generation, VLM feedback on plots, `journal2report`, data preview, and the
ablation-idea/hyperparam-idea generators. This task produces one scalar and no figures;
all of that machinery has nothing to act on.

## Summary of the issues, ranked

1. **No multi-seed evaluation.** Upstream had the answer to the campaign's central
   methodological problem; I removed it and reinvented a weaker version late.
2. **Argmin without a significance test.** I took upstream's deterministic option and
   dropped its compensating guard, producing a frontier that advances on noise.
3. **Noise floor probably understated.** Fixed-seed replicates omit initialisation and
   data-order variance, so every σ in this log is an upper bound.
4. **Audit is a whitelist of two anticipated failures**, not of the evaluation path. The
   dataloader-split hole is real though unexercised.
5. **No stage separation**, which cost the retests that Entries 31 and 33 had to run by
   hand.
