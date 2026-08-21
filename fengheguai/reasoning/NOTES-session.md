# Session notes — the reasoning arc

Written live, unlike the window files. This is where the judgement lives, including
the parts that were wrong.

## 1. Getting a researcher onto the box at all

The campaign was specified to use `codex` as the implementation agent. The H200 box
cannot reach `chatgpt.com` — DNS resolves it to an unroutable address and every
connection fails — so codex could not run there, and the operator then ruled it out
entirely. `claude` is installed on the box but reports "Not logged in", and OAuth
needs a browser the box does not have.

The campaign's `config.json` is frozen at init and I am bound not to edit it, so
switching the agent command required a **new campaign** rather than a modification.
`campaigns/h200` (codex-configured, baseline 0.9911391) was left untouched;
`campaigns/h200-claude` was initialised alongside it.

`rendezvous_agent.py` is the adapter: Fengheguai launches it as the agent command,
it publishes the prompt to an inbox and blocks until a valid structured result
appears. That makes *this session* the researcher inside an otherwise unmodified
loop — audit, measurement and promotion all run exactly as designed. The protocol is
one-directional on purpose: **edit `train.py` first, write the result JSON last**,
because the result file appearing is what unblocks the controller.

## 2. What actually produced the gains

Not re-tuning the recipe. Of roughly eighteen inherited defaults I tested, exactly
one — the output softcap — was suboptimal. The recipe was well-tuned, and the wins
came from three other places:

- **Scaling laws applied where they belonged.** Batch/LR moved together; decomposing
  the bundle showed √-batch scaling belongs to the *Muon* group alone, while the
  Adam embedding group is batch-invariant.
- **Mechanisms the recipe lacked.** Document masking, dense value embeddings, a
  wider value-embedding gate, a fourth global attention layer.
- **Reading the source for mis-specifications.** The two largest late gains came
  from this, not from search — see §4.

## 3. Where I was wrong, and what it cost

- **t0009 / t0044** — my implementation bugs (`mark_dynamic` called inside a traced
  region; bf16 × fp32 promotion rejected by the FA3 kernel). Both repaired
  in-protocol by the debug stage with the hypothesis intact.
- **t0024** — I gave a zero-initialised scalar the aggressive `x0` rate (0.5) rather
  than the `resid` rate (0.005). Cost 0.0092. The lesson generalises: *a zero init is
  not "safe" if you hand it a rate meant for a parameter that starts at a sensible
  value.*
- **t0013** — right call, wrong reason. I argued throughput was sublinear in FLOPs
  and predicted a 10–15% step cost; it was 18%, essentially the FLOP ratio. Capacity
  paid because capacity is worth more than the steps it costs, not because the steps
  were cheap. Recorded rather than quietly re-narrated.
- **t0076 → t0077** — I told a tidy coupling story (the fourth global layer *enabled*
  the narrower span). Completing the 2×2 grid showed the two effects are
  **additive**, not coupled. The narrative was tidier than the data.
- **t0080** — the code was correct; my *description* of it in the record was not
  (scrambled layout, a false monotonicity claim). I had verified after submitting
  instead of before. The correction is in the ledger, in t0084's `change_summary`.
- **t0081** — I tried to record that correction as an extra JSON key. The controller
  keeps only four fields, so it was silently dropped. I asserted it had persisted
  before checking. Same error class as t0080: **assert after verifying, not before.**

## 4. The thread that paid best

Reading `init_weights` rather than scanning coordinates.

The recipe computes one scale, `s = √3·n_embd^-0.5`, and applies it both to the
projections (fan-in 512 — correct) and to the value-embedding *tables* (fan-in 1 —
wrong). `wte` correctly gets `normal(0,1)`, which is what showed the file already
distinguishes the two cases. Measured consequence: the most load-bearing mechanism
in the model started **22.6× weaker** than the stream it feeds.

That one observation unlocked a chain:

| trial | change | gain |
|---|---|---:|
| t0087 | init matched to the `v` stream (RMS 0.044 → 1.0) | 0.00086 |
| t0088 | doubled past parity | null — parity was the target |
| t0089 | decoupled that group's LR and halved it | 0.00089 |
| t0091 | gate width re-tested at the corrected scale | no change — 128 is the gate's own optimum |

t0089 is the one worth remembering. Every earlier measurement of the
value-embedding learning rate was taken while the tables were 22.6× too small — a
defect that *rewards a fast rate*, because the path has to fund its own growth. Fix
the init and the rate can be tuned for stability instead. Neither change alone
explains the pair.

## 5. Principles that survived contact with the data

- **Time-indexed constants survive a configuration change; cumulative ones do not.**
  Warmdown, warmup and the final-LR floor are driven by `progress` and never moved.
  Weight decay is applied per step as `lr·wd`, so halving the batch silently raised
  its total by 1.37× — correcting that by arithmetic gained 0.00035, and the derived
  value *was* the measured optimum.
- **Operating-point dependence is real but not universal.** It applied to a
  cumulative quantity and to a control governing a sensitive parameter; NorMuon's
  variance horizon measured null at both batch sizes.
- **Attention span and global-layer count are different axes.** Narrow local spans
  win; *more* global layers win. Conflating them is what made me mispredict t0073.
- **Read the ledger, not the reports.** A monitor event once reported a score and
  status that matched nothing in the record — reports are regenerated files and can
  be read mid-write. The hash-chained ledger is the only authority.

## 6. The precision thread (t0092–t0093)

Correcting the value-embedding init in t0087 raised those elements from RMS 0.044 to
1.0 — and bf16 resolution scales with magnitude, so the update-quantisation floor
went from an ulp of 0.00024 to 0.0078. My own fix made update rounding 32× coarser.

t0092 tested the obvious repair, holding the tables in fp32. The result splits
cleanly in two and is worth keeping for that reason:

| | champion t0089 | t0092 |
|---|---:|---:|
| loss @ step 1000 | 3.029 | **3.013** |
| dt @ step 1000 | 105 ms | **211 ms** |
| steps | 2875 | 1828 |
| val_bpb | 0.968389 | 0.996374 |

**Quality per step improved; throughput collapsed.** Tripling memory traffic on the
model's largest parameter block (50M params as fp32 weights plus fp32 moments) cost
36% of the run, and a worse-than-baseline final score. The hypothesis was right and
the implementation was wrong — those are different failures and the evidence
distinguishes them.

t0093 relocates the precision to where it is cheap: `_step_adamw` allocates moments
with `torch.zeros_like(p)`, so a bf16 parameter silently gets **bf16 moments**, and
the moments are where small updates have to survive between steps. Parameters stay
bf16; only the buffers go fp32. `num_steps` is the whole diagnostic — above ~2700
the placement worked, near 2000 and the direction is closed.

### A process note

The first attempt at the t0093 edit failed its own assertion: `state['exp_avg']`
contains single quotes, which terminated the single-quoted ssh argument, so the
pattern reaching Python had them stripped. The assertion caught it instead of the
edit silently matching nothing. Cheap guards on every remote patch have now caught
three distinct classes of error — this one, the duplicate-source inheritance at
t0059, and the wrong-anchor case at t0069.

### Correction: there was no precision effect (t0094 closes t0092)

The §6 table above is wrong in its most important row, and the error is mine.

I read t0092 as "quality per step improved, throughput collapsed" on the basis of
loss 3.013 at step 1000 against the champion's 3.029. **That comparison is invalid.**
The learning-rate schedule is driven by elapsed *time*, not step index. t0092 ran at
half speed, so by step 1000 it had spent 55% of its budget and was already annealing
at `lrm 0.90`, while the champion was still at full rate with `lrm 1.00`. Lower loss
at equal step count is exactly what annealing produces. I compared two runs at the
same step while they sat at different points on a time schedule.

t0094 settles it. With fp32 moments the step count is preserved — 2871 against 2875 —
and the loss at step 1000 is 3.029348 against 3.029267. Identical. So:

- there was **no** precision benefit on this path, in the parameters or the moments;
- t0092 was a pure throughput loss, not a trade;
- the direction is abandoned, as pre-committed in t0094.

The general lesson is sharper than the specific one. **Under a time-based schedule,
per-step comparisons across runs of different throughput are meaningless** — the
faster run is always earlier on the LR trajectory at any given step. The only valid
cross-run comparisons here are at equal *progress* or at the final locked score. I
have been treating `num_steps` as a clean diagnostic all session, which it is; what
is not clean is reading intermediate loss at a fixed step index, and I did that
twice before catching it.

The correction is also recorded in the ledger, in t0095's `change_summary`.

## 7. Weight averaging, and expiring decisions (t0096)

Polyak averaging over the tail of training gained **0.001103** at essentially no
throughput cost — 2870 steps against 2875. It is the largest single gain in the last
forty trials.

What makes it worth recording is not the mechanism but *why it was available*. I
dismissed weight averaging early in the session, and the reasoning was sound at the
time: the schedule anneals the learning rate to zero, so the final weights are
already settled and there is nothing to denoise. Then **t0048 introduced
`FINAL_LR_FRAC = 0.1` and was promoted.** From that point the model finished training
at a tenth of peak rate, still taking real steps, and its endpoint became a single
noisy draw rather than a converged point — exactly the condition averaging exists for.

The category is worth naming. A stale *measurement* leaves a result in the ledger
that later evidence can contradict. A stale *decision not to test something* leaves
nothing at all: no record, no failure, no contradiction. It expires silently. The
only way to catch it is to periodically re-read why things were ruled out and check
whether the premise still holds.

Two of this campaign's late gains came from that habit rather than from search:

| trial | what had changed underneath | gain |
|---|---|---:|
| t0065 | batch halving raised cumulative `lr·wd` by 1.37× | 0.00035 |
| t0096 | a promoted LR floor stopped the model from annealing | 0.00110 |

## 8. What the EMA changed about the schedule (t0099)

t0099 doubled the schedule floor, `FINAL_LR_FRAC` 0.1 → 0.2, and returned
**0.967290 against the champion's 0.967286** — a difference of four parts in a
million, against a repeat noise of 1.3e-4. That is not a weak effect; it is an
exact null, and it says something specific.

Annealing does two jobs: it keeps the rate useful, and it settles the model so the
final weights are not a single noisy draw. **The EMA from t0096 now does the second
job.** t0099 shows the first is insensitive to the floor across a factor of two —
so within that range the schedule endpoint no longer influences the locked score at
all.

That reframes a setting I had treated as tuned. `FINAL_LR_FRAC` was promoted at
t0048 on a near-null median, and I have been carrying it since as though it were
measured. It was measured — but only against a model without averaging. Once
averaging existed the quantity it was trading off stopped existing.

t0100 pushes to 0.4 to find where the flat region ends. The prediction that would
falsify the whole reading is a sharp regression, which would mean averaging absorbs
endpoint *noise* but cannot replace a trajectory that has genuinely descended.

### Running count of interaction tests

Four settings have now been re-measured because a later promotion changed the
conditions under which they were first measured:

| setting | what changed underneath | outcome |
|---|---|---|
| weight decay | batch halving raised cumulative `lr·wd` 1.37× | −0.00035 |
| softcap | `lm_head` sensitivity fell at the smaller batch | −0.00021 |
| value-embed LR | init fix removed the need to fund growth | −0.00089 |
| schedule floor | EMA took over the settling job | exact null |

Three gains and one null that reframed a setting. The class is worth more than any
single coordinate scan in this campaign.

## 9. State of the search at ~100 trials

The tunable surface of `train.py` is essentially exhausted. Every coefficient is
bracketed and every component of the block has been measured:

- **optimiser** — all four learning-rate groups, scalar LR, Adam betas, Muon momentum
  and its warmup, NorMuon variance horizon
- **capacity** — depth (10/12/13/14), width, MLP ratio (3/4/5), head dim (64/128/256),
  GQA
- **attention** — short span (128/256/384/512/1024), pattern (3/4/5/6 global layers),
  graded spans both directions, attention softcap, attention temperature
- **schedule** — warmdown length, endpoint and *shape*; warmup; weight decay value and
  schedule coupling
- **value-embedding path** — density, gate width, init scale, decoupled LR
- **other** — rotary base, lm_head init, kernel autotuning, precision of parameters
  and moments, U-net skips, attention gating, per-head temperature, weight averaging

Of roughly eighteen *inherited* defaults tested, one was suboptimal (the output
softcap). The recipe was well tuned; the gains came from elsewhere.

### Where the 2.40% actually came from

| source | Δ bpb | how it was found |
|---|---:|---|
| batch/LR scaling | −0.0090 | coordinate search + decomposing a bundle |
| capacity (depth 8→12) | −0.0045 | coordinate search |
| missing mechanisms | −0.0055 | reading the model for what it lacked |
| init mis-specification | −0.0018 | reading `init_weights` |
| weight averaging | −0.0011 | re-reading a dismissal whose premise expired |
| operating-point corrections | −0.0006 | arithmetic on quantities the batch change moved |

The first two are what a hyperparameter sweep would find. The last four — about
0.009 bpb, roughly 40% of the total — came from reading the source and from
re-examining earlier decisions, not from searching a space.

### What remains

Only added mechanisms, which carry higher variance than coordinate work. t0103 tests
Lookahead. If that and a small number of similar bets fail, the honest conclusion is
that this configuration is at a strong local optimum for the 300-second contract and
further trials are spending GPU time for noise.

## 10. Trajectory averaging fails where endpoint averaging succeeded (t0103–t0104)

Averaging the *endpoint* gained 0.001103 (t0096). Averaging the *trajectory* — Lookahead,
a slow copy that the fast weights are periodically reset onto — failed twice:

| scope | Δ vs champion |
|---|---:|
| all parameters (t0103) | +0.0072 |
| AdamW groups only (t0104) | +0.0024 |

The 3× reduction confirms the diagnosis its own risk note pre-registered: **Muon keeps a
momentum buffer and a NorMuon per-row variance estimate keyed to the parameters it
updates**, and Lookahead rewrites those parameters without touching either, so after each
sync the orthogonalised update corrects a trajectory that no longer exists. Restricting
the mechanism to AdamW, whose moments are plain per-coordinate averages, removed most of
the damage — and it still lost.

So the two averaging results are not the same idea at different scales. Endpoint
averaging never influences training; it only changes which weights get evaluated, and it
works because `FINAL_LR_FRAC = 0.1` leaves the model unconverged. Trajectory averaging
changes the optimisation path, and this recipe's optimiser is not built to have its
parameters moved beneath it. Closed after two nodes rather than tuned to a third.

### A note on pre-commitments

I pre-registered a larger `k` as the follow-up and did not do it, because a larger `k`
only makes the disruption rarer — it cannot distinguish "wrong mechanism" from
"incompatible with Muon". Scoping by optimiser group tests the diagnosis directly. This
is the second time a pre-commitment was revised; both times the reason was that the
measurement changed what question was worth asking, and both revisions are recorded in
the ledger alongside the original commitment rather than replacing it.

## 11. Implementation errors, and what caught them

Five of my changes failed to run. The pattern is worth recording because none were
modelling mistakes — all were contracts between components:

| trial | error | caught by |
|---|---|---|
| t0009 | `mark_dynamic` called inside a traced region | dynamo traceback |
| t0044 | bf16 × fp32 promotion rejected by the FA3 kernel | kernel dtype check |
| t0058 | autotune search exceeded the 660s wall budget | controller timeout |
| t0093 | `lerp_` cannot promote its fp32 destination from bf16 | dynamo traceback |
| t0106 | `tokens_per_fwdbwd` still used `MAX_SEQ_LEN` | the recipe's own assertion |

Four of five were caught by an assertion, a traceback or a timeout — that is, by an
invariant that fired *before* producing a number. Only t0058 consumed real GPU time,
and even that was bounded by the controller rather than by me noticing.

The one I would most like back is t0106, because the fix was to read the file: the
sequence length is used in four places and I changed two. When repairing it I audited
every occurrence first and found a fifth use, `build_varlen(x, MAX_SEQ_LEN)`, which
would not have crashed — it would have passed a too-large hint to the varlen kernel
and quietly worked. That is the more dangerous class: an error that produces a
plausible number rather than a traceback.

The general habit that has served best all session: **verify before asserting, and
grep for every use of a symbol before changing one of them.** Both times I skipped it
(t0080's description, t0106's edit) the cost was a wasted node or a wrong record.

## 12. Why attention cuts kept disappointing (t0107)

Three separate nodes reduced attention FLOPs and each returned roughly a quarter of
the step gain the FLOP model predicted:

| trial | FLOPs cut | predicted steps | actual steps |
|---|---:|---:|---:|
| t0011 | 4% | +4% | +0.8% |
| t0030 | −9% (widened) | −9% | −1.2% |
| t0107 | 8% | +6–8% | +2% |

The explanation is simple once stated: **this model is matmul-bound, not
attention-bound.** Matrix work is about 251M of the 314M per-token FLOPs, and
attention is 63M of which the four global layers hold 50M. Trading attention span for
steps was always trading against the smaller term.

It also explains two other results that looked unrelated at the time. Kernel
autotuning (t0058, t0059) spent 162 seconds searching and returned three *fewer*
steps, because inductor's defaults were already good on the GEMMs that dominate. And
every capacity reduction lost, because reducing matrix FLOPs is the only way to buy
meaningful throughput and it costs exactly the capacity the model needs.

So the throughput side of the objective is closed: the only lever large enough to
matter is capacity, and capacity is bracketed at depth 12 in both directions.

## 13. Contention invalidated a conclusion (t0110)

I reported t0110's 1.019412 as evidence that a looser output softcap destabilises the
model, and concluded from it that every axis in the campaign was closed. **That was
wrong.** The run completed 1342 optimizer steps against 2863 for the trial immediately
before it. A 53% step collapse is a contention signature, not a modelling effect.

A foreign job — 104GB at full utilisation — had landed on the pinned GPU partway
through the run. I drew a mechanism conclusion from a corrupted measurement, and the
tell was sitting in the same evidence record I had been reading all session.

### Why I missed it

`num_steps` has been my standard diagnostic all campaign, and I checked it for nearly
every trial. For t0110 I did not: the score was so far outside the range that I
explained it with the mechanism under test instead of asking what else could produce
it. **A result that is dramatic enough to be interesting is exactly the one most worth
checking for an artefact**, and I inverted that.

### What changed

- Controller stopped cleanly while a rendezvous was open, so no in-flight measurement
  was lost, and relaunched on the emptiest GPU chosen programmatically rather than
  hardcoded.
- `gpuwatch.sh` now samples every GPU every 30 minutes for 24 hours, recording
  utilisation, memory, controller liveness and campaign state to `logs/gpu_usage.tsv`.
  Timestamps come from a single `time.time()` call so the epoch and ISO columns cannot
  disagree; controller liveness comes from our own pid file, never a name match.
- t0110's conclusion is retracted. The softcap drift question is re-opened at 15 in
  t0112, since 16 already exists as a source and would be rejected as a duplicate.

### The caveat this creates

Every measurement in this campaign up to t0111, including the champion, was taken on
GPU 7. The campaign now runs on GPU 0 — identical hardware, same host, same contract —
but a systematic device offset cannot be excluded from a single trial. Step count is
the diagnostic, since it is what both contention and device differences move, and it
has held at 2869–2875 across uncontended trials.

## 14. Pausing on contention, and a bug in the pause mechanism itself

The host filled up: all eight GPUs occupied, and our step count fell from the clean
band of 2863–2875 to 2687 and then 2549. Step count cannot be moved by any constant
these nodes were testing, so those deficits were environmental, and three trials
(t0112–t0114) measured noise rather than mechanisms. Their ledger records say so
inside the record, not just here, because a later reader would otherwise see a
discard with a plausible hypothesis attached and count it as evidence.

The campaign is now paused with automatic resume: `supervisor.sh` checks every five
minutes and relaunches the controller once a GPU is under 5000 MiB and 20 percent
utilisation **on two checks a minute apart**, so it cannot land on a GPU that is
merely between kernels.

### The bug worth remembering

My first version defined contention as GPU memory held by processes whose owner was
not `$USER`. Tested against the live box, it reported no contention — while a 77GB
job sat on our GPU. That job was `ai_scientist_v2_baseline`: **a different campaign
running under the same account.**

Owner is the wrong discriminator on a host where several campaigns share one login.
The right one is the campaign path in the process command line. Had the first version
shipped, the supervisor would have concluded "all clear" and left the campaign
running on a contended GPU for as long as it ran — a monitor that reports health
while the thing it monitors is broken.

This is the second time this exact class of error appeared. At launch, a
`pgrep -f "fengheguai run"` guard matched **its own command line** and reported a
stray controller that did not exist. Both times the fix was the same: make the check
identify *this specific work*, not a string that happens to appear near it.
