# AI-Scientist-v2 → autoresearch

An adaptation of [SakanaAI/AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2)'s
best-first tree search to Karpathy's [autoresearch](https://github.com/karpathy/autoresearch)
task: minimise `val_bpb` on a single H200 under a fixed 300-second training budget, editing
only `train.py`.

**149 trials executed** across a 24-hour window on a shared 8×H200 box.
115 scored and retained; 26 refused as environmentally compromised.

## Result

| seed | baseline | final config | improvement |
|---|---|---|---|
| **42** (the seed the search optimised on) | 0.991192 | 0.969886 (n=17) | **2.15%** |
| 43 (held out) | 0.989440 | 0.972412 | 1.72% |
| 137 (held out) | 0.990118 | 0.971373 | 1.89% |
| 2024 (held out) | 0.991566 | 0.972270 | 1.95% |

**Mean across four seeds: 1.93% ± 0.18. Held-out mean: 1.85%.**

Three numbers, answering different questions — all defensible, none interchangeable:

- **2.15%** — what this `train.py` scores on the task as specified. `train.py` fixes seed 42
  and the metric is the score of the code as written. This is what the task asks for.
- **1.93%** — what the recipe is worth as a training recipe, seed-averaged.
- **1.85%** — what it is worth on a seed never used for selection. **14% of the seed-42
  headline is attributable to having optimised on seed 42** — the winner's curse operating
  on the search as a whole.

The reported figure is the **mean of 18 byte-identical replicates**, not the minimum
(0.969615). A running minimum over noisy draws only ever moves down, which is what makes it
a biased estimator.

## The winning configuration

Nine changes to the pristine `train.py`:

```python
DEPTH = 10                     # 8  -> 10  (ASPECT_RATIO takes model_dim 512 -> 640)
TOTAL_BATCH_SIZE = 2**18       # 2**19 -> 2**18
EMBEDDING_LR = 0.9             # 0.6   -> 0.9
UNEMBEDDING_LR = 0.006         # 0.004 -> 0.006
WEIGHT_DECAY = 0.15            # 0.2   -> 0.15
FINAL_LR_FRAC = 0.05           # 0.0   -> 0.05
short_window = long_window // 8            # seq/2 -> seq/8
_precompute_rotary_embeddings(..., base=50000)   # 10000 -> 50000
torch.compile(model, dynamic=False, mode="max-autotune-no-cudagraphs")
```

## What each change is worth in the final configuration

Upstream runs a dedicated ablation stage; this port collapsed it into the improve loop, so
every lever was originally measured **at the operating point where it was found**. Removing
each from the *finished* configuration gives a different — consistently larger — answer:

| lever | at introduction | in final config | ratio |
|---|---|---|---|
| `TOTAL_BATCH_SIZE` 2¹⁸ | 0.00765 | **0.01745** | 2.3× |
| shape (`DEPTH` 10 / width 640) | 0.00423 | **0.00785** | 1.9× |
| attention span seq/8 | ~0.0025 | **0.00755** | 3.0× |
| max-autotune compile | 0.00125 | **0.00213** | 1.7× |
| `FINAL_LR_FRAC` 0.05 | 0.00053 | **0.00129** | 2.4× |
| **sum of five** | | **0.03626** | |
| **actual total improvement** | | **0.02131** | |

The five sum to **1.7× the improvement they decompose**. These are *marginal* contributions —
what a lever is worth given every other lever is present — and marginals cannot be added.
"Removing batch 2¹⁸ costs 0.0175" is true; "batch 2¹⁸ accounts for 0.0175 of the total" is
not. A real decomposition would need Shapley values over 2⁹ subsets, roughly 68 GPU-hours.

## Findings that are not the headline

- **The two largest single effects are losses from replacing a primitive the baseline chose.**
  relu² → GeLU costs **+0.0254**; removing QK normalisation costs **+0.0147** — the latter
  while producing the campaign's *highest* MFU and 4% more steps. Both are quality effects
  with throughput moving the wrong way.
- **Six of nine optimizer knobs sit on the baseline's own default.** Only settings that scale
  with step count moved. The baseline's optimizer was well tuned.
- **Capacity is capped near 86M**, confirmed five independent ways (depth 6/9/12, width
  512/768, value-embedding coverage, MLP 3×/5×, plus retests at the final operating point).
- **MFU rises monotonically with model size while quality peaks in the middle** — so
  optimising utilisation directly would have selected the worst of three shapes. Under a
  wall-clock budget a larger model is handed more actual FLOPs, which is why a
  Chinchilla-style tokens-per-parameter argument does not transfer.
- **Value embeddings are ~30% of the parameter count** (26.2M of 85.9M).
- **Grad-accum halves VRAM for 0.0013 bpb** (69.6 → 35.4 GB). Not adopted, but it prices the
  memory/quality exchange rate rather than guessing it.

## Measurement caveats

- **The noise floor is a property of the cluster, not the code.** Byte-identical replicates
  gave sd 0.00013 early and 0.00044 late, with step counts drifting 1316 → 1301 under load.
- **The devices are not interchangeable.** GPU 1 delivered ~1005 steps against ~1310
  elsewhere for identical source, with no foreign memory to explain it — worth 0.0185 bpb,
  nearly the whole campaign improvement. Trials are now rejected if a repeated configuration
  loses more than 5% of its established step count.
- **26 of 149 trials were refused, none for being wrong.** All were environmental — co-tenant
  OOM or compute contention on the shared box. Zero candidates failed on their own merits.

## Fidelity to upstream

`scripts/fidelity_check.sh` runs hourly (21 checks, verified against four injected
regressions). Divergences from AI-Scientist-v2, all deliberate and documented in
`reasoning/04-port-vs-upstream.md`:

| upstream | here | why |
|---|---|---|
| LLM-judged `get_best_node` | deterministic significance guard | `val_bpb` is exact; an LLM adds irreproducibility to ground truth. The guard restores the check that plain argmin removed. |
| LLM-judged `parse_exec_result` | regex + run audit | A hallucinated parse corrupts the objective itself. |
| multi-seed at stage boundaries | ported, seed rewritten in place | Upstream *prepends* `torch.manual_seed`; `train.py` sets it later and would override. |
| four-stage structure | **absent** | The one unaddressed divergence. Its cost is measured above: per-lever values are ~2× off when taken at introduction. |
| dedicated GPU assumed | contention gate + throughput audit | Not a divergence — a precondition. Without it the port produces numbers that measure the cluster. |

## Layout

```
ai_scientist_ar/     ported harness (journal, metric, agent, interpreter, run_bfts)
scripts/             fidelity_check, restock, supervise, run_trial, analysis tools
reasoning/           62 dated entries: hypothesis -> result -> interpretation, incl. retractions
campaign/            journal.json, results.tsv, progress.png, env_failures.json, logs
progress.html        interactive chart (progress, capacity/MFU, ablation, full table)
```

`reasoning/02-experiment-log.md` is the primary record. It contains the corrections as well
as the results — several conclusions in it are retracted by later entries, and those are left
in place rather than edited away.
