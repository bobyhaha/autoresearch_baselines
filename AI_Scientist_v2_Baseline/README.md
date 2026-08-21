# AI-Scientist-v2 → autoresearch port

An adaptation of [SakanaAI/AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2)'s
best-first tree search (BFTS) to Karpathy's
[autoresearch](https://github.com/karpathy/autoresearch) task: minimise `val_bpb` on a
single GPU under a fixed 300-second training budget.

## What was kept from AI-Scientist-v2

| Upstream | Here | Status |
|---|---|---|
| `treesearch/journal.py` — `Node`, `Journal` | `ai_scientist_ar/journal.py` | ported; plotting/VLM/seed-aggregation fields dropped |
| `treesearch/utils/metric.py` — `MetricValue` | `ai_scientist_ar/metric.py` | ported; trimmed to the scalar case |
| `parallel_agent.py` — `_select_parallel_nodes` | `agent.py::Agent._select_node` | ported; single-worker reduction |
| `parallel_agent.py` — `_draft` / `_debug` / `_improve` | `agent.py` | ported |
| `bfts_config.yaml` | `bfts_config.yaml` | adapted |
| `launch_scientist_bfts.py` | `ai_scientist_ar/run_bfts.py` | adapted |

The search semantics are unchanged: build `num_drafts` independent trees, then each
iteration either debug a buggy leaf (with probability `debug_prob`, up to
`max_debug_depth`) or improve the best node found so far.

## What was changed, and why

**1. Execution target.** Upstream writes a fresh standalone script per node and runs
it. Here a node *is* a complete `train.py`, executed in its own trial workspace on the
pinned GPU via `scripts/run_trial.sh`. See `ai_scientist_ar/interpreter.py`.

**2. Evaluation is deterministic, not LLM-judged.** Upstream asks an LLM to read stdout
and report what the metric was. The autoresearch task has an exact contract —
`train.py` prints a fixed summary block and `prepare.evaluate_bpb` is ground truth — so
the metric is parsed with a regex instead. No parse step can hallucinate a score.

**3. Runs are audited before they are scored.** `agent.py::Agent._audit` marks a trial
INVALID (buggy, unscored) if `prepare.py` no longer matches the pristine checkout by
sha256, or if `training_seconds < 295`. This is the port's main addition: it closes the
two ways a candidate could post a `val_bpb` that is not comparable to the others —
editing the read-only evaluation harness, or simply training for less time.

**4. One worker.** One GPU, so `_select_node` returns a single node. Upstream's
`processed_trees` bookkeeping existed only to keep N parallel workers off the same
tree, so it drops out.

**5. The LLM backend is a rendezvous, not an API.** There is no API key and no second
interface here: the Claude session driving the campaign is the coding model. The
harness publishes a request and blocks; the agent answers it. To stop the GPU idling
while the agent thinks, candidates can be *pre-authored* into a queue against a
specific parent node and are popped without blocking. See `ai_scientist_ar/backend.py`.

## Layout

```
ai_scientist_ar/     the ported harness
  journal.py         Node / Journal (solution forest)
  metric.py          MetricValue (compares by better, not larger)
  interpreter.py     trial workspace + run + deterministic result parsing
  backend.py         rendezvous LLM backend (queue + blocking paths)
  agent.py           search policy, draft/debug/improve, run audit
  run_bfts.py        campaign entrypoint
scripts/
  run_trial.sh       sealed launcher (full PATH; pinned GPU; per-GPU inductor cache)
  rv.py              agent-side rendezvous CLI (pending / respond / enqueue / status)
TASK_MEMO.md         the task's binding constraints, sent with every request
bfts_config.yaml     adapted config
campaign/            journal.json, results.tsv, status.json, campaign.log
```

## Running

```bash
python ai_scientist_ar/run_bfts.py --gpu 2          # runs until stopped
python scripts/rv.py status                          # leaderboard
python scripts/rv.py pending                         # open request, if any
```
