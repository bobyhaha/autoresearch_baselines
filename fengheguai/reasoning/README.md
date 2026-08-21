# Reasoning log — campaign `h200-claude`

A time-ordered record of *why* each experiment was run, at 30-minute resolution.

## What is in here

| file | contents | provenance |
|---|---|---|
| `YYYYMMDD-HHMMZ.md` | every trial completed in that 30-minute window, with its pre-registered hypothesis, change, predicted effect and stated risk | **reconstructed** from `ledger.jsonl` by `build_log.py` |
| `NOTES-session.md` | the reasoning arc: what I believed, what changed my mind, what I got wrong | **contemporaneous** — written during the session |
| `build_log.py` | the generator | — |

## The honesty caveat that matters

The window files are **reconstructed after the fact**, not written live. What makes
them trustworthy anyway is that their content is not reconstructed: each
hypothesis, predicted effect and risk was written *before* its measurement, handed
to the controller, and sealed into a SHA-256 hash-chained append-only ledger. The
generator only slices those sealed records into time order — it cannot revise them,
and a prediction cannot be edited after seeing its result.

`NOTES-session.md` is the opposite: written live, so it carries judgement the
ledger does not, including reasoning that turned out to be wrong.

## Regenerating

```bash
cd /Users/baiyu/Desktop/OPHIS/fengheguai
python3 reasoning/build_log.py     # reads campaigns/h200-claude-ledger.jsonl
```

Re-running overwrites window files from the ledger and never touches `NOTES-*.md`.
