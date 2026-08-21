# Task rules (autoresearch) — binding constraints for every candidate

**Objective:** minimise `val_bpb` (validation bits per byte). Lower is better.
It is vocab-size independent, so architectural changes compare fairly.

**Fixed budget:** training runs for exactly `TIME_BUDGET = 300` seconds of wall-clock
training time (excluding startup/compilation), then evaluates. Because the budget is
fixed, step count is *not* a constraint — it is a consequence. A change that makes
each step cheaper buys more steps; a change that makes steps more expensive must pay
for itself in per-step quality.

## What a candidate MAY change

`train.py` only. Everything in it is fair game: model architecture, optimizer,
hyperparameters, learning-rate schedule, batch size, model size, attention pattern,
initialization, the training loop itself.

## What a candidate MUST NOT change

- **`prepare.py` is read-only.** It holds the fixed constants (`TIME_BUDGET`,
  `MAX_SEQ_LEN = 2048`, `EVAL_TOKENS`, `VOCAB_SIZE = 8192`), the data loader, the
  tokenizer, and `evaluate_bpb` — the ground-truth metric. The harness sha256-checks
  it before scoring and marks any run that touched it INVALID.
- **The evaluation harness.** `evaluate_bpb` is ground truth and must be called as-is.
- **The time budget.** Training must consume the full 300s. The harness marks any run
  reporting `training_seconds < 295` INVALID. Do not shorten training to score.
- **Dependencies.** No new packages. Only what is already in `pyproject.toml`
  (torch 2.9.1 + kernels, numpy, pandas, pyarrow, tiktoken, rustbpe, matplotlib).

**VRAM** is a soft constraint (H200, 143GB, shared box). Some increase is fine for a
real gain; it should not blow up. Baseline peaks around 45GB.

**Simplicity criterion:** all else equal, simpler is better. A tiny gain that adds
ugly complexity is not worth it; equal-or-better results from *deleting* code is a
win worth keeping.

## Output contract

`train.py` must end by printing the summary block the harness parses:

```
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

A run that does not print `val_bpb:` is treated as a crash.

## How to respond to a rendezvous request

Write the **complete** new `train.py` (not a diff) plus a one-line `plan` describing
the single change. Keep changes atomic so results stay attributable.
