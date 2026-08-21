# Restricted design lineage

Fengheguai was designed only from the four sources requested for this project. Its code
is an original standard-library implementation; no source file was copied from any of
the systems below.

## 1. Karpathy's autoresearch

- Repository: [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- Inspected revision: `228791fb499afffb54b46200aca536f79142f117`
- License declaration: MIT in the upstream README
- Ideas used: one editable `train.py`, immutable preparation/evaluation, a fixed
  five-minute budget, `val_bpb` as the sole scalar objective, baseline-first operation,
  keep/discard/crash outcomes, and an indefinitely repeatable agent loop.

Fengheguai strengthens this minimal loop with an external locked evaluator, isolated
tree nodes, confirmations, and evidence auditing while preserving its 300-second
`val_bpb` contract.

## 2. The AI Scientist-v2

- Repository: [SakanaAI/AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2)
- Paper: [The AI Scientist-v2](https://arxiv.org/abs/2504.08066)
- Inspected revision: `96bd51617cfdbb494a9fc283af00fe090edfae48`
- Upstream code license: The AI Scientist Source Code License
- Ideas used: explicit search nodes with parent/child relationships, best-first
  progressive experimentation, separate draft/debug/improve actions, bounded debug
  depth, stage-aware prompts, persistent journals, and parallelizable branch semantics.

Fengheguai specializes those ideas to a single-GPU scalar optimization problem. Idea
generation can be parallelized externally, but GPU evaluation stays serialized.

## 3. DeepScientist

- Repository: [ResearAI/DeepScientist](https://github.com/ResearAI/DeepScientist)
- Inspected revision: `b36624417f0c6b8238ec02db37b94d6db2faa5b0`
- Upstream license: Apache-2.0
- Ideas used: one durable local workspace per quest, prompt-led operation, baseline
  contracts, Git-like visible research maps, structured artifacts, persistent findings
  memory, resumable execution history, and a thin runner boundary.

Fengheguai represents these concepts with a self-contained campaign directory, a locked
`program.md`, isolated source snapshots, regenerated map/report views, and an append-only
event ledger.

## 4. ScientistOne

- Paper: [ScientistOne: Towards Human-Level Autonomous Research via Chain-of-Evidence](https://arxiv.org/abs/2605.26340)
- Project: [scientist-one.github.io](https://scientist-one.github.io/)
- Ideas used: Chain-of-Evidence, provenance-bearing claims, top-K Parallel
  Explore-Exploit search, filtering specification violations before best-run selection,
  deterministic score verification, and explicit method/code/evaluator alignment.

Fengheguai applies those ideas directly to optimization evidence: every reported score
points to a nonce-bound locked evaluation and hashed log; every method node points to its
exact `train.py` hash and parent patch; and `audit` deterministically rechecks aggregate
scores, the 300-second limit, source integrity, and promotion decisions.

## Deliberate omissions

The four systems also support literature surveys, manuscript production, figures,
review, user interfaces, collaboration channels, and broad multi-domain research. Those
features are intentionally excluded because they do not directly advance the sole
objective here: lower 300-second `val_bpb`.
