# Fengheguai

Fengheguai is an autonomous search engine with one objective and one experimental
contract: minimize nanoGPT-style validation bits per byte (`val_bpb`) with training
capped at 300 seconds.

It gives an implementation agent one isolated `train.py`, evaluates the live trained
model through a locked `prepare.evaluate_bpb`, retains the best top-K branches, and
records every decision in a tamper-evident evidence chain. The source nanoGPT checkout
is never modified; every file Fengheguai creates stays inside the new campaign folder.

The default integration targets the training contract in
[Karpathy's autoresearch](https://github.com/karpathy/autoresearch): `train.py` is the
only mutable experiment surface, `prepare.py` fixes `TIME_BUDGET = 300` and the
validation evaluator, and lower `val_bpb` wins.

## What makes the loop strong

- A trusted wrapper captures the live model at the ordinary `evaluate_bpb` call and
  invokes the original immutable evaluator itself. A candidate cannot win by merely
  printing a fabricated score.
- The source audit locks `prepare.py`, dependency files, the 300-second constant, the
  seed's CUDA-synchronized timing/termination AST, the one-evaluation rule, and direct
  validation-data access.
- Progressive best-first search alternates broad exploration, UCB-guided refinement,
  scheduled recombination, and bounded debugging.
- Valid near-misses remain in a top-K archive, rather than disappearing whenever they
  fail to beat the current champion.
- The baseline and prospective winners receive confirmation runs by default. Promotion
  uses the median locked score and requires a strict decrease.
- Trial source, parentage, prompts, patches, logs, environment metadata, measurements,
  and promotion decisions are connected through a SHA-256-chained JSONL ledger.
- An interrupted campaign resumes from completed ledger events; an orphaned started
  node is reported by the audit rather than silently rewritten.

The design lineage is deliberately limited to DeepScientist, The AI Scientist-v2,
ScientistOne, and Karpathy's autoresearch. See [SOURCES.md](SOURCES.md) for the exact
mapping and pinned source revisions.

## Quick start

Requirements:

- Python 3.10 or newer;
- a CUDA environment supported by the target training repository;
- `uv` and the target's dependencies/data already prepared;
- an authenticated `codex` CLI, or another agent command supplied at initialization.

From this folder:

```bash
python -m fengheguai init \
  --target /path/to/karpathy/autoresearch \
  --campaign ./campaigns/h200-night \
  --name h200-night

python -m fengheguai baseline --campaign ./campaigns/h200-night
python -m fengheguai run --campaign ./campaigns/h200-night --trials 20
```

To keep searching until you interrupt it:

```bash
python -m fengheguai run --campaign ./campaigns/h200-night --forever
```

The first baseline command performs two 300-second measurements by default: one primary
measurement and one confirmation. A candidate is measured once; only a prospective
winner receives its confirmation run. For the faster, Karpathy-style single-run policy,
set `--confirmation-runs 0` on `init`.

The default agent command is equivalent to:

```text
codex exec --ephemeral --skip-git-repo-check --sandbox workspace-write \
  --approve-for-me --cd {trial} --output-schema {agent_schema} \
  --output-last-message {agent_result} -
```

The generated research prompt is sent on stdin. The agent is told to edit only
`train.py` and never launch training; the controller serializes access to the single GPU.

## Commands

```text
fengheguai init      create a frozen campaign snapshot
fengheguai baseline  establish the unmodified locked score
fengheguai step      propose and decide one search-tree node
fengheguai run       execute a bounded or indefinite campaign
fengheguai status    print the current champion and latest result
fengheguai audit     verify source, evidence, ledger, and promotion integrity
```

Use the package without installation by running `python -m fengheguai` from this
directory. An editable install also works:

```bash
python -m pip install -e .
fengheguai --help
```

## Campaign layout

```text
campaign/
├── config.json                    frozen runtime and search contract
├── ledger.jsonl                   append-only hash chain
├── seed/                          immutable source snapshot
├── control/
│   ├── program.md                 locked research-agent contract
│   ├── immutable_manifest.json
│   ├── training_contract.json     protected timing/termination structure
│   └── agent_output.schema.json
├── nodes/b0000|tNNNN/             isolated source state for each tree node
├── artifacts/tNNNN/               prompt, agent log, and exact source patch
├── evidence/tNNNN-rN/             run log and machine-readable evidence
└── reports/
    ├── STATUS.md
    ├── FINDINGS.md
    ├── results.tsv
    └── research_map.json
```

`ledger.jsonl` is authoritative. Reports are regenerated views.

## Search and promotion

Each node has a primary parent and, for recombination, an optional second elite. The
controller chooses among four actions:

1. `explore` starts a materially different bet from the champion;
2. `refine` expands a top-K node using metric quality plus an under-exploration bonus;
3. `recombine` periodically synthesizes the two strongest distinct branches;
4. `debug` grants one bounded repair attempt to a runtime failure.

The first locked score decides whether a node is promising. A promising node receives
the configured confirmation measurements. Its median must beat the pre-existing
champion by `minimum_improvement`; otherwise it is retained as a valid `discard` node
and can remain in the elite set if its score warrants it. Crashes, timeouts, duplicate
source, contract violations, and malformed agent outputs cannot be promoted.

The only optimized value is `val_bpb`. Runtime, VRAM, step count, and failure details are
captured solely as diagnostics and integrity evidence.

## Custom runners

Both commands are token arrays parsed with `shlex`, never shell strings. Available agent
placeholders are `{trial}`, `{campaign}`, `{agent_schema}`, `{agent_result}`, and
`{python}`. For example:

```bash
python -m fengheguai init \
  --target /path/to/autoresearch \
  --campaign ./campaigns/custom \
  --agent-command 'my-agent --workspace {trial} --result {agent_result}'
```

The training command must contain `{audit_runner}` so the controller—not `train.py`—owns
the final metric call. Its placeholders are `{trial}`, `{campaign}`, `{audit_runner}`,
and `{python}`. For a target that uses the active Python environment instead of `uv`:

```bash
python -m fengheguai init \
  --target /path/to/target \
  --campaign ./campaigns/plain-python \
  --train-command '{python} {audit_runner} --target {trial}'
```

The full config and research program are hashed at initialization. Start a new campaign
to change them; this keeps old promotion decisions interpretable.

## Verification and tests

```bash
python -m unittest discover -s tests -v
python -m fengheguai audit --campaign ./campaigns/h200-night
```

The tests prove that the locked evaluator ignores a fake printed score, budget mutation
is rejected, ledger tampering is detected, and a complete baseline-to-promotion campaign
is reproducible without a GPU through a synthetic target.

## Trust boundary

Fengheguai is defense in depth for a cooperative research agent, not a hardened hostile
code sandbox. The evaluator makes score spoofing and accidental contract drift much
harder, but arbitrary Python in `train.py` still executes with the operating-system
rights of the training process. Run autonomous candidate code in an isolated machine or
container with only the intended data and credentials available.
