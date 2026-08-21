# Reasoning log

A running record of *why* this campaign did what it did — design decisions, the
evidence behind them, and the calls that could turn out wrong. Kept separate from
`campaign/`, which holds machine-generated results (`journal.json`, `results.tsv`).

| file | what it holds |
|---|---|
| `00-port-design.md` | why the AI-Scientist-v2 port is shaped the way it is |
| `01-regime-analysis.md` | what kind of optimization problem this task actually is |
| `02-experiment-log.md` | hypothesis → result → interpretation, per trial (living) |
| `03-open-questions.md` | known weaknesses, unresolved calls, things that could invalidate conclusions |

Convention: entries are appended, never rewritten. When a conclusion is overturned
the old entry stays and a new one supersedes it, so the reasoning chain stays honest
about what was believed when.
