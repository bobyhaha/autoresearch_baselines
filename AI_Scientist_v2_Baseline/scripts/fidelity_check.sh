#!/bin/bash
# fidelity_check.sh — verify this port still matches AI-Scientist-v2, hourly.
#
# The port has drifted from upstream three times without me noticing: argmin replaced the
# non-argmin selector, multi-seed evaluation went missing, and the debug branch ran for
# 105 nodes without ever executing. Each was found by reading code on a hunch. This turns
# that into a scheduled check.
#
# Every check is designed to FAIL on a real regression, not to print reassurance. The
# ones that matter most are the invariants that already broke once: harness and restocker
# agreeing on "best" (a mismatch deadlocks the campaign), and no seed-eval node being
# adopted as the incumbent.
#
#   ./fidelity_check.sh [--once]      # default loops hourly
set -uo pipefail
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"

# Overridable so the checker itself can be tested against a sandbox copy — a check that
# has never been observed to fail is not evidence of anything.
BASE="${FIDELITY_BASE:-$HOME/ai_scientist_v2_baseline}"
PY="$BASE/task/.venv/bin/python"
UP="$BASE/AI-Scientist-v2/ai_scientist/treesearch"
LOG="$BASE/campaign/fidelity.log"
ONCE=0; [ "${1:-}" = "--once" ] && ONCE=1

pass=0; fail=0; warn=0
ok()   { echo "  PASS  $1"; pass=$((pass+1)); }
bad()  { echo "  FAIL  $1"; fail=$((fail+1)); }
note() { echo "  WARN  $1"; warn=$((warn+1)); }

run_checks () {
  pass=0; fail=0; warn=0
  echo "================ fidelity check $(date -u +%Y-%m-%dT%H:%M:%SZ) ================"

  # 1. Upstream reference must be the version this port was audited against.
  local up_hash
  up_hash=$(cat "$UP/parallel_agent.py" "$UP/journal.py" 2>/dev/null | sha256sum | cut -c1-16)
  if [ -f "$BASE/campaign/.upstream_hash" ]; then
    if [ "$up_hash" = "$(cat "$BASE/campaign/.upstream_hash")" ]; then
      ok "upstream reference unchanged ($up_hash)"
    else
      bad "upstream reference CHANGED ($up_hash) — divergence register may be stale"
    fi
  else
    echo "$up_hash" > "$BASE/campaign/.upstream_hash"; note "upstream hash recorded ($up_hash)"
  fi

  # 2. Search constants must match upstream's bfts_config.yaml.
  local cfg="$BASE/AI-Scientist-v2/bfts_config.yaml"
  for pair in "max_debug_depth:3" "debug_prob:0.5" "num_drafts:3"; do
    local k="${pair%%:*}" want="${pair##*:}" got
    got=$(grep -E "^\s*${k}:" "$cfg" | head -1 | awk '{print $2}')
    local mine
    mine=$(grep -E "^\s*${k}: (float|int) = " "$BASE/ai_scientist_ar/agent.py" | head -1 | awk '{print $NF}')
    if [ "$got" = "$want" ] && [ "$mine" = "$want" ]; then ok "$k = $want (upstream == port)"
    else bad "$k drift: upstream=$got port=$mine expected=$want"; fi
  done

  # 3. Policy semantics — the branch that never runs live is only covered here.
  if (cd "$BASE" && "$PY" tests_policy.py >/dev/null 2>&1); then
    ok "policy unit tests (significance guard, seed-eval exclusion, debug depth, persistence)"
  else
    bad "policy unit tests FAILED"; (cd "$BASE" && "$PY" tests_policy.py 2>&1 | tail -5 | sed 's/^/        /')
  fi

  # 4. Ported machinery still present. Absence is how multi-seed went missing.
  grep -q "def run_seed_eval"      "$BASE/ai_scientist_ar/agent.py"   && ok "multi-seed evaluation present"        || bad "run_seed_eval MISSING (upstream _run_multi_seed_evaluation)"
  grep -q "def _debug"             "$BASE/ai_scientist_ar/agent.py"   && ok "_debug present"                        || bad "_debug MISSING"
  grep -q "def _draft"             "$BASE/ai_scientist_ar/agent.py"   && ok "_draft present"                        || bad "_draft MISSING"
  grep -q "min_improvement"        "$BASE/ai_scientist_ar/journal.py" && ok "significance guard present"             || bad "significance guard MISSING (argmin regression)"
  grep -q "canonical train split"  "$BASE/ai_scientist_ar/agent.py"   && ok "audit whitelists the evaluation path"   || bad "eval-path whitelist MISSING"

  # 4a. Contention machinery. This is not upstream — upstream assumes a dedicated GPU —
  # but on a shared box it is what makes a run faithful at all: a co-tenant's compute comes
  # straight out of our step count, and a contended trial looks like a result rather than
  # an error.
  grep -q "CONTENTION_GATE"      "$BASE/run_trial.sh"              && ok "pre-trial contention gate installed"        || bad "contention gate MISSING from run_trial.sh"
  grep -q "FOREIGN_PEAK_MB"      "$BASE/run_trial.sh"              && ok "in-flight foreign sampling installed"       || bad "in-flight foreign sampling MISSING"
  grep -q "MAX_FOREIGN_PEAK_MB"  "$BASE/ai_scientist_ar/agent.py"  && ok "audit rejects contended trials"             || bad "contention audit MISSING"
  # The meaningful question is not "did any trial run under contention" — several did, and
  # were correctly refused. It is "did any contended trial get SCORED". Checking the logs
  # alone conflated the two and warned on eight trials the audit had already rejected.
  local scored_contended
  scored_contended=$("$PY" - "$BASE" <<'PYEOF' 2>/dev/null || echo -1
import json, re, sys
from pathlib import Path
base = Path(sys.argv[1])
state = {}
for f in ("journal.json", "env_failures.json"):
    p = base / "campaign" / f
    if p.exists():
        for n in json.loads(p.read_text())["nodes"]:
            state[n["id"]] = bool((n.get("metric") or {}).get("value")) and not n.get("is_buggy")
bad = 0
for d in (base / "trials").iterdir():
    log = d / "run.log"
    if not log.is_file() or not state.get(d.name):
        continue
    # Must use the SAME rule as the audit, which keys off GROWTH. A tenant already
    # holding memory at launch and not computing costs us nothing, so an absolute-peak
    # threshold flags trials the audit deliberately allows. The two disagreeing produced
    # a spurious FAIL on a trial whose foreign memory grew by 2 MB.
    txt = log.read_text(errors='replace')
    g = re.search(r'^FOREIGN_GROWTH_MB=(-?\d+)', txt, re.M)
    pk = re.search(r'^FOREIGN_PEAK_MB=(\d+)', txt, re.M)
    # Trials predating growth logging carry only a peak, and must be judged by the rule
    # in force when they ran (absolute peak > 10GB), not by the later growth threshold.
    if g:
        contended = int(g.group(1)) > 4000
    else:
        contended = bool(pk) and int(pk.group(1)) > 10000
    if contended:
        bad += 1
print(bad)
PYEOF
)
  if [ "$scored_contended" = "0" ]; then ok "no SCORED trial ran under >10GB foreign memory"
  elif [ "$scored_contended" = "-1" ]; then note "could not evaluate scored-contention check"
  else bad "$scored_contended contended trial(s) were scored — audit failed to reject them"; fi

  # 4b. The launcher must actually pass the guards. Verifying that the code contains a
  # feature says nothing about whether the running process uses it — a remote-only edit to
  # supervise.sh was once reverted by an rsync, and the harness silently fell back to
  # argmin while every code-level check still passed.
  if grep -q -- "--min-improvement" "$BASE/scripts/supervise.sh"; then ok "launcher passes --min-improvement"
  else bad "launcher does NOT pass --min-improvement (harness falls back to argmin)"; fi
  if grep -q -- "--num-seeds" "$BASE/scripts/supervise.sh"; then ok "launcher passes --num-seeds"
  else bad "launcher does NOT pass --num-seeds (multi-seed eval never fires)"; fi
  if pgrep -fa "run_bfts.py" | grep -q -- "--min-improvement"; then ok "running harness has the guard"
  else note "running harness lacks the guard (restart pending?)"; fi

  # 5. Harness and restocker must agree on "best". They disagreed once and deadlocked.
  local agree; agree=$("$PY" - "$BASE" <<'PYEOF' 2>/dev/null
import sys, json, importlib.util
base = sys.argv[1]; sys.path.insert(0, base)
from ai_scientist_ar.journal import Journal
spec = importlib.util.spec_from_file_location("r", base + "/scripts/restock.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
j = Journal.load(base + "/campaign/journal.json")
nodes = json.load(open(base + "/campaign/journal.json"))["nodes"]
a = j.get_best_node(min_improvement=m.MIN_IMPROVEMENT)
b = m.best_node(nodes, m.MIN_IMPROVEMENT)
print("AGREE" if (a and b and a.id == b["id"]) else "DISAGREE")
PYEOF
)
  [ "$agree" = "AGREE" ] && ok "harness and restocker agree on incumbent" || bad "harness/restocker DISAGREE on incumbent (deadlock risk)"

  # 6. Live journal invariants.
  local inv; inv=$("$PY" - "$BASE" <<'PYEOF' 2>/dev/null
import sys, json
base = sys.argv[1]; sys.path.insert(0, base)
from ai_scientist_ar.journal import Journal
j = Journal.load(base + "/campaign/journal.json")
issues = []
best = j.get_best_node(min_improvement=0.00036)
if best is not None and getattr(best, "is_seed_eval", False):
    issues.append("incumbent is a seed-eval node")
for n in j.nodes:
    if n.is_buggy and n.debug_depth > 3:
        issues.append(f"{n.id} debug_depth {n.debug_depth} exceeds max_debug_depth")
print("OK" if not issues else "; ".join(issues))
print(len(j.nodes), len(j.draft_nodes), sum(1 for n in j.nodes if n.is_buggy), sum(1 for n in j.nodes if getattr(n, "is_seed_eval", False)))
PYEOF
)
  local inv1; inv1=$(echo "$inv" | head -1)
  [ "$inv1" = "OK" ] && ok "journal invariants (no seed-eval incumbent, debug depth bounded)" || bad "journal invariant: $inv1"
  echo "  INFO  nodes/drafts/buggy/seed-eval: $(echo "$inv" | tail -1)"

  # 6b. Journal and archive must be disjoint. They were not: four purged nodes reappeared
  # because the running harness rewrote the journal from memory, silently reverting the
  # removal. The record double-counted them and the debug branch could select them again.
  local dupes
  dupes=$("$PY" - "$BASE" <<'PYEOF' 2>/dev/null || echo -1
import json, sys
from pathlib import Path
c = Path(sys.argv[1]) / "campaign"
try:
    j = {n["id"] for n in json.loads((c / "journal.json").read_text())["nodes"]}
    a = {n["id"] for n in json.loads((c / "env_failures.json").read_text())["nodes"]}
except Exception:
    print(-1); raise SystemExit
print(len(j & a))
PYEOF
)
  if [ "$dupes" = "0" ]; then ok "journal and archive are disjoint (no reverted purges)"
  elif [ "$dupes" = "-1" ]; then note "could not evaluate journal/archive disjointness"
  else bad "$dupes node(s) in BOTH journal and archive — a purge was reverted"; fi

  # 7. Every trial must have run against the pristine evaluation harness.
  local pris bad_h=0 n=0
  pris=$(sha256sum "$BASE/task/prepare.py" | cut -d' ' -f1)
  for f in "$BASE"/trials/*/prepare.py; do
    [ -f "$f" ] || continue; n=$((n+1))
    [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$pris" ] || { bad_h=$((bad_h+1)); echo "        modified: $f"; }
  done
  [ "$bad_h" -eq 0 ] && ok "prepare.py pristine in all $n trials" || bad "$bad_h of $n trials have a modified prepare.py"

  echo "-------- $pass pass, $fail fail, $warn warn --------"
  return $fail
}

while :; do
  out=$(run_checks 2>&1); rc=$?
  echo "$out"; echo "$out" >> "$LOG"
  [ "$rc" -ne 0 ] && echo "FIDELITY REGRESSION: $rc failing check(s)" | tee -a "$LOG"
  [ "$ONCE" -eq 1 ] && exit $rc
  sleep 3600
done
