"""Verify the ported policy branches, including the one that has never run live."""
import sys; sys.path.insert(0, ".")
from ai_scientist_ar.journal import Journal, Node
from ai_scientist_ar.metric import MetricValue, WorstMetricValue

def mk(j, val, parent=None, buggy=False, seed_eval=False):
    n = Node(plan="t", code=f"c{val}", parent=parent)
    n.is_buggy = buggy; n.is_seed_eval = seed_eval
    n.metric = WorstMetricValue(maximize=False, name="v") if buggy else MetricValue(val, maximize=False, name="v")
    j.append(n); return n

# 1. significance guard: a 0.04-sigma "win" must not displace the incumbent
j = Journal(); a = mk(j, 0.970000); b = mk(j, 0.969991)
assert j.get_best_node(min_improvement=0.0).id == b.id, "argmin should pick the challenger"
assert j.get_best_node(min_improvement=0.00036).id == a.id, "guard should keep incumbent"
# and a real win must still displace it
c = mk(j, 0.969000)
assert j.get_best_node(min_improvement=0.00036).id == c.id, "real win must displace"
print("PASS significance guard")

# 2. seed-eval nodes can never become the incumbent
j2 = Journal(); base = mk(j2, 0.970000); mk(j2, 0.960000, parent=base, seed_eval=True)
assert j2.get_best_node().id == base.id, "lucky seed must not be adopted"
print("PASS seed-eval excluded from candidates")

# 3. the debug branch — ported but never exercised in 105 live nodes.
# Upstream semantics: stage_name describes how a node was PRODUCED, so a failed
# improve-node has a good parent and stage_name "improve"; only a node whose parent is
# itself buggy counts as "debug", and debug_depth counts CONSECUTIVE debugging steps.
j3 = Journal(); root = mk(j3, 0.99); bug = mk(j3, 0, parent=root, buggy=True)
assert bug in j3.buggy_nodes, "buggy node not collected"
assert bug.is_leaf and bug.stage_name == "improve", f"stage_name {bug.stage_name}"
assert bug.debug_depth == 0, f"debug_depth {bug.debug_depth}"
assert [n for n in j3.buggy_nodes if n.is_leaf and n.debug_depth <= 3] == [bug]

fix1 = mk(j3, 0, parent=bug, buggy=True)          # a debug attempt that also failed
assert fix1.stage_name == "debug" and fix1.debug_depth == 1, fix1.debug_depth
fix2 = mk(j3, 0, parent=fix1, buggy=True)
assert fix2.debug_depth == 2, fix2.debug_depth
# bug and fix1 now have children, so only the deepest leaf is debuggable
assert [n for n in j3.buggy_nodes if n.is_leaf and n.debug_depth <= 3] == [fix2]
# max_debug_depth must eventually cut the chain off
fix3 = mk(j3, 0, parent=fix2, buggy=True); fix4 = mk(j3, 0, parent=fix3, buggy=True)
assert fix4.debug_depth == 4
assert [n for n in j3.buggy_nodes if n.is_leaf and n.debug_depth <= 3] == [], \
    "max_debug_depth=3 must exclude a depth-4 chain"
print("PASS debug branch (stage_name, consecutive depth, leaf filter, depth bound)")

# 4. round-trip persistence of the new flag
import tempfile, pathlib
with tempfile.TemporaryDirectory() as d:
    p = pathlib.Path(d) / "j.json"; j2.save(p)
    assert any(n.is_seed_eval for n in Journal.load(p).nodes), "is_seed_eval lost on reload"
print("PASS persistence")
