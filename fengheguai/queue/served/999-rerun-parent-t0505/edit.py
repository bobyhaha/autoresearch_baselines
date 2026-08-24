"""Last-resort entry: re-run this node's inherited configuration under a fresh hash.

Every other queue entry asserts champion state, so all of them correctly refuse a debug trial
that inherited a modified parent -- and when that happens the request gets nothing and expires
as an agent_error. This entry applies to any node.

It is not filler. When a parent died to contention, a timeout or a guard abort, its
configuration was never measured, and the ledger records a failed trial whose source hash is
burned. Re-running the same configuration under a distinct hash is exactly the right repair,
and it is what I have been doing by hand for t0433, t0437, t0439, t0455, t0468 and others.

The node id makes the inserted comment unique, so the hash differs from the parent's without
any state or randomness.

Usage: edit.py <node_id>
"""
import pathlib, ast, sys
node = sys.argv[1]
p = pathlib.Path(f"/data3/zhubaiyu/fengheguai/campaigns/h200-claude/nodes/{node}/train.py")
s = p.read_text()
marker = f"# RE-RUN as {node}."
assert marker not in s, "already marked for this node"
# Refuse configurations already proven unmeasurable, so this does not become a loop. The engine
# opens debug children of failed trials, and re-running a config that fails by construction just
# burns another slot the same way. This has now happened twice -- EXHAUSTIVE at t0482 and CUDA
# graphs at t0503 -- so the list is explicit and meant to be extended rather than rediscovered.
KNOWN_UNMEASURABLE = [
    ("max_autotune_gemm_search_space",
     "EXHAUSTIVE autotune: search exceeds the wall clock even warm (t0479/t0481/t0482)"),
    ("triton.cudagraphs = True",
     "CUDA graphs: segfault (rc 139) during compile (t0502/t0503)"),
]
for _marker, _why in KNOWN_UNMEASURABLE:
    assert _marker not in s, f"inherited a config known to fail -- {_why}"

anchor = 'torch.set_float32_matmul_precision("high")'
assert anchor in s, "expected anchor not found"
s = s.replace(anchor,
  f"{marker} The parent's configuration was never measured -- it died to contention, a\n"
  f"# timeout or a guard abort before producing a score -- and its source hash is burned by the\n"
  f"# failed record. Nothing about the model changes here; this re-runs the inherited\n"
  f"# configuration under a distinct coordinate so the measurement can actually happen.\n"
  + anchor, 1)
ast.parse(s)
p.write_text(s)
print(f"{node}: re-running inherited configuration under a fresh hash")
