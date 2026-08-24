"""Label smoothing on the cross-entropy loss.

One of the three standard mechanisms absent from this model in 508 trials. The prior is poor
and stated as such: smoothing deliberately biases predictions away from the true distribution,
and bpb measures exactly that distance. It is queued because the alternative is an agent_error
that burns the same coordinate and yields no measurement at all.

Usage: edit.py <node_id> <smoothing>
"""
import pathlib, ast, sys
node, eps = sys.argv[1], float(sys.argv[2])
assert 0.0 < eps < 0.2, "implausible smoothing"
p = pathlib.Path(f"/data3/zhubaiyu/fengheguai/campaigns/h200-claude/nodes/{node}/train.py")
s = p.read_text()
assert "label_smoothing" not in s, "already present"
old = """            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)"""
assert old in s, "cross_entropy call not in the expected form"
s = s.replace(old,
  f"            # Label smoothing {eps}. Absent from all 508 nodes. The prior is poor -- smoothing\n"
  f"            # moves the target off the true distribution and bpb measures that distance -- so\n"
  f"            # this is a genuine test of an untried mechanism rather than an expected win.\n"
  "            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),\n"
  f"                                   ignore_index=-1, reduction=reduction, label_smoothing={eps})", 1)
ast.parse(s)
p.write_text(s)
print(f"{node}: label_smoothing={eps}")
