"""Vary the token-embedding init scale, which no trial has ever touched.

init_weights sets wte to normal(0, 1.0) in all 457 nodes. Note the forward pass is
scale-INVARIANT to this: x = wte(idx) is immediately rms_norm'd, and x0 is taken after the
norm, so at initialisation the network sees the same activations at any wte scale. The
effect is purely optimisation-side -- the embedding group runs at lr 0.6 with weight_decay 0,
so the init scale sets how large each update is relative to the weight it modifies.

Keeps max-autotune: initialisation runs before torch.compile, the traced graph is unchanged
and the cache hits, so this is an unhandicapped comparison.

Usage: edit.py <node_id> <std>
"""
import pathlib, ast, re, sys
node, std = sys.argv[1], float(sys.argv[2])
assert std != 1.0, "1.0 is the incumbent"
assert 0.1 <= std <= 10.0, "implausible"
p = pathlib.Path(f"/data3/zhubaiyu/fengheguai/campaigns/h200-claude/nodes/{node}/train.py")
s = p.read_text()
old = "        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)"
assert old in s, "wte init not in expected form"
s = s.replace(old,
  f"        # wte init std 1.0 -> {std}. Untouched in all 457 nodes. The forward pass is scale-\n"
  f"        # invariant here -- wte(idx) is rms_norm'd immediately and x0 is taken after the norm --\n"
  f"        # so this cannot change what the network computes at step 0. It changes the ratio of\n"
  f"        # update size to weight size for the embedding group, which runs at lr 0.6 with no\n"
  f"        # decay, i.e. how fast the embedding can reorganise relative to where it started.\n"
  f"        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std={std})", 1)
ast.parse(s)
p.write_text(s)
print(f"{node}: wte init std 1.0 -> {std}")
