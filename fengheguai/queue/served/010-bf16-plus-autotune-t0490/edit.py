"""Cast lm_head to bf16, matching what the model already does for its other big tables.

init_weights casts wte and every value-embedding table to bfloat16 and leaves lm_head in
fp32. lm_head is 8192 x 512 and sits in the output matmul, which runs under an autocast
context -- so its weights are cast to bf16 on the way into that matmul anyway. Holding them
in bf16 removes the cast and halves the traffic for the largest single weight read per step.

No node in 482 has ever done this.

Usage: edit.py <node_id>
"""
import pathlib, ast, sys
node = sys.argv[1]
p = pathlib.Path(f"/data3/zhubaiyu/fengheguai/campaigns/h200-claude/nodes/{node}/train.py")
s = p.read_text()
anchor = "        self.transformer.wte.to(dtype=torch.bfloat16)"
assert anchor in s, "wte bf16 cast not found"
assert "lm_head.to(dtype=torch.bfloat16)" not in s, "lm_head already cast"
s = s.replace(anchor, anchor +
  "\n        # lm_head gets the same treatment as wte and the value-embedding tables. It is\n"
  "        # 8192 x 512, it feeds the output matmul, and that matmul runs under autocast -- so\n"
  "        # these weights are converted to bf16 on the way in regardless. Holding them in bf16\n"
  "        # removes the conversion and halves the traffic for the largest single weight read\n"
  "        # per step. The model already applies exactly this reasoning to its other big tables;\n"
  "        # lm_head appears to have been left out rather than deliberately excluded.\n"
  "        self.lm_head.to(dtype=torch.bfloat16)", 1)
# This arm KEEPS max-autotune. The cast changes the traced graph, so t0483 missed the cache,
# re-searched and timed out -- but it compiled and trained 2729 steps first, so the bf16
# TRAINING graph is already in the shared cache. Same warm-retry pattern that produced the
# t0433 champion. The two gains are independent and additive: the cast measured +2.73% of steps
# at t0484 and autotune measured +2.26% at the t0445 control.
ast.parse(s)
p.write_text(s)
print(f"{node}: lm_head cast to bf16")
