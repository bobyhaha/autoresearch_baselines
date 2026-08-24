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
# A dtype cast changes the TRACED GRAPH, unlike the attn_gate zero-init which changed only
# weight values. The first attempt kept max-autotune on that mistaken reasoning, missed the
# inductor cache, re-searched, and exceeded the 660s wall clock at 2729 steps. Revert compile
# mode as every graph-changing bet must.
import re as _re
_m = _re.search(r'^model = torch\.compile\(model, dynamic=False, mode="max-autotune-no-cudagraphs"\)[^\n]*$', s, _re.M)
if _m:
    s = s.replace(_m.group(0),
      "# Compile mode reverted to default: casting lm_head changes the traced graph, so the\n"
      "# inductor cache misses and a max-autotune re-search exceeds the wall clock (t0483).\n"
      "model = torch.compile(model, dynamic=False)", 1)
ast.parse(s)
p.write_text(s)
print(f"{node}: lm_head cast to bf16")
