"""Pass softcap to the fa3 attention kernels.

flash_attn_func and flash_attn_varlen_func both accept softcap=0.0 (disabled) and apply
sc * tanh(scores / sc) to the attention LOGITS inside the kernel. The model already
softcaps its OUTPUT logits at 13, but has never capped attention scores. Because the cap
happens inside the fused kernel, no intermediate reaches memory and the step cost is nil.

Usage: edit.py <node_id> <softcap>
"""
import pathlib, ast, re, sys
node, sc = sys.argv[1], float(sys.argv[2])
p = pathlib.Path(f"/data3/zhubaiyu/fengheguai/campaigns/h200-claude/nodes/{node}/train.py")
s = p.read_text()

sites = len(re.findall(r"window_size=window_size(?!, softcap)", s))
assert sites == 2, f"expected 2 un-softcapped attention call sites, found {sites}"
assert "window_size=window_size, softcap=" not in s, "attention softcap already present"
assert sc > 0, "softcap must be positive; 0.0 disables it in fa3 and would be a no-op"

s = s.replace("window_size=window_size", f"window_size=window_size, softcap={sc}")
s = s.replace("    def forward(self, x, ve, cos_sin, window_size, varlen=None):",
  f"    # Attention-score softcap {sc}: the fa3 kernels apply {sc}*tanh(scores/{sc}) to the\n"
  f"    # attention logits internally, so no intermediate reaches memory and the step cost is nil.\n"
  f"    # The model already caps its OUTPUT logits at 13 (tuned over 7 values) but has never\n"
  f"    # capped attention scores, which is a different quantity and a different failure mode.\n"
  "    def forward(self, x, ve, cos_sin, window_size, varlen=None):", 1)
ast.parse(s)
p.write_text(s)
print(f"{node}: attention softcap -> {sc} at {sites} call sites")
