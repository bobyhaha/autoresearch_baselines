"""Vary the MLP expansion ratio, which no trial has ever touched.

`4 * config.n_embd` appears exactly twice per file in all 432 nodes: the ratio has been
constant for the whole campaign while depth, aspect ratio and head dim were all explored.
The MLP holds ~25M of the ~46M non-value-embedding parameters and is the model's largest
FLOPs consumer, so the ratio directly trades capacity against step count under a fixed
300s budget.

Usage: edit.py <node_id> <ratio>
"""
import pathlib, ast, re, sys
node, ratio = sys.argv[1], int(sys.argv[2])
assert ratio != 4, "4 is the incumbent; that would be a no-op"
assert 1 <= ratio <= 8, "implausible ratio"
p = pathlib.Path(f"/data3/zhubaiyu/fengheguai/campaigns/h200-claude/nodes/{node}/train.py")
s = p.read_text()

n = len(re.findall(r"4 \* config\.n_embd", s))
assert n == 2, f"expected exactly 2 occurrences of '4 * config.n_embd', found {n}"
assert "MLP_RATIO" not in s, "MLP_RATIO already present"

s = s.replace("class MLP(nn.Module):",
  f"MLP_RATIO = {ratio}  # was a hardcoded 4, never varied in 432 trials. The MLP is the model's\n"
  f"# largest FLOPs consumer (~25M of ~46M non-value-embedding params), so under a fixed 300s\n"
  f"# budget this ratio trades capacity directly against step count.\n"
  f"# WARM RETRY: the first 3x attempt reached its full step count but was killed by the 660s\n"
  f"# wall-clock safety timeout before the locked eval could run, because changing the MLP GEMM\n"
  f"# shapes is a genuine inductor cache miss under the champion max-autotune mode. That run\n"
  f"# left the 3x shapes cached, so this one should compile from them and finish in time.\n"
  f"# budget this ratio trades capacity directly against step count.\n\n"
  "class MLP(nn.Module):", 1)
s = s.replace("4 * config.n_embd", "MLP_RATIO * config.n_embd")
ast.parse(s)
p.write_text(s)
print(f"{node}: MLP ratio 4 -> {ratio} (hidden {512*4} -> {512*ratio} per layer)")
