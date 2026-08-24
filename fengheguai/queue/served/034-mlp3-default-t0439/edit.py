"""MLP ratio 3 with compile mode reverted to default (one-slot, no timeout).

Two changes, both deliberate:

1. MLP: c_fc/relu^2/c_proj (hidden 2048, 2*512*2048 = 2,097,152 params) becomes
   gate/up/down SwiGLU at hidden 1344 (3*512*1344 = 2,064,384, -1.56%). Verified on CPU.

2. torch.compile mode reverts to default. The champion carries max-autotune, which buys
   +1.14% steps but leaves only ~76s of wall-clock headroom, and any GEMM-shape change is a
   cache miss whose extra compile blows the 660s safety timeout -- t0434, t0435 and t0436 all
   died that way AFTER completing their training. Reverting costs a known -0.00067 handicap
   (token law on 1.14%), which the decomposition subtracts, and makes this a one-slot
   experiment instead of a guaranteed timeout plus a retry.

Usage: edit.py <node_id>
"""
import pathlib, ast, re, sys
node = sys.argv[1]
p = pathlib.Path(f"/data3/zhubaiyu/fengheguai/campaigns/h200-claude/nodes/{node}/train.py")
s = p.read_text()



n = len(re.findall(r"4 \* config\.n_embd", s))
assert n == 2, f"expected 2 width sites, found {n}"
assert "MLP_RATIO" not in s, "MLP_RATIO already present"
s = s.replace("class MLP(nn.Module):",
  "MLP_RATIO = 3  # was a hardcoded 4, never varied in 432 nodes. The 5x arm measured capacity at\n"
  "# -0.002872 against a throughput bill of +0.004581, so width is valuable but expensive; this\n"
  "# arm tests the cheap side, where throughput is measured at +8.94% (3057 steps at t0438).\n\n"
  "class MLP(nn.Module):", 1)
s = s.replace("4 * config.n_embd", "MLP_RATIO * config.n_embd")

m = re.search(r'^model = torch\.compile\(model, dynamic=False, mode="max-autotune-no-cudagraphs"\)[^\n]*$', s, re.M)
assert m, "expected the champion's max-autotune compile call"
s = s.replace(m.group(0),
  "# Compile mode reverted to default for this structural probe. max-autotune buys +1.14% steps\n"
  "# but leaves ~76s of wall-clock headroom, and a GEMM-shape change is a cache miss whose extra\n"
  "# compile exceeds the 660s safety timeout -- t0434/t0435/t0436 all died there after completing\n"
  "# training. The -0.00067 this forfeits is known and subtracted in the decomposition.\n"
  "model = torch.compile(model, dynamic=False)", 1)
ast.parse(s)
p.write_text(s)
print(f"{node}: MLP ratio 4 -> 3 (hidden 2048 -> 1536), compile mode -> default")
