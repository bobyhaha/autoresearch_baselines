"""Swap ReLU^2 MLP for SwiGLU at matched parameter count, and revert compile mode.

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

assert "c_gate" not in s, "SwiGLU already present"
n = len(re.findall(r"4 \* config\.n_embd", s))
assert n == 2, f"expected the stock ReLU^2 MLP with 2 width sites, found {n}"

old_mlp = """        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x"""
assert old_mlp in s, "MLP body does not match the expected ReLU^2 form"
new_mlp = """        # SwiGLU at matched parameter count. ReLU^2 used 2 matrices of width 4*n_embd
        # (2*512*2048 = 2,097,152); this uses 3 of width 1344 (3*512*1344 = 2,064,384, -1.56%),
        # so capacity and per-step FLOPs are held ~constant and only the functional form moves.
        # 1344 is a multiple of 64 to stay tensor-core friendly.
        hidden = 1344
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        x = F.silu(self.c_gate(x)) * self.c_fc(x)
        x = self.c_proj(x)
        return x"""
s = s.replace(old_mlp, new_mlp, 1)

old_init = "            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)"
assert old_init in s, "MLP c_fc init line not found"
s = s.replace(old_init, old_init +
  "\n            # c_gate takes the same fan-in scale as c_fc. t0440 omitted this line, left the\n"
  "\n            # gate on PyTorch's default nn.Linear init, and scored 1.0446 -- worse than the\n"
  "\n            # original baseline. That result measured an uninitialised gate, not SwiGLU.\n"
  "            torch.nn.init.uniform_(block.mlp.c_gate.weight, -s, s)", 1)

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
print(f"{node}: SwiGLU hidden 1344 + c_gate init, compile mode -> default")
