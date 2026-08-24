"""U-net style skip connections between the first and second halves of the stack.

The model injects the embedding x0 into every layer via x0_lambdas, but has no layer-to-layer
skips. This adds them: layer i in the first half pushes its input, layer n-1-i in the second
half adds it back, scaled by a learned scalar initialised to zero so the model starts exactly
where it is now and only departs if the skip earns it.

Four coordinated edits are required because setup_optimizer asserts that every parameter
belongs to exactly one group -- a new nn.Parameter that is not grouped crashes loudly at init.
That assert is the reason this is safe to attempt.

Also reverts compile mode to default: max-autotune leaves ~76s of wall-clock headroom and any
graph change is a cache miss that exceeds the 660s timeout (t0434/t0435/t0436/t0438 all died
there). The forfeited throughput is a known -0.00067.

Usage: edit.py <node_id>
"""
import pathlib, ast, re, sys
node = sys.argv[1]
p = pathlib.Path(f"/data3/zhubaiyu/fengheguai/campaigns/h200-claude/nodes/{node}/train.py")
s = p.read_text()
assert "skip_lambdas" not in s, "skip connections already present"

# 1. declare the parameter next to the existing scalars
old = "        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))"
assert old in s, "x0_lambdas declaration not found"
s = s.replace(old, old +
  "\n        # U-net skips: the first half of the stack pushes its input, the second half adds it\n"
  "        # back. Zero-init means the model starts exactly at the current champion and only\n"
  "        # departs if the skip earns it.\n"
  "        self.skip_lambdas = nn.Parameter(torch.zeros(config.n_layer // 2))", 1)

# 2. keep the zero-init explicit alongside the other scalar inits
old = "        self.x0_lambdas.fill_(0.1)"
assert old in s, "x0_lambdas init not found"
s = s.replace(old, old + "\n        self.skip_lambdas.zero_()", 1)

# 3. group it, and extend the completeness assert so the arithmetic stays true
old = "        x0_params = [self.x0_lambdas]"
assert old in s, "x0_params not found"
s = s.replace(old, old + "\n        skip_params = [self.skip_lambdas]", 1)
old = "len(resid_params) + len(x0_params))"
assert old in s, "parameter-count assert not found"
s = s.replace(old, "len(resid_params) + len(x0_params) + len(skip_params))", 1)
old = "            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),"
assert old in s, "x0 param group not found"
s = s.replace(old, old +
  "\n            dict(kind='adamw', params=skip_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),", 1)

# 4. wire the forward pass
old = """        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], varlen)"""
assert old in s, "forward loop does not match the expected form"
new = """        n_layer = len(self.transformer.h)
        half = n_layer // 2
        skips = []
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            if i < half:
                skips.append(x)
            else:
                # layer i pairs with layer n-1-i; skips is a stack so pop() gives that partner
                x = x + self.skip_lambdas[i - half] * skips.pop()
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], varlen)"""
s = s.replace(old, new, 1)

# 5. compile revert
m = re.search(r'^model = torch\.compile\(model, dynamic=False, mode="max-autotune-no-cudagraphs"\)[^\n]*$', s, re.M)
assert m, "expected the champion's max-autotune compile call"
s = s.replace(m.group(0),
  "# Compile mode reverted to default for this structural probe; see the wall-clock analysis.\n"
  "model = torch.compile(model, dynamic=False)", 1)

ast.parse(s)
p.write_text(s)
print(f"{node}: U-net skips added (6 learned scalars, zero-init), compile mode -> default")
