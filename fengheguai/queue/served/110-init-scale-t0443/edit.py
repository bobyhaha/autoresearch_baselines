"""Scale the shared fan-in initialisation constant.

init_weights uses s = sqrt(3) * n_embd^-0.5 for every attention projection and for mlp.c_fc.
That expression is byte-identical in all 441 nodes of this campaign -- the scale has never
been varied, while depth, width, head dim, every learning rate and every schedule have been.

Compile mode also reverts to default (wall-clock reason; known -0.00067 handicap).

Usage: edit.py <node_id> <multiplier>
"""
import pathlib, ast, re, sys
node, mult = sys.argv[1], float(sys.argv[2])
assert mult != 1.0, "1.0 is the incumbent; that would be a no-op"
assert 0.25 <= mult <= 4.0, "implausible multiplier"
p = pathlib.Path(f"/data3/zhubaiyu/fengheguai/campaigns/h200-claude/nodes/{node}/train.py")
s = p.read_text()

old = "        s = 3**0.5 * n_embd**-0.5"
assert old in s, "init scale expression not found in its expected form"
assert "INIT_SCALE" not in s, "INIT_SCALE already present"
s = s.replace(old,
  f"        # Fan-in init scale, multiplied by {mult}. The base expression sqrt(3)*n_embd^-0.5 is\n"
  f"        # byte-identical in all 441 nodes -- never varied, while every other structural and\n"
  f"        # optimisation constant has been. It sets c_q, c_k, c_v and mlp.c_fc; the two c_proj\n"
  f"        # matrices stay zero-initialised so each residual branch still starts neutral.\n"
  f"        s = {mult} * 3**0.5 * n_embd**-0.5", 1)

m = re.search(r'^model = torch\.compile\(model, dynamic=False, mode="max-autotune-no-cudagraphs"\)[^\n]*$', s, re.M)
if m:
    s = s.replace(m.group(0),
      "# Compile mode reverted to default for this probe; see the wall-clock analysis.\n"
      "model = torch.compile(model, dynamic=False)", 1)
ast.parse(s)
p.write_text(s)
print(f"{node}: init scale x{mult} (s = {mult} * sqrt(3) * n_embd^-0.5)")
