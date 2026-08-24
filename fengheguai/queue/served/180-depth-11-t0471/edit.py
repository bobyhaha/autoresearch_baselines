"""Test depth 11 -- the other side of the depth bracket against the current champion, with compile reverted.

DEPTH was explored at 8/10/11/12/13/14 early in the campaign and 12 held. The operating
point has moved a long way since: value embeddings on every layer, the attention gate,
four reshaped decay schedules, warmdown 0.7, and max-autotune. The champion is 3% better
than the model those depth trials were run against.

Usage: edit.py <node_id>
"""
import pathlib, ast, re, sys
node = sys.argv[1]
p = pathlib.Path(f"/data3/zhubaiyu/fengheguai/campaigns/h200-claude/nodes/{node}/train.py")
s = p.read_text()
m = re.search(r"^DEPTH = (\d+)[^\n]*", s, re.M)
assert m, "DEPTH constant not found"
old = m.group(1)
assert old == "12", f"expected the champion's depth 12, found {old}"
s = s.replace(m.group(0),
  "DEPTH = 11              # 12 -> 11, the arm below the champion, completing the depth\n"
  "# bracket against depth 13. Depth was explored at 8/10/11/12/13/14 early on and 12 held,\n"
  "# but that was against a model 3% worse, before value embeddings on every layer, the\n"
  "# attention gate, four reshaped decay schedules and max-autotune.\n"
  "# The MLP bracket found capacity strongly convex: removing width cost +0.006633 while\n"
  "# adding it gained only -0.002872. If depth behaves the same way this should lose by more\n"
  "# than depth 13 gains. If the two arms are symmetric instead, capacity is not one scalar\n"
  "# for this model and where the parameters go matters.\n"
  "# Note this removes a value-embedding table as well as a block, so the residual bundles\n"
  "# FLOP-bearing and FLOP-free capacity -- symmetrically with the depth-13 arm.", 1)
mm = re.search(r'^model = torch\.compile\(model, dynamic=False, mode="max-autotune-no-cudagraphs"\)[^\n]*$', s, re.M)
if mm:
    s = s.replace(mm.group(0),
      "# Compile mode reverted to default: depth changes the graph, and a cache miss under\n"
      "# max-autotune exceeds the 660s wall clock (t0434/t0435/t0436/t0438).\n"
      "model = torch.compile(model, dynamic=False)", 1)
ast.parse(s)
p.write_text(s)
print(f"{node}: DEPTH {old} -> 11, compile mode -> default")
