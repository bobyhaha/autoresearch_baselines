"""Retest depth 13 against the current champion, with compile reverted.

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
  "DEPTH = 13              # 12 -> 13. RETRY: the first attempt was aborted by the throughput\n"
  "# guard at step 105 for running 6.56 steps/s, below its old 8.0 floor -- while reporting MFU\n"
  "# 41.4%, HIGHER than the champion's healthy 39.8%. It was a legitimately slower model running\n"
  "# at full efficiency; the guard could not tell that from contention because it watched step\n"
  "# rate. It now watches MFU, which separates the two cleanly (contended runs sat at 28.8%).\n"
  "# Depth was explored at 8/10/11/12/13/14 early on and 12 held,\n"
  "# but that was against a model 3% worse, before value embeddings on every layer, the attention\n"
  "# gate, four reshaped decay schedules and max-autotune. The MLP bracket since showed capacity is\n"
  "# binding and strongly convex here (removing width costs 2.3x what adding it gains), and depth\n"
  "# is the other way to buy capacity.", 1)
mm = re.search(r'^model = torch\.compile\(model, dynamic=False, mode="max-autotune-no-cudagraphs"\)[^\n]*$', s, re.M)
if mm:
    s = s.replace(mm.group(0),
      "# Compile mode reverted to default: depth changes the graph, and a cache miss under\n"
      "# max-autotune exceeds the 660s wall clock (t0434/t0435/t0436/t0438).\n"
      "model = torch.compile(model, dynamic=False)", 1)
ast.parse(s)
p.write_text(s)
print(f"{node}: DEPTH {old} -> 13, compile mode -> default")
