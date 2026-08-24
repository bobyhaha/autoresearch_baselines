"""Control: the champion, unchanged, with compile mode reverted to default.

Every structural probe since t0439 runs under default compile and is compared against a
max-autotune champion using an ESTIMATED handicap. That estimate is doing real work -- it
decided that init scale 1.25 was a null rather than a loss -- and it has never been measured.
This run measures it: identical model, identical data, identical everything except the
compile mode.

Usage: edit.py <node_id>
"""
import pathlib, ast, re, sys
node = sys.argv[1]
p = pathlib.Path(f"/data3/zhubaiyu/fengheguai/campaigns/h200-claude/nodes/{node}/train.py")
s = p.read_text()
m = re.search(r'^model = torch\.compile\(model, dynamic=False, mode="max-autotune-no-cudagraphs"\)[^\n]*$', s, re.M)
assert m, "expected the champion's max-autotune compile call"
assert "CONTROL" not in s, "already marked as the control"
s = s.replace(m.group(0),
  "# CONTROL. The model is byte-identical to the champion; only the compile mode changes.\n"
  "# Every probe since t0439 has been compared to the champion through an ESTIMATED -0.00067\n"
  "# autotune handicap, derived from t0433's +1.14% step gain. That estimate decided whether\n"
  "# init scale 1.25 was a null or a loss, and it has never been measured directly. Default-\n"
  "# compile runs today spanned 2643-2978 steps across different graphs, so the correction also\n"
  "# carries run-to-run variance that no single trial can separate from mechanism.\n"
  "model = torch.compile(model, dynamic=False)", 1)
ast.parse(s)
p.write_text(s)
print(f"{node}: CONTROL — champion model, default compile")
