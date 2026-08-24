"""Depth 13 at width 512 -- a pure depth increase, which this config makes awkward to express.

model_dim is derived from depth: ceil(depth*ASPECT_RATIO / 128)*128. At ASPECT_RATIO 42,
depth 13 rounds to width 640 and 5 heads, making t0459 a 47.8%-larger model rather than a
deeper one. Pairing depth 13 with ASPECT_RATIO 39 gives 13*39 = 507 -> 512, 4 heads: the
champion's exact width and head count with one extra layer.

The two constants are one conceptual change; ASPECT_RATIO exists only to encode the
depth->width relation, and holding width fixed while depth moves requires moving it.

Usage: edit.py <node_id>
"""
import pathlib, ast, re, sys
node = sys.argv[1]
p = pathlib.Path(f"/data3/zhubaiyu/fengheguai/campaigns/h200-claude/nodes/{node}/train.py")
s = p.read_text()
md = re.search(r"^DEPTH = (\d+)[^\n]*", s, re.M)
ar = re.search(r"^ASPECT_RATIO = (\d+)[^\n]*", s, re.M)
assert md and ar, "DEPTH / ASPECT_RATIO not found"
assert md.group(1) == "12" and ar.group(1) == "42", f"expected champion 12/42, found {md.group(1)}/{ar.group(1)}"
# verify the arithmetic the config will do
assert ((13*39 + 127)//128)*128 == 512, "aspect 39 does not round to 512 at depth 13"

s = s.replace(ar.group(0),
  "ASPECT_RATIO = 39       # 42 -> 39, paired with DEPTH 13 below so that 13*39 = 507 rounds up to\n"
  "# width 512 with 4 heads -- the champion's exact width. At the old 42, depth 13 gives 546 -> 640\n"
  "# and 5 heads, which is what made t0459 a 47.8%-larger model (142.6M vs 96.5M) rather than a\n"
  "# deeper one, and why its steps fell 33.9%.", 1)
s = s.replace(md.group(0),
  "DEPTH = 13              # 12 -> 13 at CONSTANT WIDTH. Within the 512 plateau deeper has always\n"
  "# won -- depths 10, 11 and 12 all round to width 512 and 12 took it -- but 12 is the largest\n"
  "# depth that still fits 512 at aspect 42, so the champion sits at the top of a width plateau by\n"
  "# accident of rounding rather than by measurement. This is the first test of depth 13 that is\n"
  "# actually about depth.", 1)

mm = re.search(r'^model = torch\.compile\(model, dynamic=False, mode="max-autotune-no-cudagraphs"\)[^\n]*$', s, re.M)
if mm:
    s = s.replace(mm.group(0),
      "# Compile mode reverted to default: depth changes the graph and a cache miss under\n"
      "# max-autotune exceeds the 660s wall clock.\n"
      "model = torch.compile(model, dynamic=False)", 1)
ast.parse(s)
p.write_text(s)
print(f"{node}: DEPTH 13 @ ASPECT_RATIO 39 -> width 512, 4 heads (pure depth change)")
