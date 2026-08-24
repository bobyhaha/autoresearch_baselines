"""Trade width for depth at roughly constant compute: 12 layers x 512 -> 21 layers x 384.

The two capacity brackets measured this session point in opposite directions:
    ADDING params    width pays ~2x depth   (-0.000442 vs -0.000209 per % of params)
    REMOVING params  depth hurts more       (3.61x convexity vs 2.3x for width)

Those were measured one step either side of the champion. This asks whether the aspect ratio
itself is wrong, by moving several steps along the depth-width trade at held compute:
depth*width^2 goes 12*512^2 = 3145728 to 8*640^2 = 3276800, +4.2%.

Usage: edit.py <node_id>
"""
import pathlib, ast, re, sys
node = sys.argv[1]
p = pathlib.Path(f"/data3/zhubaiyu/fengheguai/campaigns/h200-claude/nodes/{node}/train.py")
s = p.read_text()
d = re.search(r"^DEPTH = (\d+)[^\n]*", s, re.M)
a = re.search(r"^ASPECT_RATIO = (\d+)[^\n]*", s, re.M)
assert d and a, "DEPTH / ASPECT_RATIO not found"
assert d.group(1) == "12" and a.group(1) == "42", f"expected champion 12/42, got {d.group(1)}/{a.group(1)}"
assert ((21*18 + 127)//128)*128 == 384, "aspect 18 does not round to 384 at depth 21"

s = s.replace(a.group(0),
  "ASPECT_RATIO = 18       # 42 -> 18, paired with DEPTH 21 below: 21*18 = 378 rounds up to width\n"
  "# 384 with 3 heads. depth*width^2 goes 12*512^2 = 3145728 -> 21*384^2 = 3096576, -1.6%, so this\n"
  "# is a constant-compute reallocation in the OPPOSITE direction from the 8x640 arm.", 1)
s = s.replace(d.group(0),
  "DEPTH = 21              # 12 -> 21, trading width for depth at held compute. This is the other\n"
  "# side of the aspect-ratio bracket: 8 layers at 640 tests wider-and-shallower, this tests\n"
  "# deeper-and-narrower. Width must be a multiple of HEAD_DIM 128, so the reachable\n"
  "# constant-compute points are sparse: 12x512, 8x640, 21x384, 5x768.", 1)

mm = re.search(r'^model = torch\.compile\(model, dynamic=False, mode="max-autotune-no-cudagraphs"\)[^\n]*$', s, re.M)
if mm:
    s = s.replace(mm.group(0),
      "# Compile mode reverted to default: this changes every GEMM shape in the model.\n"
      "model = torch.compile(model, dynamic=False)", 1)
ast.parse(s)
p.write_text(s)
print(f"{node}: DEPTH 21 @ ASPECT_RATIO 18 -> width 384, 3 heads (constant-compute trade)")
