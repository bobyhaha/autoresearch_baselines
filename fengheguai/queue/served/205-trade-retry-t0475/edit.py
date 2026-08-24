"""Trade depth for width at roughly constant compute: 12 layers x 512 -> 8 layers x 640.

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
assert ((8*72 + 127)//128)*128 == 640, "aspect 72 does not round to 640 at depth 8"

s = s.replace(a.group(0),
  "ASPECT_RATIO = 72       # 42 -> 72, paired with DEPTH 8 below: 8*72 = 576 rounds up to width\n"
  "# 640 with 5 heads. depth*width^2 goes 12*512^2 = 3145728 -> 8*640^2 = 3276800, +4.2%, so this\n"
  "# is close to a constant-compute reallocation rather than a size change.", 1)
s = s.replace(d.group(0),
  "DEPTH = 8               # 12 -> 8, trading four layers for 25% more width at held compute.\n"
  "# RETRY: the first attempt was aborted by the throughput guard on a single transient MFU dip\n"
  "# to 31.3%, while the run was averaging 9.63 steps/s at 43.5% MFU -- FASTER than the champion.\n"
  "# That was a guard defect, not a property of this configuration: MFU is instantaneous and\n"
  "# noisy where the step rate it replaced was cumulative. The guard now judges the median of\n"
  "# the last 20 readings, which still aborts sustained contention (t0451/t0453 at ~29%) and no\n"
  "# longer kills a healthy run on one spike.\n"
  "# The two capacity brackets measured this session disagree about where a parameter is worth\n"
  "# most: ADDING favours width ~2:1 (-0.000442 vs -0.000209 per % of params), while REMOVING\n"
  "# punishes depth harder (3.61x convexity vs 2.3x). Both were measured one step from the\n"
  "# champion. This moves four steps to ask whether the aspect ratio itself is mis-set.", 1)

mm = re.search(r'^model = torch\.compile\(model, dynamic=False, mode="max-autotune-no-cudagraphs"\)[^\n]*$', s, re.M)
if mm:
    s = s.replace(mm.group(0),
      "# Compile mode reverted to default: this changes every GEMM shape in the model.\n"
      "model = torch.compile(model, dynamic=False)", 1)
ast.parse(s)
p.write_text(s)
print(f"{node}: DEPTH 8 @ ASPECT_RATIO 72 -> width 640, 5 heads (constant-compute trade)")
