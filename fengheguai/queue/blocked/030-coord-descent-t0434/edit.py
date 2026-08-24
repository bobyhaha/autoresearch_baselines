"""Enable inductor coordinate-descent tuning.

t0058 showed mode="max-autotune-no-cudagraphs" reaches 2921 steps against the champion's
2774 (+5.3%), but it ended as a timeout: the 300s training budget excludes compilation, so
a full GEMM autotune search blows the process wall clock. coordinate_descent_tuning is the
cheap subset -- it tunes block/stage configs for already-selected kernels rather than
searching backends -- so it should capture part of that gain at a small fraction of the
compile cost.

Usage: edit.py <node_id>
"""
import pathlib, ast, re, sys
node = sys.argv[1]
p = pathlib.Path(f"/data3/zhubaiyu/fengheguai/campaigns/h200-claude/nodes/{node}/train.py")
s = p.read_text()

anchor = 'torch.set_float32_matmul_precision("high")'
assert anchor in s, "float32 matmul precision anchor not found"
assert "coordinate_descent" not in s, "coordinate-descent tuning already enabled"
assert "max-autotune" not in s, "an autotune mode is already set; not a fresh coordinate"

s = s.replace(anchor,
  anchor + "\n"
  "# t0058 measured mode='max-autotune-no-cudagraphs' at 2921 steps vs the champion's 2774\n"
  "# (+5.3%, worth ~0.0029 bpb through the token law) but ended as a timeout: the 300s training\n"
  "# budget excludes compilation, so a full backend search blows the process wall clock.\n"
  "# coordinate_descent_tuning is the cheap subset -- it tunes block and stage configs for\n"
  "# kernels inductor has already chosen, instead of searching backends -- so it should buy\n"
  "# part of that throughput at a small fraction of the compile time.\n"
  "import torch._inductor.config as _inductor_config\n"
  "_inductor_config.coordinate_descent_tuning = True\n"
  "_inductor_config.coordinate_descent_check_all_directions = False", 1)
ast.parse(s)
p.write_text(s)
print(f"{node}: coordinate_descent_tuning enabled")
