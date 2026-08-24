"""Widen the autotune search space to EXHAUSTIVE, keeping the champion's mode and cache.

The champion uses mode="max-autotune-no-cudagraphs", whose DEFAULT gemm search space is a
curated subset of block/stage configurations. The control run measured what that is worth:
2806 steps against 2744 without it, +2.26%, or -0.001309 of bpb through the token law.
EXHAUSTIVE searches a much larger config set for the same kernels.

Nothing about the model changes, so the eval graph is already cached from the champion's own
runs -- only the training GEMMs get re-searched.

Usage: edit.py <node_id>
"""
import pathlib, ast, re, sys
node = sys.argv[1]
p = pathlib.Path(f"/data3/zhubaiyu/fengheguai/campaigns/h200-claude/nodes/{node}/train.py")
s = p.read_text()
assert "max-autotune-no-cudagraphs" in s, "expected the champion's autotune mode"
assert "search_space" not in s, "search space already set"
anchor = 'torch.set_float32_matmul_precision("high")'
assert anchor in s, "anchor not found"
s = s.replace(anchor, anchor + "\n"
  "# EXHAUSTIVE gemm autotune. The control (t0445) measured the champion's DEFAULT search space\n"
  "# at +2.26% steps over no autotune, worth -0.001309 by the token law -- the single largest\n"
  "# throughput lever found in this campaign. EXHAUSTIVE searches a much larger config set for\n"
  "# the same kernels. The model is unchanged, so the eval graph stays cached from the\n"
  "# champion's runs and only the training GEMMs are re-searched.\n"
  "import torch._inductor.config as _inductor_config\n"
  '_inductor_config.max_autotune_gemm_search_space = "EXHAUSTIVE"', 1)
ast.parse(s)
p.write_text(s)
print(f"{node}: autotune search space -> EXHAUSTIVE")
