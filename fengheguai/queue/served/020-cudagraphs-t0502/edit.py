"""Enable CUDA graphs on top of the champion's autotune.

The champion runs mode="max-autotune-no-cudagraphs" at ~111ms/step and 37-40% MFU. For a
12-layer model with a 512-wide residual stream, much of the gap to peak is kernel LAUNCH
overhead rather than arithmetic -- exactly what CUDA graphs remove, by replaying a captured
graph instead of dispatching each kernel.

Throughput is the only axis still paying in this campaign: all three promotions today were
throughput (autotune +2.26%, bf16 lm_head +2.73%, and their +5.05% combination).

Usage: edit.py <node_id>
"""
import pathlib, ast, sys
node = sys.argv[1]
p = pathlib.Path(f"/data3/zhubaiyu/fengheguai/campaigns/h200-claude/nodes/{node}/train.py")
s = p.read_text()
assert "max-autotune-no-cudagraphs" in s, "expected the champion's compile mode"
assert "cudagraphs = True" not in s, "cudagraphs already enabled"
anchor = 'torch.set_float32_matmul_precision("high")'
assert anchor in s, "anchor not found"
s = s.replace(anchor, anchor + "\n"
  "# CUDA graphs on top of the existing autotune. At ~111ms/step and 37-40% MFU with a 512-wide\n"
  "# residual stream, a large part of the gap to peak is kernel launch overhead rather than\n"
  "# arithmetic, and graph replay removes exactly that. The compile mode stays\n"
  "# max-autotune-no-cudagraphs so the tuned kernel selection is kept; this adds capture on top.\n"
  "# A memory from a different campaign (VPA) records plain max-autotune segfaulting through\n"
  "# CUDA graphs -- no node in THIS campaign has ever tried them, so that is a warning rather\n"
  "# than a result, and this trial is what turns it into one.\n"
  "import torch._inductor.config as _inductor_config\n"
  "_inductor_config.triton.cudagraphs = True", 1)
ast.parse(s)
p.write_text(s)
print(f"{node}: CUDA graphs enabled on top of max-autotune-no-cudagraphs")
