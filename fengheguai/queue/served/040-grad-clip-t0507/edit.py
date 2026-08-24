"""Clip gradient norm on the AdamW groups only, before the optimizer step.

Gradient clipping appears in zero of 504 nodes. It is standard practice everywhere else, and
this model carries a `loss > 100` fast-fail, which suggests spikes were a live concern at some
point in its history.

Clipping globally would be close to a no-op for the matrices: Muon orthogonalises its updates,
so their scale is set by the algorithm rather than by the gradient norm. The groups that could
actually benefit are the AdamW ones -- embeddings at lr 0.6, value embeddings at 0.3, lm_head,
and the per-layer scalars at 0.5 -- which take raw Adam steps at unusually large learning rates.

Usage: edit.py <node_id> <max_norm>
"""
import pathlib, ast, sys
node, max_norm = sys.argv[1], float(sys.argv[2])
assert max_norm > 0, "max_norm must be positive"
p = pathlib.Path(f"/data3/zhubaiyu/fengheguai/campaigns/h200-claude/nodes/{node}/train.py")
s = p.read_text()
assert "clip_grad_norm_" not in s, "clipping already present"
anchor = "    optimizer.step()\n    model.zero_grad(set_to_none=True)"
assert anchor in s, "optimizer step block not in the expected form"
s = s.replace(anchor,
  f"    # Clip only the AdamW groups. Muon orthogonalises its updates, so a matrix's step size is\n"
  f"    # set by the algorithm rather than the gradient norm and clipping there changes little.\n"
  f"    # The AdamW groups take raw steps at large learning rates -- embeddings 0.6, value\n"
  f"    # embeddings 0.3, scalars 0.5 -- which is where an outlier batch can actually move a\n"
  f"    # weight too far. Zero nodes of 504 have ever clipped anything.\n"
  f"    _adamw_params = [q for g in optimizer.param_groups if g['kind'] != 'muon' for q in g['params']]\n"
  f"    torch.nn.utils.clip_grad_norm_(_adamw_params, {max_norm})\n"
  + anchor, 1)
ast.parse(s)
p.write_text(s)
print(f"{node}: clip_grad_norm_({max_norm}) on AdamW groups only")
