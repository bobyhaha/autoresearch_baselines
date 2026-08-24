"""Partial rotary: rotate only the first fraction of each head-dim half.

The model applies RoPE across the full 128-dim head. Partial rotary leaves the tail
position-independent, giving each head both position-sensitive and position-invariant
channels. Verified numerically on CPU first: frac=1.0 reproduces the current function
exactly, untouched channels pass through bit-exact, rotated channels match the full
rotation, and the rotated block preserves norm.

Usage: edit.py <node_id> <frac>
"""
import pathlib, ast, re, sys
node, frac = sys.argv[1], float(sys.argv[2])
assert 0.0 < frac < 1.0, "frac must be a strict fraction; 1.0 is the current behaviour"
p = pathlib.Path(f"/data3/zhubaiyu/fengheguai/campaigns/h200-claude/nodes/{node}/train.py")
s = p.read_text()

old = """def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)"""
assert old in s, "apply_rotary_emb does not match the expected full-rotary form"

new = f'''ROTARY_FRAC = {frac}  # rotate only this fraction of each head-dim half

def apply_rotary_emb(x, cos, sin):
    # Partial rotary. The first ROTARY_FRAC of each half is rotated as before; the tail is
    # passed through untouched, so every head carries both position-sensitive and
    # position-invariant channels. qk-norm is applied after this and sees the whole vector.
    assert x.ndim == 4
    d = x.shape[3] // 2
    r = int(d * ROTARY_FRAC)
    x1, x2 = x[..., :d], x[..., d:]
    c, s_ = cos[..., :r], sin[..., :r]
    a1, b1 = x1[..., :r], x1[..., r:]
    a2, b2 = x2[..., :r], x2[..., r:]
    y1 = torch.cat([a1 * c + a2 * s_, b1], 3)
    y2 = torch.cat([a1 * (-s_) + a2 * c, b2], 3)
    return torch.cat([y1, y2], 3)'''

s = s.replace(old, new, 1)
ast.parse(s)
p.write_text(s)
print(f"{node}: partial rotary frac={frac} (rotates {int(64*frac)} of 64 pairs)")
