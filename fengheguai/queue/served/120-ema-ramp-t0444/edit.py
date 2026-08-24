"""Shape the EMA decay instead of holding it constant.

Weight averaging starts at EMA_START of the budget and uses a constant EMA_DECAY = 0.99
via e_.lerp_(q_, 1 - EMA_DECAY). Decay SHAPE at held mean has won four times in this
campaign -- table decay flat->linear->squared, matrix decay linear->squared, the
value-embedding decay ramp, and warmdown -- but the EMA's own decay has never been shaped.

A ramp from 1.5*(1-d) down to 0.5*(1-d) has the same mean (1-d) as the constant, so the total amount
of averaging is held and only its distribution over the averaging window moves. Early in the
window the average tracks the live weights closely; late it becomes nearly frozen, which is
the front-loaded profile the other four wins share.

Compile mode also reverts to default (wall-clock reason; known -0.00067 handicap).

Usage: edit.py <node_id>
"""
import pathlib, ast, re, sys
node = sys.argv[1]
p = pathlib.Path(f"/data3/zhubaiyu/fengheguai/campaigns/h200-claude/nodes/{node}/train.py")
s = p.read_text()

old = "                    e_.lerp_(q_.detach().float(), 1 - EMA_DECAY)"
assert old in s, "EMA lerp not found in its expected form"
assert "ema_alpha" not in s, "EMA ramp already present"

new = """                    # Shaped EMA. The constant 1-EMA_DECAY is replaced by a linear ramp
                    # from 1.5x to 0.5x of it across the averaging window, whose mean
                    # is exactly 1-EMA_DECAY -- so the total averaging is unchanged and only
                    # its distribution moves. Early in the window the average tracks the live
                    # weights more; late it has longer memory. Flat -> front-loaded at held mean has
                    # won four times here on other decay schedules.
                    ema_progress = min(max((progress - EMA_START) / max(1.0 - EMA_START, 1e-9), 0.0), 1.0)
                    # 1.5x down to 0.5x rather than 2x down to 0: an alpha reaching zero would
                    # freeze the average outright, so the final steps -- the best-annealed ones
                    # under warmdown -- would contribute nothing and the result would be a
                    # mid-window snapshot. Mean is still exactly 1-EMA_DECAY.
                    ema_alpha = (1 - EMA_DECAY) * (1.5 - ema_progress)
                    e_.lerp_(q_.detach().float(), ema_alpha)"""
s = s.replace(old, new, 1)

m = re.search(r'^model = torch\.compile\(model, dynamic=False, mode="max-autotune-no-cudagraphs"\)[^\n]*$', s, re.M)
if m:
    s = s.replace(m.group(0),
      "# Compile mode reverted to default for this probe; see the wall-clock analysis.\n"
      "model = torch.compile(model, dynamic=False)", 1)
ast.parse(s)
p.write_text(s)
print(f"{node}: EMA decay shaped (ramp at held mean), compile mode -> default")
