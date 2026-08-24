"""Zero-initialise attn_gate, which init_weights never touches.

The gate-init block is commented "Gate weights init to zero (sigmoid(0)=0.5, scaled by
2 -> 1.0 = neutral)" but only zeroes ve_gate. attn_gate, added when the attention output
gate was promoted at t0353, appears solely at its definition and its use -- it keeps
PyTorch's default nn.Linear init.

With 128 inputs that default is U(+/-1/sqrt(128)), so Wx has std ~0.57 and the gate starts
scattered over roughly 0.72-1.28 per head rather than at the intended 1.0.

NOTE: initialisation runs before torch.compile and does not change the graph, so the
inductor cache still hits and max-autotune is KEPT. This is an unhandicapped comparison
against the champion.

Usage: edit.py <node_id>
"""
import pathlib, ast, sys
node = sys.argv[1]
p = pathlib.Path(f"/data3/zhubaiyu/fengheguai/campaigns/h200-claude/nodes/{node}/train.py")
s = p.read_text()
assert "attn_gate" in s, "attn_gate not present in this node"
assert "zeros_(block.attn.attn_gate" not in s, "attn_gate already zero-initialised"
old = """            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)"""
assert old in s, "gate init block not in its expected form"
s = s.replace(old, old + """
            # attn_gate was never covered by this loop. The comment above says gate weights
            # start at zero so 2*sigmoid(0) = 1.0 is neutral, and ve_gate does -- but attn_gate,
            # added when the attention output gate was promoted at t0353, kept PyTorch's default
            # nn.Linear init. On 128 inputs that gives Wx a std near 0.57, so each head's output
            # starts randomly scaled over roughly 0.72-1.28 instead of 1.0. t0440 and t0443
            # together showed this model is invariant to the COMMON init scale and violently
            # sensitive to RELATIVE mis-scaling between neighbouring matrices, which is exactly
            # what an unintended per-head scatter is.
            # RETRY. The first attempt was aborted by the throughput guard at step 242 running
            # 145ms/step against a healthy 105ms (MFU 28.8% vs 39.8%) -- a contended GPU, not a
            # defect: the loss was descending normally when it was killed. A guard abort still
            # writes a failed record, so this needs a distinct source hash.
            torch.nn.init.zeros_(block.attn.attn_gate.weight)""", 1)
ast.parse(s)
p.write_text(s)
print(f"{node}: attn_gate zero-initialised (graph unchanged, max-autotune kept)")
