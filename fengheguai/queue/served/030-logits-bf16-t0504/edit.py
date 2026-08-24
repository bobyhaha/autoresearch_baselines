"""Drop the explicit fp32 upcast of the logits, keeping softcap in bf16.

The logits tensor is 131072 x 8192 = 1.07G elements: 4.29GB in fp32 against 2.15GB in bf16.
`logits.float()` writes the fp32 copy and every downstream op reads it. F.cross_entropy
upcasts internally for its log_softmax, so the explicit materialisation is not required for
numerical stability of the loss itself -- only the softcap tanh would now run in bf16.

Note t0382 tested a bf16 softcap reorder and measured it FREE, but that was with an fp32
lm_head, where the upcast was fused into the matmul epilogue. The champion's lm_head is now
bf16 (t0484/t0500), so the upcast is a standalone conversion of a 2.15GB tensor.

Usage: edit.py <node_id>
"""
import pathlib, ast, sys
node = sys.argv[1]
p = pathlib.Path(f"/data3/zhubaiyu/fengheguai/campaigns/h200-claude/nodes/{node}/train.py")
s = p.read_text()
assert "self.lm_head.to(dtype=torch.bfloat16)" in s, "expected the champion's bf16 lm_head"
old = """        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)"""
assert old in s, "logits path not in the expected form"
new = """        logits = self.lm_head(x)
        # No explicit .float() here. With lm_head held in bf16 this upcast is a standalone
        # conversion of a 1.07G-element tensor -- 2.15GB read, 4.29GB written -- and
        # F.cross_entropy upcasts internally for its log_softmax anyway. The softcap tanh now
        # runs in bf16. t0382 measured a softcap reorder as free, but that was with an fp32
        # lm_head where the conversion fused into the matmul epilogue; it does not now.
        logits = softcap * torch.tanh(logits / softcap)"""
s = s.replace(old, new, 1)
ast.parse(s)
p.write_text(s)
print(f"{node}: dropped the fp32 logits upcast (softcap now in bf16)")
