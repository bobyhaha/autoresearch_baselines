#!/usr/bin/env python3
"""Regenerate progress.html from the campaign journal.

The page's data block used to be hand-maintained, which does not survive a campaign
that adds a trial every eight minutes. This rewrites the `const D = [...]` array in
place from `campaign/journal.json`, so refreshing the chart is one command and the
page can never drift from the journal.

Short labels for the chart are derived from each node's plan text: the full plan goes
in the tooltip and the table, while the chart carries something that fits at 86 points.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
NODES = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

# Plan text -> a label short enough to sit beside a marker.
RULES = [
    (r"baseline", "baseline"),
    (r"RETEST.*2\*\*19", "retest batch 2^19"), (r"RETEST.*2\*\*17", "retest batch 2^17"),
    (r"RETEST.*DEPTH", "retest depth 12"),     (r"RETEST.*width", "retest width 768"),
    (r"CONSTANT-CAPACITY.*3x", "mlp3 + depth12"), (r"CONSTANT-CAPACITY.*5x", "mlp5 + depth9"),
    (r"AUTO-REPLICATE|REPLICATE", "replicate"),
    (r"TOTAL_BATCH_SIZE 2\*\*19 -> 2\*\*18", "batch 524K->262K"),
    (r"TOTAL_BATCH_SIZE 2\*\*18 -> 2\*\*17", "batch 262K->131K"),
    (r"TOTAL_BATCH_SIZE 2\*\*18 -> 2\*\*19", "batch 262K->524K"),
    (r"short-attention span \(seq/2", "short window 1/4"),
    (r"seq/4=512 -> seq/8", "short window 1/8"), (r"seq/8 -> seq/4", "short window 1/4"),
    (r"seq/8 -> seq/16", "short window 1/16"),
    (r"max-autotune", "max-autotune compile"),
    (r"DEPTH 8 -> 6", "depth 6"), (r"DEPTH 8 -> 10", "depth 10"),
    (r"DEPTH 10 -> 12", "depth 12"), (r"DEPTH 10 -> 9", "depth 9"),
    (r"ASPECT_RATIO 72|width 640 -> 768|model_dim 640 -> 768", "width 768"),
    (r"ASPECT_RATIO 48|model_dim 640 -> 512", "width 512"),
    (r"HEAD_DIM", "head dim 64"), (r"WINDOW_PATTERN", "pattern SSSSSSSL"),
    (r"EMBEDDING_LR 0?\.?6 -> 0\.8|EMBEDDING_LR probe", "embed LR 0.8"),
    (r"embedding-LR axis: 0\.8 -> 1\.0", "embed LR 1.0"),
    (r"embedding LR probe at 0\.9|EMBEDDING_LR fine probe at 0\.9", "embed LR 0.9"),
    (r"at 0\.85", "embed LR 0.85"), (r"0\.9 -> 0\.95", "embed LR 0.95"),
    (r"UNEMBEDDING_LR 0\.004 -> 0\.006", "unembed LR 0.006"),
    (r"UNEMBEDDING_LR.*0\.008", "unembed LR 0.008"),
    (r"UNEMBEDDING_LR 0\.006 -> 0\.005", "unembed LR 0.005"),
    (r"MATRIX_LR 0\.04 -> 0\.05|Muon LR", "matrix LR 0.05"),
    (r"MATRIX_LR 0\.04 -> 0\.03", "matrix LR 0.03"), (r"MATRIX_LR 0\.04 -> 0\.035", "matrix LR 0.035"),
    (r"WARMDOWN_RATIO 0\.5 -> 0\.35", "warmdown 0.35"), (r"WARMDOWN_RATIO 0\.5 -> 0\.7", "warmdown 0.7"),
    (r"WARMUP_RATIO", "5% warmup"), (r"SCALAR_LR 0\.5 -> 0\.7", "scalar LR 0.7"),
    (r"SCALAR_LR 0\.5 -> 0\.3", "scalar LR 0.3"),
    (r"WEIGHT_DECAY 0\.2 -> 0\.1", "weight decay 0.1"), (r"WEIGHT_DECAY 0\.1 -> 0\.05", "weight decay 0.05"),
    (r"WEIGHT_DECAY 0\.1 -> 0\.15", "weight decay 0.15"), (r"WEIGHT_DECAY 0\.15 -> 0\.2", "weight decay 0.2"),
    (r"FINAL_LR_FRAC 0\.0 -> 0\.1", "final LR 0.1"), (r"FINAL_LR_FRAC 0\.0 -> 0\.05", "final LR 0.05"),
    (r"FINAL_LR_FRAC 0\.05 -> 0\.025", "final LR 0.025"), (r"FINAL_LR_FRAC 0\.05 -> 0\.075", "final LR 0.075"),
    (r"RoPE base frequency 10000 -> 50000", "RoPE base 50k"), (r"-> 100000", "RoPE base 100k"),
    (r"25000", "RoPE base 25k"), (r"75000", "RoPE base 75k"),
    (r"x0_lambda init 0\.1 -> 0\.05", "x0_lambda 0.05"), (r"x0_lambda init 0\.1 -> 0\.15", "x0_lambda 0.15"),
    (r"beta1 0\.8 -> 0\.9", "adam beta1 0.9"), (r"beta1 0\.8 -> 0\.7", "adam beta1 0.7"),
    (r"beta2", "adam beta2 0.99"),
    (r"momentum warmup stretched", "muon warmup 500"), (r"momentum warmup shortened", "muon warmup 150"),
    (r"momentum range 0\.85->0\.95 raised", "muon mom 0.90-0.98"), (r"momentum range lowered", "muon mom 0.80-0.92"),
    (r"Value embeddings on every layer", "value emb all layers"),
    (r"relu-squared -> gelu", "activation gelu"), (r"relu-squared -> relu-cubed", "activation relu^3"),
    (r"QK normalisation", "no QK norm"), (r"rotary table", "rotary table 4x"),
    (r"MLP expansion 4x -> 3x", "mlp 3x"), (r"MLP expansion 4x -> 5x", "mlp 5x"),
    (r"DEVICE_BATCH_SIZE 128 -> 64", "grad accum 2"),
    (r"matmul precision", "fp32 matmul highest"),
]


def short(plan: str) -> str:
    for pat, lab in RULES:
        if re.search(pat, plan, re.I):
            return lab
    return plan.split(".")[0][:34]


rows = []
for n in NODES:
    if n["bpb"] is None:
        continue
    plan = n["plan"].replace("\\", " ").replace('"', "'").replace("\n", " ")
    rows.append(
        f' {{i:{n["i"]}, id:"{n["id"]}", bpb:{n["bpb"]:.6f}, p:{n["p"]}, s:{n["s"]}, '
        f'mfu:{n["mfu"]}, vram:{n["vram"]}, short:"{short(plan)}", plan:"{plan[:150]}"}},'
    )

page = (HERE / "progress.html").read_text(encoding="utf-8")
block = "const D = [\n" + "\n".join(rows) + "\n];"
page = re.sub(r"const D = \[.*?\n\];", block, page, flags=re.S)
(HERE / "progress.html").write_text(page, encoding="utf-8")
print(f"rebuilt progress.html with {len(rows)} scored nodes")
