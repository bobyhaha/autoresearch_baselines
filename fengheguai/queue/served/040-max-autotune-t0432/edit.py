"""max-autotune-no-cudagraphs with a persistent inductor cache.

t0058 measured this mode at 2921 steps vs the champion's 2774 (+5.3%) but ended as a timeout:
the 300s TRAINING budget excludes compilation, so a full backend search blows the process
wall clock while leaving the training window intact. A cache directory shared across nodes
means the autotune search is paid once rather than per trial.

Setting TORCHINDUCTOR_CACHE_DIR does NOT buy training time -- compilation is outside the
measured 300s either way. It only affects whether the process finishes inside the harness
wall clock.

Usage: edit.py <node_id>
"""
import pathlib, ast, re, sys
node = sys.argv[1]
p = pathlib.Path(f"/data3/zhubaiyu/fengheguai/campaigns/h200-claude/nodes/{node}/train.py")
s = p.read_text()

assert "max-autotune" not in s, "an autotune mode is already set"
assert "TORCHINDUCTOR_CACHE_DIR" not in s, "cache dir already set"
m = re.search(r"^model = torch\.compile\(model, dynamic=False\)[^\n]*$", s, re.M)
assert m, "torch.compile(model, dynamic=False) call site not found"

first_import = re.search(r"^import torch$", s, re.M)
assert first_import, "no bare 'import torch' to anchor the cache-dir setting"
s = s.replace(first_import.group(0),
  "import os as _os\n"
  "# Share the inductor autotune cache across nodes so the search is paid once, not per trial.\n"
  "# This does not buy training time: compilation sits outside the measured 300s either way.\n"
  "# It only decides whether the process finishes inside the harness wall clock, which is what\n"
  "# ended t0058 as a timeout despite it reaching 2921 steps against the champion's 2774.\n"
  '_os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data3/zhubaiyu/fengheguai/.inductor_cache")\n'
  "import torch", 1)

s = s.replace(m.group(0),
  'model = torch.compile(model, dynamic=False, mode="max-autotune-no-cudagraphs")'
  '  # t0058: 2921 steps vs 2774 (+5.3%)', 1)
ast.parse(s)
p.write_text(s)
print(f"{node}: max-autotune-no-cudagraphs + shared inductor cache")
