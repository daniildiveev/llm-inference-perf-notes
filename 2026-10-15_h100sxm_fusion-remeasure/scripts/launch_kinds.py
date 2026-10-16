"""Count CUDA runtime launch API calls inside profiled engine steps (graph replay vs eager kernel launches).
Usage: python launch_kinds.py <trace_dir> [...]"""
import collections, sys
sys.path.insert(0, "/workspace/fusion-study/scripts")
from summarize_trace import load

for p in sys.argv[1:]:
    _, ev = load(p)
    c = collections.Counter(e["name"] for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver")
                            and ("Launch" in e["name"] or "Graph" in e["name"]))
    print(p, dict(c))
