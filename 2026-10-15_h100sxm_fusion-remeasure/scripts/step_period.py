"""Print start-to-start period of consecutive engine steps (gpu_user_annotation) and GPU busy time per step.
Usage: python step_period.py <trace_dir> [...]"""
import sys
sys.path.insert(0, "/workspace/fusion-study/scripts")
from summarize_trace import load

for p in sys.argv[1:]:
    path, ev = load(p)
    st = sorted((e["ts"], e["dur"]) for e in ev
                if e.get("cat") == "gpu_user_annotation" and e.get("name", "").startswith("execute_"))
    cpu = sorted((e["ts"], e["dur"]) for e in ev
                 if e.get("cat") == "user_annotation" and e.get("name", "").startswith("execute_"))
    per = [round(b[0] - a[0]) for a, b in zip(st, st[1:])]
    print(p, "gpu step period us:", per, "gpu ann dur:", [round(d) for _, d in st],
          "cpu ann dur:", [round(d) for _, d in cpu])
