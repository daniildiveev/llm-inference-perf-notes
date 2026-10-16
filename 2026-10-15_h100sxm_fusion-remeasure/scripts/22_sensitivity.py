"""Sensitivity check for Phase 2: recompute decode medians/Δ from repeats 1+2 only (repeat 0 ran while host-side
trace analysis was active). Appends a section to results/phase2_latency.md. Run after 21_aggregate_latency.py."""
import json, statistics

ROOT = "/workspace/fusion-study"
S = json.load(open(f"{ROOT}/raw/p2_summary.json"))
L = ["", "## Sensitivity: repeats 1+2 only (repeat 0 overlapped host-side analysis; see RESULTS.md)", "",
     "| bs | A median | B median | B Δ | C median | C Δ | spread (r1 vs r2) A/B/C |", "|---|---|---|---|---|---|---|"]
for bs in sorted({int(k.split("|")[2]) for k in S if k.startswith("decode")}):
    med, spr = {}, {}
    for c in "ABC":
        reps = [r["median"] for r in S[f"decode|{c}|{bs}"]["reps"] if r["rep"] in ("1", "2")]
        med[c] = statistics.median(reps)
        spr[c] = (max(reps) - min(reps)) / med[c]
    fl = lambda x: f"{100 * x:.1f}%" + (" ⚠" if x > 0.03 else "")
    L.append(f"| {bs} | {med['A']:.4f} | {med['B']:.4f} | {100 * (med['B'] / med['A'] - 1):+.1f}% | {med['C']:.4f} | "
             f"{100 * (med['C'] / med['A'] - 1):+.1f}% | {fl(spr['A'])} / {fl(spr['B'])} / {fl(spr['C'])} |")
open(f"{ROOT}/results/phase2_latency.md", "a").write("\n".join(L) + "\n")
print("\n".join(L))
