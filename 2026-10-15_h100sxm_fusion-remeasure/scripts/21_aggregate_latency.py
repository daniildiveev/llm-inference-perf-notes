"""Aggregate Phase 2 vllm bench latency JSONs -> results/phase2_latency.md + raw/p2_summary.json.

Per (cfg, bs): each repeat gives median and p90 over its 20 iterations. Reported:
  median = median of the per-repeat medians, p90 = median of per-repeat p90s,
  spread = (max - min) / median of per-repeat medians, flagged when > 3%.
"""
import glob, json, os, re, statistics

ROOT = "/workspace/fusion-study"
cells = {}
for f in glob.glob(f"{ROOT}/raw/p2/*.json"):
    b = os.path.basename(f)[:-5]
    d = json.load(open(f))
    lat = sorted(d["latencies"])
    p90 = statistics.quantiles(lat, n=10)[-1]
    m = re.match(r"(prefill_)?([ABC])_bs(\d+)(?:_r(\d))?", b)
    key = ("prefill" if m.group(1) else "decode", m.group(2), int(m.group(3)))
    cells.setdefault(key, []).append(dict(rep=m.group(4), median=statistics.median(lat), p90=p90, n=len(lat)))

summary = {}
for key, reps in cells.items():
    meds = [r["median"] for r in reps]
    med = statistics.median(meds)
    summary[key] = dict(median=med, p90=statistics.median(r["p90"] for r in reps),
                        spread=(max(meds) - min(meds)) / med, n_reps=len(reps), reps=reps)

json.dump({"|".join(map(str, k)): v for k, v in sorted(summary.items())},
          open(f"{ROOT}/raw/p2_summary.json", "w"), indent=1)

lines = ["# Phase 2 — wall-clock latency (`vllm bench latency`)", "",
         "Decode: input 128, output 256, warmup 5, iters 20, 3 repeats (config order rotated per repeat).",
         "median / p90 in seconds per batch (median over repeats of per-repeat median / p90).",
         "Δ = median vs A. ⚠ = spread across repeats > 3%.", "",
         "| bs | A median | A p90 | B median | B p90 | B Δ | C median | C p90 | C Δ | spread A/B/C |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for mode in ("decode", "prefill"):
    if mode == "prefill":
        lines += ["", "Prefill-heavy: input 2048, output 1, warmup 5, iters 20, one run per cell (no spread).", "",
                  "| bs | A median | A p90 | B median | B p90 | B Δ | C median | C p90 | C Δ | — |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
    for bs in sorted({k[2] for k in summary if k[0] == mode}):
        row = [str(bs)]
        a = summary.get((mode, "A", bs))
        spreads = []
        for c in "ABC":
            s = summary.get((mode, c, bs))
            if s is None:
                row += ["—", "—"] + (["—"] if c != "A" else []); spreads.append("—"); continue
            row += [f"{s['median']:.4f}", f"{s['p90']:.4f}"]
            if c != "A":
                row.append(f"{100 * (s['median'] / a['median'] - 1):+.1f}%" if a else "—")
            flag = " ⚠" if s["spread"] > 0.03 else ""
            spreads.append(f"{100 * s['spread']:.1f}%{flag}" + ("" if s["n_reps"] == 3 or mode == "prefill" else f" (n={s['n_reps']})"))
        row.append(" / ".join(spreads) if mode == "decode" else "—")
        lines.append("| " + " | ".join(row) + " |")
open(f"{ROOT}/results/phase2_latency.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
