"""Phase 3: attribute A->B and A->C per-step GPU time differences to gemm / quant_norm / other.

Reads traces/<tag>/<cfg>_bs<bs>/ (default tag p3), writes raw/<tag>_attribution.json and results/phase3_attribution.md.
Per-step numbers are means over the profiled steady-state decode steps; forward-pass kernels only.
"""
import json, sys
sys.path.insert(0, "/workspace/fusion-study/scripts")
from summarize_trace import summarize, mean_step

ROOT = "/workspace/fusion-study"
TAG = sys.argv[1] if len(sys.argv) > 1 else "p3"
BSS = [int(b) for b in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["1", "32", "256"])]
SLOTS = ["qkv", "o", "gate_up", "down"]

data, out = {}, {}
for bs in BSS:
    for c in "ABC":
        s = summarize(f"{ROOT}/traces/{TAG}/{c}_bs{bs}")
        assert all(st["n_gemm_fp8"] == 144 for st in s["steps"]), (c, bs, [st["n_gemm_fp8"] for st in s["steps"]])
        m = mean_step(s)
        gemm = {}
        for k, (n, us) in m["kernels"].items():
            for sl in SLOTS:
                if k.endswith(f"@{sl}"):
                    gemm[sl] = dict(kernel=k.rsplit(" @", 1)[0][:70], us_per_launch=us / n, us_per_step=us)
        data[(c, bs)] = dict(total=m["total_us"], rollup=dict(m["rollup"]), category=dict(m["category"]),
                             gemm=gemm, n_steps=s["n_steps"])
        out[f"{c}_bs{bs}"] = data[(c, bs)]
json.dump(out, open(f"{ROOT}/raw/{TAG}_attribution.json", "w"), indent=1)

L = ["# Phase 3 — trace attribution (GPU µs per decode step, forward-pass kernels; lm_head + sampler excluded)", "",
     "gemm = FP8 linear GEMMs; quant_norm = rmsnorm, activation quant (scale_1x128 / per_token_group_quant), fused",
     "rms_norm_per_block_quant, and the B/C-only residual copy inductor inserts before the in-place fused op;",
     "other = attention, qk-norm+rope, silu_mul, misc.", "",
     "| bs | cfg | total | gemm | quant_norm | other | Δtotal vs A | Δgemm | Δquant_norm | Δother |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for bs in BSS:
    a = data[("A", bs)]
    for c in "ABC":
        d = data[(c, bs)]
        r = d["rollup"]
        row = [str(bs), c, f"{d['total']:.0f}", f"{r.get('gemm', 0):.0f}", f"{r.get('quant_norm', 0):.0f}",
               f"{r.get('other', 0):.0f}"]
        if c == "A":
            row += ["—"] * 4
        else:
            dt = d["total"] - a["total"]
            row += [f"{dt:+.0f} ({100 * dt / a['total']:+.1f}%)"] + [
                f"{r.get(k, 0) - a['rollup'].get(k, 0):+.0f}" for k in ("gemm", "quant_norm", "other")]
        L.append("| " + " | ".join(row) + " |")
L += ["", "## GEMM per-launch averages (µs/launch; kernel)", "",
      "| bs | cfg | qkv 4096→6144 | o 4096→4096 | gate_up 4096→24576 | down 12288→4096 | kernel |", "|---|---|---|---|---|---|---|"]
for bs in BSS:
    for c in "ABC":
        g = data[(c, bs)]["gemm"]
        L.append(f"| {bs} | {c} | " + " | ".join(f"{g[s]['us_per_launch']:.2f}" for s in SLOTS)
                 + f" | `{g['qkv']['kernel']}` |")
L += ["", "## quant_norm breakdown (µs/step)", "", "| bs | cfg | category µs/step |", "|---|---|---|"]
for bs in BSS:
    for c in "ABC":
        cat = data[(c, bs)]["category"]
        keep = {k: round(v) for k, v in cat.items() if k in ("quant", "rmsnorm", "norm_fused_quant", "residual_copy")}
        L.append(f"| {bs} | {c} | {keep} |")
open(f"{ROOT}/results/phase3_attribution.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
