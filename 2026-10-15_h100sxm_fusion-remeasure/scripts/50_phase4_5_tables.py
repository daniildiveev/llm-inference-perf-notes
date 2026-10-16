"""Render Phase 4 and Phase 5 JSON into results/phase4_microbench.md and results/phase5_proto.md."""
import json, os

ROOT = "/workspace/fusion-study"

p4 = f"{ROOT}/raw/p4_microbench.json"
if os.path.exists(p4):
    d = json.load(open(p4))
    L = ["# Phase 4 — `rms_norm_per_block_quant` vs `rms_norm` + `per_token_group_quant` (group 128)", "",
         f"GPU {d['gpu']}, hidden {d['hidden']}, BF16 in, FP8 e4m3 out, fp32 column-major TMA-aligned scales, no UE8M0.",
         "CUDA-graph timed (scripts/40_microbench.py). Unfused = vLLM `_C.rms_norm` (or `_C.fused_add_rms_norm` with residual)",
         "+ `per_token_group_quant_fp8`. Note: in-model config A uses an Inductor Triton rmsnorm instead of `_C.rms_norm`.",
         f"%BW = ideal fused bytes / time / {d['peak_bw'] / 1e12:.2f} TB/s (same byte count for both columns).",
         "max rel diff = max |dequant(fused) - dequant(unfused)| / max |dequant(unfused)| (a max-element metric). 5-6% is one",
         "e4m3 ulp (3 mantissa bits) on a single element: the unfused path rounds the normalized row to BF16 before quantizing,",
         "the fused kernel quantizes from fp32, so isolated elements land on adjacent FP8 codes. Correctness evidence vs a BF16",
         "reference is in Phase 5 (rel. Frobenius error 2.5e-2, fused <= unfused).", "",
         "| tokens | residual | fused µs | unfused µs | fused speedup (unfused/fused) | fused %BW | unfused %BW | max rel diff |",
         "|---|---|---|---|---|---|---|---|"]
    for r in d["rows"]:
        L.append(f"| {r['tokens']} | {'yes' if r['residual'] else 'no'} | {r['fused_us']:.2f} | {r['unfused_us']:.2f} | "
                 f"{r['speedup']:.2f}× | {r['fused_bw_pct']:.1f}% | {r['unfused_bw_pct']:.1f}% | {r['max_rel_diff']:.1e} |")
    open(f"{ROOT}/results/phase4_microbench.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))

p5 = f"{ROOT}/raw/p5_proto.json"
if os.path.exists(p5):
    d = json.load(open(p5))
    L = ["# Phase 5 — can the GEMM consume the fused norm+quant output directly? (proto/proto_gemm.py)", "",
         "µs = one CUDA-graph replay of the whole path (norm/quant + GEMM). qkv / gate_up include the residual-add RMSNorm",
         "(today: `_C.fused_add_rms_norm` → BF16 → FlashInfer/DeepGEMM op with internal quant; proposed: fused",
         "`rms_norm_per_block_quant`). o_proj / down have no norm in front: proposed = standalone `per_token_group_quant`.",
         "prop_dg = vLLM `deep_gemm.fp8_gemm_nt`; prop_fi = FlashInfer `fp8_blockscale_gemm_sm90` with FP8 input + scales",
         "(swapAB for M<32). Transform cost = 0: the fused op writes the required layout (see kernel lists).",
         "rel err = ‖y − y_ref‖_F / ‖y_ref‖_F against BF16-activation × dequantized-FP8-weight reference.",
         "Baseline caveat: 'today' uses vLLM _C.fused_add_rms_norm (~2.5-3 us here); in-model config A uses an Inductor Triton",
         "add+rmsnorm (~1.9 us), so in-model the fused kernel's deficit is slightly larger than shown.",
         "Only qkv and gate_up have a norm in front; the o_proj/down rows compare quant+GEMM backends (swapAB vs 1d2d), not fusion.", "",
         "| shape | M | today µs | prop_dg µs | prop_fi µs | best prop vs today | rel err today | rel err dg | rel err fi | max abs today / dg / fi |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in d["rows"]:
        t, g, f = r["today"], r["prop_dg"], r["prop_fi"]
        best = min(g["us"], f["us"])
        L.append(f"| {r['shape']} {r['K']}→{r['N']} | {r['M']} | {t['us']:.2f} | {g['us']:.2f} | {f['us']:.2f} | "
                 f"{100 * (best / t['us'] - 1):+.1f}% | {t['rel_fro']:.2e} | {g['rel_fro']:.2e} | {f['rel_fro']:.2e} | "
                 f"{t['max_abs']:.2e} / {g['max_abs']:.2e} / {f['max_abs']:.2e} |")
    L += ["", "## Kernels launched per path (one call, M=1 and M=64, qkv)", ""]
    for r in d["rows"]:
        if r["shape"] == "qkv" and r["M"] in (1, 64):
            for p in ("today", "prop_dg", "prop_fi"):
                L.append(f"- M={r['M']} {p}: " + ", ".join(f"`{k[:70]}`" for k in r[p]["kernels"]))
    open(f"{ROOT}/results/phase5_proto.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
