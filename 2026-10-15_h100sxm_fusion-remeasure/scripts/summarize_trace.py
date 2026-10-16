"""Sum GPU kernel time per decode step from a vLLM torch-profiler trace.

Usage: python summarize_trace.py <trace_dir_or_json.gz> [--json out.json] [--top N]
Steps are delimited by gpu_user_annotation events (one per engine step).
Kernels are bucketed by name into categories, rolled up to gemm / quant_norm / other.
FP8 GEMM launches are tagged with their position in the decoder layer
(qkv, o, gate_up, down) when a step has exactly 4 per layer (Qwen3-8B: 36 layers).
"""
import collections, glob, gzip, json, os, re, sys

N_LAYERS = 36
GEMM_SLOTS = ["qkv", "o", "gate_up", "down"]
CATS = [  # (category, rollup, regex) — first match wins
    ("gemm_fp8", "gemm", r"deep_gemm::|fp8_gemm|cutlass_3x_gemm_fp8_blockwise|blockwise|sm90_fp8|gemm_fp8|Fp8.*Gemm|fp8_blockscale"),
    ("lm_head", "other", r"nvjet|cublas|sm90_xmma_gemm_bf16|gemm_bf16"),
    ("quant", "quant_norm", r"per_token_group|group_quant|quant_8bit|_per_block_quant|transpose_fp32|tma_aligned|mn_major|scaled_fp8_quant"),
    ("rmsnorm", "quant_norm", r"fused_add_rms_norm|rms_norm_kernel|^triton_red_fused.*rms_norm"),
    ("qknorm_rope", "other", r"triton_per_fused_add_cat_index_select_mul_rms_norm|rotary|rope"),
    ("attn", "other", r"flash::|FlashAttn|reshape_and_cache|attention"),
    ("act", "other", r"silu"),
]
CATS = [(c, r, re.compile(p)) for c, r, p in CATS]


def classify(name):
    if "silu" in name:  # silu_mul (B/C inductor names it after the quant buffer it feeds)
        return "act", "other"
    if name == "triton_poi_fused_0":  # B/C only: inductor copy of graph-input residual before in-place fused op
        return "residual_copy", "quant_norm"
    if "scale_1x128" in name:  # TRT-LLM quant inside FlashInfer; name also matches fp8_blockscale
        return "quant", "quant_norm"
    if re.search(r"rms_norm.*quant|per_block_quant", name) and "silu" not in name:
        return "norm_fused_quant", "quant_norm"
    for c, r, p in CATS:
        if p.search(name):
            return c, r
    return "misc", "other"


def short(name):
    m = re.match(r"void deep_gemm::(\w+)<(\d+)u, (\d+)u", name)
    if m:
        return f"deep_gemm::{m.group(1)}<N={m.group(2)},K={m.group(3)}>"
    return name.split("(")[0][:110]


def load(path):
    if os.path.isdir(path):
        path = sorted(glob.glob(os.path.join(path, "*.json.gz")))[0]
    return path, json.load(gzip.open(path))["traceEvents"]


def summarize(path):
    path, ev = load(path)
    steps = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev
                   if e.get("cat") == "gpu_user_annotation" and e.get("name", "").startswith("execute_"))
    kernels = sorted((e for e in ev if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    per_step = []
    for s0, s1 in steps:
        ks = [k for k in kernels if s0 <= k["ts"] < s1]
        cat = collections.Counter(); roll = collections.Counter()
        kn = collections.defaultdict(lambda: [0, 0.0])
        gemms = [k for k in ks if classify(k["name"])[0] == "gemm_fp8"]
        slot = {id(k): GEMM_SLOTS[i % 4] for i, k in enumerate(gemms)} if len(gemms) == 4 * N_LAYERS else {}
        for k in ks:
            c, r = classify(k["name"])
            cat[c] += k["dur"]; roll[r] += k["dur"]
            key = short(k["name"]) + (f" @{slot[id(k)]}" if id(k) in slot else "")
            kn[key][0] += 1; kn[key][1] += k["dur"]
        per_step.append({"total_us": sum(k["dur"] for k in ks), "rollup": dict(roll),
                         "category": dict(cat), "kernels": {n: v for n, v in kn.items()},
                         "n_kernels": len(ks), "n_gemm_fp8": len(gemms), "wall_us": s1 - s0})
    unassigned = len(kernels) - sum(p["n_kernels"] for p in per_step)
    return {"trace": path, "n_steps": len(steps), "unassigned_kernels": unassigned, "steps": per_step}


def mean_step(summary):
    """Average over steps: rollup, category, per-kernel (count, us) per step."""
    n = summary["n_steps"]
    out = {"total_us": 0.0, "rollup": collections.Counter(), "category": collections.Counter(),
           "kernels": collections.defaultdict(lambda: [0.0, 0.0])}
    for s in summary["steps"]:
        out["total_us"] += s["total_us"] / n
        for k, v in s["rollup"].items(): out["rollup"][k] += v / n
        for k, v in s["category"].items(): out["category"][k] += v / n
        for k, (c, d) in s["kernels"].items():
            out["kernels"][k][0] += c / n; out["kernels"][k][1] += d / n
    return out


if __name__ == "__main__":
    path = sys.argv[1]
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 25
    s = summarize(path)
    if "--json" in sys.argv:
        json.dump(s, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
    m = mean_step(s)
    print(f"{s['trace']}: steps={s['n_steps']} unassigned_kernels={s['unassigned_kernels']} "
          f"gemm/step={[x['n_gemm_fp8'] for x in s['steps']]} "
          f"per-step GPU={m['total_us']:.1f}us wall={[round(x['wall_us']) for x in s['steps']]}")
    print("rollup:", {k: round(v, 1) for k, v in m["rollup"].items()})
    print("category:", {k: round(v, 1) for k, v in sorted(m["category"].items(), key=lambda x: -x[1])})
    print(f"{'us/step':>9} {'n/step':>7} {'us/launch':>9}  [cat] kernel")
    for k, (c, d) in sorted(m["kernels"].items(), key=lambda x: -x[1][1])[:top]:
        print(f"{d:9.1f} {c:7.1f} {d / c:9.2f}  [{classify(k)[0]}] {k}")
