"""Phase 4: rms_norm_per_block_quant vs rms_norm + per_token_group_quant (group 128), CUDA-graph timed.

hidden=4096, BF16 in, FP8 e4m3 out, scales fp32 column-major TMA-aligned (the layout DeepGEMM / B uses), no UE8M0.
Each variant: capture REPS calls in one CUDA graph, replay ITERS times, time with CUDA events -> µs/call (median of 5).
Bandwidth = ideal fused-kernel bytes / time (same byte count for both variants, so it is "effective" BW):
  no-res: read x (2TH) + w (2H) + write q (TH) + scales (4TH/128); res: + read r (2TH) + write r (2TH).
Usage: python 40_microbench.py out.json
"""
import json, statistics, sys
import torch
import vllm._custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

H, GS, EPS = 4096, 128, 1e-6
TOKENS = [1, 16, 64, 256, 1024, 4096, 16384]
PEAK_BW = 3.35e12  # H100 SXM5 HBM3 (this run's GPU; brief assumed NVL 3.9e12)
FP8 = torch.float8_e4m3fn
dev = "cuda"


def fused(x, w, r):
    return ops.rms_norm_per_block_quant(x, w, EPS, FP8, [1, GS], residual=r,
                                        is_scale_transposed=True, tma_alignment=4)


def unfused(x, w, r, out):
    if r is None:
        ops.rms_norm(out, x, w, EPS); y = out
    else:
        ops.fused_add_rms_norm(x, r, w, EPS); y = x  # in place: x <- norm(x + r), r <- x + r
    return per_token_group_quant_fp8(y, GS, column_major_scales=True, tma_aligned_scales=True,
                                     use_ue8m0=False)


def time_graph(fn, reps, iters=20):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay(); torch.cuda.synchronize()
    out = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            g.replay()
        b.record(); torch.cuda.synchronize()
        out.append(a.elapsed_time(b) * 1e3 / (iters * reps))
    return statistics.median(out)


def check(x, w, r):
    """Numerical sanity: fused and unfused produce matching dequantized output."""
    x1, x2 = x.clone(), x.clone()
    r1 = r.clone() if r is not None else None
    r2 = r.clone() if r is not None else None
    q1, s1 = fused(x1, w, r1)
    q2, s2 = unfused(x2, w, r2, torch.empty_like(x2))
    d1 = q1.float() * s1.repeat_interleave(GS, dim=1)
    d2 = q2.float() * s2.repeat_interleave(GS, dim=1)
    return (d1 - d2).abs().max().item() / d2.abs().max().item()


def main():
    torch.manual_seed(0)
    rows = []
    for T in TOKENS:
        reps = 200 if T <= 1024 else 20
        for res in (False, True):
            x = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
            w = torch.rand(H, device=dev, dtype=torch.bfloat16) + 0.5
            r = torch.randn(T, H, device=dev, dtype=torch.bfloat16) if res else None
            out = torch.empty_like(x)
            rel = check(x, w, r)
            tf = time_graph(lambda: fused(x, w, r), reps)
            tu = time_graph(lambda: unfused(x, w, r, out), reps)
            nbytes = 2 * T * H + 2 * H + T * H + 4 * T * H // GS + (4 * T * H if res else 0)
            row = dict(tokens=T, residual=res, fused_us=tf, unfused_us=tu, speedup=tu / tf,
                       fused_bw_pct=100 * nbytes / (tf * 1e-6) / PEAK_BW,
                       unfused_bw_pct=100 * nbytes / (tu * 1e-6) / PEAK_BW,
                       bytes=nbytes, max_rel_diff=rel)
            rows.append(row)
            print(f"T={T:6d} res={int(res)} fused={tf:8.2f}us unfused={tu:8.2f}us "
                  f"unfused/fused={tu / tf:5.2f} BW fused={row['fused_bw_pct']:5.1f}% "
                  f"unfused={row['unfused_bw_pct']:5.1f}% relδ={rel:.2e}", flush=True)
    json.dump(dict(hidden=H, group=GS, peak_bw=PEAK_BW, gpu=torch.cuda.get_device_name(0), rows=rows),
              open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
