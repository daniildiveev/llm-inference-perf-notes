"""Phase 5 prototype: can the GEMM consume the fused norm+quant output directly?

Per Qwen3-8B linear shape (K->N) and M, three paths (all timed in one CUDA graph incl. the norm/quant):
  today   : fused_add_rms_norm (BF16)  -> torch.ops.vllm.dynamic_flashinfer_deepgemm_blockscale_gemm
            (= config A: M<32 FlashInfer scale_1x128 + swapAB, M>=32 per_token_group_quant + DeepGEMM)
  prop_dg : rms_norm_per_block_quant(residual, col-major, tma_alignment=4) -> vllm deep_gemm fp8_gemm_nt
            (= config B's GEMM path, called directly)
  prop_fi : rms_norm_per_block_quant(same output, no transform) -> flashinfer fp8_blockscale_gemm_sm90(FP8 in + scales)
            (TRT-LLM runner: M<32 swapAB, M>=32 its DeepGEMM port)
For o_proj / down (inputs are attention output / silu_mul output, not a norm) the "norm" step is omitted:
today = GEMM op on BF16; prop_* = per_token_group_quant (col-major, TMA-aligned, fp32 scales) -> GEMM.

Scale layout: rms_norm_per_block_quant(is_scale_transposed=True, tma_alignment=4) returns an (M, K/128) fp32 view with
strides (1, align4(M)) == scales[kblk*align4(M)+m], which is what both DeepGEMM sm90 (MN-major TMA-aligned SFA) and
TRT-LLM scale_1x128 use. So the "transform" step is the identity; the profiler kernel list per path verifies that no
hidden layout/cast kernel runs.

Weights: random BF16 -> per_block_cast_to_fp8 (128x128) -> deepgemm_post_process_fp8_weight_block(use_e8m0=True),
exactly what DeepGemmFp8BlockScaledMMKernel / FlashInferFp8DeepGEMMDynamicBlockScaledKernel do at load time.
Numerics: vs BF16 reference y_ref = norm_out(bf16) @ dequant(W)^T in fp32 (W dequantized from the FP8 weights, so the
error isolates activation quantization + GEMM accumulation).
Usage: python proto_gemm.py out.json
"""
import json, statistics, sys
import torch
import vllm.envs  # noqa: F401
from vllm.config import VllmConfig, set_current_vllm_config
import vllm._custom_ops as ops
import vllm.model_executor.kernels.linear.scaled_mm.flashinfer  # noqa: F401  registers torch.ops.vllm.*
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    deepgemm_post_process_fp8_weight_block, per_token_group_quant_fp8)
from vllm.utils.deep_gemm import fp8_gemm_nt, is_deep_gemm_e8m0_used, per_block_cast_to_fp8
from flashinfer.gemm import fp8_blockscale_gemm_sm90

SHAPES = {"qkv": (4096, 6144), "o_proj": (4096, 4096), "gate_up": (4096, 24576), "down": (12288, 4096)}
NORM_SHAPES = {"qkv", "gate_up"}
MS = [1, 8, 16, 32, 64, 256]
EPS, GS = 1e-6, 128
FP8 = torch.float8_e4m3fn
dev = "cuda"


def time_graph(fn, reps=50, iters=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay(); torch.cuda.synchronize()
    t = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            g.replay()
        b.record(); torch.cuda.synchronize()
        t.append(a.elapsed_time(b) * 1e3 / (iters * reps))
    return statistics.median(t)


def kernels_of(fn):
    fn(); torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        fn(); torch.cuda.synchronize()
    return [e.name[:90] for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA]


def errs(y, ref):
    d = (y.float() - ref)
    return dict(max_abs=d.abs().max().item(), rel_fro=(d.norm() / ref.norm()).item(),
                ref_max=ref.abs().max().item())


def main():
    torch.manual_seed(0)
    e8m0 = is_deep_gemm_e8m0_used()
    rows = []
    for name, (K, N) in SHAPES.items():
        Wb = torch.randn(N, K, device=dev, dtype=torch.bfloat16) / K ** 0.5
        Wq, Ws = per_block_cast_to_fp8(Wb, [128, 128], use_ue8m0=False)
        Wq, Ws = deepgemm_post_process_fp8_weight_block(Wq, Ws, (128, 128), use_e8m0=e8m0)
        assert Ws.shape == ((N + 127) // 128, K // 128) and Ws.dtype == torch.float32, (Ws.shape, Ws.dtype)
        Wdq = (Wq.float() * Ws.repeat_interleave(128, 0)[:N].repeat_interleave(128, 1)[:, :K])
        w_norm = torch.rand(K, device=dev, dtype=torch.bfloat16) + 0.5
        is_norm = name in NORM_SHAPES
        for M in MS:
            x0 = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            r0 = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            x, r = x0.clone(), r0.clone()
            out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)

            def today():
                if is_norm:
                    ops.fused_add_rms_norm(x, r, w_norm, EPS)  # in place: x <- norm(x + r), r <- x + r
                    a = x
                else:
                    a = x
                return torch.ops.vllm.dynamic_flashinfer_deepgemm_blockscale_gemm(a, Wq, Ws, GS, e8m0)

            def quant():
                if is_norm:
                    return ops.rms_norm_per_block_quant(x, w_norm, EPS, FP8, [1, GS], residual=r,
                                                        is_scale_transposed=True, tma_alignment=4)
                return per_token_group_quant_fp8(x, GS, column_major_scales=True, tma_aligned_scales=True,
                                                 use_ue8m0=False)

            def prop_dg():
                q, s = quant()
                fp8_gemm_nt((q, s), (Wq, Ws), out, is_deep_gemm_e8m0_used=e8m0)
                return out

            def prop_fi():
                q, s = quant()
                return fp8_blockscale_gemm_sm90(q, Wq, s, Ws, out=out)

            # --- numerics: every path starts from the same x0, r0 (residual mutation reset each call)
            if is_norm:
                xs = (x0.float() + r0.float())
                a_ref = (xs * torch.rsqrt(xs.pow(2).mean(-1, keepdim=True) + EPS)).to(torch.bfloat16) * w_norm
            else:
                a_ref = x0
            ref = a_ref.float() @ Wdq.t()
            res = {}
            for pname, fn in (("today", today), ("prop_dg", prop_dg), ("prop_fi", prop_fi)):
                x.copy_(x0); r.copy_(r0)
                y = fn().clone()
                res[pname] = errs(y, ref)
            # --- kernels + timing (residual drifts during replay; irrelevant for timing)
            for pname, fn in (("today", today), ("prop_dg", prop_dg), ("prop_fi", prop_fi)):
                x.copy_(x0); r.copy_(r0)
                res[pname]["kernels"] = kernels_of(fn)
                x.copy_(x0); r.copy_(r0)
                res[pname]["us"] = time_graph(fn)
            row = dict(shape=name, K=K, N=N, M=M, norm_in_path=is_norm, e8m0=e8m0, **res)
            rows.append(row)
            print(f"{name:8s} M={M:4d} today={res['today']['us']:7.2f}us prop_dg={res['prop_dg']['us']:7.2f}us "
                  f"prop_fi={res['prop_fi']['us']:7.2f}us | rel err today={res['today']['rel_fro']:.2e} "
                  f"dg={res['prop_dg']['rel_fro']:.2e} fi={res['prop_fi']['rel_fro']:.2e}", flush=True)
            for p in ("today", "prop_dg", "prop_fi"):
                print(f"    {p:8s} kernels: {res[p]['kernels']}")
    json.dump(dict(rows=rows, gpu=torch.cuda.get_device_name(0)), open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    with set_current_vllm_config(VllmConfig()):
        main()
