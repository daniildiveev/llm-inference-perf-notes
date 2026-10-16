"""One-off text corrections to results/00_overview.md and results/99_conclusions.md after review
(range labelling, full bs=1 quant_norm breakdown, Phase 5 scope, five one-line bullets)."""
R = "/workspace/fusion-study/results/"


def sub(path, old, new):
    s = open(R + path).read()
    if new in s:
        return
    assert old in s, (path, old[:60])
    open(R + path, "w").write(s.replace(old, new))


sub("00_overview.md",
    "decode is **+9 to +11.5% at bs=1** and\n  **+1 to +5% for bs 8–256** in wall-clock time. With CUTLASS blockwise (config C), it is **+9% at bs=1 and +30 to +61% at bs ≥ 8**.",
    "wall-clock decode is **+11.5% at bs=1** and **−0.5% to +7.2% for bs 8–256** (median of 3 repeats; the main table).\n"
    "  Using only repeats 1+2 (repeat 0 was disturbed by host load), it is +9.3% at bs=1 and +1.2% to +5.2% for bs 8–256, positive at\n"
    "  every batch size. With CUTLASS blockwise (config C): +9.2% at bs=1 and +30.0% to +57.8% at bs ≥ 8 (median of 3), or +9.5% and\n"
    "  +30.8% to +60.7% (repeats 1+2).")
sub("00_overview.md",
    "- **At bs=1, losing swapAB now costs about +285 µs/step** (it was +362 in 0.23.x). The fused kernel and residual copy add another\n  +262 µs/step.",
    "- **At bs=1, losing swapAB now costs about +285 µs/step** (it was +362 in 0.23.x). Norm/quant adds +262 µs/step: the fused\n"
    "  kernel and residual copy account for ~+200, and ~+60 comes from `per_token_group_quant` (~2.4 µs) replacing `scale_1x128` (~1.6 µs) on\n"
    "  the o_proj/down inputs. That last part is not caused by norm fusion; it comes from leaving the FlashInfer kernel.")
sub("00_overview.md",
    "so the end-to-end path is **+3 to +8.5% at\n    M=1** for qkv/gate_up and roughly neutral (−4% to +4%) at M=8–256.",
    "so for the two shapes that have a norm in front (qkv, gate_up), the end-to-end path is\n"
    "    **+3.4% to +8.5% at M=1** and −4.2% to +4.3% at M=8–256. The o_proj/down rows have no norm; they show that DeepGEMM `1d2d`\n"
    "    beats today's `scale_1x128` + swapAB by 11–13% on o_proj at M=8/16.")

sub("99_conclusions.md",
    "| B +9.3 to +11.5% wall; GPU +558 µs/step = GEMM +285, quant/norm +262 (fused 418 + residual copy 45 replace rmsnorm 140 + `scale_1x128` 235). C +9% wall | holds |",
    "| B +11.5% wall (median of 3; +9.3% repeats 1+2); GPU +558 µs/step = GEMM +285 + quant/norm +262. quant/norm: B 637 = fused 418 + `per_token_group_quant` on o_proj/down inputs 172 + residual copy 45 + 2, vs A 375 = Triton rmsnorm 140 + `scale_1x128` ×144 235. So ~+60 µs of it is `per_token_group_quant` (2.4 µs) vs `scale_1x128` (1.6 µs) on non-norm GEMM inputs, not fusion. C +9.2% wall | holds |")
s = open(R + "99_conclusions.md").read()
i = s.index("## Five bullets")
s = s[:i] + """## Five bullets

1. **Changed since 0.23.x:** fusion passes on by default; profiler is configured via `--profiler-config`; DeepGEMM bundled (2.8.1); the M≥32 branch uses vLLM's UE8M0 `per_token_group_quant`; #54111 is a race fix only (kernel structure unchanged); DeepGEMM `1d2d` at M=1 is now only 5–15% slower per launch than swapAB.
2. **The regression still holds:** B +11.5% at bs=1 and −0.5% to +7.2% at bs 8–256 (median of 3; repeats 1+2: +1.2% to +9.3%, positive everywhere); C +9% to +58%; prefill (2048 tok) B ≈ A at bs 8/32, C +12% to +17%.
3. **At bs ≥ 32 the GEMM is identical (±1%); the whole cost is norm/quant (+130 µs/step):** a fused kernel slower than Triton rmsnorm + group quant, plus an Inductor residual copy. At bs=1, losing swapAB adds +285 µs.
4. **The proposed fix is mechanically viable with zero transform cost** (FlashInfer swapAB takes FP8 + scales; `tma_alignment=4` gives the exact layout; numerics equal or better), **but it is not a net win:** +3.4% to +8.5% at M=1 for qkv/gate_up, ±4% elsewhere.
5. **Recommendation:** fix `rms_norm_per_block_quant` first (it is latency-bound at 0.1–32% of HBM peak for ≤256 tokens, 5–7 µs vs ~1.9 µs Triton rmsnorm), then route its output via an `apply_input_quant=True` kernel to `fp8_blockscale_gemm_sm90` (M<32) and `fp8_gemm_nt` (M≥32); keep the unfused default until then.
"""
open(R + "99_conclusions.md", "w").write(s)
print("ok")
