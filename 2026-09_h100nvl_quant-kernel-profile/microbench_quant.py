"""Microbenchmark: fused rms_norm+block-quant vs the separate norm/quant pair.

Answers three things the trace-level A/B/C could not:
  1. where the launch-latency floor ends -- sweep runs past one full wave of
     blocks (H100 NVL has 132 SMs, so a 1-block-per-token grid fills a wave at
     a few hundred tokens; the knee cannot be seen at num_tokens <= 256)
  2. whether is_scale_transposed (strided scale writes) explains part of the
     gap, i.e. whether fused and separate are doing the same work
  3. both has_residual configurations, since production issues 71 launches
     with residual and 1 without

Timed with CUDA events, not the profiler, so launch overhead is included the
same way it is in a real decode step.
"""

import argparse

import torch
from torch.profiler import ProfilerActivity, profile

import vllm._custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)

HIDDEN = 4096  # Qwen3-8B hidden size
GROUP = 128  # weight_block_size[1]
EPS = 1e-6
FP8 = torch.float8_e4m3fn
TOKENS = [1, 8, 64, 256, 512, 1024, 4096, 16384]


def bench_wall(fn, warmup=20, iters=100):
    """Mean microseconds per call, wall clock.

    Includes host dispatch, so below a few thousand tokens this measures the
    Python/launch floor (~15us) rather than the kernel. Kept because it is the
    number that bounds a real decode step, but read bench_gpu for kernel cost.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def bench_gpu(fn, warmup=20, inner=20, reps=5):
    """Mean microseconds of *device* time per call.

    Captures `inner` calls into a CUDA graph so replay costs no host time, then
    sums per-kernel durations from the profiler. Summing durations rather than
    measuring the wall span means overlap between independent back-to-back
    launches cannot deflate the result -- the same quantity an2.py reports off
    a real trace, so it is directly comparable to the 7.07us decode number.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner):
            fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            graph.replay()
        torch.cuda.synchronize()

    total = 0.0
    for evt in prof.key_averages():
        for attr in ("self_device_time_total", "self_cuda_time_total"):
            if hasattr(evt, attr):
                total += getattr(evt, attr)
                break
    return total / (reps * inner)


def make_fused(n, residual, transposed):
    x = torch.randn(n, HIDDEN, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(HIDDEN, dtype=torch.bfloat16, device="cuda")
    r = torch.randn_like(x) if residual else None

    def run():
        ops.rms_norm_per_block_quant(
            x, w, EPS, FP8, [1, GROUP], None, r, transposed, 0
        )

    return run


def make_separate(n, residual, transposed):
    x = torch.randn(n, HIDDEN, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(HIDDEN, dtype=torch.bfloat16, device="cuda")
    r = torch.randn_like(x) if residual else None
    out = torch.empty_like(x)

    def run():
        if residual:
            ops.fused_add_rms_norm(x, r, w, EPS)
            y = x
        else:
            ops.rms_norm(out, x, w, EPS)
            y = out
        per_token_group_quant_fp8(y, GROUP, column_major_scales=transposed)

    return run


def make_norm_only(n, residual):
    x = torch.randn(n, HIDDEN, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(HIDDEN, dtype=torch.bfloat16, device="cuda")
    r = torch.randn_like(x) if residual else None
    out = torch.empty_like(x)

    def run():
        if residual:
            ops.fused_add_rms_norm(x, r, w, EPS)
        else:
            ops.rms_norm(out, x, w, EPS)

    return run


def make_quant_only(n, transposed):
    y = torch.randn(n, HIDDEN, dtype=torch.bfloat16, device="cuda")

    def run():
        per_token_group_quant_fp8(y, GROUP, column_major_scales=transposed)

    return run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    args = p.parse_args()

    dev = torch.cuda.get_device_properties(0)
    sms = dev.multi_processor_count
    print(f"# {dev.name}, {sms} SMs, hidden={HIDDEN}, group={GROUP}")
    print(f"# torch {torch.__version__}, iters={args.iters}, warmup={args.warmup}")
    print()

    for residual in (True, False):
        for transposed in (True, False):
            print(f"=== has_residual={residual}  transpose_scale={transposed} ===")
            print(
                f"{'tokens':>7} {'blk/SM':>7} | {'fused':>8} {'separate':>8} "
                f"{'norm':>8} {'quant':>8} {'f/sep':>6} | {'wall_f':>7} {'wall_s':>7}"
            )
            for n in TOKENS:
                try:
                    fused = bench_gpu(make_fused(n, residual, transposed), args.warmup)
                    sep = bench_gpu(make_separate(n, residual, transposed), args.warmup)
                    nrm = bench_gpu(make_norm_only(n, residual), args.warmup)
                    qnt = bench_gpu(make_quant_only(n, transposed), args.warmup)
                    wf = bench_wall(
                        make_fused(n, residual, transposed), args.warmup, args.iters
                    )
                    ws = bench_wall(
                        make_separate(n, residual, transposed), args.warmup, args.iters
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"{n:>7} FAILED: {type(exc).__name__}: {exc}")
                    continue
                print(
                    f"{n:>7} {n / sms:>7.1f} | {fused:>8.2f} {sep:>8.2f} "
                    f"{nrm:>8.2f} {qnt:>8.2f} {fused / sep:>6.2f} | "
                    f"{wf:>7.2f} {ws:>7.2f}"
                )
            print()


if __name__ == "__main__":
    main()
