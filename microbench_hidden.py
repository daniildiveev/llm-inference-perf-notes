"""Locate the fused kernel's fixed cost by shape alone, with no counters.

Hypothesis under test: the ~3.3us of extra floor the fused kernel carries over
the separate pair comes from the hidden/group amax reductions running in
sequence rather than in parallel. At hidden=4096, group=128 that is 32 groups,
so ~100ns per group.

Everything runs at n=1, where there is no data to speak of (32KB even at
hidden=16384), so any growth is structure, not bandwidth.

Two axes, because the obvious sweep confounds them:
  - vary hidden at fixed group  -> group count and hidden both grow
  - vary group at fixed hidden  -> group count alone changes
Linear in the first and flat in the second would mean the cost tracks hidden,
not the reduction chain. Linear in both means the reduction chain.

The standalone quant kernel is measured on the same shapes as a control: it
does the same per-group reductions without the norm, so if it shows the same
slope the behaviour is not specific to fusion.
"""

import torch

import vllm._custom_ops as ops
from microbench_quant import EPS, FP8, bench_gpu
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)

N = 1
HIDDENS = [1024, 2048, 4096, 8192, 16384, 32768]
GROUPS = [128, 64]
FIXED_HIDDEN = 4096


def fused_fn(n, hidden, group, residual, transposed):
    x = torch.randn(n, hidden, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(hidden, dtype=torch.bfloat16, device="cuda")
    r = torch.randn_like(x) if residual else None

    def run():
        ops.rms_norm_per_block_quant(
            x, w, EPS, FP8, [1, group], None, r, transposed, 0
        )

    return run


def quant_fn(n, hidden, group, transposed):
    y = torch.randn(n, hidden, dtype=torch.bfloat16, device="cuda")

    def run():
        per_token_group_quant_fp8(y, group, column_major_scales=transposed)

    return run


def norm_fn(n, hidden, residual):
    x = torch.randn(n, hidden, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(hidden, dtype=torch.bfloat16, device="cuda")
    r = torch.randn_like(x) if residual else None
    out = torch.empty_like(x)

    def run():
        if residual:
            ops.fused_add_rms_norm(x, r, w, EPS)
        else:
            ops.rms_norm(out, x, w, EPS)

    return run


def measure(fn):
    try:
        return bench_gpu(fn(), warmup=20)
    except Exception as exc:  # noqa: BLE001
        return f"ERR:{type(exc).__name__}"


def fmt(v):
    return f"{v:>8.2f}" if isinstance(v, float) else f"{v:>8}"


def slope(xs, ys):
    """Least-squares slope, ns per unit x. Returns None if any point failed."""
    pts = [(x, y) for x, y in zip(xs, ys) if isinstance(y, float)]
    if len(pts) < 2:
        return None
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    den = sum((p[0] - mx) ** 2 for p in pts)
    if den == 0:
        return None
    return sum((p[0] - mx) * (p[1] - my) for p in pts) / den * 1000.0


def main():
    dev = torch.cuda.get_device_properties(0)
    print(f"# {dev.name}, {dev.multi_processor_count} SMs, n={N}, transposed=True")
    print(f"# torch {torch.__version__}")

    for residual in (True, False):
        for group in GROUPS:
            print(f"\n=== vary hidden | group={group} has_residual={residual} ===")
            print(
                f"{'hidden':>7} {'groups':>7} {'KB':>6} | {'fused':>8} "
                f"{'quant':>8} {'norm':>8}"
            )
            ng, fu, qu, no = [], [], [], []
            for h in HIDDENS:
                f = measure(lambda: fused_fn(N, h, group, residual, True))
                q = measure(lambda: quant_fn(N, h, group, True))
                m = measure(lambda: norm_fn(N, h, residual))
                ng.append(h // group)
                fu.append(f)
                qu.append(q)
                no.append(m)
                print(
                    f"{h:>7} {h // group:>7} {N * h * 2 / 1024:>6.0f} | "
                    f"{fmt(f)} {fmt(q)} {fmt(m)}"
                )
            for name, ys in (("fused", fu), ("quant", qu), ("norm", no)):
                s = slope(ng, ys)
                if s is not None:
                    print(f"  slope {name:<6} = {s:>7.1f} ns / group")

    print(f"\n=== vary group | hidden={FIXED_HIDDEN} (group count moves alone) ===")
    print(f"{'group':>7} {'groups':>7} | {'fused_res':>10} {'fused_nores':>12} {'quant':>8}")
    for group in (256, 128, 64, 32):
        fr = measure(lambda: fused_fn(N, FIXED_HIDDEN, group, True, True))
        fn_ = measure(lambda: fused_fn(N, FIXED_HIDDEN, group, False, True))
        q = measure(lambda: quant_fn(N, FIXED_HIDDEN, group, True))
        print(
            f"{group:>7} {FIXED_HIDDEN // group:>7} | {fmt(fr):>10} "
            f"{fmt(fn_):>12} {fmt(q)}"
        )


if __name__ == "__main__":
    main()
