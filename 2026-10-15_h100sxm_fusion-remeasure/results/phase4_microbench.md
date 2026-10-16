# Phase 4 — `rms_norm_per_block_quant` vs `rms_norm` + `per_token_group_quant` (group 128)

GPU NVIDIA H100 80GB HBM3, hidden 4096, BF16 in, FP8 e4m3 out, fp32 column-major TMA-aligned scales, no UE8M0.
CUDA-graph timed (scripts/40_microbench.py). Unfused = vLLM `_C.rms_norm` (or `_C.fused_add_rms_norm` with residual)
+ `per_token_group_quant_fp8`. Note: in-model config A uses an Inductor Triton rmsnorm instead of `_C.rms_norm`.
%BW = ideal fused bytes / time / 3.35 TB/s (same byte count for both columns).
max rel diff = max |dequant(fused) - dequant(unfused)| / max |dequant(unfused)| (a max-element metric). 5-6% is one
e4m3 ulp (3 mantissa bits) on a single element: the unfused path rounds the normalized row to BF16 before quantizing,
the fused kernel quantizes from fp32, so isolated elements land on adjacent FP8 codes. Correctness evidence vs a BF16
reference is in Phase 5 (rel. Frobenius error 2.5e-2, fused <= unfused).

| tokens | residual | fused µs | unfused µs | fused speedup (unfused/fused) | fused %BW | unfused %BW | max rel diff |
|---|---|---|---|---|---|---|---|
| 1 | no | 4.72 | 4.05 | 0.86× | 0.1% | 0.2% | 6.3e-02 |
| 1 | yes | 5.06 | 4.79 | 0.95× | 0.2% | 0.2% | 5.1e-02 |
| 16 | no | 4.88 | 4.69 | 0.96× | 1.3% | 1.3% | 5.9e-02 |
| 16 | yes | 5.07 | 4.95 | 0.98× | 2.8% | 2.8% | 6.2e-02 |
| 64 | no | 4.97 | 4.86 | 0.98× | 4.8% | 4.9% | 5.4e-02 |
| 64 | yes | 5.17 | 5.15 | 1.00× | 10.7% | 10.7% | 5.1e-02 |
| 256 | no | 6.40 | 6.22 | 0.97× | 14.9% | 15.3% | 5.5e-02 |
| 256 | yes | 6.90 | 6.53 | 0.95× | 31.9% | 33.7% | 5.6e-02 |
| 1024 | no | 16.75 | 12.35 | 0.74× | 22.7% | 30.7% | 4.7e-02 |
| 1024 | yes | 18.97 | 13.95 | 0.74× | 46.4% | 63.1% | 5.9e-02 |
| 4096 | no | 60.78 | 49.93 | 0.82× | 25.0% | 30.4% | 5.0e-02 |
| 4096 | yes | 71.57 | 71.09 | 0.99× | 49.2% | 49.5% | 6.2e-02 |
| 16384 | no | 222.44 | 181.35 | 0.82× | 27.3% | 33.5% | 5.8e-02 |
| 16384 | yes | 260.94 | 269.67 | 1.03× | 54.0% | 52.2% | 6.2e-02 |
