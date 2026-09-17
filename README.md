# vLLM FP8 quant fusion: `rms_norm_per_block_quant` is memory-bound at 17–34% of peak

Investigation of activation quantization overhead in vLLM's FP8 block-scaled decode path (Qwen3-8B-FP8, H100 NVL). Started as "why are fusion passes not firing", ended as a measured, backend-independent claim about one kernel — plus a larger GEMM finding set aside for follow-up.

Scripts and raw sweeps are in this repo; everything reproduces with CUDA events only, no `ncu` required, and the microbenchmarks run on a consumer GPU.

## The claim

`vllm::rms_norm_per_block_quant_kernel` walks a row ~5.5× slower than the two kernels it replaces, doing exactly their work:

| kernel | ns / element | HBM3 utilization (n=16384) |
|---|---|---|
| `rms_norm` alone | 0.105 | 81–89% |
| block quant alone | ~0.005 | 32% |
| **sum of parts** | **0.110** | — |
| **fused** | **0.609** | **17–34%** |

At n=1, one block, otherwise-idle GPU: fused = 5.5 µs vs 2.2 µs for a comparable single kernel. Occupancy is not the explanation.

Cost model for the production shape (hidden=4096, group=128), 5.70 µs total: 2.40 fixed (42%) + 0.609 ns/element (44%) + 25.3 ns/group (14%). Ceiling from bandwidth alone: 203 MB at 3.3 TB/s is ~62 µs instead of 314 — ~5×.

Likely cause (from reading the kernel on `main`): three separate grid-stride passes over the row, each re-reading input; 8-byte vector loads for bf16 (`VEC_SIZE=4`) where 16-byte are available; no `__launch_bounds__`. This turns the finding from "issue with a measurement" into "PR with a diff".

## How it was established

1. **Profile** decode step of Qwen3-8B-FP8 on H100: 568 kernel launches/step, 144 (25.4%) are activation quant (`scale_1x128_kernel`, TRT-LLM), ~6% of step time. Step boundaries found two independent ways (autocorrelation of kernel names + sampler-marker segmentation), cross-checked by kernel census against the 36-layer model.
2. **Hypothesis 1** — fusion passes broken, ~6% on the table — **refuted by own A/B/C experiment** via `VLLM_DISABLED_KERNELS`: passes are fine, they just can't see inside TRT-LLM C++; forcing fusion costs ~13% end-to-end. Trace-predicted regression (+11.1 ms) matched wall clock (+12.2 ms).
3. **Decomposition** of the regression: two-thirds is losing DeepGEMM's M=1 `swapAB` specialization, one-third is the fused kernel itself — which uses 108 fewer launches and 186 µs more time.
4. **Three false explanations eliminated** — SM occupancy (flat across 256× batch and loses at n=1 on idle GPU), transposed scale writes (3–8% everywhere), sequential group reductions (real, but 14% of cost, isolated via `group_size` control).
5. **Microbench sweeps** by token count, hidden size, group size → per-element vs per-group vs fixed cost.

## Why it matters

vLLM's Q3 roadmap moves quant fusion from `torch.compile` passes to explicit fusion in model code via `QuantizedActivation` (RFC #42770, #43224). Everything that goes through the pattern matcher today will go through kernels of this type. A slow fused kernel obstructs their own plan rather than a side path.

## Context that arrived during the work

- **vLLM #54111** (merged 2026-08-27) rewrote `compute_dynamic_per_token_scales` (race fix). Measurements here predate it; the per-group component (14%) should be re-measured on HEAD. The three-pass structure and vec4 loads are untouched.
- **flashinfer #4480** (merged 2026-08-26) ships `FusedAddRMSNormFP8BlockQuant` — single pass, ~75% of HBM peak. An upstream fused kernel now exists; the remaining gap is on the vLLM side.

## The larger prize, set aside

`cutlass_3x_gemm_fp8_blockwise` vs `deep_gemm::fp8_gemm_kernel_swapAB`: 33.6 vs 19.9 µs/launch (1.69×), gap **grows** with batch — ≈1972 µs/step at bs=256, 12× the quant kernel. Whether this path is the default on sm89 (Ada) is unverified and decides the weight of both findings.

## Reproduce

```bash
vllm bench latency --model Qwen/Qwen3-8B-FP8 --input-len 1024 --output-len 16 --batch-size 1 \
  --profile --profiler-config.profiler=torch --profiler-config.torch_profiler_dir=./traces/bs1
# arms: VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel[,DeepGemmFp8BlockScaledMMKernel]
python microbench_hidden.py   # hidden sweep at n=1
python microbench_quant.py    # standalone quant kernel
```

Environment notes (noexec `/dev/shm`, nightly wheel pinning, `uv` disk peak, profiler flags, no `ncu` in unprivileged containers) are in the report.
