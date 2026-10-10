# FP8 norm+quant fusion in vLLM on H100: a GEMM-selection regression and an underperforming fused kernel

Investigation of activation-quantization overhead in vLLM's FP8 block-scaled decode path. It started as "why are the fusion passes not firing" and ended with two findings: forcing norm+quant fusion on sm90 is a net loss because of which GEMM backend runs afterwards, and the fused `rms_norm_per_block_quant` kernel is slow on its own.

**Setup.** Qwen3-8B-FP8 (W8A8, 1×128 block scales), H100 NVL (132 SMs, HBM3 peak 3.9 TB/s), unprivileged vast.ai container. Builds: bs=1 end-to-end runs on vLLM 0.23.1rc1.dev1458 (nightly), bs=256 runs and all microbenchmarks on vLLM 0.23.0 (PyPI). Within each column of each table, all arms come from the same build. Kernel-level numbers predate vLLM #54111 (see [Context](#context-that-arrived-during-the-work)).

## TL;DR

- **Forcing norm+quant fusion is a net loss today:** +13% decode latency at bs=1 (trace prediction confirmed by wall clock) and +56% GPU time per decode step at bs=256 (from the trace; wall clock not captured). GEMM selection accounts for ~1970 of the ~2700 µs/step gap at bs=256; the quant kernel for ~160.
- **The fused kernel underperforms on its own:** `rms_norm_per_block_quant` moves less memory than the rms_norm + quant pair it replaces, yet is 1.08–1.36× slower in isolation. Its marginal cost per element is ~5.5× the pair's, and it reaches 17–34% of HBM3 peak.

| Decode step, Qwen3-8B-FP8, H100 NVL | bs=1 | bs=256 |
|---|---|---|
| A: unfused, default backend | baseline | 4782 µs |
| B: norm+quant fused, DeepGEMM | ≈ +13% | — |
| C: norm+quant fused, CUTLASS blockwise | ≈ +13% | 7478 µs (+56%) |

bs=1 deltas are end-to-end latency; bs=256 values are summed GPU kernel time per decode step from the trace.

## 1. Fusion is net-negative on sm90 today

**Why the passes don't fire.** The default sm90 linear backend, `FlashInferFp8DeepGEMMDynamicBlockScaledKernel`, takes BF16 activations and quantizes them internally with TRT-LLM's `scale_1x128_kernel`. `RMSNormQuantFusionPass` matches vLLM's own quant op, so it finds nothing to fuse, even though the logs report the pass as enabled. This is consistent with the skips in `tests/compile/fusions_e2e/test_tp1_quant.py`.

**What forcing it costs.** Arms B and C disable the default backend via `VLLM_DISABLED_KERNELS`. The linear then falls back to backends that take pre-quantized input, and the existing fusion pass fires.

- **bs=1, A→C:** +548 µs per step. Of that, +362 µs comes from the GEMM (losing the `deep_gemm::fp8_gemm_kernel_swapAB` path) and +186 µs from the fused norm+quant kernel, which makes 108 fewer launches but takes more time. For this run, the trace-predicted regression (+11.1 ms) matched the wall-clock regression (+12.2 ms).
- **bs=256:** `cutlass_3x_gemm_fp8_blockwise` takes 33.6 µs/launch vs 19.9 µs for `deep_gemm::fp8_gemm_kernel_swapAB` (1.69×). That is ≈1972 µs per step (≈26% of arm C's step), while the quant kernel accounts for ≈163 µs (2.2%). The GEMM gap grows with batch: +362 µs/step at bs=1, ≈1972 µs/step at bs=256.

**Why it matters.** vLLM's Q3 roadmap moves quant fusion from `torch.compile` passes into model code via `QuantizedActivation` (RFC #42770, #43224; the RMSNorm+quant port is tracked in #43500). Once the norm emits pre-quantized activations, the linear can no longer use the default sm90 backend unless that backend accepts external quantization. Without that, the port inherits the regression above. Two ways out: gate fused norm+quant on sm90 by the selected linear backend, or let the default backend accept pre-quantized input so that fusion removes the quant step without giving up the GEMM.

## 2. The fused kernel underperforms on its own

| kernel | ns / element (hidden sweep, n=1) | HBM3 utilization (n=16384, peak 3.9 TB/s) |
|---|---|---|
| `rms_norm` alone | 0.105 | 81–89% |
| block quant alone | ~0.005 | 32% |
| **sum of parts** | **0.110** | — |
| separate pair, measured | — | 49–58% |
| **fused** | **0.609** | **17–34%** |

- **Total time in isolation.** On the same work, the fused kernel is 1.08–1.36× slower than the separate pair. At n=16384 without residual, it moves 43% of the pair's bytes in 128% of the pair's time; with residual the ratio is 1.09×.
- **Floor at n=1.** On an idle GPU, the fused kernel's floor is 5.5 µs vs 4.4 µs for the separate pair (2.2 + 2.2). Both run as single blocks here, so occupancy is not the explanation.
- **Cost model** for the production shape (hidden=4096, group=128), fitted on the n=1 hidden sweep: 5.70 µs = 2.40 µs fixed (42%) + 0.609 ns/element (44%) + 25.3 ns/group (14%). The isolated floor (5.5 µs) and the fit (5.70 µs) come from different sweeps. The in-model average from the bs=1 trace (6.85 µs/launch) is dominated by the with-residual variant under real-model conditions.
- **Bandwidth ceiling.** Without residual at n=16384 the kernel moves 203 MB. At an achievable ~3.3 TB/s that is ~62 µs; measured is 314 µs, about 5× off.
- **Production comparison, with a caveat.** Inside the model, the fused kernel takes 6.85 µs/launch at bs=1 and 7.08 at bs=256, vs 4.04 and 4.82 for the unfused path (1.70× and 1.47×). That baseline is TRT-LLM's `scale_1x128_kernel` plus an Inductor Triton norm, not vLLM's own pair, so part of this gap is "TRT-LLM's quant kernel is faster than vLLM's" rather than "fusion is slow". The isolated 1.08–1.36× above is the like-for-like number.

**Ruled out:**
- **SM occupancy.** Per-launch time is flat across a 256× batch increase (6.85 → 7.08 µs), and the kernel loses at n=1 on an idle GPU.
- **Transposed scale writes.** `transpose_scale` True vs False costs 3–8% everywhere.
- **Sequential group reductions.** Real, but 14% of cost, isolated by controlling `group_size`.

**Likely causes**, from reading the kernel on `main` (not yet confirmed with `ncu` or SASS):
- three separate grid-stride passes over the row, each re-reading the input (and the residual, when present);
- 8-byte vector loads for bf16 (`VEC_SIZE=4`) where 16-byte loads are available;
- no `__launch_bounds__`.

## Context that arrived during the work

- **vLLM #54111** (merged 2026-08-27) rewrote `compute_dynamic_per_token_scales` (race fix). Measurements here predate it, so the per-group component (14%) should be re-measured on HEAD. The three-pass structure and vec4 loads are untouched.
- **vLLM #36895** adds an opt-in Helion implementation of `rms_norm_per_block_quant`, reported at 1.67–2.06× faster than the C++ kernel on H100. The C++ kernel remains the default. This is an independent confirmation that the C++ kernel leaves performance on the table.
- **flashinfer #4480** (merged 2026-08-26) ships `FusedAddRMSNormFP8BlockQuant`: a single pass, ~75% of HBM peak on H200, with scales emitted in the layout DeepGEMM consumes.

## Known gaps

- Wall-clock latency at bs=256 was not captured; the bs=256 numbers are trace-derived GPU time.
- No `ncu`: it fails with `ERR_NVGPUCTRPERM` in an unprivileged container. Bandwidth figures are derived from bytes moved and measured time.
- Raw profiler traces were not preserved (they lived in `/dev/shm` on the GPU host). `run_bs256.sh` regenerates them.
- Only sm90 (H100 NVL) was measured. Other architectures are unchecked.
- A rerun on current nightly is planned, to cover #54111 and to capture wall clock at bs=256.

## Files

| path | contents |
|---|---|
| `microbench_quant.py` | Fused op vs the rms_norm + quant pair; defines `bench_gpu` (CUDA events) |
| `microbench_hidden.py` | Hidden-size sweep at n=1 with a `group_size` control; imports from `microbench_quant.py`, keep them together |
| `an.py` | Loads a torch profiler trace (chrome JSON, `.gz`), finds the decode-step period by self-similarity of the kernel-name sequence, categorizes kernels by regex |
| `an2.py` | Segments steps by the sampler marker and prints a per-step kernel census; imports `an.py` |
| `run_bs256.sh` | Driver for arms C and A at bs=256 with profiling |
| `sweep_hidden_n1.txt` | Raw hidden sweep. `ERR` rows at group 256 and 32 are expected: the fused op is instantiated only for 64 and 128 |
| `sweep_and_trace_bs256.txt` | Raw bs=256 output. The latency sections are empty (see Known gaps) |
| `vllm_qwen3_fp8_profile.md` | Original profiling report (2026-07-28). Predates the microbenchmarks |

**Tooling caveats.** The `an2.py` marker can match an ArgMax `reduce_kernel` that fires about twice per step, which yields half-step segments. Windows were verified by a kernel census against the 36-layer model: 568 launches per step at bs=1 (arm A), 1120 at bs=256 (arm C). Kernel categories are regex-based, and `scale_1x128_kernel` lives under the `fp8_blockscale_gemm` namespace, so match it by `scale_1x128` explicitly.

## Reproduce

```bash
# Arm A (default). For B and C, prefix with:
#   B: VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel
#   C: VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel
vllm bench latency --model Qwen/Qwen3-8B-FP8 \
  --input-len 1024 --output-len 16 --batch-size 1 \
  --num-iters-warmup 1 --num-iters 1 --profile \
  --profiler-config.profiler=torch \
  --profiler-config.torch_profiler_with_stack=false \
  --profiler-config.torch_profiler_dir=./traces/A

# bs=256, arms C and A. Written for a vast.ai H100 container: installs vLLM 0.23.0
# into /root/venv, expects an.py and an2.py in /root, writes /root/RESULTS256.txt
cp an.py an2.py /root/ && bash run_bs256.sh

# Per-step census from a trace
python an2.py <trace.json.gz>

# Microbenchmarks (need a vLLM build with the _C ops; numbers above are from H100 NVL)
python microbench_quant.py
python microbench_hidden.py
```

**Environment notes** (vast.ai container; details in `run_bs256.sh`):
- `/dev/shm` is tmpfs mounted `noexec`, so the venv must live on the overlay: torch's `.so` files fail to `dlopen` from tmpfs.
- `uv` unpacks wheels into `TMPDIR` before moving them into the venv. On the 16 GB overlay this double-counts every package and hits `ENOSPC` on torch, so `TMPDIR` goes to `/dev/shm`.
- Inductor and Triton JIT their launchers and `dlopen` them, so `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` must point at the exec-capable overlay, not at `TMPDIR`.
- The nightly wheel used for bs=1 later returned 404, so the rest was pinned to vLLM 0.23.0 from PyPI.
- `ncu` needs GPU performance-counter permissions that unprivileged containers don't have.
