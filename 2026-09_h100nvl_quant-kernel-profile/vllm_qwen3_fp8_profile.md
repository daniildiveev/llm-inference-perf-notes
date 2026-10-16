# Qwen3-8B-FP8 decode profile — vLLM on H100 NVL

Torch-profiler trace analysis of `vllm bench latency` running **Qwen3-8B-FP8**
(W8A8, block-scaled) on an **H100 NVL 94 GB**, at batch size 1 and 256.

The driving question: **how many distinct GPU kernels make up one decode step,
and which of them are activation quantization?**

Answer up front: **568 kernel launches per decode step, identical at bs=1 and
bs=256.** Of those, **144 (25.4%) are activation quantization** — one
`scale_1x128_kernel` before every one of the 4 linear projections in each of the
36 layers. They account for only ~6% of GPU time, and they are *not* fused into
the preceding norm/activation kernels despite vLLM reporting the fusion as
enabled. A follow-up A/B/C forcing the fusion to engage showed this is the *fast*
configuration — fusing costs ~13% end-to-end, not less. Two thirds of that is
the loss of an M=1-specialized GEMM; one third is the fused quant kernel itself
being slower than the separate kernels it replaces.

## Setup

| | |
|---|---|
| GPU | NVIDIA H100 NVL, 94 GB HBM3, sm90, driver 580.159.03 / CUDA 13.0 |
| Host | 40 vCPU, 314 GB RAM, Ubuntu 24.04 (vast.ai container) |
| vLLM | `e222c33f2f568b54b41d2ebfc44f6c85ea072737` → `0.23.1rc1.dev1458` |
| Install | editable source install + precompiled kernel wheel |
| torch | 2.13.0+cu130 |
| Model | `Qwen/Qwen3-8B-FP8` |

Model config, as read from `config.json`:

```json
"quantization_config": {
  "activation_scheme": "dynamic",
  "fmt": "e4m3",
  "quant_method": "fp8",
  "weight_block_size": [128, 128]
}
```

36 layers, `hidden_size` 4096, `intermediate_size` 12288, GQA 32 Q-heads / 8
KV-heads, `head_dim` 128, vocab 151936, `tie_word_embeddings: false`.

`activation_scheme: dynamic` is the reason the question is interesting —
activations are quantized **at runtime, on every step**, rather than with a
precomputed static scale.

Engine configuration picked up at boot:

```
Selected FlashInferFp8DeepGEMMDynamicBlockScaledKernel for Fp8LinearMethod
DeepGEMM E8M0 enabled on current platform
Enabled custom fusions: norm_quant, act_quant
custom_ops: ['+quant_fp8', 'none', '+quant_fp8']
cudagraph_mode: FULL_AND_PIECEWISE
Asynchronous scheduling is enabled
Available KV cache memory: 74.41 GiB  (541,840 tokens)
```

## Build notes

Four things that cost time and are worth recording.

**1. `/dev/shm` is `noexec`.** The container has 157 GB of tmpfs and 314 GB of
RAM, which looks like an ideal home for the venv on a 16 GB disk. It is not —
loading any `.so` from it fails with `failed to map segment from shared object`.
Neither `mount -o remount,exec /dev/shm` nor mounting a fresh tmpfs works; the
container has no mount capabilities. tmpfs is still fine for *data* (model
weights, traces), just not for anything executable.

**2. vLLM main often has no published wheels.** `VLLM_USE_PRECOMPILED=1` resolves
a wheel from `wheels.vllm.ai/<commit>/`. At the time of this run:

- the `cu129/` and `cu130/` variant directories were **404 for every commit tested**;
  only the default (no-variant) layout is populated
- the default layout **lagged main by ~3 days** — every commit from the current
  day 404'd

`setup.py` maps `nvidia-smi`'s reported CUDA version to a variant
(`{12: "cu129", 13: "cu130"}`), so a CUDA 13 box asks for `cu130` and gets a 404;
it then falls back to the default variant, which also 404s if the commit is too
new. The fix is to pin to the commit that `nightly` points at:

```bash
curl -s https://wheels.vllm.ai/nightly/vllm/metadata.json   # -> commit + filename
git checkout <that-commit>
export VLLM_PRECOMPILED_WHEEL_LOCATION="https://wheels.vllm.ai/<commit>/<wheel>.whl"
```

Pinning the source to the *same* commit as the wheel keeps the Python and the
compiled `.so` in sync.

**3. `UV_NO_CACHE=1` doubles peak disk.** vLLM's dependency tree is 162 packages
— torch, torchvision, flashinfer, tilelang, z3-solver, opencv, and *both*
`nvidia-cutlass-dsl-libs-cu12` and `-cu13`. With no cache, uv stages into
`TMPDIR` and then **copies** into the venv, so both exist at once and a 16 GB
disk dies with `No space left on device`. With a persistent cache on the same
filesystem, uv **hardlinks** instead:

```bash
export UV_CACHE_DIR=/root/uvcache
export UV_NO_CACHE=0
export UV_LINK_MODE=hardlink
```

Peak usage dropped from >16 GB (failure) to 7.9 GB (success).

**4. `VLLM_TORCH_PROFILER_DIR` is dead.** It now warns
`Unknown vLLM environment variable detected` and the run fails at profile start
with `RuntimeError: Profiling is not enabled`. Current flags:

```
--profiler-config.profiler=torch
--profiler-config.torch_profiler_dir=<dir>
--profiler-config.torch_profiler_with_stack=false
```

`with_stack` defaults to **true** and inflates traces substantially without
helping a kernel-level analysis.

Also of note: the first boot spends ~90 s JIT-compiling **1735 DeepGEMM kernel
variants** (`ptxas -arch sm_90a`) plus ~30 s in torch.compile. Both are cached
under `/root/.cache`, so subsequent boots skip them. Loading the 8.83 GiB of
weights from tmpfs took **1.14 s**.

## Latency

Clean runs, profiler disabled, 1024 input / 128 output tokens, 3 warmup + 5
measured iterations.

| batch | avg latency | p10 | p99 | output tok/s |
|---|---|---|---|---|
| 1 | **0.5798 s** | 0.5786 | 0.5826 | ~221 |
| 256 | **9.2148 s** | — | — | ~3556 |

These are end-to-end and **include prefill**, which at bs=256 means 262,144
prompt tokens — the dominant term in the 9.21 s. The per-decode-step numbers
below come from the traces and are prefill-free; they are the more meaningful
figure for decode analysis.

## Method: finding a decode step

Trace step boundaries were derived two independent ways, and they agree.

**Autocorrelation.** Take the time-ordered sequence of GPU kernel names and find
the smallest period `p` where `names[i] == names[i+p]` holds across the
steady-state middle of the trace. This assumes nothing about vLLM's internals.
For bs=1 it returned **p = 568 at 0.996 self-similarity**.

This method alone was *not* sufficient. At bs=256 it returned **p = 15 at 0.871**
— that is one *layer*, not one step, and the window it measured was contaminated
by the very large prefill. Reported blindly it would have given a decode step of
"15 kernels, 10.4 ms", which fails an obvious sanity check: 10.4 ms × 36 layers
is 374 ms/step, irreconcilable with the measured ~72 ms/token.

**Sampler-delimited segmentation.** The sampler runs exactly once per decode
step, so `_combine_sampled_and_draft_tokens_kernel` is a reliable delimiter.
Cutting the kernel stream on it and taking the last full segment gives the
steady-state decode step directly, immune to prefill contamination:

```
bs=1    marker count 16   last 12 segment sizes: [568]*12
bs=256  marker count 33   last 12 segment sizes: [568,568,568,568,568,568,568,569,569,569,569,568]
```

Both methods return **568** for bs=1. Segmentation returns 568 for bs=256 too.
(The occasional 569 is a stray one-off kernel, not a structural difference.)

## Kernels per decode step

**568 launches — the same at bs=1 and bs=256.** Batch size changes kernel
*durations*, never the count. Expected, given a captured CUDA graph with fixed
topology, but worth confirming rather than assuming.

| category | launches | bs=1 µs | bs=1 % | bs=256 µs | bs=256 % |
|---|---|---|---|---|---|
| gemm | 149–150 | 2981.4 | 71.6% | 3486.7 | 55.0% |
| attention | 109 | 602.7 | 14.5% | 2053.0 | 32.4% |
| **act_quant** | **144** | **253.3** | **6.1%** | **361.2** | **5.7%** |
| norm | 73 | 165.9 | 4.0% | 198.0 | 3.1% |
| other | 42 | 91.9 | 2.2% | 137.9 | 2.2% |
| activation | 36 | 46.0 | 1.1% | 70.1 | 1.1% |
| mem | 7 | 10.0 | 0.2% | 10.6 | 0.2% |
| elem | 6–7 | 8.7 | 0.2% | 12.2 | 0.2% |
| sample | 1 | 4.4 | 0.1% | 5.5 | 0.1% |
| **TOTAL** | **568** | **4164.3** | | **6335.2** | |

### Per-layer anatomy

The 568 decomposes as **15 kernels × 36 layers = 540**, plus 28 for embedding,
`lm_head`, sampling, and block-table/slot-mapping bookkeeping.

| # | kernel | role |
|---|---|---|
| 1 | `triton_red_fused_fused_add_rms_norm_0` | input RMSNorm + residual add |
| 2 | `scale_1x128_kernel` | **quantize activations → fp8** |
| 3 | `deep_gemm::fp8_gemm_kernel_swapAB<6144, 4096>` | qkv_proj (4096+1024+1024) |
| 4 | `triton_red_fused_3` | QK-norm (Qwen3 per-head q_norm/k_norm) |
| 5 | `vllm::reshape_and_cache_flash_kernel` | write K/V into paged cache |
| 6 | `flash::FlashAttnFwdSm90` | attention |
| 7 | `flash::FlashAttnFwdCombine` | split-K combine |
| 8 | `scale_1x128_kernel` | **quantize activations → fp8** |
| 9 | `deep_gemm::fp8_gemm_kernel_swapAB<4096, 4096>` | o_proj |
| 10 | `triton_red_fused_fused_add_rms_norm_2` | post-attention RMSNorm |
| 11 | `scale_1x128_kernel` | **quantize activations → fp8** |
| 12 | `deep_gemm::fp8_gemm_kernel_swapAB<24576, 4096>` | gate_up_proj (2×12288) |
| 13 | `triton_poi_fused_mul_silu_slice_1` | SwiGLU |
| 14 | `scale_1x128_kernel` | **quantize activations → fp8** |
| 15 | `deep_gemm::fp8_gemm_kernel_swapAB<4096, 12288>` | down_proj |

The four GEMM template shapes map cleanly onto the config: `6144 = 32·128 +
8·128 + 8·128` (qkv with GQA), `4096×4096` (o_proj), `24576 = 2×12288`
(gate+up fused), `4096×12288` (down_proj).

At bs=256, DeepGEMM selects a different tile configuration for the gate_up GEMM
(`128u, 24u, ..., 6u` vs `128u, 16u, ..., 8u` at bs=1) — the only kernel whose
*identity*, not just duration, changes with batch.

## Activation quantization

One kernel, 144 launches per step:

```
void tensorrt_llm::kernels::fp8_blockscale_gemm::scale_1x128_kernel<__nv_bfloat16, __nv_fp8_e4m3>
```

**144 = 36 layers × 4 linear projections.** Every GEMM is immediately preceded by
a quantization of its input activations into fp8 with 1×128 block scales.

| | bs=1 | bs=256 |
|---|---|---|
| launches | 144 / 568 (**25.4%**) | 144 / 568 (**25.4%**) |
| total time | 253.3 µs (6.1% of step) | 361.2 µs (5.7% of step) |
| **per launch** | **1.76 µs** | **2.51 µs** |

> A note on classification: an automated categorizer will mis-bucket this kernel
> as a GEMM, because its namespace is `fp8_blockscale_gemm`. The first pass of
> this analysis reported "0 activation-quant kernels" for exactly that reason.
> Match on `scale_1x128`, not on `quant`.

## Insights

**1. Quantization is not fused, despite the config saying it is.** Boot logs
report `Enabled custom fusions: norm_quant, act_quant`, `fuse_norm_quant: True`,
`fuse_act_quant: True`, and `custom_ops: ['+quant_fp8', ...]`. The trace shows
otherwise: quantization appears as 144 **standalone** `scale_1x128_kernel`
launches, and the surrounding triton kernels are named
`triton_red_fused_fused_add_rms_norm_0` and `triton_poi_fused_mul_silu_slice_1`
— neither contains `quant`. Had `norm_quant` fusion engaged, the quantization
would be in the norm kernel's epilogue and would not appear separately at all.

The cause is a backend mismatch: the fusion passes rewrite vLLM's own
`quant_fp8` custom op, but the selected backend
(`FlashInferFp8DeepGEMMDynamicBlockScaledKernel`) supplies TensorRT-LLM's
`scale_1x128_kernel` instead, which those passes don't recognize.

**This was verified by A/B/C experiment, and the headroom estimate it originally
carried was wrong — see "Fusion A/B/C" below.** Forcing the fusion to engage
costs ~13% end-to-end; it does not save 6%.

### Fusion A/B/C

`VLLM_DISABLED_KERNELS` makes the kernel selector skip a candidate class
(`vllm/model_executor/kernels/linear/__init__.py:952`), so disabling kernels
walks down `_POSSIBLE_FP8_BLOCK_KERNELS` (`__init__.py:359`). Three arms, all
bs=1, in-len 1024, out-len 16, same commit:

```
A:  (nothing disabled)     -> FlashInferFp8DeepGEMMDynamicBlockScaledKernel
B:  disable FlashInfer     -> DeepGemmFp8BlockScaledMMKernel
C:  disable FlashInfer,    -> CutlassFp8BlockScaledMMKernel
       DeepGemm
```

Arm C is the de-confounder: it keeps B's vLLM quant op but changes the GEMM
again, and it matches A's GEMM *launch count* (150), so A↔C isolates the two
effects that A↔B alone conflates.

| arm | kernel                       | launches | GPU/step  | wall, 10 iters |
| --- | ---------------------------- | -------: | --------: | -------------: |
| A   | `FlashInferFp8DeepGEMM…`     |      568 | 4176.1 µs | **0.09363 s**  |
| B   | `DeepGemmFp8BlockScaledMM`   |      533 | 4872.6 µs |    0.10577 s   |
| C   | `CutlassFp8BlockScaledMM`    |      497 | 4768.0 µs |    0.10643 s   |

**The fusion passes are not broken.** In B and C the quantization does move into
the epilogues, exactly as predicted (arm C shown):

```
 71x  486.2us  vllm::rms_norm_per_block_quant_kernel<BFloat16, Float8_e4m3fn, true, true, 128>
 36x   85.4us  per_token_group_quant_8bit_kernel<BFloat16, Float8_e4m3fn, false, false, …>
 36x   71.2us  vllm::silu_and_mul_per_block_quant_kernel<BFloat16, Float8_e4m3fn, true, 128>
  1x    7.2us  vllm::rms_norm_per_block_quant_kernel<…, false, true, 128>
```

Confirmed by what disappears: `norm` drops 73 launches → 1, `act` drops 36 → 0.
The 36 surviving standalone `per_token_group_quant_8bit` are the o_proj inputs,
which follow attention rather than a norm or activation and so have no epilogue
to fuse into. The passes simply cannot see into TensorRT-LLM C++, which is why
they never fired in A.

**Decomposition, A → C** (GEMM launch count held at 150 in both):

| block              |      A |      C |       Δ |
| ------------------ | -----: | -----: | ------: |
| gemm               | 2976.6 | 3338.9 | +362.3 µs|
| quant + norm + act |  466.7 |  652.8 | +186.1 µs|
| attn               |  617.2 |  613.8 |   −3.4 µs|
| other              |   92.8 |  137.3 |  +44.5 µs|
| **total**          | 4176.1 | 4768.0 | +591.9 µs|

So the regression is **two-thirds GEMM, one-third fusion** — two independent
effects, not one:

- **GEMM (~362 µs/step).** `deep_gemm::fp8_gemm_kernel_swapAB` is the M=1 decode
  specialization; both replacements lose it. B additionally pays 36×
  `deep_gemm::transpose_fp32` (54.9 µs) to reformat scales. This is the larger
  prize and has nothing to do with quantization.
- **Fusion (~186 µs/step).** Isolated:

  ```
  A:  144 scale_1x128 (255.1) + 73 norm (165.7) + 36 silu (45.9)  = 253 launches, 466.7us
  C:  144 fused        (650.0) +  1 norm   (2.8)                  = 145 launches, 652.8us
  ```

  C uses **108 fewer launches and takes 186 µs more**. Per launch, A's
  norm+quant pair costs 2.27 + 1.77 = 4.04 µs; C's fused
  `rms_norm_per_block_quant` costs 6.85 µs for the same work — **~70% slower than
  the two separate kernels it replaces.** Launch count was never the bottleneck.

**Wall clock confirms it end-to-end.** Predicted vs measured, 16 decode steps
per iteration:

|   | GPU/step  | ×16 steps | wall measured |
| - | --------: | --------: | ------------: |
| A | 4176.1 µs |   66.8 ms |       93.6 ms |
| B | 4872.6 µs |   78.0 ms |      105.8 ms |
| Δ | +696.5 µs |  +11.1 ms |      **+12.2 ms** |

The trace-predicted regression (+11.1 ms) matches the measured one (+12.2 ms) to
within ~1 ms, the residual being prefill and host overhead, which are constant
across arms.

Caveats: traces and wall clock were measured on two different (identical-model)
H100 NVL boxes, so sub-1% cross-arm differences are not meaningful — in
particular C shows 104.6 µs/step *less* GPU time than B yet measures 0.6%
*slower* in wall clock, consistently across percentiles. That inversion is
unexplained and too small to read anything into. Arm A reproduced to within 0.5%
across three separate boxes (4157.3 / 4176.1 µs), so the large effects are solid.

**2. Activation quant is launch-bound, not bandwidth-bound.** 1.76 µs per launch
at bs=1 against 2.51 µs at bs=256 — **256× the data for 1.43× the time**. At
bs=1 each launch quantizes a single `[1, 4096]` or `[1, 12288]` row; the kernel
is essentially all fixed overhead. This is why the fusion opportunity in (1)
matters disproportionately at low batch.

**3. bs=1 decode is squarely memory-bandwidth-bound.** The fp8 linear weights
total 192.9 M params/layer × 36 = **6.95 G params = 6.95 GB**. The four
DeepGEMM kernels sum to 2617.6 µs at bs=1:

```
6.95 GB / 2617.6 µs = ~2.65 TB/s     (H100 NVL peak ~3.9 TB/s, so ~68%)
```

The confirming measurement is the batch sweep: GEMM time rises only
**2981 → 3487 µs (1.17×)** while performing **256× the arithmetic**. At bs=1 the
tensor cores are idle and the GPU is doing nothing but streaming weights.

**4. `lm_head` is a fixed tax that dominates at low batch.** The vocabulary
projection runs as a single cuBLAS kernel — `nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT`
at bs=1, `..._384x24_...` at bs=256 — and costs **351.0 µs at bs=1 vs 361.1 µs
at bs=256**. Essentially flat across a 256× batch increase.

It is left in **bf16** (`quantization_config` covers the linear layers, not the
head): 151936 × 4096 × 2 B = **1.245 GB**.

```
1.245 GB / 351.0 µs = ~3.55 TB/s     (~91% of peak — fully bandwidth-bound)
```

At bs=1 this one unquantized kernel is **8.4% of the entire decode step**, and
you pay it identically whether serving 1 request or 256.

**5. Attention is what actually scales with batch.** 14.5% → 32.4% of the step;
`FlashAttnFwdSm90` goes **400.4 → 1832.5 µs (4.6×)**. Meanwhile
`FlashAttnFwdCombine` — the split-K reduction — stays flat at 117.3 → 111.4 µs.
At bs=1 that combine step is 117 µs of pure overhead for splitting work that was
already too small to need splitting; at bs=256 it is amortized to noise.

**6. Overall step scaling.** 4164 → 6335 µs, i.e. **1.52× the time for 256× the
tokens**: 240 tok/s of decode at bs=1 against 40,410 tok/s at bs=256, a **168×**
improvement in decode throughput. The gap between end-to-end per-token latency
(4.53 ms) and GPU-busy time in the trace (4.16 ms) covers prefill amortization
and any non-overlapped CPU work.

## Optimization leads

Ranked by expected bs=1 payoff:

1. ~~**Fuse `scale_1x128` into the RMSNorm/SiLU epilogues** — ~250 µs/step (6%)~~
   **Refuted by A/B/C measurement.** Routing to a backend that uses vLLM's
   `quant_fp8` op does fuse it, and costs ~13% end-to-end rather than saving 6%.
   See "Fusion A/B/C". Two live leads replace it:

   1a. **Make `vllm::rms_norm_per_block_quant_kernel` competitive** — ~186 µs/step
   (4.5%). It costs 6.85 µs/launch where the separate norm + `scale_1x128` pair
   it replaces costs 4.04 µs. A fused kernel doing strictly less memory traffic
   should not be 70% slower; this looks like an unoptimized epilogue rather than
   an inherent limit. Fixing it would make fusion a win instead of a loss.

   1b. **Keep the M=1 GEMM specialization** — ~362 µs/step (8.7%), the larger
   prize and independent of quantization. `deep_gemm::fp8_gemm_kernel_swapAB`
   beats both `sm90_fp8_gemm_1d2d_impl` and Cutlass blockwise at decode shapes.
   Any backend change that silently drops it pays this immediately.
2. **Quantize `lm_head`** — up to ~175 µs/step (4%) if fp8 halves the 1.245 GB
   read. Watch output quality; the head is quantization-sensitive.
3. **Revisit split-K at bs=1** — `FlashAttnFwdCombine` is 117 µs (2.8%) for a
   1024-token context with a single sequence.

Items 1 and 2 together are ~10% of the bs=1 decode step, and both are pure
overhead in the sense that neither moves any useful additional data.

## Caveats

- **Profiled runs used `--output-len 16`**, not 128, to keep traces openable.
  They establish per-step *structure*; the latency table comes from separate
  clean runs with profiling off.
- **Prefill is not decomposed.** The bs=256 end-to-end 9.21 s is dominated by
  prefilling 262,144 tokens, and nothing here analyses that phase. All trace
  figures are steady-state decode.
- Kernel-category assignment is regex-based over kernel names. The categories
  `gemm`/`elem`/`other` contain a handful of sub-2 µs PyTorch utility kernels
  whose exact bucket is arbitrary; this does not affect any figure quoted above.
- `triton_red_fused_3` (36×, 62.7 µs) is identified as Qwen3's per-head QK-norm
  by position and count. The name is opaque, so treat that specific attribution
  as inference rather than confirmed — `enable_qk_norm_rope_fusion` was `False`,
  so RoPE is presumably elsewhere in the same region.
- Single run per configuration. Latency percentiles were tight (bs=1 p10→p99
  spans 0.7%), but the trace figures are from one profiled iteration.

## Reproduction

```bash
# build
git clone https://github.com/vllm-project/vllm.git && cd vllm
curl -s https://wheels.vllm.ai/nightly/vllm/metadata.json    # -> COMMIT, WHEEL
git checkout $COMMIT
uv venv --python 3.12 /root/venv
export UV_CACHE_DIR=/root/uvcache UV_NO_CACHE=0 UV_LINK_MODE=hardlink
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_LOCATION="https://wheels.vllm.ai/$COMMIT/$WHEEL"
uv pip install -e . --python /root/venv/bin/python

# latency (clean)
vllm bench latency --model Qwen/Qwen3-8B-FP8 \
  --input-len 1024 --output-len 128 --batch-size 1 \
  --num-iters-warmup 3 --num-iters 5

# latency (profiled)
vllm bench latency --model Qwen/Qwen3-8B-FP8 \
  --input-len 1024 --output-len 16 --batch-size 1 \
  --num-iters-warmup 1 --num-iters 1 --profile \
  --profiler-config.profiler=torch \
  --profiler-config.torch_profiler_with_stack=false \
  --profiler-config.torch_profiler_dir=./traces/bs1
```

Analysis scripts used: `an.py` (autocorrelation period detection + kernel
categorization) and `an2.py` (sampler-delimited step segmentation).
