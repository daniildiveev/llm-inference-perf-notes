# llm-inference-perf-notes

One directory per experiment run. Each has its own README / RESULTS with setup, numbers and repro steps.

| Run | GPU | What | Start here |
|---|---|---|---|
| [`2026-09_h100nvl_quant-kernel-profile/`](2026-09_h100nvl_quant-kernel-profile/) | H100 NVL | vLLM 0.23.x Qwen3-8B-FP8 decode profile; `rms_norm_per_block_quant` is memory-bound; A/B/C fusion regression | [README.md](2026-09_h100nvl_quant-kernel-profile/README.md) ([ru](2026-09_h100nvl_quant-kernel-profile/README.ru.md)) |
| [`2026-10-15_h100sxm_fusion-remeasure/`](2026-10-15_h100sxm_fusion-remeasure/) | H100 SXM5 | Re-measure of norm+quant fusion on vLLM main `e8e1e961`, plus swapAB FP8-input prototype | [RESULTS.md](2026-10-15_h100sxm_fusion-remeasure/RESULTS.md) |
