#!/bin/bash
# Assemble RESULTS.md from results/*.md and RUN.md (headings of included parts demoted one level).
cd /workspace/fusion-study
demote() { sed 's/^#/##/' "$1"; echo; }
{
  cat results/00_overview.md; echo
  echo "# 1. Environment (RUN.md)"; echo; demote RUN.md
  echo "# 2. Phase 1 — kernels"; echo; demote results/phase1_kernels.md
  echo "# 3. Phase 2 — latency"; echo; demote results/phase2_latency.md
  echo "# 4. Phase 3 — attribution"; echo; demote results/phase3_attribution.md
  echo "# 5. Phase 4 — microbenchmark"; echo; demote results/phase4_microbench.md
  echo "# 6. Phase 5 — prototype"; echo; demote results/phase5_proto.md
  echo "# 7. Conclusions"; echo; demote results/99_conclusions.md
  echo "# Reproduce"; echo
  echo '```'
  echo 'bash scripts/00_setup.sh; bash scripts/00_download.sh; bash scripts/01_env.sh'
  echo "bash scripts/10_phase1_profile.sh p1 'A B C' '1 64'           # Phase 1"
  echo 'bash scripts/20_phase2_latency.sh && python scripts/21_aggregate_latency.py && python scripts/22_sensitivity.py'
  echo 'bash scripts/23_prefill_split.sh                               # Phase 2b'
  echo "bash scripts/10_phase1_profile.sh p3 'A B C' '1 32 256' && python scripts/30_attribution.py p3 1,32,256"
  echo "DELAY=200 OUTLEN=256 bash scripts/10_phase1_profile.sh p3late 'A C' 256"
  echo 'python scripts/40_microbench.py raw/p4_microbench.json; python proto/proto_gemm.py raw/p5_proto.json'
  echo 'python scripts/50_phase4_5_tables.py; bash scripts/60_assemble.sh'
  echo '```'
} > RESULTS.md
wc -l RESULTS.md
