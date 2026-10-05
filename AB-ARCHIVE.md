
## Moved out of AGENTS.md (2026-10-05 compaction) - pre-R36 sweeps on the OLD 2.38 baseline
All NEUTRAL, none adopted; all superseded by the R36-era baseline (5.6ish) unless noted.
- rolloutTicks 64 / 128 / 16; `lr` flat over 4x; `clip` .3; `lambda` .99; `valueCoef`; `latestBias`; `obsNorm` (measured twice - realm stays bias-free); `minibatches` 1; R29 difficulty curriculum; R31 (`poolSize` 24, `sigmaInit` .7).
- Batch/world count on that baseline: 1024 = 2.27, 2048 = 2.44, 512 stayed. Re-measured post-R36 as R40 (256 = 5.47, 128 = 4.74, 512 = 5.62).
