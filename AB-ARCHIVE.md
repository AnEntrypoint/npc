
## Moved out of AGENTS.md (2026-10-05 compaction) - pre-R36 sweeps on the OLD 2.38 baseline
All NEUTRAL, none adopted; all superseded by the R36-era baseline (5.6ish) unless noted.
- rolloutTicks 64 / 128 / 16; `lr` flat over 4x; `clip` .3; `lambda` .99; `valueCoef`; `latestBias`; `obsNorm` (measured twice - realm stays bias-free); `minibatches` 1; R29 difficulty curriculum; R31 (`poolSize` 24, `sigmaInit` .7).
- Batch/world count on that baseline: 1024 = 2.27, 2048 = 2.44, 512 stayed. Re-measured post-R36 as R40 (256 = 5.47, 128 = 4.74, 512 = 5.62).

## Blob PPO A/B detail (moved out of AGENTS.md 2026-10-05; blob-side tuning is CLOSED)
512 worlds, 300k ticks, 3 seeds; judge EVAL at ~250k ticks (evalBase flips sign, so ratio is meaningless), seed sd 0.01. Shipped 0.74.
- base 0.52; `rolloutTicks` 16 BETTER 0.56; `gamma` .99 0.48 and `rolloutTicks` 128 0.45 WORSE; `rolloutTicks` 16 + `gamma` .999 BEST 0.59 +- 0.01. `hidden` 192 does not compile (35984 B).
- Both are blob defaults via `RL_GAME_DEFAULTS` (src/rl.js), applied after URL params; confirmed shipped (0.58 vs 0.52).
- Entropy .01 NEUTRAL-to-worse (0.56/0.55 vs 0.58, 3 seeds at 250k), so blob stays pinned at .001 while the global default rose to .02.
- R26 `obsNorm` 1 BIG WIN: 0.73 +- 0.00 vs 0.58 +- 0.01 (3 seeds at 250k; 0.74 shipped) -> blob default (implies bias).
- R28: `bias` alone 0.58 (the win is the normaliser); `hidden` 128 NEUTRAL 0.73; entropy .01 WORSE 0.68. blob keeps `curriculum` 0.
- NOTE: the R30 entropy-anneal result (5.10 +- 0.06 vs entdef 4.97) is a REALM number, not blob; it was misfiled here. It is already stated correctly in AGENTS.md's realm section ("the shipped anneal .02 -> .002 gives 5.10").
