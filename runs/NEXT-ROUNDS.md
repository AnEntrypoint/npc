# Next three PPO rounds (realm v4) — proposal

Baseline to beat: `popabL-single` 2.37 ± 0.08 at the 360k horizon (2.43 / 2.41 at
385k), `horiz64` 2.37 ± 0.04, `v4-final` 2.38. Compare with
`node tools/report.mjs --at-tick=360000`. Queue job shape:
`{name, url, minutes, seeds:[..]}` (tools/queue.mjs expandJobs).

## Why this plateau is not a tuning plateau

Every long run is asymptotically flat: popabL-single s1 goes 2.43 (341k) -> 2.44
(385k), and `H2H[ratio]` is 1.00-1.05 against a snapshot ~2000 iterations old.
That is convergence, not a budget shortfall, so "train longer" is dead and every
remaining optimiser knob has a low prior — which is exactly what the last six A/Bs
found. The three picks below are the only levers left that are (a) untested,
(b) mechanistically tied to a measured failure, and (c) reachable by URL params.

Note what a URL param cannot reach: `rl_train_fx` is gated on `POPK > 1u`
(rlshader.js:1206), so reward-channel reweighting through `popWeights` is a no-op
at `policies=1`. The obs space also has real holes (no action-cooldown/readiness
signal, no target HP, no absolute position, no per-node regrow state) — those
need a src change and belong in a separate round.

## R1 — credit-assignment horizon (lambda is the one knob never swept)

`/dev/rllong.html?name=lam99&game=realm&worlds=512&ticks=400000&minutes=25&lambda=0.99`
`/dev/rllong.html?name=g999lam99&...&lambda=0.99&gamma=0.999`

gamma .997 x lambda .97 gives gamma*lambda = 0.967, a ~30-tick GAE horizon, and
gamma^3000 = 1.2e-4, so the -1024 fx death penalty is invisible at the start of a
3000-tick life — while 41-58% of lives end in starvation. gamma .997 was only ever
tested against .995/.99; lambda has never been moved at all.

3 seeds each, 400k ticks. 6 x ~17 min = ~1.7 GPU h.
Adopt if mean ratio >= 2.50 with no seed < 2.40 and eval life up.
Kill if mean <= 2.45, or if `vl` inflates and eval life drops (camping).
Objection: the shaping (need compass, town-approach, distance) exists precisely so
a 30-tick horizon suffices, and rolloutTicks 31 truncates the GAE carry anyway, so
raising lambda mostly amplifies bootstrap error.

## R2 — batch scale (worlds 1024 at matched ticks)

`...&worlds=1024&ticks=400000&minutes=40`
`...&worlds=1024&ticks=400000&minutes=40&lr=0.002`

At 512 worlds each update sees ~207k samples from ~417 training worlds; the run is
converged, so what is left is gradient noise plus overfitting to a finite set of
randomCfg worlds. The old realm's best numbers (4.1-5.0x) were at 1024 worlds, and
throughput rises with worlds, so 1024 costs only ~1.7x wall clock per seed.

3 seeds (lr 1e-3) + 2 seeds (lr 2e-3), 400k ticks. ~2.4 GPU h.
Adopt the arm whose mean ratio >= 2.50 with no seed < 2.40; if only the lr arm
wins, the finding is step size, not batch.
Kill if both arms sit inside 2.37 ± 0.10.
Objection: convergence implies a bias limit, and larger batch attacks variance;
r4-w1024 (2.15) was only measured at 26k ticks and is not evidence either way.

## R3 — exploration sharpness (sigmaInit)

`...&worlds=512&ticks=400000&minutes=25&sigmaInit=0.25`
`...&sigmaInit=0.15`

8 of 12 outputs are thresholded at 0.5 (realm.js:586-596), logStd never moves
(diag ent 0.75 vs init 0.726), and eval deploys mu — so the behaviour policy is
mu + 0.5*N(0,1) while the measured policy is mu. Sharp on-policy data should track
the evaluated policy better.

3 seeds (0.25) + 2 seeds (0.15), 400k ticks. ~1.4 GPU h.
Adopt if mean ratio >= 2.50 with no seed < 2.40.
Kill if <= 2.45.
Objection: entropy 0.01 drove sigma 0.5 -> the 1.0 cap and cost ~1% (2.41 vs
2.44), so sensitivity to sigma looks weak; and less exploration may cement the
camp-and-spam optimum.

Order: R1, R2, R3. Do not edit src/ while a round is queued.

### R3 landed 2026-10-03 — `sigmaInit` .25 is a KILL, not a tie

`node tools/verdict.mjs sig025 --at-tick=360000`: ratio 1.98 +- 0.23 (2.14 / 1.82) vs
baseline 2.38 +- 0.08, EVAL 3.31 +- 0.41 vs 3.90 +- 0.19, `bots` 1.67 vs 1.64. This one
loses on the raw column, not the denominator — the four drift arms held EVAL at 3.90 and
moved only `evalBase`; here `evalBase` is flat and EVAL drops 15%. R3's own objection
predicted weak sensitivity; it was right about direction and wrong about which way:
cutting exploration costs 16%, while 10x the entropy bonus cost nothing, so the ceiling is
not exploration-limited upward either way. It was the last unrun key in `RL_DEFAULTS`.
Not run: .75/1.0. Full row in runs/AB-ARCHIVE.md.

## R4 — optimiser and league knobs (queued 2026-10-03, `epochs` first)

`epochs=3`, `lr=2e-3`, `lr=5e-4`, `clip=0.3`, then `valueCoef=0.25`,
`leagueFraction=0.25`, `latestBias=0.8`. Same protocol (512 worlds, 400k ticks,
2 seeds, judged at 360k against popabL-single 2.37). Decision rule as above:
adopt at >= 2.50 with no seed < 2.40, kill at <= 2.45.

`epochs` is the only one of the seven with a mechanism not yet probed:
`encodeUpdates` does epochs x minibatches Adam steps per rollout and re-salts the
minibatch hash each epoch, so at epochs 1 (default) every sample is used once in
one minibatch; at 3 the same rollout is reused three times in three different
partitions. Cost is ~30% throughput (ep3-s1 measured 143k vs 213k world-ticks/s).

## R5 — observation space (the only lever left that is not a knob)

Four flat results in a row (epochs 3 worse, lr 2e-3 and 5e-4 neutral, plus the
earlier T/hidden/batch/entropy rounds) say this is a bias limit, and a bias limit
from partial observability is not reachable by tuning. What `observe()`
(src/games/realm.js:408-490) does NOT expose:

- **Distance magnitudes.** out[88]/[89] give the town *compass* and out[87] the
  in-town flag, but never `isqrt(town.d2)`; out[78]/[92] give the spring/berry
  compass *multiplied by urge/8192*, so they are 0 whenever urge is 0 and the
  magnitude is unrecoverable. The policy cannot tell "town is 200 away" from
  "town is 2000 away" except through the shaping reward.
- **Nearest-threat HP.** Only class sector distances (out 0..71) and boss
  distance (out[96]); nothing says whether the mob in front is nearly dead, so
  disengage/finish decisions are blind.
- **Action readiness.** 8 of 12 outputs are thresholded at 0.5 (realm.js:586-596)
  with no cooldown or "was the last action accepted" signal fed back.
- **Node regrow state.** `nodeTimer[j]` is consulted but never exposed, so a node
  that is 1 tick from ready looks identical to one that is 300 away.

Every slot 0..104 is already taken (out[85] is the great-node compass, not a
spare), so this raises nIn and invalidates every champion and every pruned genome
(`evalsuite` refuses a dim mismatch against the game): the teacher has to be
retrained and `tools/prune.mjs` rerun (~25 min for the six targets) afterwards.

Protocol: add the fields to `observe()` and to the WGSL `g_observe` mirror in the
same change — the JS-env lockstep test compares word for word, so a mismatch
fails there rather than silently. First arm adds only the two distance magnitudes
(cheapest, most defensible); a second adds threat HP and action readiness.
Retrain the baseline at the new nIn in the same round: the old 2.37 is not
comparable across an observation change. 2 seeds, 400k ticks, judged at 360k.

## Deferred until the queue drains — wire blob's per-game defaults

Blob's `rolloutTicks 16 + gamma 0.999` (0.59 vs 0.52, 3 seeds) is documented in
README/AGENTS.md but not enforced: `dev/rllong.html?game=blob` still runs the
realm defaults. Neither half transfers to realm (realm T16 neutral, realm
gamma .999 worse), so this must be per-game, not a global default.

Both files are off limits while a realm round is queued (later arms load the page
fresh and would run different code). When the queue is empty: add
`RL_GAME_DEFAULTS = { blob: { rolloutTicks: 16, gamma: 0.999 } }` next to
`RL_DEFAULTS` in src/rl.js and apply it in dev/rllong.html after the URL overlay
on line 28, skipping keys the URL already set:

```js
const fromUrl = new Set([...q.keys()]);
for (const [key, value] of Object.entries(RL_GAME_DEFAULTS[common.game] || {})) if (!fromUrl.has(key)) common[key] = value;
```

`!(key in common)` guards line 28, so the per-game map has to go on after it —
placed before, it would swallow the URL param.

## R5 observation round (2026-10-03) — decision tree

obs4 = realm nIn 109 with the four new inputs (105 townDist, 106 needDist,
107 threatHp, 108 ready), 512 worlds / 400k ticks, seeds 1-3, judged at the
360k horizon against `popabL-single` 2.37 +- 0.08 (kill <= 2.45, adopt >= 2.50,
`node tools/verdict.mjs obs4 --row`).

- obs4 >= 2.50 -> adopt nIn 109 as the realm default, then split (2 seeds each):
  `obs-need` (105 + 106, the two distances) vs `obs-threat` (107 + 108). If the
  split is inconclusive, keep all four — the cost is 512 B of workgroup and
  4 x 96 = 384 extra first-layer weights.
- 2.45 < obs4 < 2.50 -> NEUTRAL. Keep nIn 105, spend the GPU on `sigmaInit` .25
  instead.
- obs4 <= 2.45 -> revert realm to nIn 105 / nNodes 147 in `src/games/realm.js`
  and `realm.wgsl.js` (four edits: dims, inputNames, observe, WGSL set_obs) and
  do not retry observation growth: nIn 109 sits exactly on the evolution
  ceiling `(nNodes|1) + (nIn|1) <= 258`, so ANY further input needs a smaller
  evolution hidden or a bigger game workgroup budget.

## R6: channel-weighted training at policies=1 (written 2026-10-03, to run after blobdef)

Why: every realm arm that loses with the "survives instead of earns" signature
(`gamma` .999 eval life 7567 vs 2902; obs4 nIn 109 life 3420 vs 2902) holds eval
life UP and reward flat — the policy farms the dense survival channel while the
sparse progress/combat channels stay unlearned. The trainer already owns the knob
that fixes exactly that (a weight vector over the game's four reward channels);
it is wired for the population path only, and the population path is off by
default because K>1 loses with bots in training. So un-gate the knob for K=1.

The change (all guarded so blob is untouched):
- `src/rlshader.js:1206` is `if (POPK > 1u) { trainFx = rl_train_fx(t, u32(lane[b * LS + LS_POL])); }`; drop the gate. `rl_train_fx` is already generated for any game with `g_reward_channel` + `rewardChannels` (rlshader.js:583-588) and reads `opt[pol * OPTS + OPT_W + ch]`. When `hasChannels` is false it is `return f32(rewardBuf[a]);`, identical to `rl_reward_fx`, so blob cannot change.
- The weights are already correct at K=1: `resetOptimizer` writes `pop.weights[p]` into the opt block for every policy including p=0 (rlengine.js:212-216), `rlRoleDefaults(0, C)` is 1/1/1/1, and the sum of the four channels is the total reward — so the un-gated default is the current number and 21/21 should stay green with no JS mirror.
- `popWeights` already reaches the opt block at K=1: `rlPopConfig` parses it (rl.js:720) and rllong forwards any `RL_POP_DEFAULTS` key from the URL (rllong.html:28). Only the shader gate blocks it today.
- JS mirror: needed only for a test that asserts a weighted training reward at K=1. The existing "weighted rewards" population test (rlengine.js:1581) covers K>1; extend it rather than add a new one.
- No interaction with `rl_rescale` (it normalises by return std) or with channel caps (they clamp per channel before the weight).

Arms (realm, 512 worlds, 400k ticks, 2 seeds; judge with
`node tools/verdict.mjs chw-a --at-tick=360000` against popabL-single 2.37):
- `chw-a`  `popWeights=0.5,1.5,1.5,1.5`  (halve survival)
- `chw-b`  `popWeights=0.25,1.5,2,1.5`   (stronger, only if chw-a >= 2.45)
Decisions:
- chw-a >= 2.50 with no seed < 2.40 -> adopt as the realm default (set it in
  `rlRoleDefaults(0, C)`), then run `chw-b`.
- 2.45 < chw-a < 2.50 -> NEUTRAL, stop the round.
- chw-a <= 2.45 -> survival is not what to down-weight; stop the round and spend
  the GPU on `sigmaInit` .25 (the one queued knob left).

### R6 landed 2026-10-03 (before the arms ran)
- `src/rlshader.js:1205` is now `var trainFx = rl_train_fx(t, u32(lane[b * LS + LS_POL]));` — the
  `if (POPK > 1u)` gate is gone. For a game with no reward channels `rl_train_fx` is
  `return f32(rewardBuf[a]);`, identical to `rl_reward_fx`, so blob is bit-identical.
- `LS_POL` is 0 at K=1 (rlshader.js:996) and `resetOptimizer` writes `pop.weights[0]` into
  `opt[OPT_W + c]` for policy 0 too (rlengine.js:212-216), so the un-gated default (1/1/1/1)
  reproduces the old number exactly; the sum of the four channels is `env.reward`.
- `popWeights` added to `RL_TRACKED_KEYS` so `tempBackend` inherits it and the suite sees the
  weighted path; the lockstep and channel-caps tests now expect
  `rlWeightedReward(channels, pop.weights[0], capFx)` (rl.js:779, fround-ordered to match the
  GPU) — exact when every weight is 1, 1e-3 fx tolerance otherwise. Verified in Node: unit
  weights reproduce `env.reward` on 9600 realm samples with 0 mismatches, and 0.5/1.5/1.5/1.5
  changes |reward| 8487 -> 13430 over the same window.
- Stats, league score and the eval benchmark still use the UNWEIGHTED `rewardFx`
  (rlshader.js:1203/1213), so the A/B stays measured on the same yardstick as the baseline.

## R7 — the two opt-in flags never A/B'd on realm: `obsNorm` and `bias` (queued 2026-10-03)

`AGENTS.md` lists the PPO opt-ins as `recurrent`, `bias`, `obsNorm` (implies bias),
`learnLeak`, `leakTauMax`, `rewardClip`, `valueClip`, `channelCaps`, `curriculum*`.
Sweeping `runs/` for each name: every one of them appears only inside `tests=1`
jobs (`jobs-p17-tests.json`, `jobs-normfix*.json`) or in the *recurrent* round's
ablations (`learnLeak=0`, AB-ARCHIVE) — never as a 512-world training arm. So the
recurrence round tested `recurrent+bias+obsNorm` as a bundle and concluded
"recurrence is not a clear win", which says nothing about the two flags alone on
the feed-forward default path that actually ships.

Protocol: realm, 512 worlds, 400k ticks, 2 seeds each, judged at the 360k horizon
against `popabL-single` 2.37 (`node tools/verdict.mjs obsn --at-tick=360000`,
kill <= 2.45, adopt >= 2.50 with no seed < 2.40).

- `obsn`  `obsNorm=1`   — running mean/std over the 105 inputs, clamped to +-10,
  folded into W1/b1 on export; adds 2*nIn theta words and a variance/count block
  to opt (memory, not workgroup, so 512 worlds is unaffected). Mechanism: realm's
  observation scales are wildly heterogeneous (sector distances, wood/ore prices,
  needs, boss hp) and the first layer is shared across all of them.
- `bias1` `bias=1`      — the isolating arm: `obsNorm` turns bias on, so if
  `obsn` wins this says whether it was the normalization or just the intercepts.

Decisions:
- `obsn` >= 2.50 with no seed < 2.40 -> adopt `obsNorm` as the realm default and
  re-run `tools/prune.mjs` (the export folds inputNorm into W1/b1, so pruned
  genomes and every champion before it are stale).
- 2.45 < `obsn` < 2.50 -> NEUTRAL, keep it off (it costs theta words and a
  non-stationary preprocessing step for nothing).
- `obsn` <= 2.45 -> both flags stay off. Together with R1-R6 that closes the
  URL-param space on realm v4 completely: the remaining levers are structural
  (realm's reward composition or the algorithm), and the reward-composition
  analysis in the section below is what to spend the GPU on next.

`bias1` is read as a control, not as a candidate: if it lands inside noise while
`obsn` moves, the effect is normalization; if both move together, it is the
intercepts (and the cheaper `bias=1` is what to adopt).

## R8 — close the great-node camping exploit (measured 2026-10-03)

The single biggest thing the reward composition still gives away for free. Verified
by execution, not by reading (`node runs/great-camp-check.mjs`, JS env, difficulty
100, two learners teleported onto great node 204 with tool tier 2, needs topped up,
INTERACT held):

```
campers=2 gate>=2 | coop/tick 2.16 | channels 0.98/0.00/-0.03/2.16 | EVAL-units 3.10 | gprog 90
campers=2 gate>=3 | coop/tick 0.21 | channels 0.98/0.00/-0.03/0.21 | EVAL-units 1.15 | gprog 90
campers=3 gate>=2 | coop/tick 4.44 | channels 0.98/0.03/-0.02/4.44 | EVAL-units 5.43 | gprog 0
```

Two learners standing still earn **3.10 EVAL-units/tick — 79% of the trained
champion's 3.90 — for doing nothing**. It is also sustainable: the same run with
the need top-ups removed (only position, velocity and tool tier forced) survives
all 1200 ticks with no death at 2.98/tick, so the great node sits close enough to
water and food that a camper never has to leave. A passive, immortal policy is
worth 76% of the trained one.

The node can never complete:
`greatPhase` (realm.js:847) gates completion on
`harvesters >= 1 + trunc(2*difficulty/100)` = 3 at difficulty 100, while
`interactPhase` (realm.js:901) pays `GREAT_TOGETHER` = 2 fx/tick of COOP whenever
`gcount >= 2` and the node is not done. Progress caps at `GREAT_PROGRESS_CAP` 90 >=
`GREAT_PROGRESS_NEED` 60, so with exactly 2 harvesters it sits at 90 forever: an
unbounded, cooldown-free dense term paying 2x `ALIVE_REWARD`, booked into the one
channel that is supposed to mean something. Three campers complete it (gprog 0,
5.43) — the gate is what makes 2 the degenerate number.

Why this is the best remaining lever rather than another knob:
- It is learner-only. Bots hard-code `allowGreat = false` (realm.js:659,
  wgsl:954), so `evalBase` cannot drift — the confounder that has made four arms
  unreadable (`entropy` .01, both `lr` arms, `lambda` .99).
- It is nIn/nOut-neutral, so champions and pruned genomes still load.
- It matches the measured failure signature exactly: 41% of lives end in
  starvation and every losing arm holds eval life UP and earned reward flat.

The change (two lines, must land word for word in both files or the JS-env
lockstep test fails) — the gate is the COMPLETION gate, not the literal 3, so it
tracks curriculum difficulty and a group that can actually finish still gets paid:
- `src/games/realm.js:901`  `>= 2` -> `>= 1 + Math.trunc((2 * this.difficulty) / 100)`
- `src/games/realm.wgsl.js:1191`  `>= 2` -> `>= 1 + (2 * world_difficulty()) / 100`
(`world_difficulty()` is already called by `r_great` at wgsl:1152; WGSL i32 `/`
truncates, so the two agree.)

Protocol (`runs/jobs-r13.json`, queued only once the queue is empty so no later
arm loads different code): `camp-tests-rl` (21/21) + `camp-tests-gpu`, then
`campgate3` = defaults at 512 worlds / 400k ticks / 2 seeds, judged at the 360k
horizon. Because bots are unaffected, judge on EVAL as well as ratio:
- EVAL >= 4.10 (vs 3.90) with no seed < 3.70 -> adopt. The policy was spending
  capacity on a free lunch and now has to earn.
- 3.70 <= EVAL <= 4.10 -> NEUTRAL: learners were not camping, keep the gate as is.
- EVAL < 3.70 -> the camp was load-bearing for the metric; revert and record that
  the 2.4x is partly a camping artefact rather than skill.

Risk to state honestly: the camp pays ~79% of the champion rate, so if learners
ARE camping today, closing it first LOWERS the headline ratio even though the
resulting policy is better. That is the reason to judge on EVAL and life, and to
read `EG[greatHarvests]` — it is 3-33 per run today against ~98k harvests, which
is what "completions never happen" looks like.

Not proposed: touching `ALIVE_REWARD`. R6 already ran that direction with
`popWeights` (2x faster, peaked 2.55 at 190k, decayed to a tie) — it moves speed,
not the ceiling.

### Small cleanup noticed on 2026-10-03 (do it when the queue drains)
- `dev/ab.html`'s `tests=1` mode finishes by throwing, so every GPU test job's log ends with
  `ERR Uncaught Error: tests finished` (r105-tests-gpu and chw-tests-gpu both show it after
  `tests done 8/8`). The queue still marks the job `done`, but the line reads as a failure and
  would trip anything stricter later. Log and return instead of throwing; `tests done N/N` is
  already the real signal.

### R6 seed-1 read (2026-10-03, s2 pending)
chw-a-s1 (`popWeights=0.5,1.5,1.5,1.5`) at the 360k horizon: ratio 2.39, EVAL 4.26,
evalBase 1.78, train/eval life 1540/2528, entropy 0.679 — against popabL-single 2.37 /
EVAL 3.90 / evalBase 1.64 / life 1889/2902 / entropy 0.76.

Two things are real and neither is the hoped-for one:
1. It learns about twice as fast (2.20 at 52k where the baseline is ~1.3) and peaks at
   2.55 around 190k — ABOVE the baseline's final 2.43 — then decays to ~2.3 by 240-300k and
   recovers to 2.39. The baseline climbs monotonically through it. Judged at 190k this arm
   would have been adopted; at the agreed 360k horizon it is a tie.
2. EVAL rose 9% but evalBase rose 8.5% in the same worlds, so the ratio is flat. The learners
   made the world richer for the bots too rather than beating it by more. Same drift signature
   as `entropy` .01 / both `lr` arms / `lambda` .99, except those held EVAL constant — this one
   does not, so it is a richer world, not just a drifting denominator.

Diagnostics say the knob does what it says: cheaper death -> train life 1540 vs 1889, entropy
0.679 vs 0.76 (sharper policy), higher reward/tick. Shorter lives at a higher rate is exactly
"down-weight survival".

Dead end to record so nobody re-runs it: a uniform scale on all four weights is a no-op after
`rl_rescale` (it normalises by return std), so `1/3/3/3` is the SAME experiment as
`0.5/1.5/1.5/1.5`. There is no way to separate "survival is cheaper" from "progress pays more"
— they are one tilt, and the tilt is what was measured.

If s2 confirms <= 2.45 the round stops (decision tree) and `sigmaInit` .25 runs next. Note that
sigmaInit is the LAST unrun knob in RL_DEFAULTS: after it, every default has been measured flat
or worse and the remaining levers are structural (realm's reward composition, or the algorithm),
not knob-tuning. An idea worth banking but not building now: chw-a's early speed is real, so a
weight ANNEALING schedule (start tilted, decay to 1/1/1/1) might bank the 190k peak without the
decay — that is a new mechanism, not a param, and needs its own JS mirror and test.

### R7 landed 2026-10-03 — `obsNorm` is NEUTRAL, `bias` dropped unrun
2 seeds, 360k horizon: obsNorm 2.35 +- 0.04 vs popabL-single 2.38, EVAL 4.05 vs 3.90, evalBase
1.72 vs 1.64, life 3301 vs 2902, entropy 1.13 vs 0.77. Mechanical verdict was KILL (<= 2.45) but
the raw column moved UP and `evalBase` drifted again, so it is the sixth drift arm: NEUTRAL.
Not adopted (extra params, extra failure mode, no reward). `bias` alone was dropped unrun — a
flat hypothesis arm does not justify spending a GPU-hour on its control. Every PPO default and
opt-in flag has now been measured; the remaining levers are structural. R8 (great-node gate) is
running: src patched, `camp-tests-rl`/`camp-tests-gpu` gate it, `campgate3` is the verdict.

## R9 — animal kills are a second learner-only free lunch (measured 2026-10-03)

Found by the reward audit, then reproduced here: `node runs/animal-audit2.mjs` drives a
hand-scripted policy that does nothing but walk to the nearest animal and kill it (HP left
natural, so mob damage and deaths are counted), 2400 ticks x 3 seeds:

```
seed 4242  HUNT  kills 107 (44.6/1k) deaths 4  dmg 0.39/tick  S/P/C/O 1.379/-0.470/1.218/0.000  TOTAL 2.126
seed 777   HUNT  kills 113 (47.1/1k) deaths 3  dmg 0.32/tick  S/P/C/O 1.932/-0.321/1.560/0.015  TOTAL 3.186
seed 31337 HUNT  kills 106 (44.2/1k) deaths 3  dmg 0.27/tick  S/P/C/O 1.796/-0.332/1.525/0.000  TOTAL 2.990
           idle  kills   0           deaths 0                 S/P/C/O 0.977/0/0/0.02            TOTAL 0.996
```

Idle is 1.00 and the champion is 3.90, so one repeated action is worth **53-82% of the trained
policy**. Why it is free: realm.js:1080-1085 pays +1 ration (cap 4), +1 gold, `gainXp(XP3,4)`
and a flat `add(SURVIVAL, 32)` — ~48 fx per kill — with no diminishing, no consumable and no
risk (animals never attack: 0 `ACT.ATTACK` from e in 56..63 over 600 ticks). The only ceiling is
respawn (8 every 150 ticks, ~45 kills/1k ticks at the measured rate). Bots never target class 6,
so like the great-node gate `evalBase` cannot drift.

The champion does not seem to be farming it yet (EG `mobKills` counts animals too, and the rate
implied is an order of magnitude below 44/1k), which is exactly why this is cheap to close now:
it is an unclaimed attractor, not current income.

The change (`diminish` already has spare bits — `F.ACH` uses 5-7 pvp, 8-11 boss, 12-15 great, so
16-19 are free; no layout or `worldWords` change, so champions and the lockstep test are safe):
- add `COUNT_ANIMAL = 16` next to `COUNT_BOSS`/`COUNT_GREAT` (realm.js:95-96) and
  `RW_COUNT_ANIMAL` (realm.wgsl.js)
- realm.js:1084  `this.add(e, CH.SURVIVAL, 32)` -> `this.diminish(e, COUNT_ANIMAL, 32, CH.SURVIVAL)`
- the same call in realm.wgsl.js (`r_diminish(..., RW_CH_SURVIVAL)`)
Effect: 32/16/8 then 4 fx per kill for the rest of the life — hunting for food still pays, a
45-kills/1k farm drops to ~0.07 fx/tick.

Protocol (`runs/jobs-r14.json`): `anim-tests-rl` + `anim-tests-gpu` gate it, then `animaldim`
2 seeds at the 360k horizon. Judge on raw EVAL (bots immune): adopt >= 3.90, keep 3.70-3.90,
revert < 3.70. Queue it only after campgate3's verdict lands (src must not change mid-round).

### Audit leftovers worth remembering
- `runs/shaping-audit.mjs`: town/spring/berry distance shaping pays **0** to an oscillate-in-view
  policy (`gain` needs `prev > 0 && cur > 0`, and leaving the radius zeroes both) — not a leak.
- town-approach shuttle (leg 25) pays up to 1.30 fx/tick, but bots run the same loop, so it is
  inside `evalBase`; a `prevTown` fix is cosmetic for the ratio. NEEDS-MORE-WORK, low priority.
- Bounded by construction, no patch needed: `gainXp` (cap 1296x4 per field per life), mob/boss/
  PvP kills (risk- or diminish-gated), sale and craft (consumable-gated), milestones (once/life).
- Harness bug, not a reward bug: a dead learner is never respawned by the JS env (`applyPhase`
  sets `dead[e]=1` but leaves HP <= 0 with DEAD 0), so any script driving `createEnv` directly
  banks -1024 every tick unless it calls `env.spawn` itself.

### R9 patch revised 2026-10-03 — no free `F.ACH` bits; the ration gate instead
`diminish` was the wrong tool: `F.ACH` is fully packed (milestones 0-4, pvp 5-7, boss 8-11,
great 12-15, and DEALT_SHIFT 16 with DEALT_MASK 32767 = bits 16-30 for boss damage dealt), so
there is no counter slot left for a prey counter.

And the prey reward is three terms, not one — all three fire per kill:
- `add(SURVIVAL, 32)`            2.14 fx/tick at 44.6 kills/1k   (realm.js:1084, wgsl:1451)
- `gainXp(XP3, 4, SURVIVAL)`     ~0.7 fx/tick until the 5184 fx/life cap (realm.js:1083, wgsl:1450)
- attack xp `gainXp(XP0+style, max(1, atkDmg>>1), COMBAT)` 1.22 fx/tick measured
  (realm.js:1045-1049, wgsl:1409-1413) — paid for ANY attack, so it pays for stabbing prey

Revised patch, both files, two edits each:
1. Prey pays only when it actually feeds you — wrap the animal-kill block in the ration check
   (`RAT` caps at 4 and is consumed by auto-eat at FOOD < 40, realm.js:1124 / wgsl:1493):
   JS realm.js:1080-1085  `if (f[b + F.RAT] < 4) { <ration, gold, xp3, add 32> }`
   WGSL wgsl:1447-1452    `if (r_f[b + RX_RAT] < 4) { ... }`
   Paying 4 kills per eating cycle is ~0.06-0.13 fx/tick: hunting stays a survival activity
   (the eaten ration still pays `restore`) and stops being an income stream.
2. Prey does not train you — skip the per-attack combat xp when the target is an animal:
   JS realm.js:1049  `if (this.atkTgt[e] < R_ANIMAL0) this.gainXp(...)`   (-1 < 56 still pays)
   WGSL wgsl:1413    `let t = r_atk_t(e); if (t < 0 || u32(t) < RW_ANIMAL0) { r_gain_xp(...) }`
   Mob/boss/PvP xp is untouched, and bots never target class 6 so `evalBase` cannot drift.

Both halves are needed: measured HUNT extra over idle is 1.13 net (gross ~2.8-4.0 minus 1.7 of
death penalties), and it is split roughly 2.14 / 0.7 / 1.22 across the three terms.

### R9 measurement caveat (measured, don't trust the harness totals)
Splitting the two halves on patched copies of realm.js (`runs/realm-r9a.mjs`, `realm-r9b.mjs`)
shows the `animal-audit2` hunter is too chaotic to attribute a ~1 fx/tick effect: it teleports
next to the nearest animal every tick with FOOD/WATER forced to >= 60, so it drifts into mobs and
bosses, and the COMBAT column swings -1.7 / +1.2 / +8.1 across variants with deaths 3-8. The
net totals (2.13-3.19 unpatched vs 4.08-6.35 patched) are therefore NOT evidence that the patch
helps; they are noise. Do not cite them.

What is solid is the analytic rate, because it counts only the prey terms:
- kill income = 48 fx/kill (32 flat + `gainXp(XP3,4)` = 16) x ~44.6 kills/1k ticks = **2.1 fx/tick**
- attack income = `max(1, atkDmg>>1)` x 4 fx per attack tick, ~1 fx/tick sustained
- total ~3.1 fx/tick gross, ~1.1 net above idle once the death penalties of a real life are paid
Against 3.99 fx/tick for the champion and 1.00 for idle, prey alone is ~53% of the champion.

Post-patch expectation, from the code and not from the harness: rations cap at 4 and are consumed
only by auto-eat at FOOD < 40 (realm.js:1124), and food drains ~1 point per 32 ticks, so a full
eating cycle is ~1500-2000 ticks — payable kills fall to ~4 per cycle = 0.13 fx/tick, a ~94% cut,
and prey attack xp goes to 0. Hunting stays worth doing (the eaten ration still pays `restore`).
Judged on the GPU, not here: if `animaldim` EVAL drops below 3.70 the whole thing reverts.

### R9 verdict: NO (measured 2026-10-03) — prey is 14% of the champion, not 53-82%
Tagged measurement (`runs/prey-income.mjs`, 6 seeds, walking hunter, honest HP and death penalties): prey income **0.55 +- 0.20 units/tick** (kill 0.28 +- 0.10, attack xp 0.27 +- 0.10) against idle 1.00 and champion 3.99 — ~14%, not the 53-82% the teleport harness implied. The 8x gap is the harness: `animal-audit2.mjs` teleports onto the target and reaches 44.6 kills/1k; a body that walks gets **5.8 kills/1k** (64% of ticks travelling, 9% of ticks inside melee, travel-limited not respawn-limited). Payout per kill is exact at 96 fx (48 kill bucket + 4 x 12 attack fx).
Patch test: ration gate + prey attack-xp skip cuts the total 60% (0.55 -> 0.22) — attack xp 100% (0.27 -> 0.00), ration gate only 21% (0.28 -> 0.22), because at 6 kills/1k the 4-per-eating-cycle budget plus a free reset on every death (RAT = 0 after respawn, 1.5 deaths/1k) still pays most kills; the gate was sized for 45 kills/1k. 60% < the 80% bar and the prize is 8% of the champion, so **R9 is not run** (jobs-r14 stays unqueued).
Lesson: a scripted-policy audit that teleports or holds needs constant overestimates a farm's rate by up to 8x. Size every gate from a walking-policy measurement and state the prize as a fraction of the champion (3.99 units/tick) before spending a GPU round.

## Champion income decomposition (measured 2026-10-03, runs/champ-decomp.mjs)
Replay of `runs/campgate3-s1-base-best` in the JS env (4 worlds x 3000 ticks, 191,959 alive learner ticks), every `env.add` bucketed by call site. TOTAL **4.33 units/tick** (GPU eval of the same champion: 4.08) = survival 1.23 / progress 1.05 / combat 1.08 / cooperation 0.97.
Top sources (units/tick): alive tick 0.80 | **attack xp 0.60** (realm.js:1049, `xp = max(1, atkDmg >> 1)` every tick the action is ATTACK — the game pays for swinging, not for killing) | boss group `diminish` 0.49 | interact progress (craft/sale) 0.47 | mob kill xp 0.32 | coop milestone 0.27 | need restore 0.22 | town/need shaping 0.18 | boss top-hit 0.15 | mob gold 0.12 | harvest xp 0.21 across three resource types. Costs: damage taken -0.40, death -0.21.
Consequences:
- R9 is confirmed dead from the other side: the prey lines (realm.js:1083-1084) do not clear the 0.02 units/tick cutoff, so the champion earns ~nothing from animals.
- No single source exceeds 0.8 and the four channels sit within 1.23/0.97 of each other. The champion is diversified: there is no second farm to close, so further rounds must raise the ceiling rather than remove income.

## R10 candidate (NOT yet measured): the flagship cooperative mechanic is unused
Great harvests are ~0 (39 in a 400k-tick eval, 0.004 per life) even though cooperation already pays 0.97 units/tick, almost all of it boss-group `diminish`. A great node needs 3 harvesters with tool >= 2 at difficulty 100. Question to answer before spending a GPU round: do learners ever co-locate at a great node (gcount >= 2) and fail on the tool gate, or do they never co-locate at all? Co-location argues for approach shaping; no co-location means the mechanic is invisible and shaping cannot work.

### R8 verdict (2026-10-03): KEPT, despite the script saying KILL
campgate3 (2 seeds, 360k horizon) vs popabL-single: ratio 2.34 +- 0.06 vs 2.38 +- 0.08, EVAL 4.02 vs 3.90, `evalBase` 1.72 vs 1.64, life 3323 vs 2902, h2h 0.98, seeds 2.30/2.38. `tools/verdict.mjs` prints KILL because it applies the knob rule (adopt >= 2.50, kill <= 2.45), but the whole delta is `evalBase` drift — the seventh arm with that signature — and raw EVAL is HIGHER. It is a fix, not a knob: it removes a passive two-learner camp worth 3.10 EVAL-units (79% of the champion) at no measured cost.

## R10 — great-node tool gate 2 -> 1 (designed 2026-10-03, queued as `greattool`)
Why: the champion decomposition plus `runs/great-probe.mjs` show the flagship cooperative mechanic is unreachable, not invisible. Learners put `tgtNode` on a great node and press INTERACT **13.27% of learner-ticks**, but tool >= 2 is held only 17.64% of learner-ticks and `gcount >= 3` (what difficulty 100 demands) occurs in ~0% of node-ticks: 16 great completions per 4 worlds x 3000 ticks.
Patch: `GREAT_TOOL` 2 -> 1 (src/games/realm.js) and `RW_GREAT_TOOL` (src/games/realm.wgsl.js), with the town `banking` rule pinned to a new `BANK_TOOL = 2` / `RW_BANK_TOOL` so the wood/ore hoarding that funds tool crafting does not move. The coupled variant (banking also at 1) was measured and REJECTED: tool >= 2 collapsed 17.6% -> 2.9% because learners sell instead of saving for tools.
Measured effect on the UNTRAINED champion (same genome, 4 worlds x 3000 ticks): completions 16 -> 57 (3.6x), `gcount >= 2` 0.034% -> 0.319%, `gcount >= 3` ~0% -> 0.029%, tool >= 2 unchanged at 18.03%. The applied patch reproduces the isolated copy exactly (build-hash 034d2f5a25979f51).
Exploit check: R8 still demands 3 simultaneous tool-gated harvesters for both `GREAT_TOGETHER` and completion, so solo and duo camping pay nothing at tool 1 either; `diminish` halves the 1024 fx share per repeat to a 1/8 floor and `GREAT_TIMER` 1500 caps each node's rate.
Judge: 2 seeds, 360k horizon, vs campgate3 (2.34). Adopt if ratio >= 2.45 with the greatHarvests stat up; keep as a fix if EVAL >= 3.70 and the cooperation channel rises; revert if EVAL < 3.70.

## Blob `rolloutTicks` round (2026-10-03) — per-seed numbers at the ~250k judging horizon
Baseline `blobdef` (rt16 + `gamma` .999 through `RL_GAME_DEFAULTS`, 3 seeds): EVAL 0.58 / 0.58 / 0.57 (mean 0.577 +- 0.006); `blobpp-t16g999-s1` 0.59. Adoption bar: >= 0.62 (baseline + 3 sd).
- `blobt8-s1` (rolloutTicks 8): EVAL **0.59** at tick 243,232, life 2250, entropy 0.549 — +0.013 over the baseline mean, inside seed noise, so NEUTRAL on this seed. Verdict waits for s2/s3.
- `blobt24` (rolloutTicks 24, 2 seeds) is queued behind it: 8-vs-24 brackets whether 16 is a real optimum or a plateau.
- `blobt8-s2`: EVAL **0.58** at tick 251,768 (life 2241). 2-seed mean 0.00059 +- 0.00001 vs blobdef 0.00058 +- 0.00001: +1.4%, inside noise, below the 0.62 bar. **`rolloutTicks` 8 is NEUTRAL** — blob's 16 stays unless `blobt24` (2 seeds) wins, which would mean 16 is not the optimum either way.

## R11 — entropy annealing (added 2026-10-03, queued as `entanneal` + `enthi`)
Entropy regularization already exists (`U.entropy`, default 0.001) and is already swept: 0.01 NEUTRAL (2.38, raw EVAL 4.25 vs 3.90 — the largest raw gain of any single knob, but `evalBase` rose with it) and 0.0003 MUCH WORSE (1.36, entropy collapses 1.42 -> 0.16). What was never tested is a SCHEDULE, and the 0.01 result is the reason to try one: it buys exploration early and the flat-0.01 tail is what dilutes it.
Implementation (opt-in, bit-inert by default): `entropyEnd` in `RL_DEFAULTS` (default 0.001 == `entropy`), `RlEngine.entropyCoef()` mirroring the existing `learningRate()`/`lrDecayIterations` schedule, `writeUniforms` uses it, and `entropyEnd` was added to `RL_TRACKED_KEYS`. rllong.html line 28 accepts any `RL_DEFAULTS` key as a URL param, so `?entropy=0.01&entropyEnd=0.001` needs no plumbing. Guarded for old checkpoints that lack `entropyEnd`.
Caveat: the JS reference path in rl.js still uses `hp.entropy` as a constant, so a JS-vs-GPU parity test run WITH a schedule would disagree — the 21-test suite runs on defaults, where the schedule is inert.
Arms (realm, 512 worlds, 400k ticks, 2 seeds each, judged at 360k vs `greattool` since both run the R10 code): `entanneal` = 0.01 -> 0.001 over the first 2000 iterations (~62k ticks, 15% of the run); `enthi` = flat 0.01 as the control, so a win is attributable to the decay and not to more entropy. Adopt if ratio >= 2.50 with no seed < 2.40.

## P20 — perf: host-side skipping and per-learner layout (2026-10-03)
Not an accuracy round: every item here is plumbing or index permutation, so training numerics are bit-identical and the arms above stay comparable.

Measurements that started it (`prof-kern`, realm 512, `profile=1`, bumped ahead of the queue):
per-kernel timestamps from `RlBackend.profileKernels()`/`profileIteration()`, to find what the 232 ms/iteration is actually made of. The sim share implied by the evolution number (sim_step 512 worlds K=16 = 8.18 ms -> ~16 ms for 31 ticks) leaves ~200 ms unaccounted for, so the host and the RL kernels are the suspects, not the game sim.

Host side (analysis + patch landed, gated behind `hostPipeline: 0` in `RL_DEFAULTS`, so the running rounds run the old path):
- `iterate()` awaited `queue.onSubmittedWorkDone()` every iteration, so the next iteration's encode never overlapped the GPU. src/engine.js:369-372 already has the fix (store the completion, await the PREVIOUS one); rlengine.js now does the same under the flag.
- `pushErrorScope`/`popErrorScope` ran every iteration; Dawn resolves a scope by draining the device. Gated to every 16th pass (`RL_ERROR_SCOPE_EVERY`).
- Rollout and update were two submits; under the flag they are one encoder, one submit (WebGPU guarantees the ordering either way).
- `readWords` created and destroyed a staging buffer per call; now cached per byte length on `this`. Not gated: pure churn removal.
Not done: `RL_STATS_FLUSH_ITERATIONS` 8 -> 64. It looks free but the flush feeds `rl_rescale` and the curriculum, so changing the cadence changes training, not just timing.

Kernel side (the premise I started with was wrong): the genome edge arrays are ALREADY edge-major/brain-minor (`edgePk[e*BR+b]`, packed that way at src/engine.js:543-544), so consecutive lanes read consecutive `b` — coalesced. There is nothing to CSR-ify. The real waste is the per-learner state: `lane` (LS=239) gives 4-8 B per 32 B sector across 32 learners, and `rRec`/`rObs` scatter the same way. Fix = AoS -> SoA reindex (`field*trainLearners*K + b*K + i`) so the 32 learners land in one 128 B window; it is a pure index permutation, zero extra workgroup bytes, and every reduction is over `j`/`c`/`i` and never over `b`, so it is bit-identical. Being implemented behind `soa: 0`.
Rejected for now: CSC/destination-sorted edges (needs a new parity test; evolution path only, and `structural`'s swap-remove breaks the grouping), row-cooperative rollout matvec (no workgroup headroom: 1292 B free), and splitting one brain across 4 lanes to fill the 16/64 idle lanes (changes `acc` summation order).

### Blob `rolloutTicks` final numbers (3 seeds for the baseline and for 8, s2 of 24 pending)
At the ~250k horizon: `blobdef` (rt16 + `gamma` .999) 0.58 / 0.58 / 0.57 = 0.577 +- 0.006; `blobt8` 0.59 / 0.58 / **0.55** = 0.573 +- 0.021 (the third seed drops it — the earlier 2-seed 0.585 did not survive); `blobt24-s1` 0.56. Neither neighbour beats 16 and 8's spread is 4x the baseline's, so 16 stays the blob default. `rolloutTicks` 8 is NEUTRAL-to-worse, not an improvement: my earlier note recorded only 2 seeds.

### P20 measurement 1 — where an iteration actually goes (realm, 512 worlds, `prof-kern`, `profile=1`)
`iterationMs=62.9`; GPU timestampWrites: **rl_rollout 42.68**, rl_grad_blk **16.72**, rl_advstats 1.21, rl_select 0.32, rl_gae 0.19, rl_reduce_grad 0.17, rl_adam 0.16, rl_rescale 0.00, total 61.47.
Two corrections to the standing notes: (a) the 232 ms/iteration figure in AGENTS.md is a contention artifact — the real number is 63 ms, so its "ff 232 ms / recurrent 438 ms" pair is ~3.7x too high; (b) therefore the host is only ~1.4 ms of an iteration, which caps the whole host-side patch at ~2%. `profileKernels()` disagrees with the timestamps on the gradient (4.65 vs 16.72 ms) — trust the timestamps, they are per-pass GPU time.
Consequence: the target is `rl_rollout`, and its arithmetic is not the problem. The forward is ~1 ms of FLOPs (508k learner-ticks x ~21 kFLOP = 11 GFLOP), so ~27 ms of the 42.7 is traffic: `lane` alone is 239 words x 508k learner-ticks = ~485 MB of stores per iteration, stored learner-major so the 32 learners are 956 B apart and every store under-fills a sector. That is why the SoA reindex (and any dead field in `lane`) is the lever, not the sim and not the FLOPs.

### P20 measurement 2 — two ideas killed by measurement
**Active-set compaction (the literal nvGraph analogy) is NOT worth it.** Measured over 16 worlds x 20k ticks (runs/active-frac.mjs): only **3.65%** of entity-slots are inactive per tick (players 0.00% — respawn is same-tick; mobs 11.57% at a 200-tick respawn; bosses 0.47%; animals 0.06%), and the per-entity kernels already gate on `r_alive` (realm.js:363-365, WGSL r_alive at realm.wgsl.js:417) both in the phase entries (JS :1164/:1170/:1174, WGSL g_step :1539/:1545/:1552/:1561) and inside every interaction loop. Only 2.11% of phase entries run with a dead entity. So compaction would buy ~1-2% of sim time at the cost of a per-tick append scan, lane remapping and a break of the JS/WGSL lockstep contract. Even doubling mob kills gets to ~7%. **Do not implement.**
**The edge arrays need no CSR work.** `edgePk[e*BR+b]` is already edge-major/brain-minor (src/shader.js:390, packed at src/engine.js:543-544), so consecutive lanes already read consecutive `b`. The "make it nvGraph-shaped" instinct was wrong here; measuring first saved a large no-op refactor.

### P20 hypothesis — the rollout is occupancy/latency-bound, not instruction-bound
Budget check: the forward is ~5040 MAC per lane per tick x ~3 instructions = ~15k warp-instructions per workgroup-tick, so an iteration is ~480 M warp-instructions; at 30 SMs x 4 schedulers x 1.7 GHz the device can issue 204 G/s, i.e. the arithmetic needs ~2.4 ms of the measured 42.7 ms. We are at ~5% of issue capacity, so the rollout is stalled on memory latency, not on work.
The reason is probably occupancy: the module declares 31,476 B of workgroup storage (rolloutBytes, rlshader.js:55 — gradBytes is only 23,044 so the rollout sets the size) against a 32,768 B grant, and an SM has ~96-128 KB of shared memory, so only ~3-4 workgroups (6-8 warps) are resident per SM. **13,440 B of that is `obsBuf`, the per-learner observation array** (rlshader.js:775, `array<atomic<u32>, learners*obsStride>` = 32x105), reached only through `set_obs`/`obs_get` (rlshader.js:834-835). Moving it to a global `array<atomic<u32>>` (512x32x105 = 6.9 MB) would drop the workgroup to ~18 KB and roughly double resident warps. Cost: the compass's `atomicMin` (P19 order-independent scans) becomes a global L2 atomic instead of a shared one.
This is a hypothesis, not a measurement — it is worth one gated experiment (`wgObs`), and it is the only idea left with a plausible ~1.5x on the dominant kernel.

### P20 — the rollout's real cost: a function-local array indexed by the loop variable
The occupancy hypothesis above is probably NOT the main lever. The bigger one is in the forward itself: `var xr: array<f32, NIN>` (rlshader.js, rl_rollout + rl_value_of_shared_obs + the bootstrap value) is a **function-scope array indexed by a dynamic loop variable**, which on NVIDIA lowers to *local memory* — private per-thread storage in global RAM. Every one of the 105 x HID/QL = 5040 MACs per lane-tick then pays a local load, and a warp's 32 threads hit 32 different private addresses (one 420 B stack frame each), so each load is ~20-32 uncoalesced sectors of L1 traffic.
Scale: 512 worlds x 64 lanes x 5040 x 31 ticks = **5.1 G thread loads per iteration**, ~159 M warp-instructions, about half of everything the rollout issues. It is the only place in the kernel where per-MAC memory traffic exists — the grad kernel was already written the other way (`xs[e*XS + j]` is a workgroup array), which is why rl_grad_blk is only 16.7 ms.
Fix (`fwdDirect`, opt-in, default 0, src/rl.js + src/rlshader.js + RL_TRACKED_KEYS): when obs normalisation is off, `rl_norm` is the identity, so the input *is* the observation already sitting in workgroup `obsBuf`; read it inline (`obs_get(ln, j)`) instead of staging it into `xr`. Bit-identical (same values, same ascending-j accumulation order, and nothing writes obs between the g_observe barrier and the forward). Shared load instead of local: 16 unique addresses (lane = t/QL, stride OSTR=105 -> banks 9*ln mod 32, all distinct) so no conflicts.
Also converts the two *writes* that used to come from the array (rObs per tick, lane XLAST at the last tick).
Not converted: `p2`/`pre2` (nOut = 12, ~576 accesses per lane-tick, 11% of xr's traffic) and `acc: array<f32, ACCN>` in rl_grad_blk — do those only if the first win is confirmed.
Ablation hazard: `noPolicyForward` in dev/rllong.html matched the old source string; it now strips both forms, so an ablation job still works at either setting.

### P20 — lane coalescing: the SoA permutation currently shipped is the wrong axis
`rlLanePlanes`/`rlLanePermute` (src/rlshader.js) lay each plane out as `off*blocks + b*len + i` — planes contiguous, **learner contiguous inside a plane**. Call sites index `lane[LA_F + lb*LM_F + i]`, so that is consistent, but it does not buy coalescing: the hot pattern is *one field across many learners* (`lane[hnew + i] = hn`, `let hp = lane[hold + i]`, the XLAST copy into rObs). A warp is 32 threads = 16 learners x QL=2 subs, so at a fixed i it currently touches 16 addresses spaced `len` apart (768 B for the H0 plane, 1340 B in AoS) — 16 chunks of 8 B, i.e. ~16 x 32 B sectors for 128 B of useful data, ~25% efficiency either way.
What actually coalesces is **field-major, learner-minor**: `off*blocks + i*blocks + b`. Then a warp at fixed i writes two contiguous 64 B runs (the two subs sit in adjacent i rows) = 128 B useful in 128 B touched, 100% efficiency: a 4x cut in lane store traffic. Cost: per-learner sequential scans (`rl_zero_hidden`, respawn) become strided instead of contiguous — those are rare, the per-tick writes are not.
Implementation note (why it is not a one-line change): AoS needs `b*laneStride + i` and field-major SoA needs `i*blocks + b`, so the call-site expression shape differs between modes. It has to go through an emitter like the existing `rlRix` for rRec — `lix(plane, field, learner)` emitting `lb * LM_F + i` (AoS) or `i * LM_F + lb` (SoA) — not through a shared literal, or soa=0 codegen changes.
Size of the prize: lane stores are ~49 M x 4 B per iteration; at 25% efficiency that is ~1.3 GB of write traffic per 42.7 ms rollout (~30 GB/s, ~9% of the laptop's bandwidth), so expect single-digit percent, not a multiple. Do this only after `fwdDirect` is measured.

## P20 — compute reduction: what was measured (2026-10-03)

Starting point (quiet machine, realm 512 worlds, ff, hidden 96, T=31): iteration 62.9 ms =
rl_rollout 42.68, rl_grad_blk 16.72, rl_advstats 1.21, gae 0.19, select 0.32, reduce 0.17, adam 0.16,
host ~1.4 ms. The rollout is the target: 68% of the iteration.

### Killed by arithmetic before any code
- Occupancy: 31476 B of workgroup storage per CTA, ~100 KB shared per SM => 3 CTAs x 2 warps =
  6 warps per SM out of 64 slots (~9%). Cutting to <=16384 B would need removing obsBuf (13440 B)
  almost entirely, and the realm chunk atomics on obsBuf at ~14 sites (compass scans), so it cannot
  move to global. Not reachable without redesigning the game chunk.
- Active-set compaction: only 3.65% of lanes are inactive in a rollout, so any skip-list buys <4%.
- Edge re-layout (edge-major/brain-minor): already the case.

### Built and measured: both NEUTRAL (0%)
- `fwdDirect` (opt-in, default off): the forward staged its 105 inputs in `var xr: array<f32, NIN>`
  indexed by the loop variable, which lowers to *local memory* (private per-thread storage in
  global RAM, 32 uncoalesced sectors per warp access). It now reads `obs_get(ln, j)` inline.
  Legal only with `obsNorm` off, because `rl_norm` is the identity then, so the staged copy *is* the
  observation. 21/21 tests pass with it on (fwd-tests).
  A/B (fwd-ab, 24k ticks, 512 worlds, both variants interleaved tick-for-tick in one job):
  **on 89177 vs off 89274 wt/s — 0.1%, i.e. nothing.**
- `soa` (opt-in, default off): lane field planes laid out field-major, `(off + i) * blocks + b`, so a
  warp's lane indices are contiguous (AoS puts the lane in the multiplier slot; the emitter swaps the
  operands). Bug found: rRec SoA planes were sized `trainLearners` while the rollout indexes rec by
  a lane number over ALL worlds, so an eval-lane write wrapped into the next field instead of falling
  off the end (AoS drops it) — that is the 11/21 NaN storm in soa-tests. Fixed by sizing the rec
  planes for every lane (`recBlocks`, 31.7 -> 40.6 MB at 512 worlds); after the fix the soa variant
  tracks the baseline. A/B (soa-ab): **soa 43809 vs off 43673 wt/s — 0.3%, i.e. nothing.**

### What that means
Two independent memory-layout rewrites of the learner's hottest loops moved throughput 0%. The
learner is ~2.4% of FLOP peak and ~4% of DRAM bandwidth, so the rollout is NOT limited by the
policy's arithmetic or its traffic: with 6 warps per SM it is pure latency, and the thing that fills
the latency is the 31-tick game simulation plus the rollout's obs/rec bookkeeping, not the network.
Conclusion: **kernel micro-optimisation of the learner is exhausted. Remaining compute levers are
(1) ticks-to-quality — sample efficiency, i.e. algorithmic, and (2) sim cost per tick, which is
game-chunk work (P19 got 1.69x there; more may exist).** A third, not yet measured: the gradient
kernel at 16.7 ms (26% of the iteration) which is pure learner math with no sim in it.
Next measurement: prof-nofwd (`?profile=1&ablate=noPolicyForward`) splits sim+bookkeeping from the
policy forward inside the rollout; prof-hp1 measures the host pipeline.

### Compute-per-NPC (the product side, not the trainer)
The same Brain runs in runtime/npc.js for the live game, so per-NPC inference cost is a product
constraint. Pruning (P18) says a trained net keeps 99% of teacher quality at 2090 of 11232 edges;
training sparse from the start would cut both the trainer's MACs and the live per-tick cost. The
trainer-side throughput win looks small (learner is 2.4% of peak), but the live-game win is real and
independent. See the sparse-training research.

## P21 — compute: where the 81 ms actually goes (2026-10-03, measured)

prof-nofwd (512 realm worlds, `?profile=1&ablate=noPolicyForward`) vs prof-hp1 (same page, host
pipeline, no ablation): `rl_rollout` 62.19 ms ablated vs 61.16 ms not ablated, `rl_grad_blk` 16.77 vs
16.56, rest < 1.5 ms each, total 81.0 vs 79.8 ms. So **removing the policy forward from the rollout
changes nothing**: the forward is under 2% of the rollout, confirming fwdDirect/soa neutral. The
rollout is game sim + observe + obs/rec bookkeeping. Cross-check against P19's evolution `sim_step`
(512 worlds K=16 = 8.18 ms = ~1.0 us/world-tick): 31 ticks at that rate is ~16 ms, so ~46 ms of the
62 ms rollout is RL-side (observe + obs/rec writes), not the game chunk. The sim is the smaller half.

Consequence: the "cut the sim" plan is the smaller lever; **observe + rollout bookkeeping is the
biggest single block, and it is latency-bound (3 CTAs/SM at 31476 B of workgroup storage)**.

### Gradient-path audit (subagent, ranked; realm defaults, partials buffer 226 MB)
1. `rlshader.js:370,505,496` — in the FF config `SB_OFF` is a byte-copy of `HB_OFF` and `DH_OFF` is
   dead (line 361 `var hn = sv;` only RECUR changes it; the only other SB/DH read, :523, is
   RECUR-gated). ~137 MB/iter, 47% of scratch. Gate the writes on RECUR and drop both planes from
   `blockStride` (:99): 209,504 -> 111,200 f32, partials 226 MB -> ~126 MB. Bit-inert.
2. `rlshader.js:345` in the `for j` at :330 — `theta[TB+OFF_W1+c*HID+ii]` is j-invariant but
   re-loaded 32x per block: ~150 M LDGs/iter (601 MB of L1/L2 traffic, all L2-resident = pure
   latency). Unroll j by 2 (two `P_XS` planes, half the barriers): +6,720 B workgroup (23,044 ->
   29,764; the declared max stays 31,476 because the rollout dominates). Est. 2-4 ms/iter. Bit-inert.
3. `rlshader.js:1155-1162` — the h0 pre-pass recomputes what `lane[LA_H0]` already holds in FF
   (hread/hwrite both alias LA_H0), dead whenever validSh==1. ~3.2% of the rollout forward.
   Medium-high risk: depends on LA_H0 surviving between dispatches on every path.
4. `rlshader.js:443-446` — `mu`, `sigma`, `eps` recomputed/re-read (:445 redoes exp already done at
   :421). Keep in locals. ~25 MB/iter. Bit-inert, trivial.
5. `rlshader.js:1430` — `rl_mine(s/ROLL)` per sample: 462,272 lane reads where 14,912 suffice.
   Loop learners outer / ticks inner. NOT bit-inert (FP reduction order) — needs a flag.
6. `var acc: array<f32,45>` at :1498/:1699 is the only large dynamically-indexed local array (the
   `blockGrad:0` path only): 2.7 GB/iter of LOCAL traffic if it lands in local memory.
9. `gradGroups` 256 -> 128 halves `rl_reduce_grad` traffic and the scratch (214 -> 107 MB). Bit-inert,
   but fewer CTAs could hurt latency hiding — measure.

### Eval-world budget (the cheapest lever, P21)
`evalStart = worlds - evalCount - 2*leagueEvalCount` (rlengine.js:79); defaults evalFraction 0.125,
leagueEvalFraction 0.03, so at 512 worlds ~94 of 512 (~18%) are eval-only: they run the full sim and
policy forward and produce no gradient. Queued: `p21-evalbench` (interleaved default vs
evalFraction 0.0625/leagueEvalFraction 0.015 vs 0.03125/0.0075 — wt/s) and `p21-evalq` (2 seeds,
360k, halved eval) against the greattool baseline 4.48 +- 0.20 / ratio 2.69. Halving eval is +11%
training worlds at the same wall clock, i.e. +11% samples/s at an 11% larger batch (the 1024-world
batch sweep was inside noise, so quality risk is low).

## R11 landed 2026-10-04 — entropy .01 annealed to .001 is the realm default

Per seed at the 360k horizon (EVAL x1e3 / ratio / eval life), from the log window nearest tick 360000:

```
greattool (R10 baseline)  4.34 / 2.56 / 3366   4.62 / 2.81 / 2912   -> 4.48 +- 0.20, 2.69 +- 0.18
enthi  (flat .01)         5.01 / 2.93 / 3693   4.81 / 2.79 / 3373   -> 4.91 +- 0.14, 2.86
entanneal (.01 -> .001)   5.00 / 2.92 / 3708   4.92 / 2.84 / 3541   4.78 / 2.95 / 3482 (s3)
                                                                    -> 4.90 +- 0.11, 2.90, life 3577
```

`evalBase` is 1.69 vs the baseline's 1.67, so this is not the drift signature: +9% EVAL on a
flat denominator, 3 seeds (s3 was queued as `entan-s3`, the same config). `enthi` and `entanneal`
are within each other's noise, so the win is having more entropy, not the decay — the anneal is
adopted because it is the cheaper of the two to keep (the tail is the default .001 either way).

Changes: `RL_DEFAULTS.entropy` 0.001 -> 0.01 with `entropyEnd` 0.001 (anneal over
`lrDecayIterations` = 2000 passes, ~62k ticks); blob pinned to `entropy: 0.001` in
`RL_GAME_DEFAULTS` because entropy earns nothing there (3 seeds each at ~250k: blobdef 0.58 +- 0.01,
blobent01 0.56 +- 0.02, blobentan 0.55 +- 0.02).

One non-obvious consequence: the 21-test suite is no longer inert. `rlLossAndGrad` reads
`hp.entropy` as a constant while the GPU gets `entropyCoef()` from the uniforms, so with a schedule
at the default the gradient/Adam parity tests would compare different coefficients (4.5e-6 per pass
against a 4e-8 tolerance). Fixed by recording the value actually written (`this.lastEntropy` in
`writeUniforms`, also seeded in the constructor) and passing `{ entropy: t.lastEntropy }` into the
three JS-reference call sites (testUpdateParity, population league test, population gradient
compare). 21/21 on realm and blob after the change (`entdef-tests-rl/bl`, build-hash 9320258c4710a900).

`entdef` (realm defaults, no URL overrides, seeds 1-2, 400k) is the confirmation run that the
shipped default path reproduces these numbers.

## P21 eval-budget verdict (2026-10-04): quality-neutral, NOT adopted

`p21-evalq` (`evalFraction` .0625 / `leagueEvalFraction` .015, 2 seeds, 360k): EVAL 4.59 +- 0.51,
ratio 2.69 +- 0.20, life 3409 against greattool 4.48 +- 0.20 / 2.69 +- 0.18 / 3139. The point
estimate is higher and the ratio identical, so quality is neutral — but the seed sd is 2.5x the
baseline's (0.51 vs 0.20), which is exactly the noise that has made seven earlier arms unreadable.
+11% training worlds (arithmetic: 0.18 -> 0.09 of worlds are eval-only) does not pay for a noisier
yardstick, so 0.125 / 0.03 stay. `p21-evalbench` did not measure what it was meant to: three
variants ran concurrently in one job, so each page got ~57k wt/s of a shared GPU and the per-variant
throughput is contention, not cost.

## Streaming AC (arXiv 2410.14606) — landed in runtime/npc.js, 2026-10-04

Opt-in `learnMode:'ac'` (Stream AC(λ) + AdaptiveObGD): one update per tick, no replay, exact
within-tick backprop of the policy score through a CSR-by-destination index, one eligibility trace
per parameter, critic trained on its own head only, warm-started from `meta.valueWeights` or a
previous session's `acValue`. Measured on blob (24 learners, reward/tick/learner x1e3):

- converged PPO champions, 2 champions x 50k ticks: frozen .623/.647, sigma .48 (the champion's own
  training logStd) .589/.606, .25 .613/.626, .10 .639/.661, **.05 .669/.658**, .02 .642/.654,
  .05 with `acCritic:0` .648/.660 — the deployment sigma must be far below the training one.
- untrained random brains, 2 seeds x 35k ticks: frozen -4.43, Hebbian -4.46, sigma .05 -4.38,
  **sigma .25 -4.17** (~2 fewer deaths per learner) — there the larger sigma wins by 6%.
- realm, dense 11k-edge champion, 20k ticks, n=1: frozen 4.79 vs AC .05 4.85, drift steady not
  growing, `bad` 0. Neutral-to-slightly-positive, not an adoption at one seed.

So `sigma` defaults to .05 (keep improving a champion without damaging it); start an untrained brain
at ~.25. Harness kept as `tools/acab.mjs` (CPU, JS env, `--champ` or `--random`).

Next for this thread: trainer-side ObGD/AdaptiveObGD as an opt-in optimiser in `src/rlshader.js` +
`src/rl.js` (needs a JS mirror and a 2-seed A/B vs Adam at 360k), and a 2-seed realm runtime-AC
measurement to promote it off n=1.

## Next GPU round after `entdef` — channel-weight annealing (banked from R6)

`chw-a` (`popWeights=0.5,1.5,1.5,1.5`) learns ~2x faster and peaks 2.55 at 190k, then decays to a
tie by 360k. The mechanism to test is a schedule, not a knob: start tilted (0.5/1.5/1.5/1.5) and
decay to 1/1/1/1 over `lrDecayIterations`, mirroring `entropyCoef`. Needs its own JS mirror
(`rlWeightedReward` + the reference path reads `pop.weights` from opt, not from hp) and an extension
of the existing "weighted rewards" population test, so it is a src round: land it, run
`chw-tests-rl`/`chw-tests-bl` (21/21), then 2 seeds at 400k judged at 360k against the new
`entdef` baseline (~4.90 / 2.90). Adopt if EVAL >= 5.00 with no seed < 4.70.

## R24 channel-weight annealing (2026-10-04, VERDICT: NEUTRAL, not adopted)

Motivation (R6, AB-ARCHIVE): at K=1 `popWeights=0.5,1.5,1.5,1.5` (halve survival) is NEUTRAL overall (2.46 vs 2.38 at 360k) but learns ~2x faster and peaks 2.55 at 190k, then decays. Reading: the tilt is a curriculum on the credit assignment, not a better objective - so schedule it away, exactly like entropy.

Implementation (opt-in, shipped path untouched):
- `RL_POP_DEFAULTS.popWeightsEnd` (URL param works: dev/rllong.html accepts every RL_POP_DEFAULTS key). `rlPopConfig` parses it per role ('/' separated like `popWeights`) and returns `weightsEnd` (null when unset, or when the game has no channels: C == 0, so blob is inert).
- `pop.weights` = LIVE weights, `pop.weightsStart` = configured start. `RlBackend.applyChannelSchedule()` lerps start -> end over `lrDecayIterations` (same clock as lr and entropy) and rewrites `opt[OPT_W+c]` per policy in every `writeUniforms`, i.e. before the rollout that uses them. The JS references read `pop.weights`, so they follow with no extra plumbing; the checkpoint carries `weightsStart`/`weightsEnd` and restore rebuilds both plus the live row.
- Inert unless `popWeightsEnd` is set: `weightsEnd === null` returns immediately.

Test design lesson (cost one failed 21/21): a from-scratch JS env replay matches the GPU rollout only at the FIRST iteration, because the training worlds keep advancing - replaying a later rollout from tick 0 against a fresh env compares different world states (it failed at world 0 tick 0 with 'gpu weighted reward 1 js 5'). So the annealed window cannot be checked against a fresh env. Instead: three temp backends with the same seed (annealing, pinned at the target weights, pinned at the start weights), all stepped with `update:false` (the policy never changes, so trajectories are bit-identical), then compare `rec[.. + R.reward]` sample by sample: the annealed backend must equal the pinned-target backend on every sample and differ from the pinned-start one on many. `testPopulationCheckpoint` additionally asserts the targets survive the round trip and that weights actually moved (guarded on channels > 0, so blob is unaffected).

Round: `chwa` = realm, 512 worlds, 400k ticks, seeds 1-2, popWeights 0.5,1.5,1.5,1.5 -> popWeightsEnd 1,1,1,1, judged at the 360k horizon against `entdef` (4.97 +- 0.05, ratio 2.90 +- 0.03, 2 seeds). Early read at ~89k ticks: chwa 4.69 vs entdef 3.29 (the tilt front-loads learning, as R6 predicted). Adopt if EVAL >= 5.00 with no seed below 4.70; adoption = `RL_GAME_DEFAULTS.realm = { popWeights: '0.5,1.5,1.5,1.5', popWeightsEnd: '1,1,1,1' }`, keeping K=1 un-gated.

Verdict (360k horizon, 2 seeds): chwa 4.89 +- 0.04 (ratio 2.93 +- 0.03, `evalBase` 1.67, eval life 3251 +- 91, self-play 5.27) vs entdef 4.97 +- 0.05 (2.90 +- 0.03, 1.71, 3827 +- 126, 5.40). Per seed: 4.77 vs 4.83, 4.96 vs 4.88 - a wash in the wrong direction on EVAL and +0.03 on ratio only because `evalBase` drifted down. It misses the 5.00 bar, so NOT adopted: defaults stay 1/1/1/1 with no anneal. The mechanism stays in the code (inert unless `popWeightsEnd` is set) because the negative result is the useful one: the tilt really does front-load learning (4.69 vs 3.29 at 89k ticks, 1.4x) and annealing it away does NOT convert that head start into a higher ceiling - by 360k both curves are flat at the same level. So channel weighting on realm v4 is a speed knob only, and with `entanneal` already spending the early budget on exploration there is nothing left to buy with it. Next lever is the optimiser, not the objective.

## R25 ObGD as an opt-in PPO optimiser (2026-10-04, VERDICT: arm A loses badly, arm B a wash - not adopted)

Motivation: the streaming-drl stack is in the runtime (`learnMode:'ac'` = Stream AC(lambda) + AdaptiveObGD) but the trainer still used plain Adam. PPO here is already a streaming method - one pass per rollout, one epoch, no replay - which is exactly the non-stationary regime where ObGD's authors argue momentum-based Adam is the wrong tool. R6/R24 showed the objective is exhausted (channel weights are a speed knob only), so the remaining lever is the update rule.

Math ported from `runtime/npc.js` `StreamingAc.update` (per-parameter trace of squared gradients, debiased; step normalised by it; optional global trust step), at batch cadence:
- `rlObgdStep` (src/rl.js): v <- beta2 v + (1-beta2) g^2, step per parameter = stepSize * g / (sqrt(v/c2) + eps) - no momentum, no m slot. `z = sum |g| / (sqrt(v/c2) + eps)` and `stepSize = lr / max(1, lr*z/obgdBudget)` when `obgdBudget > 0`: the total L1 parameter movement per update is then capped at `obgdBudget`. `obgdBudget` 0 = plain normalised step (per-parameter step = lr, i.e. Adam's magnitude without momentum).
- WGSL: `rl_adam` branches on the compile-time `OBGD` constant; the z pass writes `redB`, reduces it in-workgroup (barriers stay in uniform control flow), then the apply pass reads the already-updated v. Adam's path is byte-for-byte the old arithmetic.
- Plumbing: `RL_DEFAULTS.optimizer` ('adam' | 'obgd') and `RL_DEFAULTS.obgdBudget`, both in `RL_TRACKED_KEYS` so tempBackend/checkpoint/restore carry them, both in `shaderConfig` so the shader recompiles per variant. `testUpdateParity` now runs three backends (adam, obgd, obgd with a budget) and each compares the GPU theta against its own JS reference.

Scale note: at realm hidden 96, LEARN ~ 10953, so z ~ LEARN (at step 1 the debias makes every ratio exactly 1) and Adam's implied total L1 step per update is lr*z ~ 11. So `obgdBudget` 11 is the Adam-equivalent cap.

Round (realm, 512 worlds, 400k ticks, seeds 1-2, judged at 360k vs `entdef` 4.97 +- 0.05 / 2.90 +- 0.03):
- `obgdA`: optimizer=obgd, obgdBudget=0, lr 1e-3 - isolates removing momentum at Adam's own step size.
- `obgdB`: optimizer=obgd, obgdBudget=11, lr 1e-2 - trust region at the Adam-equivalent total step with 10x headroom, so the cap governs the magnitude and the direction is the raw gradient.
Adopt the better arm if EVAL >= 5.20 with no seed below 4.90 (a >4% move, outside the 0.05 seed sd and the ~0.04 `evalBase` drift); record the optimiser in RL_DEFAULTS only if an arm wins, otherwise `optimizer` stays 'adam'.

R25 pivot (same day, before the arms finished): obgdA-s1 is far behind at matched ticks - EVAL 0.00117 at 43k and 0.00120 at 75k vs entdef-s1 0.00286 at 43k and 0.00273 at 75k (ratio 0.77 vs 1.72/1.52). Not a bug: the parity test pins the GPU to `rlObgdStep` at 5e-8, and at iteration 1 ObGD's per-parameter step is exactly Adam's (v/c2 makes every ratio 1). The mechanism is signal-to-noise: without momentum each update follows one minibatch gradient at full normalised magnitude lr, so the useful drift is lr * SNR, and Adam's beta1 = 0.9 buys ~sqrt(10) of SNR by averaging ~10 iterations. ObGD is built for *streaming* cadence, where thousands of tiny trust-region updates per episode supply the averaging that momentum supplies here - at one update per rollout, momentum is the variance reducer and removing it costs ~2-3x the learning speed.

Consequence: arm B as designed (budget 11 + lr 1e-2) would have been redundant - with the cap binding, its per-parameter step is budget/z ~ 1e-3, i.e. arm A again. Replaced by `mb1` = realm defaults with `minibatches: 1` (2 seeds, 400k, 360k horizon): one update per rollout over ALL samples instead of four over a quarter each, the direct test of the same noise hypothesis (4x fewer, 4x less noisy updates at equal data and equal lr). Judged the same way; adopt minibatches 1 only if EVAL >= 5.20 with no seed below 4.90.

## R26 blob-side PPO tuning (2026-10-04, blobentan VERDICT: NEUTRAL-to-worse, not adopted; blobobsn running)

Motivation: every realm knob that moved the needle has been measured, and the "Open work" list has blob tuning as the remaining item. Blob already carries its own defaults through `RL_GAME_DEFAULTS` (rolloutTicks 16 + gamma .999, entropy pinned .001) and sits at 0.58 +- 0.01 at the 250k horizon (`blobdef`, 3 seeds) vs 0.52 for the pre-tuning defaults. Two knobs tested on realm but never on blob:
- `blobentan`: entropy .01 -> entropyEnd .001 over lrDecayIterations. On realm the same anneal is +9% EVAL; on blob flat .01 was NEUTRAL-to-worse (0.56 vs 0.58), so this isolates the decay rather than the level.
- `blobobsn`: obsNorm 1 (implies bias). NEUTRAL on realm (2.35 vs 2.38) but never run on blob, where observation scales differ and the game is much smaller - a generality check on the normaliser, not an expected win.

Protocol: 512 worlds, 300k ticks, 3 seeds each, judged at the COMMON 250k horizon against `blobdef` (0.58 +- 0.01). Adopt an arm into `RL_GAME_DEFAULTS.blob` only if EVAL >= 0.62 with no seed below 0.55 (a >7% move on a 0.01 seed sd). Blob has no reward channels (C == 0), so channel-weight and population knobs are inert here.
R25 verdict, arm A (obgdA, n=1, judged at its last window ~390k): EVAL 2.68, ratio 1.61, `evalBase` 1.66, eval life 4306 vs entdef 4.97 +- 0.05 / 2.90 +- 0.03 / 3827. A 46% loss - 45x the 0.05 seed sd and far outside any `evalBase` drift - so the arm is dead at one seed and obgdA-s2 was dropped rather than re-run; its slot became mb1-s3 (3 seeds for the minibatch arm instead of 2). ObGD at batch cadence is therefore a negative result: removing momentum costs ~2x the learning speed even though the GPU matches `rlObgdStep` at 5e-8 and the first update is bit-comparable to Adam. The optimiser stays adam; if mb1 wins, the lever is update granularity under Adam, not the update rule.
## R27 lr anneal + reward clipping (2026-10-04, queued after R26)

Motivation: the two axes that ever moved realm are schedules (entropy .01 -> .001 over `lrDecayIterations` = +9% EVAL) and gating (R8/R10), while the update rule (R25 ObGD), the update granularity (R25 mb1: EVAL 5.00 / 2.91 at 390k vs entdef 4.97 / 2.90 - NEUTRAL), the objective (R24) and the architecture (T/hidden/batch) are all flat. So the remaining cheap lever is the other schedule: `lrEnd` is still equal to `lr` (1e-3), i.e. no lr anneal at all, even though the entropy anneal proved this clock matters.

Arms (realm, 512 worlds, 400k ticks, 2 seeds each, judged at the 360k horizon vs `entdef` 4.97 +- 0.05 / 2.90 +- 0.03):
- `lran`: lrEnd 1e-4, so lr decays 10x over the same 2000 iterations as the entropy anneal (~62k ticks, 15% of the run). Adam keeps its normalised per-parameter magnitude; this only shrinks late-stage churn.
- `rclip`: rewardClip 2. Scaled per-tick reward averages ~0.006 (train 0.00476 env/tick at rewardScale ~1.27), while a boss kill is ~0.95 scaled, so 2 clips only the stacked multi-event spikes and leaves ordinary rewards untouched - a tail lever on the reward distribution, not on minibatch noise (which mb1 showed is not binding).
Adopt an arm if EVAL >= 5.20 with no seed below 4.90 (same bar as R25). `valueClip` stays untested unless a clip arm wins, in which case the critic tail is the obvious next one.
mb1 seed 1 (390k): EVAL 4.88, ratio 2.90, `evalBase` 1.68, eval life 2958; seed 3: 5.00 / 2.91 / 1.72 / 2614. Mean over the two finished seeds 4.94 +- 0.08 vs entdef 4.97 +- 0.05 (ratio 2.91 vs 2.90) - NEUTRAL. Worth noting for the record: mb1 earns the same reward rate with ~27% shorter eval lives (2786 vs 3827), i.e. it trades survival for intensity and lands in the same place, so neither the update rule (ObGD) nor the update count (minibatches 4 -> 1) is the binding constraint on realm v4.
R25 final verdict (judged at the 360k horizon, report matched windows, vs entdef 4.97 +- 0.05 / 2.90 +- 0.03, 2 seeds):
- obgdA (optimizer=obgd, obgdBudget=0, n=1): EVAL 2.84, ratio 1.71, `evalBase` 1.66, eval life 4854. A 43% loss - the arm was killed at one seed rather than re-run, and obgdA-s2 became mb1-s3.
- mb1 (minibatches 1, n=3): EVAL 5.00 +- 0.07, ratio 2.96 +- 0.11, `evalBase` 1.69, eval life 2815 +- 459, self-play 5.36 +- 0.06. Per seed 4.88 / 4.77 / 5.00 at their final windows (~390k). Raw EVAL is entdef's within 1 sd and the +0.06 ratio is `evalBase` drifting 1.71 -> 1.69, so NEUTRAL: not adopted, and it also costs 27% of eval life at the same reward rate.
Conclusion: on realm v4 at one update per rollout, neither the update rule (momentum is the variance reducer, not the problem) nor the update count (4 -> 1) is binding. Defaults stay optimizer=adam and minibatches=4; both mechanisms stay in the code as opt-in flags, and the negative result is the useful one - it closes the whole streaming-optimiser family for batch-cadence PPO here, so the remaining levers are schedules (R27) and the game/objective, not the optimiser.
R26 arm 1 verdict (blobentan, 512 worlds, judged at the COMMON 250k horizon, 3 seeds): EVAL 0.56 +- 0.02 (`evalBase` -0.29, eval life 2993 +- 178, self-play 0.56) vs blobdef 0.58 +- 0.01 (-0.33, 2988, 0.58) - a 3% move, about 1.5 seed sd, and 6 sd below the pre-registered 0.62 bar. Per-seed final windows: 0.59 at 283k, 0.61 at 300k, 0.57 at 282k. So the realm entropy anneal (+9% EVAL) does NOT transfer: blob is smaller, its reward has no channels, and the flat .01 level was already neutral-to-worse here (0.56 vs 0.58) - the decay inherits that and adds nothing. Blob stays pinned to entropy .001 with no anneal.
R26 arm 2 verdict (blobobsn, obsNorm 1, judged at the COMMON 250k horizon, 3 seeds): EVAL 0.73 +- 0.00 (`evalBase` -0.06 +- 0.02, eval life 2983 +- 26, self-play 0.73 +- 0.00, h2h 1.05) vs blobdef 0.58 +- 0.01 (-0.33, 2988, 0.58) at matched ticks 250032 vs 252421. Per seed 0.73 / 0.72 / 0.74 - every seed far above the 0.62 bar and none near the 0.55 floor, and the seed sd is ZERO, so this is a +26% win and the first arm to clear its bar since R11.

ADOPTED: `RL_GAME_DEFAULTS.blob` gains `obsNorm: 1` (implies bias, so blob champions now export `npc-brain/3` with `inputNorm` folded into W1/b1). Build `a76d01e29b9223e8`. Because adoption must be verified on the shipped path, the gate (`obsn-tests-rl`/`obsn-tests-bl`, both must be 21/21 with obsNorm now ON by default for blob) plus `blobdef2` (3 seeds, no URL overrides, must reproduce ~0.73) were bumped ahead of R27; R27 (lran, rclip) follows.

Why blob and not realm: `obsNorm` was NEUTRAL on realm (2.35 vs 2.38, R7) and is +26% here. The obvious candidate is input scale - realm's 105 inputs are bounded compass/sector/need values, blob's are raw masses and distances, and the policy's softsign(2x) saturates on wide inputs. UNVERIFIED: it predicts obsNorm should be the FIRST knob tried on any new game, and it is worth a direct test (log the pre-normalisation input range per game) before we trust the story.
Adoption verified: `obsn-tests-rl` and `obsn-tests-bl` are both 21 PASS / 0 FAIL on build a76d01e29b9223e8 (so obsNorm being ON by default for blob breaks nothing, including the population and export paths), and `blobdef2` - the defaults path with no URL overrides, 3 seeds - gives EVAL 0.74 +- 0.01 at 246k vs the arm's 0.73 +- 0.00 at 250k: the shipped default reproduces the arm, so the adoption is real and not an artefact of the URL param. Blob's new baseline is therefore ~0.73-0.74 (was 0.58), i.e. +27% over the pre-R26 default and +17% over blobpp-t16g999's 0.59 at the 250k horizon - the largest single gain on blob since the T/gamma defaults landed.
## R28 candidates (not queued; pick after R27 lands)

1. Realm `obsNorm` retest, 3 seeds. R7 measured it NEUTRAL on 2 seeds (2.35 vs 2.38 ratio, but raw EVAL 4.05 vs 3.90 - the whole gap was `evalBase` drifting 1.72 vs 1.64). Blob just gave +26% with zero seed spread, which raises the prior enough that a 3-seed rerun is worth 1.9 h.
2. Blob defaults retune under obsNorm. Blob's rolloutTicks 16 / gamma .999 / entropy .001 were all tuned BEFORE normalisation existed; the interaction is unknown (a normaliser changes the effective input scale, which is exactly what gamma and T trade against).
3. Test the input-scale story instead of assuming it: log the pre-normalisation per-input range and mean for realm and blob for a few thousand ticks (a dev page, no training) and see whether blob's inputs are the wide ones. Cheap, and it decides whether obsNorm should be the first knob on any new game.
R27 arm 1 verdict (lran, lr 1e-3 -> 1e-4 over `lrDecayIterations` 2000, n=1, judged at the 360k horizon): EVAL 4.05, ratio 2.38, `evalBase` 1.70, eval life 4382, entropy 1.223 vs entdef 4.97 +- 0.05 / 2.90 +- 0.03 / 1.71 / 3827. A 19% regression on BOTH columns with `evalBase` flat, so it is not drift - killed at one seed (the deficit is ~18 seed sd) and lran-s2 was dropped so rclip could start 38 min earlier.
Reading: the entropy anneal works because exploration is a level you want early and not late; lr is not. At 400k ticks (~13k iterations) the run is nowhere near a noise floor - the eval curve is still climbing every window - so a 10x decay applied 15% of the way in removes learning speed and buys no convergence benefit. Combined with mb1 (4x fewer updates, NEUTRAL) and ObGD (much worse), the picture is consistent: this regime is learning-rate-limited, not noise-limited, so anything that reduces effective step size or step count costs quality roughly in proportion.
R27 arm 2 verdict (rclip, rewardClip 2, n=1): EVAL 3.00 at the matched 359k window (3.98 at its own final 400k window, still climbing), ratio 2.19, `evalBase` 1.37, eval life 2639 vs entdef 4.97 / 2.90 / 1.71 / 3827. Raw EVAL is 20% down even though `evalBase` also fell, and it misses the 5.20 bar by miles, so the arm is killed and rclip-s2 was dropped. Reading: clipping the scaled per-tick reward at 2 does not just trim spikes - it removes the very events (boss kills, milestones, big sales) that the value function and the advantage carry, and it slows learning the same way a smaller lr does. With lran also worse, R27 is closed: no clipping, no lr anneal, lr stays flat 1e-3.

## R28 (2026-10-04, queued)

- `obsnrl` (realm, 3 seeds, 400k, judged at 360k vs entdef): obsNorm 1. R7 called it NEUTRAL on 2 seeds but the raw numbers were EVAL 4.05 vs 3.90 with `evalBase` 1.72 vs 1.64 - i.e. obsNorm was AHEAD on raw EVAL and only "neutral" once the drifting denominator was divided in. Blob then gave +26% with zero seed spread, so a 3-seed rerun is worth 1.9 h. Adopt if EVAL >= 5.20 with no seed below 4.90; if it lands 5.0-5.2 with `evalBase` flat, treat that as the same drift trap and demand a 4th seed before touching `RL_GAME_DEFAULTS.realm`.
- `blobh128` (blob, 3 seeds, 300k, judged at 250k vs blobdef2 0.74): hidden 128. Pre-obsNorm it was NEUTRAL (0.52 vs 0.52 base) at 25% less throughput; with the inputs normalised, extra capacity may finally pay.
- `blobei01` (blob, 3 seeds, same): entropy .01 flat. Pre-obsNorm it was NEUTRAL-to-worse (0.56/0.55 vs 0.58), but the realm result says .01 is the better level and the anneal's failure here may have been a scale artefact rather than a level preference.

## Input-scale hypothesis: FALSIFIED (2026-10-04)

The stated reason for expecting obsNorm to pay more on blob than realm was "blob's inputs are raw masses and distances while realm's are bounded compass/need values, and softsign(2x) saturates on wide inputs". Measured directly from the learned `inputNorm` stats of the champions that carry them (`runs/blobobsn-s1-base`, `runs/obsnrl-s1-base`; `obsNormFloor` is 0.1, so std 0.100 means "at the floor"):

| game  | nIn | std p25 / median / p75 / max | n with std > 1 | n with std > 3 | |mean| median / max |
| ----- | --- | ---------------------------- | -------------- | -------------- | ----------------- |
| blob  | 30  | 0.134 / 0.143 / 0.286 / 0.43 | 0              | 0              | 0.06 / 0.76       |
| realm | 105 | 0.152 / 0.195 / 0.252 / 0.71 | 0              | 0              | 0.09 / 0.78       |

Both games feed the policy inputs with std well under 1, and realm's are slightly WIDER, not narrower. So the input-width story is wrong and it no longer predicts that obsNorm is the first knob to try on a new game. What obsNorm actually does here is (a) mean-centre inputs whose |mean| reaches 0.76-0.78 against a std of only ~0.15 - i.e. strongly one-sided, near-constant channels - and (b) rescale them by ~1/std, a 4-7x gain that is the same order in both games; and it implies `bias`. The blob/realm difference must come from something other than width: number of inputs (30 vs 105 - per-input noise in the running mean/std is 3.5x worse on realm for the same sample count, and a mis-normalised input is amplified by the same 1/std), or the fact that realm's 105 inputs are already near-standardised by construction.

Consequence: `blobbias` (blob, 3 seeds, `bias: 1, obsNorm: 0`) was queued to separate the bias from the normaliser. If bias-only reproduces ~0.73, ship bias alone - fewer learned params, no running-mean state to fold at export, no `npc-brain/3` upgrade needed on the blob path.

## R29 (pre-registered 2026-10-04, NOT queued until the obsnrl verdict is in)

Realm `curriculum: 1`, 3 seeds, 512 worlds, 400k ticks, judged at the 360k window vs `entdef` 4.97 +- 0.05 / 2.90 +- 0.03 (life 3827). Knobs: `curriculumDifficultyStart` 30, `curriculumDifficultyStep` 5, `curriculumLifeUp` 0.12 (the notes' fix - realm maxAge is 6000, so the default 0.2 is rarely reached), `curriculumLifeDown` 0.05, `curriculumPatience` 2, `curriculumMetric` 0 (eval life), `curriculumSelfPlay` 0.

Why it is the right next lever: every cheap PPO knob is now measured, and the wins that moved realm were environment-side (R8/R10 gates) plus the entropy anneal (R11). Curriculum is the one remaining mechanism that attacks ticks-to-quality directly rather than asymptotic quality - the same axis as the compute-efficiency goal - and it is already implemented and test-gated, just never measured at scale.

Bar: adopt only at EVAL >= 5.20 with no seed below 4.90 (identical bar to obsnrl). A win here is worth more than a knob win because it changes the shape of the learning curve, not just its ceiling; a loss is cheap because it stays opt-in. Confound control: queue it only after obsnrl is decided, so whichever observation pipeline wins is the one the curriculum arm inherits.

## R28 realm verdict (2026-10-04): obsNorm NEUTRAL, not adopted

`obsnrl` at the matched 360k window: 4.98 +- 0.09 (ratio 2.96, evalBase 1.68, life 3417, self-play 5.36) vs `entdef` 4.97 +- 0.05 (2.90, 1.71, 3827, 5.40), n=2. Dead even, and 5.20 was unreachable, so `obsnrl-s3` was skipped (status `skipped`, chrome killed by `queue stop`) rather than burning 38 min of GPU on a seed that could not change the decision. Realm stays bias-free; `RL_GAME_DEFAULTS.realm` is untouched.

This is now the cleanest controlled comparison in the project: the SAME mechanism is NEUTRAL on realm (4.98 vs 4.97, two independent 2-seed rounds on two different baselines) and a +26% win on blob (0.73 vs 0.58, 3 seeds, zero seed spread). So obsNorm is not a general "always turn it on" knob - it is game-dependent, and the input-width explanation for that is falsified (see above). The `blobbias` arm (bias only, obsNorm 0) is the next cut at why.

## R29 (2026-10-04, queued as `curr`)

Realm `curriculum: 1` + `curriculumLifeUp: 0.12`, 3 seeds, 512 worlds, 400k, judged at 360k vs `entdef` 4.97. Pre-registered above; queued now that the observation pipeline question is settled (realm keeps the plain bias-free path, so the arm inherits ship defaults). Same bar: adopt at EVAL >= 5.20 with no seed below 4.90.

## R28 blob arm 1 verdict (2026-10-04): hidden 128 NEUTRAL, rejected

`blobh128` 0.73 +- 0.01 at the matched 250k window (2 seeds, life 3804, self-play 0.74) vs `blobdef2` 0.74 +- 0.01 (3 seeds, life 3313). Identical within noise, and it costs ~25% throughput, so it is rejected; seed 3 was skipped. This is the same answer as the pre-obsNorm measurement (0.52 vs 0.52 base), so normalising the inputs did not unlock extra capacity on blob either - capacity is not what either game is short of.

## R28 blob arm 2 verdict (2026-10-04): entropy .01 WORSE, rejected

`blobei01` 0.68 at the matched 250k window (n=1, life 3257, evalBase -0.18) vs `blobdef2` 0.74. Seeds 2-3 were skipped: the direction matches the two pre-obsNorm 3-seed arms (0.56/0.55 vs 0.58), the gap is 6x the blob seed sd, and blob's negative `evalBase` makes only EVAL judgeable. Blob keeps entropy .001; the realm .01 result does not transfer on either try, so the entropy level is genuinely game-dependent and both defaults are now measured twice.

## R28 blob arm 3 verdict (2026-10-04): the win is the normaliser, not the bias

`blobbias` (bias 1, obsNorm 0) 0.58 +- 0.00 at the matched 250k window, 2 seeds, zero spread, life 3018 - exactly the old pre-obsNorm blob level - vs `blobobsn` 0.73 +- 0.00 and `blobdef2` 0.74 +- 0.01. So bias alone reproduces none of the +26%, and the normaliser (running mean/std, rescaling each input by ~1/std against a typical std of 0.14, i.e. a ~7x gain) is the mechanism. Practical consequence: blob keeps `obsNorm: 1` and its `npc-brain/3` export with folded `inputNorm`; there is no cheaper bias-only substitute.

Also settled on blob: `hidden` 128 NEUTRAL (0.73 vs 0.74 at ~25% less throughput, 2 seeds), entropy .01 WORSE (0.68, n=1, corroborated by two earlier 3-seed arms). Blob's measured defaults are now rolloutTicks 16 + gamma .999 + obsNorm 1 + entropy .001, and every one of them has been measured at least twice or at 3 seeds. Blob tuning - the last item in "Open work" - is closed.

## R28 close-out

Three of four arms rejected, one adopted earlier (obsNorm on blob). Realm is unchanged: obsNorm NEUTRAL twice, so the shipped realm path is still plain feed-forward, bias-free, entropy .01 annealed to .001, 512 worlds, hidden 96, rolloutTicks 31. The round also killed the input-width explanation, which was the only story we had for why the same mechanism pays on one game and not the other; the honest statement now is that obsNorm's value is game-dependent and must be measured per game, not assumed.

## R29 verdict (2026-10-04): curriculum NEUTRAL, not adopted

`curr` 4.91 +- 0.13 at the matched 360k window (2 seeds, ratio 2.87, evalBase 1.71, life 3372, self-play 5.33) vs `entdef` 4.97 +- 0.05 (2.90, 1.71, 3827, 5.40). Seed 3 skipped: the 5.20 bar was unreachable. The mechanism works as designed - `CUR` walks d=30 -> 40 -> ... -> 100 as smoothed eval life climbs from 1511 to 3786, and it is throughput-neutral (173.8k wt/s vs entdef 165.7k/159.7k) - it just does not improve quality at equal ticks. Reading: realm v4 at difficulty 100 is not hard enough early for an easier ramp to help, so the curriculum buys nothing the fixed-difficulty run does not already get; it stays opt-in.

## R30 (2026-10-04, queued)

Two arms, realm, 512 worlds, 400k, judged at 360k vs `entdef` 4.97, same bar (EVAL >= 5.20, no seed below 4.90):
- `enthi2`: `entropy` .02 -> `entropyEnd` .002. Entropy is the only knob with a measured monotone gradient (.0003 much worse, .001 old default, .01 annealed = the shipped default), so extending the trend one step is the highest-EV cheap experiment left.
- `currsp`: `curriculum` 1 + `curriculumSelfPlay` 1 + `curriculumDifficultyStart` 100 - the self-play ramp with the difficulty ramp pinned at 100, so it isolates the "learners compete with each other more than with bots" mechanism that R29's difficulty ramp could not test. Self-play share follows the eval/bot ratio between 0.15 and 0.9.

## R30 arm 1 interim (2026-10-04): enthi2 seed 1 = 5.06

`enthi2` (entropy .02 -> .002) 5.06 at the matched 364k window, n=1, ratio 2.99, evalBase 1.69, life 3422, self-play 5.44, vs `entdef` 4.97 / 2.90 / 1.71 / 3827 / 5.40. Raw EVAL is up 0.09 on a denominator that MOVED THE OTHER WAY (1.69 vs 1.71), so this is not the drift signature - but 0.09 is about one seed sd (enthi 4.91 vs entanneal 4.90 vs entdef 4.97 are all the same shipped anneal measured on different seeds), so n=1 decides nothing.

Decision rule, pre-registered before seeds 2-3 land: adopt entropy .02 -> .002 only if mean EVAL >= 5.05 AND every seed >= 4.97 (the baseline mean) AND mean `evalBase` within 0.03 of entdef's 1.71. Otherwise reject as noise. If it is adopted, the follow-up is `entropy` .03 -> .003 to see whether the gradient continues; if it is rejected, the entropy level is a plateau over .01-.02 and the knob is closed.

## R30 arm 1 at 2 seeds (2026-10-04): enthi2 5.11 +- 0.08 vs entdef 4.97

Per seed 5.06 / 5.16, ratio 2.98 +- 0.02, evalBase 1.71 +- 0.04 (identical to entdef's 1.71), life 3481 +- 83, self-play 5.44 +- 0.00. All three pre-registered conditions hold: mean >= 5.05, every seed >= 4.97, `evalBase` within 0.03. Seed 3 is left to run for n=3 before any src edit.

Sequencing constraint: `currsp` (already queued) does not set entropy, so it inherits `RL_DEFAULTS`. Changing the default while those jobs are pending would silently re-baseline them, so the adoption edit waits until every queued job that inherits entropy is finished. `enthi3` (.03 -> .003) is queued after `currsp` with EXPLICIT entropy params for the same reason - it stays comparable to entdef whatever the default becomes.

## R30 arm 1 verdict (2026-10-04): entropy .02 -> .002 ADOPTED

`enthi2` 5.10 +- 0.06 at the matched 360k window, 3 seeds (5.06 / 5.16 / ~5.08), ratio 2.98 +- 0.01, evalBase 1.71 +- 0.03, life 3428 +- 108, self-play 5.44 +- 0.01, vs `entdef` 4.97 / 2.90 / 1.71 / 3827 / 5.40. Every pre-registered condition held on a denominator that did not move, so `RL_DEFAULTS.entropy` .01 -> .02 and `entropyEnd` .001 -> .002 (src/rl.js), rebuild build-hash 436f670f.

Two traps handled during the edit:
1. `currsp` was already queued and inherits entropy from `RL_DEFAULTS`, so raising the default mid-round would have silently re-baselined the self-play mechanism test. All three `currsp` jobs were pinned to `entropy` .01 / `entropyEnd` .001 before the src edit, so their comparison against entdef stays valid.
2. `RL_GAME_DEFAULTS.blob` overrides `entropy` (.001) but not `entropyEnd`, so the raised global end would have made blob anneal UPWARD (.001 -> .002). Added `entropyEnd: 0.001` to the blob override; blob is flat .001 as before.

Follow-ups queued: `enthi3` (.03 -> .003, explicit, so it stays comparable to entdef whatever the default is) to see whether the gradient continues past .02, and `entdef2` (2 seeds, no URL overrides) to confirm the shipped default path now measures ~5.10 rather than 4.97 - the same confirmation `entdef` provided for .01.

## R30 arm 2 interim (2026-10-04): currsp seed 1 = 4.68

`currsp` (curriculumSelfPlay 1, difficulty pinned at 100, entropy pinned .01/.001) 4.68 at the matched 355k window, n=1, ratio 2.79, evalBase 1.68, life 3956, self-play 5.05, vs `entdef` 4.97 / 2.90 / 1.71 / 3827 / 5.40. The ramp behaves as designed - `sp` climbs to 0.84 because the eval/bot ratio (~2.7) sits near the top of the 0.5-3.0 mapping window - but learners end up LONGER-LIVED and LOWER-EARNING (life 3956 vs 3827, rate 4.68 vs 4.97): more self-play teaches survival without teaching earning, the same shape as the realm `gamma` .999 failure.

Seed 2 runs for the 2-seed minimum. If it confirms, the follow-up arm is `curriculumSelfPlayEnd` 0.5 (a capped ramp) rather than abandoning the mechanism - the user goal is that learners compete with each other, so a milder share is the interesting variant, not none.

## R30 arm 2 verdict (2026-10-04): curriculumSelfPlay rejected

`currsp` 4.94 +- 0.37 at the matched 360k window, 2 seeds (4.68 / ~5.20), ratio 2.90 +- 0.16, evalBase 1.70 +- 0.04, life 3789 +- 236, self-play 5.24 +- 0.28, vs `entdef` 4.97 / 2.90 / 1.71 / 3827 / 5.40. Mean is neutral, but the seed spread is 0.37 against entdef's 0.05 - the mechanism does not move the average, it makes the outcome depend on the seed, which is worse than a plain loss for a shipped default. Seed 3 skipped; `curriculumSelfPlay` stays opt-in/off.

Follow-up queued: `currsp5` (`curriculumSelfPlayEnd` 0.5, 2 seeds, inherits the new .02/.002 entropy so it is judged against enthi2/entdef2 ~5.10 rather than entdef's 4.97). The full ramp drives `sp` to 0.84 because the eval/bot ratio sits near the top of the 0.5-3.0 mapping; capping it at 0.5 is the variant worth one arm, since learners competing with each other is a user goal even though the uncapped version does not pay.

## R30 arm 3 verdict (2026-10-04): entropy .03 is worse - .02 is the plateau

`enthi3` (.03 -> .003) 5.01 at the matched 360k window (n=1, ratio 2.80, evalBase 1.79, life 3440, self-play 5.52) vs `enthi2` 5.10 (2.98, 1.71, 3428, 5.44) and `entdef` 4.97. Seeds 2-3 skipped: it needed to BEAT enthi2 and it is 0.09 below it, with `evalBase` drifting up to 1.79 (so the ratio gap 2.80 vs 2.98 is partly denominator, but the raw EVAL gap is the one that matters and it points down).

Entropy is now a measured plateau, not a gradient: .0003 much worse, .001 old default, .01 -> .001 shipped since R11, .02 -> .002 the new default (+2.6%), .03 -> .003 worse. The knob is closed unless the game changes.

## Process finding (2026-10-04): run 300k ticks, not 400k

Every recent realm run is flat from ~200k to 400k (`entdef-s1` 5.03 at 200k then 4.90/5.00/4.83; `entdef-s2` 4.91/5.08/4.93/4.88; `curr-s1` 4.71/4.91/5.00/4.96; `enthi2-s1` 4.71/4.87/5.17/5.06/4.86; `enthi2-s2` 4.95/5.15/5.17/5.02; `enthi2-s3` 5.11/5.01/5.06/5.14; `currsp-s1` 4.31/4.56/4.68/4.49; `entdef2-s1` 4.91/5.10/5.07/5.13/5.08). The last 100k ticks buy nothing and sometimes lose a little. So A/B jobs drop to `ticks: 300000` / 30 min and are judged at --at-tick=300000: 25% cheaper per arm, 25% more arms per GPU-hour, same decision power. Baselines have 300k windows too, so nothing has to be re-measured.

## R31 (2026-10-04, queued; first round at the new 300k length)

Judged at 300k vs `enthi2`/`entdef2` (~5.10). Two arms, 3 seeds each:
- `pool24`: `poolSize` 8 -> 24. A deeper league reservoir means later snapshots meet a wider range of opponents; the self-play family has been neutral so far, but this is the one league knob never swept (`leagueFraction` was cancelled as flat).
- `sig07`: `sigmaInit` .5 -> .7. The exploration axis is the only one that paid recently (entropy .02 > .01), so pushing initial action noise the same direction is the cheapest test of whether that result generalises beyond the entropy bonus.

## R30 adoption confirmed on the shipped default path (2026-10-04)

`entdef2` (no URL overrides, so it exercises `RL_DEFAULTS` exactly as shipped) 5.10 +- 0.05 at the matched 360k window, 2 seeds, ratio 2.91, evalBase 1.75, life 3575 - identical to `enthi2` 5.10 +- 0.06 and +0.13 over the old `entdef` 4.97. The .02 -> .002 anneal is therefore a property of the shipped configuration, not of an explicit URL param.

New 300k reference for R31: `enthi2` 5.11 +- 0.09 and `entdef2` 5.01 +- 0.08 at the 300k window (same config, two runs) - i.e. ~5.05 +- 0.08 with roughly 0.10 of run-to-run spread. Pre-registered bar for R31 arms: adopt at EVAL >= 5.25 with no seed below 4.95.

## Tooling fix (2026-10-04): report.mjs ignores deliberately-skipped seeds

When a seed is skipped mid-flight (`queue stop` + status `skipped`), its log keeps whatever partial windows it wrote - e.g. `currsp-s3` left 60 s of data at ~24k ticks. `report.mjs` averaged that in, which made the `currsp` row read "3 seeds, 3.49 +- 2.52" and hid the real 2-seed result (4.94 +- 0.37). Fixed: `skippedJobs()` reads `runs/queue-state.json` and `main()` drops any log whose job is `skipped`, the same way `IGNORED_LOGS` drops `queue.log`. A skipped seed is a deliberate discard, so its numbers must never reach a mean.

## R31 pre-arm interim: currsp5 seed 1 = 5.25

`currsp5` (`curriculumSelfPlay` 1 with `curriculumSelfPlayEnd` 0.5, difficulty pinned 100, inheriting the .02/.002 anneal) 5.25 at the matched 362k window, n=1: ratio 3.09, evalBase 1.70, life 2858, self-play 5.76, vs `entdef2` 5.10 (2.91 / 1.75 / 3575 / 5.46) and `enthi2` 5.10. Two things to note: it BEATS the uncapped ramp by a lot (4.94 with sd 0.37 -> 5.25), and it earns more per tick while living LESS (2858 vs 3575) - the opposite of the uncapped version, which produced longer-lived, lower-earning learners. So the cap is not a milder version of the same failure, it is a different regime.

Bar, fixed now before seed 2 lands: adopt the capped self-play ramp only if 2 seeds give mean EVAL >= 5.20 with no seed below 5.05 (run-to-run spread at this horizon is ~0.10). If it holds, `RL_DEFAULTS` gains `curriculum` 1 + `curriculumSelfPlay` 1 + `curriculumSelfPlayEnd` 0.5 + `curriculumDifficultyStart` 100.

## R32 currsp5 - capped self-play curriculum ramp (VERDICT: ADOPT at n=2)
- 2 seeds vs the 5-seed entdef2/enthi2 reference (~5.07-5.10): 360k 5.20 +- 0.08 (ratio 3.01, evalBase 1.72, life 3203, SP 5.62); 300k 5.15 +- 0.02 vs 5.01/5.11 (SP 5.60 vs 5.27/5.42).
- Pre-registered bar (mean >= 5.20, no seed < 5.05) met at BOTH horizons; seed sd 0.02-0.08 vs baseline 0.08-0.09, and the mechanism-targeted self-play column moves further than the headline (+0.18..+0.25), which is the signature of a real effect rather than drift.
- Uncapped R30 currsp was 4.94 +- 0.37: capping the self-play share at 0.5 (with difficulty pinned at 100) is what keeps the ramp from destabilising seeds.
- Confirmation queued: seeds 3-5 at the 300k/30-min length, so all 5 currsp5 seeds can be judged at 300k against the 5 baseline seeds there.
- src edit PENDING (confound): pool24 x3 and sig07 x3 are queued WITHOUT curriculum params, so they inherit RL_DEFAULTS. Landing `curriculum` 1 + `curriculumSelfPlay` 1 + `curriculumDifficultyStart` 100 + `curriculumSelfPlayEnd` 0.5 + `curriculumLifeUp` 0.12 now would silently re-baseline them against the ~5.05 reference. Chosen: wait for them to drain (the queue is full either way, so no GPU time is lost), then edit src/rl.js, `node build.mjs`, and add the README clause. currsp5 s3-s5 pass the knobs explicitly, so they are immune either way.

## R33 (queued behind R32 confirmation): shape of the self-play ramp
- Mechanism: `RlCurriculum.update` sets the training self-play share to Start + (End - Start) * progress, progress tracking the eval/bot ratio normalised between ratioLow .5 and ratioHigh 3 (src/rl.js:696). At the observed ratio ~2.9, progress ~= .96, so R32s End .5 means ~48% of training worlds in 32-learner self-play.
- R30s uncapped variant (End .9, difficulty ramping from 30) was 4.94 +- 0.37; R32 pinned difficulty at 100 and capped End at .5 for 5.20 +- 0.08. The arm that differs on both axes at once cannot say which mattered, so R33 brackets End on the R32 config: `sp25` (.25 -> ~29% self-play) and `sp75` (.75 -> ~73%), 2 seeds each at 300k.
- Bar (pre-registered): move the shipped End away from .5 only if that arms 2-seed mean beats currsp5s 5-seed mean at 300k by >= 0.10 and no seed falls below currsp5s worst seed. If currsp5 seeds 3-5 come in below ~5.05, mark these two jobs `skipped` before they start rather than sweeping a knob whose reference just moved.

## R34 (queued): hidden 64 on the R32 config - a compute win if quality holds
- The old T/hidden sweep judged hidden 64 NEUTRAL-to-worse (2.29 vs 2.38 ratio) on a pre-R10 baseline; it has not been re-judged since entropy .02 and the self-play ramp. Per-NPC inference cost is a shipped-product constraint (the brain keeps improving live), so 96 -> 64 is a third off live inference and rollout cost.
- Bar (pre-registered): adopt hidden 64 only if 3 seeds at 300k land within 0.05 of currsp5s 5-seed mean (i.e. quality-neutral, not 2% worse). Anything worse than that keeps 96.
- Dormancy check that killed the ReDO idea (no GPU, champion weights only): per-hidden-unit incoming |W1| mass on the R32/R30 champions has min 9.1 vs mean 23.9 and a bottom decile holding ~5% of total mass, with no unit under 5% of the mean. There is nothing dormant to recycle, so recycling-style mechanisms are not worth a build. Caveat: weight mass is not activation; true dormancy would need a replay, and replays must not run while a GPU job is live.

## R31 verdict (2026-10-04, 3 seeds each, judged at 300k vs the ~5.07 enthi2/entdef2 reference)
- `pool24` (`poolSize` 8 -> 24): 5.15 / 5.13 / 4.96 = 5.08 +- 0.09, ratio 2.98, self-play 5.41. NEUTRAL - a deeper league reservoir does not pay, and it misses the pre-registered 5.25 bar.
- `sig07` (`sigmaInit` .5 -> .7): 5.02 / 4.95 / 5.09 = 5.02 +- 0.06, ratio 3.02 but `evalBase` drifted to 1.66 vs 1.71-1.74, so the ratio flatters it. NEUTRAL-to-worse on EVAL.
- Neither is adopted: `poolSize` 8 and `sigmaInit` .5 stay. The exploration axis that paid was the entropy bonus, not initial action noise.

## R32 landed (2026-10-04): capped self-play curriculum is now the shipped default
- `RL_CURRICULUM_DEFAULTS` in src/rl.js: `curriculum` 0 -> 1, `curriculumDifficultyStart` 30 -> 100, `curriculumLifeUp` .2 -> .12, `curriculumSelfPlay` 0 -> 1, `curriculumSelfPlayEnd` .9 -> .5. `RL_GAME_DEFAULTS.blob` gains `curriculum` 0 / `curriculumSelfPlay` 0 so blob keeps the behaviour its numbers were measured under (never measured there).
- Live check after the edit: a fresh `RlCurriculum({}, 6000)` starts at difficulty 100 with share 0.15 and converges to 0.499 at ratio 2.99 - exactly the ~48% self-play of the measured arm.
- Rebuilt: index.html build-hash a77303ca06538162 (was 436f670f499618d8). README PPO paragraph now states the curriculum default and the .02 -> .002 entropy anneal.
- Verification queued: `curdef` seed 3 passes NO curriculum params and is bumped ahead of the sweep, so it should come out bit-identical to `currsp5-s3` (explicit params) - the same proof entdef2 gave for the entropy adoption. Compare `edges`, `params` and `meta.valueWeights` of the two champions.

## R32 interim at n=3 (2026-10-04): the n=2 read was optimistic, and curriculum costs determinism
- Same window rule as report.mjs (single log window nearest 300k), currsp5 seeds 1-3: 5.13 / 5.16 / 5.00 = 5.10 +- 0.07, SP 5.53 +- 0.11. Reference pooled 5 seeds: 5.07 +- 0.08, SP 5.36 +- 0.13. So EVAL is +0.03 (inside noise) and the mechanism-targeted self-play column is +0.17 (t ~ 2).
- The 2-seed read (5.20 vs 5.10 at 360k, 5.15 vs 5.07 at 300k) came from two seeds that happened to agree; seed 3 is the first below the reference.
- Determinism cost, measured: curdef-s3 (shipped path, NO curriculum params) and currsp5-s3 (explicit params) have identical resolved configs - verified by merging the job params against RL_DEFAULTS + RL_CURRICULUM_DEFAULTS + RL_POP_DEFAULTS, zero mismatches - yet scored 5.08 vs 5.00. Cause: the curriculum steps at every stats report, reports fire on wall clock (logSeconds 30), and the two runs crossed those steps ~200-300 ticks apart (15035 vs 14818, 29171 vs 29481), so world re-inits on flips land differently. Curriculum-OFF runs stay bit-identical (entdef2-s1 == enthi2-s1).
- Decision rule (pre-registered before seeds 4-5 land): KEEP the curriculum default only if the 5-seed mean at 300k is >= 5.17 (>= +0.10 over the reference). Rationale: the gain must exceed the ~0.08 reproducibility cost it introduces, or every future A/B gets a worse noise floor for nothing. If it lands below, revert RL_CURRICULUM_DEFAULTS to `curriculum` 0 / `curriculumSelfPlay` 0, rebuild, fix README + AGENTS.md, and re-queue hid64 without curriculum params (sp25/sp75 become moot and get skipped).
- If it is kept, the follow-up is to make curriculum updates tick-driven instead of report-driven (a tick counter, not logSeconds) so bit-identity comes back; that needs the queue drained first, since sp25/sp75/hid64 all pass curriculum 1 explicitly and would silently change.

## R36 spfix0 - train in the eval composition (2026-10-04)

Hypothesis: R30/R32's "self-play ramp" arms were misread. `rl_init_mode` shows the
reference already trains 75% of worlds at 32 brain slots (all-learner, selfplay rules on,
randomized regrow/mob/hunger/pvp from `randomize == 1u`), and turning `spOn` on moves
non-selfplay worlds to mode 0 = `defaultCfg`, 16 learners + 16 bots, no randomization. So
End .5 did not ADD self-play, it cut all-learner worlds 75% -> ~49% and moved the rest to
the eval-A composition - and EVAL still went 5.04 -> 5.14 (+0.07). If composition matching
is the driver, the full dose should beat the half dose.

Arm: `curriculum 1, curriculumSelfPlay 1, Start 0, End 0, curriculumDifficultyStart 100`
= every training world is defaultCfg 16 learners vs 16 bots (exactly eval A), constant
selfPlay 0 and difficulty pinned at 100, so no flips, no reinit: the arm is DETERMINISTIC
unlike R30/R32 (same-seed reruns must agree).

Bar (pre-registered): adopt only if EVAL at 300k >= 5.13 (+2% over the ~5.03 shipped-default
reference: entdef2 5.01 +- 0.08, curdef-s3 5.08) on 3 seeds, and self-play EVAL does not fall
more than 5%. A win is "default cfg + eval composition" jointly (mode 0 changes both); a
follow-up arm separates them.

## R34 hid64 status (2026-10-04)

hid64b-s1: EVAL 5.00 at 289k (SP 5.54), 258k wt/s vs 214k for curdef-s3 on the same 30-min
job shape = ~+20% throughput and a third off live inference per NPC. Reference at 300k:
entdef2 5.01 +- 0.08 (2 seeds), curdef-s3 5.08. Adopt hidden 64 if the 3-seed mean lands
within 0.05 of the reference (pre-registered R34 bar).

## R36 (spfix0) - every training world in the eval-A composition
Design: `curriculum:1, curriculumSelfPlay:1, curriculumSelfPlayStart=End=0` -> `rl_init_mode` returns 0 for EVERY training world (mode 0 = realm `defaultCfg`, 16 learners + 16 scripted bots, no domain randomization). Eval groups untouched: A stays `U.randomize` (mode 1, already defaultCfg/16 slots), B/C stay mode 2. Difficulty pinned 100 and the share is constant, so no flips, no reinit, deterministic.
- seed 1 @296k: EVAL 5.53, evalBase 1.75, ratio 3.17, eval life 1955, H2H 1.14, SP 6.33 (SP life 1217), wt/s 372k.
- reference hid64b (hidden 64 = shipped default, 3 seeds @293k): 5.00 +- 0.02, ratio 2.92 +- 0.05, SP 5.41 +- 0.11, life 3178, wt/s 263k.
- So +0.53 EVAL (+10.6%) at +41% throughput (half the entities are scripted bots -> half the policy forwards), and eval C (pure 32-learner self-play) +17%, H2H 1.14 vs 1.02.
- Caveat: training composition now equals the eval-A composition, so eval A is no longer composition-held-out (worlds, respawn and freshness still are). Eval C and H2H rising with it says this is not benchmark overfitting alone.
- Bar (pre-registered, restated on the hidden-64 baseline): adopt at EVAL >= 5.10 on >= 3 seeds with SP not down > 5%. Seeds 2,3 running; 4,5 queued.
- Adoption form if it wins: `RL_GAME_DEFAULTS.realm.randomize: 0`, which IS spfix0 without the curriculum flags (randomize 0 -> mode 0 for training worlds; eval worlds unchanged because mode 1 already ran them at defaultCfg/16 slots).

## R37 (spfix1) - isolates composition from domain randomization
`curriculumSelfPlayStart=End=1` -> every training world mode 2 = defaultCfg at 32 slots, no bots, no domain randomization. 3 seeds, 300k, same protocol (jobs-r47.json).
- spfix1 ~= 5.5 -> removing randomization is the driver (write a real randomization fix next: randomize regrow/mob/hunger WITHOUT moving slots off 16).
- spfix1 ~= 5.0 -> the 16-learner-vs-16-bot composition is the driver, and randomization is free.
Either way the arm decides what the realm default should be and what the next round writes.

## R36 result so far (spfix0, 2 seeds @~297k)
- s1 5.53, s2 5.71 -> 5.62 +- 0.09; base ~1.74 (flat), ratio 3.17/3.29, SP 6.33/6.08, wt/s 372k/361k.
- reference hid64b (3 seeds @~293k): 5.00 +- 0.02, ratio 2.92, SP 5.41 +- 0.11, wt/s 259k.
- Bar (5.10, SP not down >5%) CLEARED at 2 seeds: +0.62 EVAL (+12.4%), +12% self-play, +40% throughput. Seed 3 running.
- Interpretation: the reference trains 75% of worlds at 32 brain slots (all learners, `randomize == 1u`) plus domain randomization; spfix0 trains 100% at 16 learners vs 16 bots, no randomization. Both eval A (vs bots) and eval C (pure 32-learner self-play) improved, so this is not just benchmark-composition matching.

## R38 (spfix25) - the mixture endpoint
`curriculumSelfPlayStart=End=0.25`: 25% of training worlds at 32 slots (mode 2, all learners) + 75% at 16 learners vs bots (mode 0), no randomization. 2 seeds, 300k, same protocol (jobs-r48.json).
- R37 (100% all-learner) vs R38 (25%) vs spfix0 (0%) brackets how much all-learner world share the new base wants; adopt the best into the realm default.

## R36 shipped-path verification (hid64b-s5) and the open question
- hid64b-s4 started BEFORE the rl.js edit -> true reference seed 4: 4.90 (wt/s 255k, old config). Reference over 4 seeds: 4.98 +- 0.05, SP 5.39.
- hid64b-s5 started AFTER the edit, no URL params at all -> the shipped default (`randomize: 0`, curriculum off): EVAL 5.21, base 1.73, SP 5.67, wt/s 364k, tick 299305. So the default IS in effect (364k wt/s = 16 learners/bots per world) and it beats the reference by +0.23, but it is 0.44 BELOW spfix0's 5.53/5.71/5.70 while lagging at every window (5.06 vs 5.42-5.67 @235k, 5.14 vs 5.61-5.63 @256k, 5.37 vs 5.57-5.73 @278k).
- Checked for a config difference: `u[6]` difficulty is 100 either way (controller pinned at `curriculumDifficultyStart` 100 vs `difficultyAt()` over a 100->100 ramp); `u[26]` spOn is 1 vs 0 but `rl_init_mode` reaches mode 0 in both cases (spfix0 through the `spOn` branch, the default through `U.randomize` 0), eval groups are mode 0/2 in both, and `spOn` is read nowhere except `rl_init_mode`/`rl_world_selfplay`/`rl_reinit_world`. wt/s and `params=7565` match. So on paper the two are the same run.
- The discriminator is `spfix0-s5` (seed 5, spfix0 flags) vs hid64b-s5 (seed 5, bare default): equal => seed 5 is simply a low seed and the per-seed pairing holds (s1 +0.53, s2 +0.69, s3 +0.71); ~5.6 => the curriculum flags do something the bare default does not and the adoption form is wrong.
- Either way the adoption (5.21 vs 4.98 reference) is not in doubt; only its magnitude is. More bare-default seeds will be queued if the paired test says the two are equal.

## R36 final (5 seeds, 300k)
- spfix0 s1..s5: 5.53 / 5.71 / 5.70 / 5.68 / 5.38 -> 5.60 +- 0.13, SP 6.15, wt/s 356-372k, eval life ~1930-2350.
- reference (hid64b s1..s4, old config): 5.00 / 5.02 / 4.99 / 4.90 -> 4.98 +- 0.05, SP 5.39, wt/s ~256k, eval life ~3170.
- +12.4% EVAL, +14% self-play, +40% throughput. Adoption stands.
- PAIRED TEST RESOLVED: seed 5 under spfix0 flags = 5.38 vs seed 5 under the bare shipped default = 5.21 (delta 0.17, inside the 0.13 seed sd), and both are the lowest seed of their family. So `randomize: 0` (no curriculum) IS the spfix0 run; the earlier 0.44 gap was seed 5, not the flags. hid64b-s5 is therefore the shipped-path verification: 5.21 vs a 4.98 reference at 364k wt/s.
- def64 (seeds 6,7, no params at all) extends the shipped-default estimate.

## R37 answer (spfix1 = 100% all-learner worlds, no randomization)
- s1: EVAL 4.88, base 1.72, SP 5.34, wt/s 227k, eval life 3703 vs spfix0-s1 5.53/6.33/372k/1955 and reference s1 5.00.
- So removing domain randomization alone is worth ~nothing (4.88 vs reference 4.98), while the 16-learners-vs-16-bots composition is worth +0.6. The composition is the driver.
- It also costs 40% throughput (32 learners = twice the policy forwards) for slightly WORSE quality.
- Notable inversion: training against bots beat training in self-play on the PURE SELF-PLAY eval too (SP 6.15 vs 5.34-5.44). Self-play training is not what builds self-play strength here; learner opponents are non-stationary and chaotic, the scripted bots are a stable gradient.

## R39 (long64) - is 300k still the plateau?
2 seeds, 600k ticks, bare shipped default (jobs-r50.json). If EVAL at 600k exceeds 5.60 by more than 1 sd (0.13), the 300k horizon understates the new default and every future A/B must move; if flat or lower, 300k stays and the extra compute is waste.

## R37 confirmed (2 seeds): spfix1 4.88 / 5.01 -> 4.95 +- 0.07
vs spfix0 5.60 +- 0.13 and reference 4.98 +- 0.05. Randomization removal is neutral; composition is the driver.

## R40 (w256 / w128) - buy the same quality for less compute
At 16 learners/world the update batch is already half what it was (512x16x31 = 254k samples vs 508k), and the new composition made that smaller batch BETTER. So try fewer worlds: 256 and 128, 2 seeds each, 300k, bare default (jobs-r51.json). Wall clock to a fixed per-world horizon is proportional to world count, so 256 costs half and 128 a quarter of a 512-world run.
- Adopt 256 if EVAL >= 5.47 (within 1 sd of 5.60); same rule for 128.
- Risk: eval groups shrink with `worlds` (evalFraction), so EVAL gets noisier - judge on the mean and report the sd.

## R38 (spfix25 = 25% all-learner + 75% bot worlds)
- s1: EVAL 5.16, SP 5.70, wt/s 306k, eval life 2833. Seed 1 ladder: 0% all-learner 5.53 | 25% 5.16 | 100% 4.88 -> monotone, so ANY all-learner share costs quality. Seed 2 pending.
- This also explains R32's retraction from the other side: `curriculumSelfPlay` ramping up to .5 moved worlds onto the wrong side of this gradient.

## R41 (noleag / noleag2) - the league is self-play too
`RL_DEFAULTS` league 1, leagueFraction .5, leagueSlotFraction .5: in half the training worlds, half the learner slots run a frozen snapshot and their samples are MASKED (rlshader.js:1030) - so a quarter of all learner samples are discarded and a quarter of opponents are snapshot policies. Given the R37/R38 gradient, both should hurt.
- `noleag` = `leagueFraction: 0` (keeps the pool and h2h eval, no snapshot worlds), `noleag2` = `league: 0` (machinery off entirely). 2 seeds each, 300k, bare default otherwise.
- Adopt if EVAL >= 5.60 (the spfix0 mean) - i.e. equal counts as a win because it also restores the masked samples and removes snapshot bookkeeping; report h2h as unavailable for noleag2.

## R38 confirmed (2 seeds): spfix25 5.16 / 5.39 -> 5.27 vs spfix0 5.62 at the same seeds
Seed-1 ladder 0% 5.53 / 25% 5.16 / 100% 4.88; seed-2 5.71 / 5.39 / 5.01. Monotone at both seeds. Written to AGENTS.md as the R37/R38 bullet.

## Shipped default over 8 seeds
spfix0 s1-5 (5.53/5.71/5.70/5.68/5.38) + hid64b-s5 (5.21) + def64 s6,s7 (5.51/5.73) = 5.56 +- 0.17, self-play 6.09 +- 0.21, vs reference 4.98 +- 0.05 / 5.44. +11.6% EVAL, +12% self-play. AGENTS.md R36 bullet updated to the 8-seed figure.

## R39 (long64) - 600k, seed 1 done
- long64-s1: 5.47 @292k, 5.47 @310k, 5.61 @329k, 5.68 @384k, 5.76 @421k, 5.81 @551k, 5.75 @588k. SP 6.43, eval life 2050, wt/s ~312k.
- So the curve is NOT flat: +0.28 (+5%) for a doubling of ticks, with no sign of turning over by 588k. At 300k this same seed gives 5.47 (spfix0-s1: 5.53), so the config reproduces.
- Verdict pending seed 2: 300k stays the A/B horizon (every arm is judged at equal ticks, so a common slow drift cancels), but a FINAL champion run should use 600k, where the shipped default is worth ~+5% more for 2x the compute.

## R39 verdict (2 seeds)
- long64-s1 5.47 @300k -> 5.75 @588k; long64-s2 5.73 @300k -> 5.84 @584k. Mean 5.60 -> 5.80: +3.6% for 2x the compute, both seeds positive but small.
- 300k stays the A/B horizon (equal-tick comparisons cancel a common drift); 600k is for a FINAL champion run. AGENTS.md's plateau bullet updated (it wrongly said "plateaus by ~250k").

## Streaming-drl audit (2026-10-04, subagent) — what we have NOT implemented

Our R25 "ObGD MUCH WORSE" (2.84 vs 4.97) tested the 2024 GLOBAL L1-budget variant. The repo default branch is 2026, which replaces it with PER-COORDINATE bounded updates that KEEP momentum (our R25 lesson: momentum is the variance reducer at one update per rollout). Ranked candidates, none needing new buffers or workgroup storage:

1. optimizer:'bounded' — v = max(beta2*v, |u|); step = mhat/max(beta2*v, |mhat|), so |dw_i| < alpha per coordinate. Touch rlAdamStep (src/rl.js ~370) + rl_adam (src/rlshader.js ~1910; the obgd branch is the pattern), extend the existing `optimizer` key. Zero layout change; default 'adam' stays bit-identical. Arms: beta2 .999 (ours) and .99995 (2026 default).
2. Intentional step size: alpha = eta/sqrt(sigma_bar * <rho g,g>), drop the constant lr; revives valueCoef/maxGrad. Keys etaPolicy/etaValue.
3. Adaptive delta clipping: deltahat = EMA(delta^2), cap = 20*sqrt(deltahat) in rl_gae (~1400). Our rewardClip/valueClip are 1e9 = never exercised, and realm returns are heavy-tailed. Keys advClipMult/advClipBeta.
4. Sign-gated entropy (2026 Alg 2): grad += tau*sign(delta)*grad(H). Key entropySign.
5. Running |A| normalisation replacing per-batch standardisation in rl_advstats.

Excluded as too costly: bias-free LayerNorm before each hidden activation (breaks npc-brain/2|3 replay identity across src/core.js and runtime/npc.js); ERK sparse init.

Order: R42 = (1) once the queue drains (needs a src edit, so only after w128 finishes).

## R42-R45 verdicts (2026-10-05, all 2 seeds at 300k unless noted; shipped adam reference 5.62 on the same seeds)
- R42 bounded optimizer (2026 streaming-drl `BoundedOptimizer`), lr ladder 1e-4/3e-4/1e-3/2e-3 = 5.36/5.56/5.74/5.41. Inverted-U peaking on the default lr; +2% at the top, inside one sd. NEUTRAL, opt-in only.
- R43 eval budget: `evalFraction` .125 -> .0625 (on top of `league` 0) = 5.79 vs 5.81 NEUTRAL; `leagueEvalFraction` .03 -> .015 = 5.55 with the seed sd tripling to 0.45. Neither adoptable.
- R44 stacking bounded + `league` 0 = 5.79: no stack.
- R41 RETRACTED at 4 seeds: `league` 0 = 5.74/5.88/5.70/5.68 vs shipped 5.53/5.71/5.70/5.68 - +0.10 paired, sd 0.11, zero on seeds 3 and 4. The 2-seed +3.4% was seed noise. League stays (it keeps H2H/SP).
- R45 adaptive delta clipping (`advClipMult`, streaming-drl 2026): 20x RMS = 2.79, 5x = 4.07 vs 5.62. MUCH WORSE both ways; the tighter clip is less bad because clipping shrinks the return std that `rl_rescale` divides by. Killed.
- Lesson across R25/R42/R45: every streaming-drl variance-control mechanism (ObGD, bounded updates, delta clipping) is neutral-to-catastrophic here. Our per-batch advantage standardisation + Adam momentum already does that job, and realm's heavy-tailed deltas (death -1024, boss/PvP payouts) are signal, not noise.
- R46 next: sign-gated entropy (`entropySign`), then a 600k champion of the shipped config.
- R46 sign-gated entropy (`entropySign`) = 2.60 at 285k vs shipped 5.53 (seed 1): MUCH WORSE, seed 2 killed. With R42/R45 this closes the streaming-drl variance/exploration family: neutral (bounded), worse (delta clip), catastrophic (entropy sign). Remaining untried from the audit: (2) intentional step size, (5) running |A| normalisation - both optimiser-side, i.e. the family that keeps failing here.
- Preliminary, n=1 (`tools/acab.mjs --game realm --champ runs/champ600-s2-base --modes frozen,ac --ticks 6000 --windows 3 --seeds 11`): frozen 6.65/5.47/5.07 vs ac 6.85/4.86/5.08 reward/tick/learner x1e3, ac weight drift 8.9 -> 13.1 so it IS updating, but no reward gain over 6k ticks on an already-converged champion in a stationary bot env. Expected - AC is for non-stationarity (other players), not a fixed env. Worth a longer run with more seeds before claiming anything; NOT written to AGENTS.md at n=1.
- NOTE: acab needs `ticks` divisible by `windows` or `per` is fractional and every row is dropped (silent empty output). Hit with 20000/3.

## R49 candidate (evidence, 2026-10-05) - starvation is the champion's remaining failure mode
`tools/replay.mjs --game=../src/games/realm.js --ticks=6000 --worlds=4 --top=4 runs/champ600-s2-base`:
- champion: life 1641, r/t 5.79, deaths per 255 lives -> starve 48, player 14, mob 8, boss 2, age 1, alive 26. Bots: starve 7, mob 56, boss 11, player 1, alive 25.
- evalsuite on the same champion: death_starve 0.617, death_player 0.228, death_mob 0.080 - same ordering.
- Channels per 1000 ticks: champion survival 1.19 / progress 1.04 / combat 1.01 / cooperation 2.54; bots 0.73 / 0.82 / 0.16 / 0.08.
- The asymmetry is the opportunity: bots barely starve (7) and die to mobs (56), the champion is the reverse. So a game-side change that makes food easier to secure should lift learners far more than it lifts `evalBase`, which is exactly the shape R8/R10 had (+11%).
- Caveat before touching anything: 32 entities per world, and "32 players on 32 springs at regrow 700 starved everyone by tick 1000" is a recorded capacity limit - check whether this is a policy failure or a world-capacity ceiling FIRST (hand-wired policy in Node, per the "test hand-wired policies before blaming the trainer" rule). If it is capacity, raising food availability is the only lever and it moves evalBase too.
- Cheaper alternative if the above is inconclusive: anticipatory shaping on the survival channel (reward for eating while hunger is high) rather than a world change.

### R49 capacity test - RESULT: starvation is behavioural, not capacity (2026-10-05)
Same horizon, only the mix changes (`tools/replay.mjs --game=../src/games/realm.js --ticks=3000 --worlds=2`, champ600-s2-base):
- 16 learners + 16 bots: life 1352, r/t 6.45, deaths mob 6 / boss 4 / player 11 / starve 34 / alive 45.
- 32 learners (`--selfplay`): life 955, r/t 6.85, deaths mob 4 / boss 1 / player 26 / starve 37 / alive 32.
- Doubling the brain-driven population moves starvation only 34 -> 37 but doubles PvP deaths 11 -> 26, and r/t goes UP (6.45 -> 6.85, cooperation channel 3.00 -> 3.70). So the food economy is not the binding constraint at this margin - the champion has solved combat and starves because of how it allocates time (hGreat 2.7/life and the cooperation channel dominate).
- Caveat: 2 worlds, 1 seed, 3000 ticks. Direction is clear enough to act on; confirm on more seeds if the fix looks marginal.

### R49 = the untested mirror of R6
R6 halved survival (`popWeights=0.5,1.5,1.5,1.5`) and was NEUTRAL but "learned ~2x faster, then decayed" - on the OLD baseline where combat was unsolved. The champion has now solved combat (mob 6, boss 4) and starves instead, which is exactly the regime where up-weighting survival should pay. The mirror arm `popWeights=1.5,1,1,1` was never run.
- Cheap: host param only, no src change and no rebuild (popWeights is un-gated at K=1, so it works with policies 1).
- Run: 2 seeds, 300k, judge at 300k against the ref58 reference. Watch `EVAL`, eval life, and the evalsuite death_starve share - a survival-weighted policy should show fewer starvation deaths even if EVAL barely moves.
- If it wins, the follow-up is whether to land it as a default or as a `popWeightsEnd` anneal back to 1,1,1,1 (R24's machinery, which was NEUTRAL when annealing FROM the tilt).
- acab n=3 (supersedes the n=1 line above), `--champ runs/champ600-s2-base --modes frozen,ac --ticks 6000 --windows 3`, reward/tick/learner x1e3, mean of 3 windows:
  seed 11: frozen 5.729 vs ac 5.598 (-2.3%) | seed 12: frozen 5.955 vs ac 6.252 (+5.0%) | seed 13: frozen 5.878 vs ac 5.468 (-7.0%).
  Means 5.854 vs 5.773 = -1.4%, paired-difference sd ~0.36, so no effect. AC's weight drift still climbs every run (dbar > 0, dw 9 -> 13 on seed 11), so it genuinely updates - it just buys nothing on an already-converged champion in a stationary bot env, which is what the theory predicts (AC is for non-stationarity, i.e. real players). Do not spend more seeds re-testing this in a fixed env; the informative test would need a drifting opponent.
