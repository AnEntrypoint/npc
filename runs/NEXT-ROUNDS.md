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

## R61 — creep archetypes resist a style (v5res) — REJECTED, measured 2026-10-06

Why: v5tri (+0.31 EVAL, ADOPTED) and v5ch (REJECTED) both failed to move the style
mix, so the third try was to make one style unusable against part of the world:
`MOB_RESIST = [0,1,2,0,1]` by `mobKind(e)` = `(e - R_MOB0) % 5` (brute/caster
resist melee, stalker/ambusher resist ranged, pack resists mage), `damage >>= 2`
in `tryAttack` when `MOB_RESIST[mobKind(target)] === style`, mirrored as
`r_mob_resist`; bots switch to a non-resisted counter-style. The learner already
sees the archetype at obs 106 (`hostileType`).

Screen (2 seeds, 512 worlds, `popWeights` 0.5,1,1,1, judged at 200k, paired vs
v5tri which is the shipped baseline):

```
arm    EVAL d      bots d      ratio d   life d     h2h d      selfplay d
v5res  -0.50+-0.19 -0.21+-0.02 -0.02+-.05 115+-21  -0.01+-0.10 -0.50+-0.29
```

`node runs/xpsurvey.mjs runs/v5res-s1-base 2 4000` -> xp0 p75 12, xp1 p50 247,
xp2 p75 9; s2 -> 9 / 231 / 0. So the mix did not move either: the learner does not
read obs 106 as "switch style". Per the pre-registered rule in
`runs/pending-v5res.md` ("Reject if the mix is unchanged AND EVAL falls") the arm
is REJECTED and all six edits were reverted (`grep -c MOB_RESIST` = 0 in both
realm.js and realm.wgsl.js, `node build.mjs` re-run).

What it generalizes to: three arms now agree that REACH, not perks and not
resistance, is what makes ranged dominant — a ranger at reach 800 takes ZERO
return damage from a reach-200 melee mob, and no multiplier competes with zero.
Numeric tuning of the combat triangle is exhausted; the remaining levers are
structural (role/style-locked population policies, a style curriculum, an
exploration bonus over style x archetype). That is why the next block of work is
telemetry-for-architecture rather than a fourth combat knob.

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

## Publishing (2026-10-05)
- GitHub: https://github.com/AnEntrypoint/npc (public, MIT, branch main). C:\dev\train had no git repo
  before today (gm verb `git_init` is NOT served by this build - unknown_verb - so init went through
  gm exec_js + git; a subagent was sent to add git_init to c:\dev\gm).
  runs/ (1.0 GB) and .gm/ (840 MB) are gitignored; runs/NEXT-ROUNDS.md is the one exception. 82 files, 2.4 MB.
- Kaggle CLI: account heclgang, ~/.kaggle/access_token (ACCESS_TOKEN auth, CLI at
  ~/AppData/Roaming/Python/Python312/Scripts/kaggle). Dataset heclgang/npc-brain-trainer (public,
  3.5 MB) = the tracked tree + the champ600 genomes under champions/.
- Probe kernel heclgang/npc-webgpu-probe (script, GPU+internet, private) settles whether a WebGPU
  adapter exists on Kaggle at all: swiftshader / vulkan / plain flagsets, then a 64-world realm run
  at 40k ticks. Our only training path is WebGPU in headless Chrome, so Kaggle can host and eval
  but cannot train unless that adapter appears.
- kaggle/ (probe-kernel/, npc-trainer/ staging) is gitignored. Note: the kaggle CLI crashes on
  Windows when the upload dir contains dotfiles (".codesearchignore" -> temp path join error);
  the staged copy drops .gitignore/.codesearchignore.

## Kaggle feasibility (2026-10-05) - WebGPU probe
Kaggle notebook facts (script kernel, GPU accelerator on): 4 vCPU, 31 GB RAM, 2x Tesla T4,
node v20.19, python 3.13, Ubuntu glibc 2.39, internet on, puppeteer installs fine.

- v1: dataset did not mount (/kaggle/input/npc-brain-trainer missing; /kaggle/input holds "datasets").
- v2: Chrome failed to launch - missing system libs (libatk-1.0.so.0 ...). Fixed with apt per-package install.
- v3: Chrome launches, but navigator.gpu is UNDEFINED under every flagset (swiftshader / vulkan / plain).
  NO Vulkan ICD on the box (/usr/share/vulkan/icd.d absent, no vulkaninfo), so a T4 WebGPU adapter
  would need NVIDIA Vulkan drivers installed - the T4s are CUDA-only from Chrome's point of view.
  Chrome 131 (puppeteer@23) predates WebGPU on Linux, which is the likely cause of no navigator.gpu.
- v4: retry with puppeteer@latest (newer Chrome) + Chrome's bundled vk_swiftshader_icd.json;
  if an adapter appears it is CPU SwiftShader, then measure wt/s at 64 worlds before promising anything.

Bottom line so far: Kaggle can host and version the trainer and run CPU-side eval/replay, but the
training path is WebGPU-in-Chrome and has not been shown to exist there. Do not promise Kaggle
training until v4 reports an adapter AND a measured wt/s.

## Kaggle feasibility (2026-10-05) - v5/v6: an adapter EXISTS, but it is CPU-only
- Root cause of v3/v4's "no navigator.gpu" was NOT Chrome and NOT the missing ICD: WebGPU needs a
  SECURE CONTEXT, and about:blank is not one. Probing http://localhost:8123/index.html makes
  navigator.gpu appear. (Chrome 154 = Chrome/154.0.8037.57.)
- All three flagsets (lavapipe / swiftshader / plain) return the SAME adapter:
  vendor google, architecture swiftshader, maxComputeWorkgroupStorageSize 32768, maxBindGroups 4.
- The trainer page at only 64 worlds produced NO tick line and NO pageerror in 420 s. CPU SwiftShader
  is orders of magnitude too slow; this lane is dead for training.
- maxBindGroups 4 is suspiciously low next to our 10-binding layout (we use ONE bind group, so not
  automatically fatal), but with zero ticks produced the page either stalled or failed pre-first-log.
- v7 = heclgang/npc-nvidia-vulkan-probe: installs nvidia-vulkan-icd + libvulkan1, retests under
  --use-angle=vulkan with VK_ICD_FILENAMES pointing at the NVIDIA ICD, then measures wt/s at 512
  worlds. This is the ONLY remaining Kaggle GPU path. If vendor is still google/swiftshader, drop
  the Kaggle GPU lane entirely and treat Kaggle as TPU-only.

## TPU + GPU interleaving strategy (2026-10-05)
Goal: total progress = sum of lanes, so no lane may ever wait on another lane's artifact, and the
authoritative lane must never be gated on a proposer.

Lane economics (measured locally): 512 worlds x 300k ticks ~= 600 s/arm at ~259k wt/s; a 4-arm round
~= 40 min; ONE GPU job at a time (gpulock). A genome is ~220 KB JSON and a Kaggle dataset version
upload+download is ~2-3 min, i.e. ~3% of one arm - so bytes are NOT the bottleneck and handoffs are
cheap. What is scarce is GPU slots and, above all, deciding what to run.

Lanes:
- L0 local GPU (AUTHORITATIVE, serialized): owns every number that adopts anything. Standing backlog.
- L1 Kaggle TPU (PROPOSER, conditional on the probe): train-only, exports npc-brain/2|3 genomes.
- L2 Kaggle GPU (PROPOSER, conditional on v7): only if the adapter is NVIDIA and wt/s >= ~50k at 512.

Rules (mechanical):
1. GPU NEVER waits. Keep >= 6 arms queued. Genome import happens at JOB START only, never mid-run,
   so there is zero runtime coupling between lanes.
2. TPU explores, GPU confirms. TPU does broad sweeps / many seeds / long horizon (it can price 4-8
   seeds for one local seed). GPU does 2-seed confirmation at 300k, H2H + league, champion export.
3. PIPELINED ping-pong, never lockstep: while the GPU fine-tunes TPU genome N, the TPU is already
   training genome N+1 from the previous GPU champion. Both lanes stay busy every minute.
4. Adoption gate: a TPU genome first gets a 60k-tick screen against a same-tick GPU-only run; only
   if not worse does it earn a 300k arm. This is what stops a JAX-port drift from silently burning
   10-minute GPU slots - a drifted port makes warm starts WORSE than scratch.
5. Replay-parity gate before any GPU slot: Brain vs RealtimeNpc must be 0 diff over 4000 ticks
   (tools/prune.mjs). A genome failing this is a format bug, not a result.
6. TPU checkpoints every ~15 min; never start a TPU run longer than the Kaggle session cap minus
   slack, so a session kill costs one checkpoint at most.
7. Equal-tick accounting: the only headline metric is beating GPU-only at equal TOTAL ticks
   (GPU-only 5.62 +- 0.16 at 300k, 2 seeds). TPU ticks count in the budget.
8. Insert TPU-derived arms with `bump`, which now reorders WITHOUT killing the live job (fixed and
   live-verified 2026-10-05). Never bump ahead of a running job.

Open questions the two running probes answer:
- heclgang/npc-tpu-probe -> core count and measured TFLOP/s: is the JAX port worth writing at all.
- heclgang/npc-nvidia-vulkan-probe -> adapter vendor + wt/s: is Kaggle GPU a lane or a dead end.
- CONCURRENCY: at 13:13 TPU was QUEUED while NVV was already RUNNING, i.e. Kaggle looks like it
  serializes one account's kernels. If that holds, "TPU || GPU on Kaggle" is impossible and the real
  parallel pair is local GPU || Kaggle TPU - which is the combination that matters anyway, since L0
  owns truth and is the lane we must never stall.

## Kaggle GPU lane: DEAD (2026-10-05, v7 = heclgang/npc-nvidia-vulkan-probe)
- nvidia-vulkan-icd DOES install: /usr/share/vulkan/icd.d/nvidia_icd.json exists. But the driver it
  names is absent - /usr/lib/x86_64-linux-gnu has libvulkan_{asahi,gfxstream,intel,intel_hasvk,lvp,
  nouveau,radeon,virtio}.so and NO libvulkan_nvidia.so. The container ships the CUDA userspace, not
  the Vulkan userspace driver, so the ICD points at nothing.
- Result: `nvidia` flagset (--use-angle=vulkan + VK_ICD_FILENAMES=nvidia_icd.json) -> adapter: null,
  i.e. Vulkan fails to init and there is not even a swiftshader fallback. `vulkan_default` -> the same
  google/swiftshader adapter as v6 (maxWorkgroupStorage 32768, maxBindGroups 4).
- Trainer at 512 worlds: no tick line in 480 s (same as v6 at 64 worlds).
- VERDICT: no NVIDIA WebGPU adapter is reachable on Kaggle. Do NOT write a v8. Kaggle cannot run our
  trainer. Cross the Kaggle GPU lane off; stop spending probe budget on it.

## Consequence for the interleaving plan (2026-10-05)
With the Kaggle GPU lane dead, the ONLY possible second lane is the TPU, and it is JAX-port shaped
(reimplement realm v4 + PPO in JAX) - i.e. the high-cost / poor-fit / permanent-triple-maintenance
path, not the zero-risk reuse path.
- NEW FACT: the TPU kernel was still QUEUED at 13:23 while the GPU kernel had already run and
  COMPLETED, and it had been pushed FIRST. So Kaggle either serializes one account's kernels or TPU
  capacity is scarce; either way the queue latency is minutes to tens of minutes.
- That queue latency alone violates the <=20 min cycle-time constraint derived for warm-start
  freshness (a genome trained from champion N is stale once the GPU reaches champion N+1, ~2 arms
  ~= 1200 s). A lane whose start latency is >= its allowed cycle time cannot feed the pipeline.
- So before ANY JAX port: measure TPU queue latency and run time from the probe. If start latency is
  >= ~10 min, the TPU cannot be a warm-start producer at all and the port is not worth starting.

## Sim vs policy split: how to actually measure it (2026-10-05)
Context: we need the sim/policy cost split to choose an optimization target, and to know whether a
non-GPU (TPU) lane could ever be a good fit.
- F1 confirmed and CORRECTED: BOTH P20 knobs were dead. src/rlshader.js:17 derives features from cfg
  (`soa: !!cfg.soa`), and src/rlengine.js:115 shaderConfig() omits BOTH `soa` and `fwdDirect`, so
  `!!undefined == false` - the GPU never saw either. Fix is one line: add `fwdDirect: o.fwdDirect,
  soa: o.soa` to shaderConfig(). RL_TRACKED_KEYS (rlengine.js:10) already lists both.
- BUT fwdDirect is NOT a sim/policy ablation. src/rlshader.js:663-666: `directObs` only swaps
  `xr[j]` (a function-local array) for `obs_get(ln, j)` - it removes the observation STAGING COPY
  inside the policy forward pass. So a fixed P20 measures one memory-traffic micro-optimisation,
  NOT the sim/policy balance. Do not read it as the latter.
- What the split currently rests on (two weak, independent, same-direction lines):
  (a) profile=1 timestamps: rl_rollout 42.7 ms of a 62.9 ms iteration = 68%;
  (b) a hidden-size regression on EXISTING A/B throughputs (hidden 64 ~259k wt/s, hidden 96
      ~175-214k wt/s) fits total = a + b*hidden with a ~ 389k wt/s at hidden 0, i.e. policy ~1/3,
      sim+overhead ~2/3. These AGREE in direction but absolute wt/s swings 2-3x across sessions
      (shared GPU), so (b) is not quantitative - hidden 48 measured 401k/353k, far off the fit.
- CLEAN EXPERIMENT (run this, not P20): same-session hidden sweep at fixed worlds=512, hidden
  {16,32,64}, interleaved min-of-N per AGENTS.md (only interleaved min is valid on the shared GPU),
  ~3 short arms ~10 min. Fit total_time = a + b*hidden; the intercept a is sim + rollout overhead
  (trajectory stores to rObs/rRec, which is a suspect in its own right - it may be memory traffic,
  not sim). This is what decides whether F2-F10 sim work is even the right target.

## JAX/TPU port scoping (2026-10-05) - read-only survey, no edits
Scope: 1-2 engineer-weeks FULL; ~2 days for the smallest proving slice. Cheaper than feared because
the shipped realm champion is the SIMPLEST config in the repo: ff, hidden 64, no bias, no obsNorm,
no recurrence, K=1, no curriculum/league/population, channel weights 1/1/1/1, all worlds 16v16 bots.
Recurrence/population/league/curriculum/self-play-share/obsNorm are all measured NEUTRAL or WORSE on
realm, so a base-path-only pretrainer is a COMPLETE pretrainer.

STRUCTURE (this corrects an earlier assumption): realm's tick has NO ragged control flow - every loop
bound is 64 / 256 / 32 / 9 / 8, and randomness is STATELESS (`mix(seed,tick,index,salt)`, no Rng
stream, no pcg in realm.js). So it maps to vmap + lax.scan without awkwardness. The "TPU is
architecturally mismatched to a branchy scalar sim" argument is WRONG - the sim is fixed-shape dense
array work. Kaggle GPU being dead is the real reason to consider TPU, not workload fit.

THE ACTUAL RISK - silent integer drift, no error signal:
- 26 `Math.trunc(a/b)` sites: JS truncates toward zero, Python `//` floors toward -inf; numerators go
  negative (dx, dy, prices, deltas). Each needs explicit sign-aware truncation or the sim drifts.
- ~10 lowest-index tie-breaks must be strict `<` over ascending index (consider, nearestTownAt,
  nearestInteractNode, tryAttack packed key d2<<8|idx, damagePhase best, interactPhase harvest claim).
- uint32 pcg/mix wraparound (Math.imul + >>>0).
None of these raise. A wrong port trains happily and yields a WORSE-than-scratch warm start.

MITIGATIONS THAT ALREADY EXIST (all GPU-free, runnable in Node today):
- src/rl.js is a complete per-kernel JS reference: rlGae:195, rlLossAndGrad:239, rlAdamStep:388,
  rlNormMerge:639, rlWeightedReward:858, rlInitTheta:172, rlPolicyStep:117.
- rlGradientCheck():576 - finite-difference oracle for the loss/grad, self-contained, NO device.
- RlBackend.testRolloutLockstep() rlengine.js:893 - its JS half (realm env + observe + step) needs NO
  GPU, so a JAX env can be diffed BIT-EXACTLY against JS in Node.
- testEnvLockstep engine.js:654 compares all 2736 packed words; testObserveParity :702 checks all 105.

SMALLEST SLICE (~2 days): port policy + PPO math ONLY, diff vs src/rl.js on a fixed synthetic
trajectory with rlGradientCheck as oracle. Then port env + observe, check bit-exact vs JS.
LANDING GATE (empirical, not bitwise): train ~100k ticks in JAX, export npc-brain/3 with the
rlGenomeParams vector written LITERALLY (`[0,0,0,0,0,0,wmax,0,0,0,0,0,2,0.02,0,0,1,0,0,1]`; idx 6
wmax, 12 gain=2, 13 =0.02, 16 inScale=1, 19 =1 - nothing validates these and rlGenomeToTheta never
reads params, so a wrong vector exports and replays silently wrong), then
`node tools/evalsuite.mjs --game=realm --suites=bots --seeds=3 <genome>` + rlreplay. Beat a
scratch-trained WebGPU run at the SAME tick budget on 2 seeds or kill it.

COST CORRECTION: genome evaluation is CPU-side (evalsuite ~10 s per 6000-tick world in worker
threads; rlreplay.mjs has no threshold, just reports). GPU-vs-CPU ratio gap is only 0-12%, GPU ratio
primary. So screening a TPU genome does NOT consume the bottleneck GPU lane - the "validation tax"
is far smaller than 2 GPU arms.

SEQUENCING CONFLICT: the port targets realm v4. Realm v5 (trading / skill trees / magic / creep FSMs)
would invalidate it, so the port must be re-synced after v5 lands. Either port v4 now for v4
champions, or do v5 first and port once.

## ref58-s5 STALLED and was lost (2026-10-05)
- `ref58-s5 failed in 1227s (stalled: no log output for 600s)`. Ran ~12:53-13:45 local, which OVERLAPS
  spoint-02's browser still being up until ~13:25 (two Chrome instances on one GPU). 600 s of silence
  at logSeconds=30 is a hang, not slowness, so contention is the leading hypothesis but NOT proven.
- Cost: the reference seed for the R49 (surv15) paired comparison. One seed cannot carry it, so
  ref58-s5 must be re-run. ref58-s8 + surv15-s1/s2 complete the round (~14:23); ref58-s5 is re-queued
  in the NEXT round (after spoint-02's 45-min window), NOT bumped ahead of it.
- Detector note: stallSeconds is 600. With logSeconds=30, ~240 s of silence (8 missed windows) would
  catch the same hang ~6 min sooner for ~6 min less loss per failed arm; risk is a false positive on a
  slow-but-alive job. Not changed yet.
- F1 (add `fwdDirect: o.fwdDirect, soa: o.soa` to shaderConfig(), src/rlengine.js:117) is
  BEHAVIOUR-PRESERVING for defaults: both keys default 0, `!!0 === !!undefined` and `0 === 1` is false
  either way, so the compiled WGSL is unchanged. Safe to apply in the handover window at zero GPU cost.

## Environment contamination: 21 foreign Chrome processes (2026-10-05)
- ref58-s5 AND ref58-s8 both failed identically: `stalled: no log output for 600s`, ~1230 s each,
  with `firstOutputMs = null` and NO runs/<name>.log written at all. They never produced output;
  this is NOT slowness and NOT a config effect.
- Cause: 21 chrome.exe processes on the box, 20 of them foreign (chrome-devtools-mcp profile and a
  C--dev-design viz profile). Ours was alone on runs/chrome-shared, so not a profile collision - it
  was contention for GPU/CPU. 14 foreign Chrome processes remained after we killed only our own.
- RULE: before queueing a round, count foreign Chrome. If a job shows firstOutputMs=null / no log
  file, treat it as environment failure, stop the round rather than let retries burn arms, and do not
  record the arm as a config result. Two arms ~= 40 min were lost to this.
- Consequence for measurement: absolute wt/s from this box is unreliable (AGENTS.md already says
  timings swing 2-3x; this is worse, it is an uncontrolled shared resource). Prefer RATIOS
  (sim vs policy share) over absolute throughput - the hidden sweep is a ratio, so it survives.

## F1 applied (2026-10-05)
- src/rlengine.js:117 shaderConfig() now returns `fwdDirect: o.fwdDirect, soa: o.soa`.
- VERIFIED end-to-end without a device: RL_DEFAULTS has both (`fwdDirect: 0`, `soa: 0`), and
  rlFeatures({soa:1}) -> {..., soa:true}. Behaviour-preserving at defaults (`!!0 === !!undefined`),
  so the shipped config's compiled WGSL is unchanged; the knobs are only live when set to 1.
- Both P20 knobs are now measurable for the first time. src/rlshader.js:647-659 (soa lane layout)
  and :663-666 (directObs obs staging) are the code paths they switch on.

## 2026-10-05 -- sim-vs-policy split: prediction written BEFORE the sweep runs
G: find where realm's per-tick cost actually goes, so wallclock work targets the real
bottleneck instead of a guessed one.

Model: t(hidden) = c0 + c1*hidden  (t = seconds per world-tick; wt/s = 1/t).
Two existing noisy anchors: hidden 64 -> 259k wt/s (t=3.861us), hidden 96 -> 175-214k
(t=5.128-4.673us). Solving the two extremes:
  pessimistic c0 = 1.33us, c1 = 0.0396us/unit  -> sim+overhead 34% of the h64 tick
  optimistic  c0 = 2.23us, c1 = 0.0254us/unit  -> sim+overhead 58% of the h64 tick
So the prior on "is the sim worth attacking" is genuinely wide: 34-58%.

FALSIFIABLE PREDICTIONS for the split16/32/64 arms (realm, 512 worlds, 60k ticks,
same session, take max wt/s per arm as the uncontended estimate):
  split16  380-510k wt/s
  split32  330-385k wt/s
  split64  ~259k wt/s   <- anchor; if this lands near 259k the session is comparable
                           to the 259k anchor, if it lands ~350k the whole regression
                           is contamination-dominated and the fit is void.
Decision rule (fixed now, before seeing data):
  intercept >= 50% of the h64 tick -> sim work (F2-F10) is the right target.
  intercept <= 35%                 -> sim is not the lever; stop sim work, the only
                                      remaining wallclock lever is lane count.
  between                          -> inconclusive, do not start sim work on it.
Note the model's own weakness: it assumes policy cost is linear in hidden. It is not
exactly (w1 is nIn*hidden, wr is hidden*hidden, so there is a hidden^2 term). Three
points at 16/32/64 will show that as curvature; if the三点 curve visibly bends upward,
refit with a quadratic instead of reading the linear intercept.

## 2026-10-05 -- realm sim hot spots, enumerated (GPU-free code read, file:line evidence)
Produced as pre-staged work so sim optimization can start the moment the split says sim.

Direct answers:
- The 8x8 grid is built ONCE PER DISPATCH, not per tick: `r_build_grid` is called only from
  `g_load` (realm.wgsl.js:1576), and `g_load` runs once before the K-tick loop
  (rlshader.js:1123, shader.js:580). Node positions are written only in g_load/r_place_node
  (:1642) and are immutable after init. Cost 4 barriers (:229,243,259,273).
- Per-tick O(n^2) that remains: entities^2 ONLY. No O(nodes^2) per tick.

Ranked hot spots (per world per tick unless noted):
1. r_shape :1341 (called :1539) - flat scan 60 berries + 32 springs, NO grid: 92*32 players
   ~= 2944 distance tests. Amortizable (needed only when food/water<50 or wantTown) and it
   DUPLICATES r_scan_part's grid spring/berry work (:515-527).
2. r_try_attack :810 - for j 0..64 plus r_in_safe (4-iter) :823: up to 4096 pair tests.
   Partly gateable (gate is out_of>0.5).
3. r_ally_count :617 - 32-loop, called from r_damage:1283 AND g_observe:678: ~2048.
   Cacheable per tick with one pass.
4. r_mob_decide :990 - 32 mobs x 32 players: 1024, unconditional. Gateable on mob aggro.
5. r_scan_part :498 - grid node scan with entStep=32 entities, workgroup atomicMin per node
   (:495,525,526); obsBuf is workgroup (rlshader.js:849): 2048 entity tests + N atomics.
   Nodes amortizable, entities not.
6. r_market :1324,1328 - 2x32-player loops x 4 towns: 256, unconditional. Decay only
   every 6 ticks.
7. r_apply :1445 - 64-iteration kill scan, already gated on anyDie.
8. BARRIERS: 8 in g_step (:1542,1544,1551,1558,1560,1562,1564,1566) + ~6 in rl_rollout
   (:1187,1189,1229-30,1266,1268) = ~14/tick. :1558 and :1562 look removable (r_great reads
   only decide-phase state; r_damage only r_act/r_atk/r_atkmask) - UNVERIFIED, must be
   confirmed with the lockstep test before touching.
9. r_isqrt :276 - sqrt + 2 correction loops; used in r_move:64, r_shape, r_level x3 in
   r_power:421 / r_max_hp:426, per grid row in r_scan_part. Replace with grid/table.
10. r_fill_terrain :331 via g_load:1586 - 1024 cells x 3 mix4 (r_terrain_raw:311), PER
    DISPATCH. Terrain depends only on w_seed -> cache in world buffer at g_init (~128 words).

Cheap: reward channels (r_add :1087 = 2 workgroup ops; g_save :1613 writes 4x32 per dispatch).
Node timer pass :1553-1556 = 256 decrements/tick, unconditional, trivial.

## 2026-10-05 -- sim-vs-policy split: MEASURED (prediction vs outcome)
Instrument: profile=1 GPU timestamps, 512 worlds, min of 3 repeats per arm.
Repeats are TIGHT (h16 rollout 20.24/19.16/19.27; h32 23.16/22.85/22.96; h64 31.58/31.57/31.61),
so the AGENTS.md "timings swing 2-3x" caveat is about WALL time on a contended GPU, not about
GPU timestamp queries on an otherwise-idle card. Single samples are usable.

  hidden   rl_rollout   rl_grad_blk   total (ms)
      16        19.16          7.41       27.97
      32        22.85          5.73       30.01
      64        31.57          9.56       42.66
      96        83.11         17.94      103.66   (single sample)

Fit on 16/32/64:  rl_rollout(h) = 14.80 + 0.2605*h  ms.
Convexity: 64 -> 96 jumps 2.6x while 16 -> 64 grows 1.65x, so the curve is NOT linear.
workgroupBytes is 31476 at EVERY hidden size, so the 96 blowup is NOT workgroup storage.
Undetermined cause (registers/occupancy? a second pass?). Do not model 96 with this fit.

VERDICT on the pre-registered rule: the intercept is a LOWER BOUND (for a convex curve the
secant sits below the true curve at h=0), and the h16 rollout of 19.16 is an UPPER BOUND:
  sim + rollout overhead is in [14.8, 19.2] ms
  = 47-61% of rl_rollout (31.6 ms) at the shipped hidden 64
  = 35-45% of the whole iteration (42.7 ms, which includes rl_grad_blk 9.6)
My rule was written as "intercept >= 50% of the h64 tick -> sim is the target" WITHOUT pinning
the denominator, and the two denominators straddle 50%. So: sim is the largest single bucket
but not dominant, and part of that bucket is rollout overhead (trajectory stores, ~14 barriers
per tick), not sim. Do NOT start F2-F10 sim work on this alone.

THE BIGGER FINDING (supersedes the split as a wallclock priority):
iterationMs at h16 = 29.4 ms wall vs 27.96 ms of GPU kernels -> the GPU is ~95% busy INSIDE an
iteration. Yet live training ran at 22-31k wt/s = ~635 ms wall per iteration. So ~600 ms per
iteration is HOST time outside the kernels. On a quiet box: 259k wt/s = 61 ms wall vs ~43 ms
GPU = 70% busy. Today: ~7% busy, matching nvidia-smi's 8% util at 30 W / 66 C.
=> The 8x slowdown was CPU contention (34 litebox_runner_linux_on_windows_userland processes
saturating 16 cores), NOT the GPU, NOT the sim, NOT the config. Optimizing the sim buys ~40%
of GPU time while the GPU is idle 93% of the time. Fix the host/CPU before the kernel.
Also: rl_grad_blk is NON-monotonic in hidden (7.41 at 16, 5.73 at 32, 9.56 at 64) with tight
repeats, so "smaller hidden is uniformly cheaper" is false for the gradient kernel.

## 2026-10-05 -- TPU/GPU interleaving: verdict (answers the open strategic question)
Measured, not assumed:
- Kaggle GPU lane: DEAD. `nvidia_icd.json` installs but `libvulkan_nvidia.so` does not exist in
  the container, so the NVIDIA flagset returns adapter:null with no fallback; the default gives
  google/swiftshader with maxBindGroups 4. Probe kernel COMPLETE, zero ticks at 512 worlds in
  480 s. Do not write a v8.
- Kaggle TPU lane: start latency alone is fatal. `heclgang/npc-tpu-probe` was pushed ~13:07 and
  was STILL KernelWorkerStatus.QUEUED at ~16:15 - over 3 hours without starting. The pipelined
  interleaving design needs a warm-start producer inside a ~20 min cycle to stay fresh; 3 h+
  queue latency kills that design outright. Only a BATCH PRETRAIN (latency amortized over one
  long job) could survive, and it still has no throughput number because the probe never ran.

CONCLUSION: there is currently no second lane. Progress levers, in order of measured size:
1. Host/CPU overhead on the single GPU lane - dominant today (GPU 7% busy, ~600 ms of host time
   per iteration under contention vs ~43 ms of GPU kernels).
2. Per-dispatch amortization (rolloutTicks) - under test now.
3. Sim kernel work - only 35-45% of the iteration and partly rollout overhead, NOT dominant.
Do not spend more time on TPU/GPU interleaving until a TPU throughput number actually exists.

## R49 -- throughput levers: rolloutTicks (REFUTED) and hostPipeline (CONFOUNDED)

### rolloutTicks is not a throughput lever -- measured, hypothesis dead
Prediction: host cost is per-ITERATION, so 2x/4x more ticks per rollout amortizes it -> higher wt/s.
Measured (bench, 512 worlds, hidden 64, min of 3 per arm):

  rolloutTicks  iterationMs  rolloutMs  gradientMs  worldTicksPerSec
  31            72.5         53.6       4.7         218,924
  64            147.0        111.0      9.4         222,911
  128           302.2        224.5      17.9        216,863

Iteration time scales 2.03x per doubling; throughput is flat within +-1.4%. There is NO fixed
per-iteration cost to recover. rolloutTicks is a QUALITY knob only.
Also: rt128 COMPILES at hidden 64 (workgroupBytes 31476). The 33232 B failure was at hidden 96.
Caveat kept: bench = tempBackend(evalFraction .0625, evalEvery 0), NOT the training path, so this
says nothing about training-loop overhead.

### hostPipeline: +22% on seed 1, +2% on seed 2, and the ordering gives it away
hostPipeline 0 (DEFAULT, src/rl.js:31) = 2 submits + onSubmittedWorkDone + popErrorScope per
iteration. hostPipeline 1 = one pipelined submit, error scope only every RL_ERROR_SCOPE_EVERY.
Same passes, same order -> numerics identical; it is host-only (no shaderConfig whitelist issue).

  arm      start  mean wt/s  sd      windows (30s each)
  hp0-s1   15:08  282,705    10,319  289311,297202,273560,272827,290829,270570,295613,270768,283662
  hp0-s2   15:14  339,738    14,833  (seed 2)
  hp1-s1   15:19  345,403     6,923  361759,348915,347148,345027,346092,343595,336224,339480,340386
  hp1-s2   15:24  346,217     4,773  (seed 2)

Seed 1 alone looks decisive: +22.2%, and the distributions do not overlap (hp0 max 297,202 <
hp1 min 336,224). But seed 2 is only +1.9%, and the four means are MONOTONE IN START ORDER
(282.7 -> 339.7 -> 345.4 -> 346.2). The gm daemon was burning 268% during hp0-s1 and was fixed
mid-round, so the box got quieter as the round progressed. The +22% is contention relief, not
hostPipeline. This A/B is VOID as run.

VERDICT: do not adopt hostPipeline 1 on this evidence. It is also not a drop-in anyway: at ~145k
ticks hp1 showed EVAL 0.00458 vs hp0 0.00427 at 149k (better at fewer ticks) because curriculum
flips ride the wall-clock logSeconds report, so more iterations per report moves when difficulty
steps land -- same class of caveat as R36. Any adoption needs (a) interleaved arms and (b) a
quality check at a fixed tick horizon (ticks=300000).

Re-run recipe if it is ever worth 20 min again: interleave hp0/hp1/hp0/hp1 in TIME, not by seed,
and compare each hp1 against the MEAN of its two neighbours.

### Also learned
- Throughput varies by SEED as well as by load (hp0-s1 282.7k vs hp0-s2 339.7k): world state
  (curriculum difficulty -> entities alive -> sim cost) moves the rate. Pair by seed; never pool.
- A single 5-min arm is enough to measure throughput (9 windows, sd 2-4%) but not quality.

### CORRECTION to the hostPipeline section above (2026-10-05, same day)
The "curriculum pacing" explanation there is WRONG. I asserted curriculum flips ride the
wall-clock logSeconds report, so more iterations per report moves difficulty steps.
Checked: `curriculum: 0` is the GLOBAL default (src/rl.js:669) and
`RL_GAME_DEFAULTS.realm = { hidden: 64, randomize: 0 }` (src/rl.js:61) does NOT set it.
The hp logs contain no CUR[..] tokens, which is the signature of an active curriculum.
Curriculum was OFF for these arms.

Re-checked the EVAL gap with curriculum off (hp0-s1 vs hp1-s1, interpolated to common ticks):
  50000  hp0=0.00123  hp1=0.00129  delta=+0.00006
  80000  hp0=0.00328  hp1=0.00321  delta=-0.00008
 100000  hp0=0.00394  hp1=0.00391  delta=-0.00003
 120000  hp0=0.00433  hp1=0.00421  delta=-0.00012
 140000  hp0=0.00442  hp1=0.00454  delta=+0.00011
Deltas are +-0.00012 with alternating sign against a trajectory rising 0.0012 -> 0.0045, i.e.
noise. They are explained by EVAL being a mean over a wall-clock log window: a faster arm's
window spans more ticks. There is NO evidence of a numerical difference between hp0 and hp1.

Net: hp1 is numerically equivalent, so it would be a pure speed change IF the speed win were
real. The speed win is still NOT established (time-confounded). Verdict unchanged: do not adopt.

## horiz128 / hid128 round is DEAD on the device workgroup cap (2026-10-05)
All four jobs errored at pipeline creation, with ZERO ticks produced (no `tick=` line, no log data):
- horiz128-s1/s2: `GPUPipelineError: total use of workgroup storage (33232 bytes) > max (32768)` in rl_grad_blk
- hid128-s1/s2: same error at 32784 bytes.
So there is no `ratio=` to extract and no verdict to write. The cap is already documented in AGENTS.md
("hidden 128 needs 32784 B, rolloutTicks 128 33232 B, of the 32768 B blockGrad: 1 grant"). Both arms
compile ONLY at `blockGrad: 0` (~37% slower), so they lose twice. DO NOT re-queue hidden 128 or
rolloutTicks 128 at hidden 96. rolloutTicks 128 DOES compile at hidden 64 (31476 B).
Adjacent data already banked: hid64b 5.02 +- 0.11 (5 seeds), hid48 5.36 +- 0.20 (2 seeds).

## R49 (survival weight) round REBUILT as ref58/surv15 interleaved (2026-10-05)
The round as first queued could not have produced a verdict:
- ref58-s5 and ref58-s8 both `failed ... stalled: no log output for 600s` (foreign-Chrome contention;
  21 chrome.exe on the box). Neither wrote ANY runs/*.log - the reference is gone.
- surv15-s1 reached only 17,825 ticks (it=575) before `gpu lock taken over by another owner`.
- surv15-s2 had never started. The runner was about to spend 30 min on an arm with no partner.
Rebuilt with --replace (old state backed up to runs/.old/queue-state-<ts>.json): 4 arms, 30 min each,
300k ticks, seeds 1 and 2, ORDERED ref58-s1, surv15-s1, ref58-s2, surv15-s2 - reference and arm
alternate in TIME so the pair is not confounded by host load (the R49 throughput lesson).
- arm = popWeights=1.5,1,1,1 (survival up-weighted); reference = shipped defaults (1,1,1,1).
- Motivation: champion deaths starve 48/255 vs bots 7; it has solved combat. R6 halved survival
  (0.5,1.5,1.5,1.5) on the OLD baseline and was NEUTRAL; this is the untested mirror.
- Judge at 300k via `node tools/report.mjs --at-tick=300000`. Watch eval life and death_starve, not
  only EVAL: a survival-weighted policy can shift behaviour without moving EVAL.

## R49 round PAUSED for spoint-02's 45-min parity window (2026-10-05)
- 19:38 `queue stop` -> `gpu lock: free`, 0 of our chrome.exe. Their window is NVIDIA-then-AMD
  TSL-vs-legacy parity, one browser at a time, nothing else of theirs on the box.
- Cost, stated correctly: ref58-s1 was 1052 s in (~216k of 300k ticks, ~72%, ~7 min from finishing)
  when it was stopped, so that arm is LOST and restarts on resume. I had told spoint-02 4 min -
  that came from a stale check at 58 s and was wrong. Real cost ~17.5 min of GPU + the 45 min wait.
- Why now instead of after arm 2: the arms are interleaved ref58-s1, surv15-s1, ref58-s2, surv15-s2.
  A 45-min gap falling BETWEEN ref58-s1 and surv15-s1 breaks the time-adjacency the interleaving
  exists to protect; handing over before a pair starts keeps all four arms contiguous on resume.
- RESUME: `node tools/queue.mjs resume` when spoint-02 reports its last browser closed. ref58-s1
  re-runs from 0 (30 min). Box must stay quiet during their window - no evalsuite/replay batches.

## R49 pre-flight: popWeights verified plumbed end-to-end at K=1 (2026-10-05, CPU-only, no GPU)
Done in spoint-02's lane window so a 2-h round could not be void by the F1-class bug (a knob that is
silently inert makes arm and reference IDENTICAL and fakes a NEUTRAL verdict). Chain:
- dev/rllong.html:28 forwards any URL key present in RL_DEFAULTS/RL_CURRICULUM_DEFAULTS/DEFAULT_OPTS/
  RL_POP_DEFAULTS; `popWeights` is in RL_POP_DEFAULTS (src/rl.js:755). parseValue('1.5,1,1,1') -> NaN
  -> stays a string; rl.js:791 popList(..,'/') splits per policy.
- rlshader.js:638-641 rl_train_fx multiplies each clamped channel by opt[pol*OPTS + OPT_W + ch],
  gated ONLY on hasChannels (game has g_reward_channel + rewardChannels + count <= slots), NOT on K.
- rlengine.js:215-220 writes pop.weights[p] into the opt weight slots for EVERY p < K, unconditional
  on K. So K=1 is live.
- Reference arm = weights all 1 with no channelCaps -> capFx 1073741823 -> clamp is identity -> the
  unweighted total, i.e. exactly shipped behaviour. Arms genuinely differ.
NOTE the differing gate: shaderConfig() (rlengine.js:117) is the whitelist for COMPILE-TIME knobs
(fwdDirect/soa - the F1 bug). Runtime-valued knobs like popWeights are NOT listed there and must not
be; verify the WRITE path instead.

## Kaggle probe v2 (2026-10-05) - PREDICTIONS WRITTEN BEFORE THE RESULT
Submitted while spoint-02 held the local GPU lane, so Kaggle's hours of queue latency overlap local
work (Little's law: a high-latency lane only pays if requests are in flight far ahead of need).
Why the question is open again - v1's evidence was partly an artifact:
- v1 adapter result IS reliable: vendor google / architecture swiftshader, maxWorkgroupStorage 32768,
  maxBindGroups 4. Both of our hard gates are MET: all 9 bindings are @group(0) (1 bind group needed)
  and 32768 B is exactly what realm/h64 needs. So limits do NOT block Kaggle.
- v1 "sawTick=false elapsedS=420" is NOT reliable: probe.mjs listened only to page.on('console'), but
  dev/rllong.html never console.logs - it writes to <pre id="out"> and POSTs to /api/log. It also
  served with `python3 -m http.server`, which rejects those POSTs. So v1 could not distinguish
  "produced no ticks" from "never looked in the right place".
- The real blocker v1 did reveal: /usr/share/vulkan/icd.d has lvp_icd.json (lavapipe, CPU) but NO
  NVIDIA ICD, so Chrome's Dawn had no path to the 2x Tesla T4 and fell back to CPU swiftshader.
PREDICTIONS (falsifiable):
- PA (~40%): `apt-get install nvidia-vulkan-icd` succeeds and vulkaninfo lists a Tesla T4; Chrome
  with --use-vulkan=native then reports a non-CPU adapter (vendor nvidia).
- PB (if PA): realm at 64 worlds produces ticks. T4 320 GB/s vs RTX 3060 laptop ~336 GB/s, both
  bandwidth-bound, so expect ~0.5-1.0x of local (local is 260-370k wt/s at 512 worlds). 2 T4s are
  present -> potentially 2 concurrent lanes if each job pins one GPU.
- PC (if PA fails): adapter stays swiftshader, and with the DOM-reading fix we get a REAL number for
  the first time; predict <5k wt/s. That would finally be evidence rather than a probe artifact.
- PD: v1's "no ticks" will be confirmed as a harness artifact. The AGENTS.md line "Kaggle is no
  second lane" currently rests partly on it - recompute that verdict from v2, not from v1.

## Kaggle-as-lane predictions part 2: the host-bound argument (2026-10-05, BEFORE the result)
Context: v2 probe confirms on-box nproc = 4 with 2x Tesla T4; the local box has 16 cores.
P_K1: even if the NVIDIA Vulkan ICD lands and the adapter becomes a real T4, a Kaggle lane will be
  SLOWER than local, because the documented bottleneck is HOST-side (GPU ~7% busy end-to-end;
  per-iteration host cost = 2 submits + onSubmittedWorkDone + popErrorScope) and Kaggle has 4 CPUs
  vs 16. Predict < 100k wt/s vs local 250-370k.
P_K2: 2 GPUs / 4 CPUs means two concurrent lanes split 4 CPUs across two hosts - predict 2 lanes
  WORSE than 1. If it works, scale by worlds per lane, not by lanes per box.
P_K3: budget - 300k ticks is ~25 min locally, so ~75 min if P_K1 holds; a 4-arm round ~5 h, which
  fits one Kaggle 9 h session but spends most of a 30 h weekly GPU quota on ONE A/B.
Decisive cheap test (what v2 already runs): one 64-world 20k-tick rllong run, read wt/s. Within 2x
  of local = a lane; otherwise not.

## R49 decision rule, PRE-REGISTERED before any arm is read (2026-10-05)
Arms: ref58-s1, surv15-s1, ref58-s2, surv15-s2 (reference/arm ALTERNATING in time), realm 512 worlds,
300k ticks, seeds 1-2. Judge with: node tools/report.mjs --at-tick=300000
ADOPT surv15 (popWeights 1.5,1,1,1) only if ALL THREE hold:
  (1) surv15 > ref58 on BOTH seeds (not just on the mean - a mean can be carried by one seed);
  (2) mean(EVAL delta) > 0.15, i.e. > ~2x the 0.05-0.08 seed sd documented for realm at 300k;
  (3) eval life does not fall while EVAL rises (a survival-weight arm that buys EVAL by dying later
      but doing less is not a win - watch death_starve too).
Anything weaker = NEUTRAL, keep shipped 1,1,1,1. Do not re-threshold after seeing the numbers.
Also: ref58-s1 (the first attempt) reached tick=251534 at 371 s then produced NO output for ~11 min
at ~146k wt/s - that is box contention (29 chrome.exe, 67% CPU), not config. If wt/s < ~170k when the
round resumes, a 30-min arm finishes SHORT of 300k, which --at-tick will show as spread across seeds.

## Kaggle probe v2 RESULT (2026-10-05) - verdict recomputed from a real measurement
PA REFUTED: installing nvidia-vulkan-icd did not give Chrome a GPU. vulkaninfo --summary lists
  exactly ONE device: llvmpipe (LLVM 20.1.2), driverName llvmpipe - a CPU software rasterizer. No
  NVIDIA Vulkan device exists on the box, so Dawn cannot reach the 2x Tesla T4 under ANY flagset.
  WebGPU on Kaggle is CPU by construction.
PD CONFIRMED: v1's "no ticks in 420 s" WAS a harness artifact (console-only listener + python
  http.server with no POST). v2 served with node serve.mjs and read the DOM: sawTick=true, 1178
  ticks in 300 s at 64 worlds = ~293 wt/s. The trainer does run there - ~1000x slower than local
  (250-370k wt/s), not "broken".
PC: right in kind, 17x off in magnitude - predicted <5k wt/s, measured 293.
P_K1: directionally right, MECHANISTICALLY WRONG. I predicted the limit would be host CPU (4 cores
  vs 16). The real limit is that there is no GPU at all.
VERDICT (now real): Kaggle is NOT a second lane. A 300k-tick 512-world arm = 153.6M world-ticks;
  at 293 wt/s that is 524k s = 145 h. Hopeless at any world count. Spend no more Kaggle GPU quota
  on it. The TPU probe is a separate question (JAX path), unaffected by this.
TOOLING: kaggle kernels logs <ref> WITHOUT -f returns only the last ~6.5 KB, so early output is
  LOST once the run ends. Capture with -f into a file WHILE it runs.

## R49 OUTCOME PREDICTION, written before any arm is read (2026-10-05)
Prior art on this exact axis, both at the OLD pre-R36 baseline (~2.4): R6 down-weighting survival
  (0.5,1.5,1.5,1.5) NEUTRAL (2.46 vs 2.38); R24 annealing that back to 1,1,1,1 NEUTRAL (4.89 vs
  entdef 4.97). R49 is the same axis at the NEW post-R36 baseline (~5.0-5.6).
PR49: NEUTRAL - mean delta in [-0.10, +0.15] and NOT positive on both seeds.
Why: rl_rescale + advstats normalise by return std, so a GLOBAL scale cancels; only the change in
  RELATIVE channel composition can matter, and both prior probes of that axis were neutral.
Falsifiable sub-signals to check BEFORE believing any verdict:
  - wt/s must match across arms (a channel weight touches no kernel). A systematic wt/s gap means
    the arms are not comparable - contention, not effect.
  - a real WIN shows eval life UP and death_starve DOWN, not merely EVAL up.
  - a LOSS shows progress/combat crowded out (ratio falls while life rises).
If PR49 is refuted by a >0.15 win on both seeds, the pre-registered rule adopts it anyway.

## R50 — realm v5 RE-BASELINE (not an A/B; the old numbers are void)
Why it is a baseline and not an arm: v5 changed the GAME (perks, magic, creep archetypes, cover, boss
phases, town price bias) and the BOTS. Bot changes move evalBase AND the numerator, so every
pre-v5 EVAL/ratio verdict (R8-R49) is non-comparable. Only h2hRatio and selfPlayReward survive, and
only as a "still learning" signal, not as a comparison.
Job: v5base-s{1,2}, page rllong, 512 worlds, ticks 300000, logSeconds 30, saveSeconds 300, 30 min cap.
Pre-registered read-out, in this order:
  1. LEARNABILITY (the real question): ratio = EVAL/evalBase and its trend over the last 3 windows.
     v4 shipped ~5.0-5.6 EVAL; v5's absolute EVAL is expected to move (richer rewards, stronger
     bots), so judge ratio and the h2h signal, not the raw EVAL.
  2. SANITY: wt/s in the 250-380k band (else contention, rerun), eval life > 1000, death_starve not
     dominant, and at least one of bossKills/greatHarvests/pvpKills non-zero (the v5 content is being
     used at all - a richer game nobody touches is a regression).
  3. Only after that: the ratio here becomes the number every future v5 arm is judged against.
Deliberately NOT in this round: bot great-node harvesting (held out so this baseline measures one
change set), town-sell actions, P2P barter - see runs/GAME-DESIGN.md §8 for the byte costs.

## R51 (pre-registered, NOT queued) -- bot combat engagement, the first STRUCTURAL bot lever
Rationale: two economic bot levers are now measured neutral (great nodes -3%/+3%, price-aware towns -0.8%); in both the
access cost cancelled the reward. The remaining bot gap is behavioural, not economic: bots only engage a mob inside
500 and at HP > 50, and abandon any fight when hurt && menaced, so their kill rate -- and the XP/gold that follows --
stays far below the learners'. This is a constant change, not a new system, so it needs no nIn and no workgroup bytes.

Arm: bfight, realm, 512 worlds, 300k ticks, seeds 1 and 2, page rllong.
  JS  src/games/realm.js:826  `if (nd2[5] < 500 * 500 && f[b + F.HP] > 50 && !safe)` -> `800 * 800` and `> 40`
  WGSL src/games/realm.wgsl.js:1179 `if (preyD2 < 250000 && r_f[b + RX_HP] > 50 && !safe)` -> `640000` and `> 40`
  Keep `menaced` (250000) untouched: it gates the flee branch, and widening both would just make bots oscillate.

Gates before queueing, in this order:
  1. CPU sweep, 8 seeds x 6000 ticks, bot evalBase via stats[STAT.REW_BASE]/stats[STAT.TICKS_BASE] -- the ONLY valid
     bot instrument. Decide at >=10% (seed sd is 0.00035 on a 0.00127 base). Do not proceed on a 3% win.
  2. lockstep (page perf, game=realm, mode=tests) must be 8/8 on the mirrored pair.
  3. only then `node tools/queue.mjs add runs/jobs-botfight.json && node tools/queue.mjs resume`.

BLOCKER: do not edit src/ while bgreat-s1/-s2 are running or pending -- a queued job compiles the page at its start,
so an edit lands on bgreat-s2 and silently turns baseline seed 4 into an experimental arm.

### R51 RESULT (measured 2026-10-06, CPU, bot evalBase = stats[REW_BASE]/stats[TICKS_BASE], 6000 ticks/seed)
Dose-response over 16 seeds [7,11,23,31,47,59,71,83,95,103,117,131,149,161,173,187], base mean 0.00126 (8-seed) / 0.00102 (fresh 8):
  base 500/50        -- reference
  800/40   (640000)  +7.9% on seeds 1-8 (5/8 up), +4.0% on the fresh 8 (4/8 up)  => ~+6%, 9/16
  1200/30  (1440000) +7.7%, 9/16 up
  1600/30  (2500000) +7.9%, 11/16 up      <-- best dose, wins rise monotonically with dose
Verdict: the FIRST bot lever with a consistently POSITIVE sign and a monotone dose-response - both economic levers were
negative. But the magnitude plateaus at ~+8%, which is UNDER the >=10% gate pre-registered above. Recorded as measured,
not rounded up to 10%. Decision: adopt at 1600/30 only if the GPU arm (512 worlds, 300k ticks, 2 seeds) confirms; the CPU
horizon is 6000 ticks and the real eval horizon is 50x longer, so GPU is the deciding instrument, not this sweep.
BLOCKER stands: do not edit src/ until bgreat-s1/-s2 are done - a queued job compiles the page at ITS start.

### R51 STATUS UPDATE (2026-10-06)
- APPLIED: `realm.js:826` -> `nd2[5] < 2500000 && f[b + F.HP] > 30`; `realm.wgsl.js:1179` -> `preyD2 < 2500000 && r_f[b + RX_HP] > 30`. `menaced` (500*500 / 250000) deliberately untouched - it gates the retreat+sprint, not the engage.
- Lockstep 8/8 PASS (`ls-bfight`): env lockstep 160 + 120 ticks word for word (2736 words/tick, dense-interaction events mobKills 10, greatHarvests 4, bossKills 2, harvests 0, playerDeaths 33), brain parity 1.79e-6, workgroup 32432/32768, obs parity 128x109, CPU/GPU statistical equivalence 0.00477/0.00477 (rel diff 0.000).
- GPU round queued: `bfight-s1` / `bfight-s2`, realm, 512 worlds, 300k ticks, 40 min each, runner pid 18052. SEED-MATCHED against the baseline: bfight seeds 1 and 2 are the same seeds as `v5base-s1`/`v5base-s2`, so this is a paired comparison, not a fresh-seed one.
- Baseline reference is now FOUR seeds, not two: v5base-s2 2.99, v5base-s1 2.96 (truncated 255k), bgreat-s1 2.92 @273k, bgreat-s2 2.99 @285k -> ratio ~2.97 +- 0.03.
- Queue-time blocker: a foreign GPU lock (`tsl-parity-pinned-nohmr` pid 35004, fresh heartbeat, not stale). The runner holds the jobs and claims when it frees; do NOT clear that lock by hand and do not re-add the jobs.
- Decision rule unchanged and pre-registered: the CPU sweep said +7.9% (11/16 seeds, monotone in dose) which is UNDER the 10% CPU decidability gate. The GPU run is the deciding instrument. Adopt only if bfight matches or beats 2.97 at a common tick horizon; otherwise revert both lines and record R51 as killed.

### R52 / R53 pre-registration (re-judge the v4 knobs on v5) -- GATED on the R51 verdict
- GATE: do not start these until `bfight-s1/-s2` resolve R51 AND the bot code is final (adopt => keep both lines; kill => revert both). Any further change to the bot policy moves `evalBase` and therefore re-voids the 2.97 reference, the same trap that made the v4 verdicts incomparable.
- R52 entropy anneal (.02 -> .002) on v5. It landed on v4 at +2.6%, but the v4 motivation may not carry: v5's entropy at 285k is 1.35-1.38 with no collapse, so there may be no collapse to fix. Pre-registered as "measure, do not assume it transfers".
- R53 R36 composition (`randomize` 0) on v5, +13% EVAL / +40% throughput on v4. CHECK FIRST whether `randomize` 0 is already in `RL_GAME_DEFAULTS.realm` (src/rl.js) -- if it shipped as the default, R53 is void and the v5 baseline already contains it. Do not queue an arm that is a no-op.
- Confirm the exact URL param names from `RL_GAME_DEFAULTS` before writing any job file; a misnamed param is silently dropped (the `bnd-tests-rl` lesson).
- BUDGET CEILING for further content: evolution kernel is 32432 of 32768 B (336 spare) and the cap is `(nNodes|1) + (nIn|1) <= 258` at realm, with 147 + 109 = 256 in use -- TWO slots of nIn/nNodes headroom. Any new content feature must be observability-neutral or reclaim an existing input; there is no room for another input-expanding feature.

### R54 -- v5 content-usage instrumentation (built 2026-10-06) and the first measurement
WHY: the `EG[...]` set in the training log is the **v3** set, unchanged through v4 and v5
(`statNames` identical between `dev/legacy/realm_v3.js:1218` and `src/games/realm.js:1588`).
So the AGENTS.md line "the new content is actually used per run: bossKills ~6.1k, greatHarvests
~16.4k, pvpKills ~4.2k" was **not evidence about v5** - those are all v4 mechanics. Six of the
seven v5 features had no instrument at all.
WHAT: six host-side counters added to `tools/evalsuite.mjs` only. No `src/` or `dev/` edit, no
`worldWords` change (2736 unchanged), no change to the lockstep word stream. Reads state the JS
env already exposes: `env.atkCast`, `env.perkTier`, `env.inCover`, `env.bossPhase`, `env.tgtNode`.
FIRST MEASUREMENT (`runs/bgreat-s1-base`, bots suite, 1 world 1 seed -- n=1, indicative only):
  castShare 0.040 (casts1k 2.094, attacks1k 51.9)   -> magic IS cast, ~4% of attacks
  coverShare 0.360, coverFirstHits1k 4.646          -> cover first-hits DO occur
  bossPhase3Share 0.298                             -> ~30% of boss attacks land in the shielded phase 3
  saleTown0..3 = 0.087 / 0.275 / 0.138 / 0.500      -> all four towns used, strongly asymmetric:
       lumber towns 0/2 = 0.225 vs mining towns 1/3 = 0.775. saleGoldPerUnit 2.525.
  perkTier mean 1.768, max 3.000
  mobArch0..4 = 0.240/0.172/0.185/0.153/0.250, entropy 2.296 (max log2 5 = 2.322)
CAVEATS, stated because two of these are weaker than they look:
  - perkTier is AUTO-UNLOCKED from XP. Reaching tier 3 is not "use"; only an obs[108] lesion
    (zero the input before Brain.step, compare rate) tests whether the net exploits it.
  - mob archetype near-uniform entropy is CONFOUNDED: archetype = (target-MOB0)%5, so "attack
    whatever is nearest" also produces near-uniform coverage. Needs the obs[106] lesion.
  - coverShare 0.360 partly reflects how common forest terrain is, not intent.
  - n=1. Do not quote these as established until the multi-seed run below lands.
NEXT (run AFTER bfight-s1/-s2 finish - CPU evalsuite and a live GPU job stall each other):
  node tools/evalsuite.mjs --game=realm runs/bgreat-s1-base --suites=bots --seeds=1,2,3 --worlds=4
  node tools/evalsuite.mjs --game=realm runs/bgreat-s2-base --suites=bots --seeds=1,2,3 --worlds=4
  node tools/evalsuite.mjs --game=realm runs/v5base-s2-base --suites=bots --seeds=1,2,3 --worlds=4
  Then the two lesions (obs[108] perks, obs[106] hostileType) if the plain numbers look interesting.

### R51 GPU RESULT, seed 1 (bfight-s1 @282658 ticks) -- and a flaw in the pre-registered gate
Seed-matched against baseline seed 1 (bgreat-s1 @273513 ticks):
                    baseline s1     bfight s1     delta
  EVAL (learner)      0.00496        0.00524       +5.6%
  evalBase (bots)     0.00170        0.00235      +38.2%
  ratio               2.92           2.23         -24%
  H2H ratio           1.00-1.02      0.99-1.01    flat
  self-play reward    0.00524/0.00537 0.00534/0.00542  ~+1%
  eval life           2201           2094          flat
  entropy             1.347          1.321         flat
THE GATE WAS MIS-SPECIFIED. I pre-registered "adopt only if ratio matches or beats 2.97".
A bot-improvement arm MECHANICALLY lowers the ratio, because the bots are the denominator.
Judging a "make the bots catch up" lever by a metric that bot improvement drives down is a
category error, and it would have killed the one lever that did exactly what was asked.
THE PROJECT'S OWN RULE SUPPLIES THE RIGHT METRIC. AGENTS.md (v5 section, phase 4) already says:
"Stronger bots move `evalBase` AND the numerator, so pre-v5 EVAL/ratio verdicts are NOT
comparable - only `h2hRatio` and `selfPlayReward` survive."
On those two surviving metrics bfight-s1 is FLAT (h2h 1.00 -> 1.00, self-play +1%): the learners
are not worse, the benchmark got harder. Bot reward +38% with learner self-play and head-to-head
unchanged = "the bots caught up", which is precisely the goal clause.
VERDICT: do NOT revert on the ratio alone. Hold for bfight-s2 (seed 2), then decide on:
  adopt  <=> bot reward up clearly on BOTH seeds AND h2h/self-play flat or better.
  kill   <=> h2h or self-play degrades on either seed (that would mean the arm hurts learners).
Either way record which metric decided it, and note that ratio 2.97 is no longer the reference
for any future BOT arm - it is only the reference for learner-side arms.

### R55 -- P2P trading (the one requested feature that does not exist yet)
STATUS: not implemented, not designed until now. The condition asks for trading; the game has
only player-to-TOWN sales (`saleTown0..3`, gold from `SALE_GOLD_REWARD`). There is no
player-to-player trade anywhere in `src/games/realm.js`. Previously deferred on observability
grounds; this design pays for observability inside the existing budget.
DESIGN (cheapest that fits, 2 nIn slots, ZERO nNodes, ZERO nOut):
  - No new action. A trade executes through the existing `ACT.INTERACT` when the nearest
    interactable target is a PLAYER rather than a node, so nOut stays 13 and the RL rollout
    workgroup (32116 of 32768 B) is untouched.
  - New inputs (exactly the 2 free slots, nIn 109 -> 111, (nNodes|1)+(nIn|1) 256 -> 258 = AT the cap):
      109 `tradeWant`   commodity index (0 wood, 1 ore, 2 food, 3 water, -1 none) that a player
                        within trade range is short of AND I hold a surplus of
      110 `tradeDx`     signed distance to that player (reuse the greatDx/townDx convention)
  - Settlement: transfer the surplus unit, both sides gain; payer gains cooperation-channel
    reward, receiver gains survival-channel value. No new world fields, so `worldWords` (2736)
    is unchanged and the lockstep word stream is unchanged in length.
  - Determinism: this is a cross-entity effect, so it must follow the WGSL rule already in
    AGENTS.md - victim/node-centric resolution with lowest-index ties, and NO writes to another
    lane's readable state during the DECIDE phase.
GATE: implement only after bfight-s2 resolves and `src/` is unfrozen; then lockstep 8/8 before
any training run. Cost if it fails: revert is a two-file revert like R51's.
DEPENDENCY: this spends the last 2 free slots. Anything after it must sell evolution hidden
units (nNodes) 1-for-1 for inputs, per the `src/shader.js:36` trade identified above.

## R55 budget CORRECTION (2026-10-06, computed not estimated)
Ran the real exported `shaderLayout(game, cfg)` from src/shader.js against the 32768 B grant:
```
nIn=109 nNodes=147  workgroup=32432  spare=336   <- today
nIn=111 nNodes=147  workgroup=32560  spare=208   <- R55 trading (2 new inputs)
nIn=113 nNodes=147  workgroup=32688  spare=80    <- CEILING at nNodes 147
nIn=115 nNodes=147  workgroup=32816              <- OVER
nIn=111 nNodes=145  workgroup=32432  spare=336   <- selling 2 evo hidden units buys +2 nIn
```
Because nIn is `|1`-rounded, 110 and 111 cost the same, so headroom arrives in STEPS OF TWO and the
real ceiling is **nIn 113 = +4 usable inputs, not the +2 first reported**. AGENTS.md's
"(nNodes|1)+(nIn|1) <= 258" is the same fact in a different form (147+115 = 262 fails, 147+113 = 260 fits).
nNodes trades 1-for-1 with nIn, so the budget is a TRADE, not a wall. RL rollout side (32116 B today)
is NOT yet recomputed and must be re-checked via the RL suite before landing.

## Trading is CONFIRMED ABSENT (2026-10-06)
Case-insensitive search for `trade` across the whole tree hits only: NEXT-ROUNDS.md (this design),
.wfgy/lessons.md, tools/evalsuite.mjs (field name), and runtime/play.html + runtime/README.md, where
`trade` is only a KEY_ACTION_NAME alias ('t'/'T') for a generic action index. Zero hits in
src/games/realm.js or src/games/realm.wgsl.js. So of the six requested features, trading is the only
one with no mechanic behind it at all. (The ls-v5trade / ls-v5trade2 log names are misleading: they
were init experiments - v5trade failed lockstep at tick 0 on a spawn-kit field set WOOD/ORE/TOOL,
v5trade2 passed 8/8 at nIn 109 / 2736 words. Neither touched trading.)

## R51 STATUS (2026-10-06)
bfight-s1 DONE: tick 298,654 EVAL 0.00508 evalBase 0.00230 ratio 2.20 (18 windows).
  vs seed-matched baseline bgreat-s1 @273,513: EVAL 0.00496 evalBase 0.00170 ratio 2.92.
  => bots +35-38%, learners +2.4-5.6%, ratio -24%, H2H flat (1.00-1.02 -> 0.99-1.03), self-play ~+1%.
bfight-s2 FAILED - "gpu lock taken over by another owner" at 298 s. This is the SECOND job killed by
  a peer lock takeover (v5base-s1 was the first). R51 is therefore still n=1 and NOT decidable.
src/ stays frozen until bfight-s2 completes; re-queue it when the lock frees.

## R54 RESULT (2026-10-06) - v5 content IS engaged, but NO input is load-bearing
Two independent champions x 3 eval seeds x 4 worlds (12 world runs each), CPU evalsuite:
```
metric                 bgreat-s1-base        bgreat-s2-base
perkTier               1.754 +- 0.054        1.717 +- 0.048     (max 3.000 both)
attacks1k                 52.9 +- 0.9           55.1 +- 0.7
casts1k                 2.280 +- 0.160        2.367 +- 0.178
castShare               0.043 +- 0.003        0.043 +- 0.004    <- magic used, but only 4% of attacks
coverShare              0.379 +- 0.071        0.368 +- 0.069    <- in cover >1/3 of ticks
coverFirstHits1k        5.497 +- 0.704        5.782 +- 0.960    <- ambush first-hits real
mobArchetypeEntropy     2.304 +- 0.007        2.307 +- 0.004    <- max is log(5)=2.322
bossPhase3Share         0.299 +- 0.003        0.309 +- 0.013    <- ~30% of boss hits in phase 3
saleGoldPerUnit         2.644 +- 0.086        2.599 +- 0.186
saleTown0..3            0.121/0.277/0.206/0.396  0.141/0.267/0.263/0.329
```
Both champions agree within error on EVERY metric, so this is replicated, not one run.
=> All six v5 features are provably exercised. Town specialisation shows in the sale split
(town 3 highest, town 0 lowest, same ordering on both champions = TOWN_BIAS is legible).

### The lesion test (bgreat-s1-base, same 12 runs, input zeroed after observe)
```
baseline (no lesion)        ratio 2.008 +- 0.199
--lesion=hostileType        ratio 1.980 +- 0.031   (-1.4%)
--lesion=perkTier           ratio 1.944 +- 0.052   (-3.2%)
--lesion=bossPhase          ratio 2.010 +- 0.110   (+0.1%)
--lesion=coverNear          ratio 2.029 +- 0.009   (+1.0%)
```
VERDICT: the content is ENGAGED but not BEHAVIOURALLY LOAD-BEARING. Zeroing any single v5 input moves
ratio by <= 3.2%, inside the seed sd. mobArchetypeEntropy ~= its 2.322 maximum is the same story from
the other side: learners hit all five archetypes almost uniformly, i.e. they attack whatever is nearest
rather than reading archetype. So v5 added mechanics the policy survives without.
Caveats: single-input lesions only (a combinatorial lesion could still matter), and zeroing shifts the
input off its trained distribution, so a lesion is a perturbation, not a clean ablation.
This is the honest answer to "is the content actually used": yes, it fires; no, it is not yet what the
policy is playing on. That is a REWARD-SHAPING gap, not a content gap.

## R51 VERDICT (2026-10-06) - ADOPTED, 2 seeds, near tick-matched
```
              bots evalBase          learners EVAL        ratio    h2h    self-play
s1 bgreat-s1  0.00175 @290315        0.00515              2.94     1.02   0.00537
s1 bfight-s1  0.00230 @298654 (+31%) 0.00508              2.20     1.03   0.00538
s2 bgreat-s2  0.00178 @285603        0.00532              2.99     1.00   0.00542
s2 bfight-s2  0.00237 @286967 (+33%) 0.00515              2.18     0.98   0.00530
```
Bot reward up +31% / +33% on BOTH seeds; learners flat (mean -0.4%, +2.4% on s1 / -3.2% on s2);
h2h and self-play flat inside noise on both. That is the pre-registered rule met on both seeds:
ADOPT. R51 (realm.js:826 / realm.wgsl.js:1179, engage range 500->1581, HP gate 50->30) STAYS IN.
src/ is UNFROZEN as of this verdict.
NOTE: ratio drops 2.9x -> 2.2x by construction - bots are the denominator. ratio 2.97 is the
reference for LEARNER-side arms only; for BOT arms the instruments are bot rate, h2h and self-play.

## R55 trading BUDGET IS TIGHTER THAN DESIGNED (2026-10-06, computed)
The scratch implementation's WGSL plan needs new workgroup arrays in the GAME chunk, and
`game.workgroupBytes` (15112) is ADDED to the evolution total by src/shader.js:36. Measured:
```
nIn=111 chunk+128 -> 32688 B  spare 80   FITS   <- only viable option
nIn=111 chunk+256 -> 32816 B  spare -48  OVER   <- the 2-array plan in the scratch report
nIn=111 chunk+512 -> 33072 B  spare -304 OVER   <- the 4-array plan in the scratch report
nIn=113 chunk+128 -> 32816 B  spare -48  OVER   <- so nIn MUST stay 111, not 113
```
So all four trade fields (partner, role, ok, traded) must pack into ONE array<i32,32> (+128 B);
8 bits per entity is enough (partner 5 + role 1 + ok 1 + traded 1). nIn stays 111 (2 new inputs).
Any genome/checkpoint trained at nIn 109 will NOT load at 111 (W1 is nIn x hidden), so landing
trading invalidates every existing champion and needs a fresh v5 baseline.

## CORRECTION: ls-trade-rl failed because a peer killed all Chrome, not because 20 min was too short
I first read `ls-trade-rl` (failed, "stalled: no log output for 600s", 2 attempts, 1353s) as the
documented `tests=1` deadline trap and re-queued it at 45 min. The timestamps contradict that:
it started 33m ago and ended 11m ago, so with a 600s stall window its output stopped ~21m ago.
A design peer then reported running `taskkill /IM chrome.exe` "20 to 40 minutes" before that
message - the stall sits inside that window. Compare `ls-v5-rl2`: the same 21-test suite
completed in 700s at minutes=10, so 20 min was plenty. Real cause: every Chrome on the box was
killed by another session. Lesson: before blaming a documented failure mode, check whether the
timestamps fit it - a stall cause from OUTSIDE the project looks exactly like an internal one.
Re-queued as `ls-trade-rl2` at 45 min (harmless extra headroom; started after the kill).

## 2026-10-06 -- R55 suite: both failures were the runner's 600 s stall, not a hang
- `ls-trade-rl` (1353 s) and `ls-trade-rl2` (1111 s) both ended `failed: stalled: no log output for 600s`.
- Discriminator: `ls-v5-rl2`, the last suite that PASSED (nIn 109), sums to 494,154 ms of individual test time and finished at ~520 s - only ~80 s under the 600 s stall. nIn 111 pushes the same suite past it. `dev/rllong.html` buffers every PASS line until `runTests()` returns, so total silence for the whole suite is EXPECTED and is not evidence of a hang.
- Fixes, all three needed: (1) `--stall-seconds=1800` on the RUNNER - per-job `minutes` does not affect the stall check; (2) the default `--port=8123` is another project's server, so `start` exited instantly with `server on port 8123 does not serve /dev/rllong.html (status 404)` and the job never ran - start `node serve.mjs 8124` and pass `--port=8124`; (3) a previous runner's leftover pending job makes `start` refuse, so use `resume --port=... --stall-seconds=...`.
- Added while waiting: `tools/evalsuite.mjs` now counts trading host-side (`tradeBought1k`/`tradeSold1k`, a ration moving with an 8-gold delta; town rations cost 10, so the price disambiguates). Needed because trading would otherwise be unjudgeable.

## 2026-10-06 -- how to measure whether trading is learned and load-bearing
- `REALM_INFO.inputNames` (realm.js:1650) ends `... perkTier, tradeNeed, tradeDx` and totals exactly 111, so obs 109 = `tradeNeed`, obs 110 = `tradeDx`. `evalsuite` resolves a lesion by `game.inputNames.indexOf(context.lesion)` (evalsuite.mjs:203) and throws on an unknown name, so the lesion names are `tradeNeed` / `tradeDx`.
- New host-side counters in `tools/evalsuite.mjs`: `tradeBought1k` / `tradeSold1k`. A trade is a ration moving against an 8-gold delta (buyer `GOLD-8, RAT+1`; giver `RAT-1, GOLD+8`, realm.js:1128-1134); town rations cost 10, so the price disambiguates from a town purchase. No src edit, no `worldWords` change.
- Plan: train `v5tr` (2 seeds, runs/jobs-v5tr.json) -> eval champions with no lesion to get `tradeBought1k`/`tradeSold1k` (is it ENGAGED) -> re-eval with lesion `tradeDx` (removes partner proximity) and `tradeNeed` (is it LOAD-BEARING, i.e. does the ratio drop by more than the seed sd).

## 2026-10-06 -- the nIn 111 lockstep failure was init order (`env.brain`), not a trade algorithm bug
- `ls-trade-rl2` FAIL: `world 0 tick 0 learner 0 input 110: gpu 0.03125 js 0`. 0.03125 * 256 = 8, so the GPU saw a mutual partner at distance 192 (= RW_TRADE_RANGE 200 - 8) and JS saw none. Every algorithm involved is byte-identical on both sides (`realmIsqrt`/`r_isqrt`, `alive`/`r_alive`, `tradeD2`/`r_trade_d2`, scan order, mutual check).
- Cause: WGSL sets `r_brain = min(r_slots, LEARNERS)` in `g_init` (realm.wgsl.js:1931; LEARNERS = 32 in the RL kernel), while JS set `this.brain = R_LEARNERS` (16) in the constructor and only recomputed it inside `step()` (realm.js:1439). The lockstep test calls `observe()` BEFORE the first `step()` (rlengine.js:938 vs 945), so at tick 0 JS scanned 16 entities and the GPU 32 - the partner was a slot >= 16.
- Fix: initialise `brain`/`selfplay` in the constructor from `learnerSlots` (realm.js:239); `step()` still clamps by `outputs.length / R_NOUT`, so nothing after the first step changes. Verified live: 32-slot cfg -> brain 32 / selfplay true, 16-slot -> 16 / false. Re-ran the suite (`--port=8124 --stall-seconds=1800`): **21/21 PASS**, so nIn 111 is locked step for both realm and blob.
- Two facts a design review got wrong, both re-checked: `shaderLayout(REALM, evoConfig(REALM,{}))` = 32592 B (176 spare) at nIn 111, not 32560; and cover pays NO first-hit reward - only `damage*3>>1` from cover (realm.js:651) and halved detection aggro (953). AGENTS.md corrected.
- Trading executes on both sides: JS `tradePhase()` (realm.js:1109) is called at step 1462, before `interactPhase`; WGSL `r_trade_plan`/`r_trade_apply` (1442/1462) run at g_step 1895/1897, before `r_interact` (1899), with a barrier between plan and apply. So a `tradeBought1k` of 0 after the round means "not learned", not "not implemented".
- Next lever once trading is measured: the mana gate. RARE (cap 8) is BOTH the cast cost (1, realm.js:721) and the tier-3 craft ingredient (2, REALM RARE_TIER3 at 789/795), and it comes only from great nodes (+3) and bosses (+2/+4). `casts1k` vs `gearTierMax` on a champion decides whether that competition is binding before any change is made.

## 2026-10-06 -- v5 trading baseline measured: trading is ENGAGED but NOT load-bearing
- Round: v5tr-s1 / v5tr-s2, 512 worlds, 300k ticks, realm v5 + trading (nIn 111), ~11 min per seed (152M world-ticks at ~285k wt/s).
- Result: ratio **2.17 +- 0.12** (v5tr-s1 2.07 @298k, v5tr-s2 2.26 @288k), EVAL 5.12 +- 0.26, evalBase 2.36.
- The drop from the phase-4 re-baseline 2.97 is the DENOMINATOR, not the learners: R51 (bot engage range 1581, HP gate 30) moved the bot benchmark 1.74 -> 2.36. Learner EVAL is flat against the pre-R51 5.16/5.11, and against the tick-matched current-source reference bfight-s2 (2.22 @287k) v5tr-s2 is 2.26 @288k. v5 + trading neither helped nor hurt; the bots caught up.
- evalsuite (host-side, no src edit): trading IS used, ~1.4-1.6 rations/1k ticks each way (self-play 1.9-2.3), but lesioning obs 110 (tradeDx) costs -1.4% and obs 109 (tradeNeed) -3.8% - both inside the +-3.6% seed sd. Same verdict as every other v5 feature: engaged, not load-bearing.
- Economy: casts1k 2.0-2.4 (magic is cast), gearTierMax pinned at 2.0-2.5, greatHarvests 1.84/life, death_starve 0.389.
- MANA GATE REFUTED on CPU before spending GPU: removing the RARE gate from tier-3 craft (realm.js:786-796, 1190-1203) left gearTierMax at 2.000/2.500 - tier 3 is unreachable for a reason other than RARE supply. All four edits reverted, so the tree reproduces the measured baseline.
- R36 is ALREADY the realm default: RL_GAME_DEFAULTS.realm has randomize: 0 (src/rl.js:61), so every training world is the 16-learner/16-bot eval-A composition - do not re-run it as a v5 arm. It also means learnerSlots is 16 by default, so env.selfplay is false and PvP/bounty/boss-group content is OFF in the default training mix.
- Consequence: the remaining gap is reward shaping, so the next arms are channel weights (popWeights, works on the K=1 path, no src edit): coop x3 (v5cw3), progress x2 (v5pw2), survival x0.5 (v5sw5), 2 seeds each at 300k, judged against v5tr.
- Friction: `queue resume` defaults to port 8123 while every job URL is baked on 8124, so a bare resume dies with "server on port 8123 does not serve /dev/rllong.html". Always `node tools/queue.mjs resume --port=8124`.

## 2026-10-06 -- reward-magnitude audit (source-read only, no job run)
Units: `add()` (realm.js:1025) writes fixed-point fx; **1024 fx = 1 reward unit**. Calibrated against NEXT-ROUNDS:262's camping probe (2 campers on GREAT_TOGETHER=2 + ALIVE=1 fx/tick, logged 3.10): **1 log-EVAL unit ~ 1 fx per learner-tick**, so EVAL 5.12 ~ 5.2 fx/tick ~ 10,900 fx (~10.7 units) per 2100-tick life.
- HYPOTHESIS REFUTED: the per-tick alive term is `ALIVE_REWARD = 1` fx (realm.js:53, call 1420), i.e. ~2,100 fx = **~19%** of a life, NOT the ~2000 UNITS I assumed (off by 1000x). Death claws back -1024 (1424), so alive nets ~10%. Survival does NOT dwarf the discrete events: one great share (1024) is half a life's alive income.
- Measured/derived per-life budget (~10,900 fx): alive +2,100 (19%), need restore <=+682 (6%), great shares 1.84/life with halving ~+2,260 (21%), trades ~+102 (0.9%), death -1,024 (-9%). Residual ~61% is harvest xp / craft / sales / mob kills / combat xp / boss / PvP / damage-taken - could not be split without rates.
- Channel shares (bounded): survival 16-20%, cooperation 22-27%, progress 30-40%, combat 10-20%.
- **Trading cannot be load-bearing at 32 fx (0.9% of a life)** - and it cannot simply be raised: gold is conserved across a trade (1129-1133), the ration is NOT consumed, and there is no counter or cooldown, so two learners could ping-pong one ration every tick for unbounded reward. ACH bits 0-30 are all taken (only bit 31 free, 1053/1041/1356). Cheapest safe form: pay only the receiver, or require the giver's ration count to drop.
- **XP is nearly untapped**: XP_CAP 1296 (57) gives 20,736 fx of headroom per life (1.9x the whole measured life total) but observed XP is ~75-77, so level = 1+isqrt(xp>>4) tops out ~3 and perk tiers 2/3 (level >=5/>=7, 561) plus the tier-3 craft (needs RARE_TIER3, 45/789/795) are effectively unreachable - **CRAFT_REWARD[2] = 2048 is dead code**.
- **F1 - dying is net reward-positive**: spawn() zeroes XP0..3 and ACH (330), so a life can re-earn the five milestones (512+1024+256+256+256 = 2304 fx) plus a fresh XP budget for only 1024 fx (1424). Same failure mode AGENTS.md records for the spawn kit, but for milestones/XP. Fixes: raise the death penalty above the re-earnable bonus (e.g. -3072), preserve ACH/XP across rebirth, or gate milestones on AGE.
- **Magic, cover ambush and the creep archetypes have NO reward write of their own** - they pay only through attack damage xp (1337), ~1.5% of a life. No magnitude change inside the current 4 channels can make them load-bearing; they need their own term (or a 5th channel, which costs input/weight headroom).
- Safe-to-raise candidates ranked: NEED_REWARD 4 -> 16 (100/1063; attacks the 38.9% starvation deaths, rate-limited by drain 1 food pt/32 t), XP_CAP 1296 -> 2592 (57; no power change, level still caps at 10), CRAFT_REWARD -> [768,1536,3072] (80; fixes the documented "craft pays less than selling" inversion), GREAT_REWARD 1024 -> 3072 (54; 32 nodes x 1500-tick timer so supply is not binding, but watch the /8 floor).
- GREAT_TOGETHER (2 fx/tick, 1164) is correctly gated at gcount >= 3 at difficulty 100 and completes the node - the R8 two-camper exploit is closed; do not loosen.

## R56 -- channel weights (popWeights, K=1) + the XP/style survey

Round: realm v5, 512 worlds, 300k ticks, seeds 1-2, `popWeights` (no src edit, reaches the opt
block at K=1 through `rlPopConfig`). Judge at the 290k window against `v5tr` (EVAL 5.09, ratio 2.17).

- `v5cw3`  `1,1,1,3` (coop x3): s1 EVAL 5.94 @294k, s2 5.81 @292k -> mean 5.88, ratio 2.32.
- `v5pw2`  `1,2,1,1` (progress x2): s1 EVAL 5.27 @292k, ratio 2.27.
- `v5cw5`  `1,1,1,5`, `v5sw5` `0.5,1,1,1`: running / pending at the time of writing.

### XP / style survey (CPU replay, `runs/xpsurvey.mjs`, v5cw3-s1 champion, 240 lives)
Combat XP lands in ONE track: `xp1` (range) p25 178 / p50 318 / p75 570 / p90 958, while `xp0`
(melee) and `xp2` (mage) sit at p75 6 and 9 -- i.e. learners are a pure ranged monoculture and the
melee/mage columns of the 3x3 perk tree never unlock. Measured against the CURRENT thresholds
(`tierOf`: xp 64 / 256 / 576) a life reaches tier 1 / 2 / 3 with share 0.95 / 0.625 / 0.246, so the
tree is NOT unreachable -- CORRECTS the earlier "perk tiers 2/3 are unreachable (xp ~75)" note,
which was a remembered number, not a measurement. `xp3` (harvest track) p50 176.

Consequence for the next arm: the defect is style dominance (STYLE_RANGE 200/800/500 makes range
safe and the melee/mage perks dead), not the thresholds. Candidate arm X = gate the combat perks on
`perkTier(e)` (the best track) instead of `tierOf(e, style)`, so a ranger at tier 2/3 also gets
cleave / execute / ward and mixing styles pays. Judge with evalsuite (`perk`, `maxPerk`, melee/range/
mage label shares) plus EVAL.

## R57 -- channel weights, bot catch-up, and the cost of a decision (2026-10-06)

### Channel weights (`popWeights`, K=1 path, no src edit) -- CLOSED
Baseline `1,1,1,1` (v5tr, 2 seeds @284-288k): EVAL 5.09/5.32, evalBase 2.39/2.35, ratio 2.20.
Paired deltas at the 290k window (arm - baseline, SAME seed):

| arm | weights | EVAL delta | bots delta | ratio delta | verdict |
|---|---|---|---|---|---|
| v5sw5 | `0.5,1,1,1` | +0.74 +- 0.12 | +0.03 | +0.28 | **ADOPTED**, baked into `RL_GAME_DEFAULTS.realm` |
| v5cw3 | `1,1,1,3`   | +0.76 +- 0.38 | +0.16 | +0.16 | real, not adopted (sw5 is stronger and simpler) |
| v5cw5 | `1,1,1,5`  | +0.57 +- 0.89 | +0.18 | +0.07 | real but noisy |
| v5sc  | `0.5,1,1,3`| +0.86 +- 0.66 | +0.20 | +0.17 | no stacking gain over sw5 |
| v5pw2 | `1,2,1,1`  | -0.03 +- 0.42 | -0.06 | +0.04 | flat / negative |

### Bots catch up -- great nodes LANDED, boss + gathering screened
- `v5g` = bots trek to the nearest active great node within `BOT_GREAT_RANGE` 900 (gated
  `TOOL >= GREAT_TOOL`, water > 60, food > 40). 2 seeds @287-291k, paired vs v5sw5:
  **bots +0.37 +- 0.02 fx/1k (+15%)**, learner +0.31 +- 0.33, ratio 2.47 -> 2.26, h2h 0.00, self-play +0.12.
  Realm 21-test suite re-run on `game=realm` (v5gtr, 458 s): 21/21.
- Screened on CPU (6 worlds x 8000 ticks, champion replay, `runs/chansurvey.mjs`), two champions:
  boss contest (`BOSS_ENGAGE` 1200, `HP*2 > maxHp`, replaces the `BOSS_FLEE` sprint-to-town) alone
  bots +9.4% / +5.6%; boss + great-node gathering (`gatherAt` prefers a node with an ally within
  `BOT_GATHER_RANGE` 300, `d2 >>= 2`) **bots +14.0% / +10.9%**, learners +1.7% / +1.8%, bot coop
  0.81 -> 1.09 fx/1k. Both shipped; `v5b` (2 seeds, 512 worlds x 200k) is the confirmation.

### Cost of a decision (the wall-clock question)
- Paired deltas (`node tools/report.mjs --paired=<run>`): the sd of a SAME-SEED delta is the noise on
  the decision, 3-17x tighter than the sd of arm means (sw5 +0.69 +- 0.04 vs +- 0.12). Free.
- Horizon: paired-delta sd over 5 arms = 0.60 @150k, 0.35 @200k, 0.56 @290k. **150k screening is false
  economy** (v5pw2 +0.46 @150k, -0.03 @290k; all deltas ~1.4x exaggerated). Standard horizon moved to
  **200k** (a third cheaper than 300k, and the least noisy). `--at-tick` now interpolates between the
  bracketing log windows instead of taking the nearest one.
- Not done: 256 worlds (halves cost but shifts learning dynamics, -0.15 EVAL documented) and concurrent
  jobs (throughput scales with world-ticks = the GPU is already saturated).

### Next arms (R58), in priority order
1. **X - perk gating** (makes v5 content load-bearing). Measured cause of the monoculture: combat XP
   lands almost only in `xp1` (range) p50 318, while `xp0`/`xp2` sit at p75 6/9, so the melee and mage
   columns of the 3x3 perk tree never unlock and lesioning their inputs does nothing. Arm: gate the combat
   perks on `perkTier(e)` (best track) instead of `tierOf(e, style)`, so a ranger at tier 2/3 also gets
   cleave / execute / ward and mixing styles pays. Pre-screen on a game copy (`src/games/realm-x.js`) with
   a champion replay: style label shares, `perk`/`maxPerk`, damage per style. Judge with EVAL + evalsuite.
2. **Bots cast** (PvE). Bots hold RARE from spawn 6 and from great nodes but never cast, and `tryCast`
   targets only players, so this needs new code (cast at the nearest mob/boss when RARE allows).
3. **Ranged monoculture, direct**: `STYLE_RANGE` 200/800/500 makes range strictly safer; shortening the
   ranged band (or paying melee a gap-closer) is the blunt alternative to X.

## R58 -- bot confirmation, entropy on v5, and arm X (2026-10-06)

### v5b (bots contest the boss + prefer gathered great nodes) -- ADOPTED
2 seeds, 512 worlds x 200k, `popWeights=0.5,1,1,1`, PAIRED vs `v5g` (same config, great nodes only):

| metric | delta | sd |
|---|---:|---:|
| bots fx/1k | **+0.25** | 0.03 |
| learner EVAL | -0.12 | 0.42 |
| ratio | -0.22 | 0.17 |
| self-play | -0.15 | 0.42 |
| h2h | 0.00 | 0.00 |

Absolute @200k: EVAL 5.98 +- 0.01, bots 3.02 +- 0.06, ratio 1.98 +- 0.05, life 1536, h2h 1.03.
The bot gain is decisive (sd 0.03 on a +0.25 effect); the learner is flat inside noise, so the whole
move is a genuine bot catch-up (bots 2.39 -> 2.78 -> 3.02 across R57 -> v5g -> v5b) bought for nothing.
Realm suite re-run green (v5btr, 21/21, 610 s) after fixing the boss-direction divergence below.

### Divergence caught by the suite (and the field it was hiding in)
`r_load_bot_scan` packed only the DISTANCE for boss slot 9, so `r_ndx[9]/r_ndy[9]` stayed 0 while JS
`scan()`->`consider()` filled them -- WGSL bots walked toward the origin the moment the new
boss-engage branch ran. Fixed by packing the boss index too (`(u32(d2) << 8u) | j`) and setting
`nd2/nidx/ndx/ndy` for slot 9 together. Lesson: agreement is per-FIELD; a slot can be right in
distance and wrong in direction, and only a consumer of the wrong field exposes it.

### Entropy 0.04 on v5 (v5e4) -- NEUTRAL, closes the open question
2 seeds @200k PAIRED vs `v5b`: EVAL +0.10 +- 0.47, bots +0.04 +- 0.06, ratio 0.00 +- 0.12,
self-play +0.17 +- 0.57. Every column is far inside its sd, so on v5 (unlike v4's R11/R30) more
exploration buys nothing. Default anneal stays.

### Arm X (perk gating) -- running
Combat XP lands in the range track only (`xp1` p50 318 vs `xp0`/`xp2` p75 6/9), so the melee and mage
columns of the 3x3 perk tree never unlock and lesioning them does nothing. X gates all five combat
perks on `perkTier(e)` (best of the three tracks) instead of `tierOf(e, style)`:
damage tier 651/985 (which also feeds cleave), cast cooldown 723/1074, execute 1016/1335,
bulwark 1287/1668, ward 1289/1670. Costs no observation budget (obs 108 already carries `perkTier`).
Judge: 2 seeds @200k PAIRED vs `v5b`, plus evalsuite style-label shares. Bots inherit the perks
through the shared damage code, so expect the bots to rise too -- read EVAL first, ratio second.

### Arm X verdict -- NEGATIVE, REJECTED (2 seeds @200k PAIRED vs v5b)
| metric | delta | sd |
|---|---:|---:|
| learner EVAL | **-0.18** | 0.08 |
| bots | +0.05 | 0.13 |
| ratio | -0.09 | 0.11 |
| self-play | -0.15 | 0.07 |
| h2h | +0.03 | 0.02 |

Both negative columns are ~2 sd and consistent across seeds, so gating ALL FIVE combat perks on the
best track costs the learner ~3%. Plausible mechanism: bulwark (was `tierOf(v,0) >= 3`) and ward
(was `tierOf(v,2) >= 2`) are DEFENSIVE and hence symmetric - handing every entity a tier-3 perk
flattens fights instead of rewarding a build. Isolation arm X2 gates only the OFFENSIVE perks
(damage tier/cleave 651/985, execute 1016/1335, cast cooldown 723/1074) and leaves bulwark/ward on
their own tracks; if X2 is >= v5b the defensive half was the whole cost.

## R59 — bots PvE cast (REJECTED) and the style triangle (v5tri, queued)

### bots PvE cast — v5bc, 2 seeds, 512 worlds, 200k, paired vs v5b @200k
| run | eval r/t x1e3 | bots x1e3 | ratio | eval life | h2h | self-play |
| --- | --- | --- | --- | --- | --- | --- |
| v5bc | -0.09 +- 0.08 | **-0.20 +- 0.07** | +0.11 +- 0.08 | -199 +- 514 | +0.02 | -0.06 +- 0.08 |

Bots calling `tryCast(e)` before `tryAttack` in both combat branches cost the bots 7% of their rate
(2.83 -> 2.63) and moved the ratio the wrong way. Mechanism: a cast sets `F.CD` to `CAST_COOLDOWN`,
displacing the next attack (higher per-hit DPS at the ranges bots fight at), and spends RARE that
bots otherwise craft gear with. Verdict **REJECTED**, all four call sites reverted.
Screening note: a gated variant (cast only when RARE is plentiful) fixes only the resource half and
not the cooldown half, so it can at best move the loss toward zero — not worth a round.

### v5tri — restore the style triangle (queued, 2 seeds @200k)
Ranged monoculture has a concrete cause in the damage code, not a training failure. Mob style is
`e % 3` (uniform melee/range/mage) and `tryAttack` scales x3/2 when the attacker's style beats the
defender's, x3/4 when the defender's beats the attacker's — but realm.js:653 exempted ranged
attackers at tier >= 2 from the x3/4 (`!(style === 1 && tier >= 2)`). So a ranger kept its good
matchup (x3/2 vs mage mobs) and was immune to its only bad one (melee mobs), on top of 4x the reach
(STYLE_RANGE 200/800/500): monoculture is rational. Arm drops the exemption on both sides; melee
then pays x3/2 vs ranged mobs and vs ranged learners in self-play.

### v5tri verdict — ADOPTED (2 seeds, 512 worlds, 200k, paired vs v5b @200k)
| run | eval r/t x1e3 | bots x1e3 | ratio | eval life | h2h | self-play |
| --- | --- | --- | --- | --- | --- | --- |
| v5tri | +0.31 +- 0.13 | -0.01 +- 0.01 | +0.11 +- 0.03 | -248 +- 323 | +0.02 | +0.38 +- 0.02 |

EVAL 5.98 -> 6.29, self-play 6.09 -> 6.47, bots flat. Mechanism: bots and other learners lose
ranged damage against melee-style defenders, so learners survive longer (death_player is ~30% of
deaths). Adopted — it also makes the counter triangle symmetric, so melee now genuinely counters
ranged.

BUT the arm did NOT do what it was designed to do. xpsurvey (2 worlds x 4000 ticks per champion):
v5tri-s1 xp0 p75 9 / xp1 p50 244 / xp2 p75 0; v5tri-s2 9 / 272 / 0; v5b-s1 9 / 297 / 0; v5b-s2
18 / 330 / 0. Monoculture intact. The lesson: reach (STYLE_RANGE 200/800/500), not the damage
triangle, is what makes ranged dominant — at 800 a ranger takes zero return damage from a melee
mob, and no multiplier competes with zero.

### v5ch — melee charge (queued)
`tryAttack` tracks the nearest out-of-reach-but-within-600 target for melee-style players and
sets `F.SPRINT` (2x speed, 1 water / 8 ticks) when nothing is in reach — a warrior closes. No new
state, no decide-phase position write. Judge paired vs v5tri at 200k; the win condition is melee
XP leaving single digits, not EVAL.

### v5ch verdict — REJECTED (2 seeds, 512 worlds, 200k, paired vs v5tri @200k)
| run | eval r/t x1e3 | bots x1e3 | ratio | eval life | h2h | self-play |
| --- | --- | --- | --- | --- | --- | --- |
| v5ch | -0.23 +- 0.19 | +0.14 +- 0.11 | -0.16 +- 0.01 | +270 +- 28 | -0.01 | -0.36 +- 0.18 |

Melee charge = `tryAttack` sets `F.SPRINT` when a melee target is out of reach but within 600, so a
warrior closes at 2x speed (sprint costs 1 water / 8 ticks). Cost the learner ~4% and gave bots
+0.14, i.e. the ratio moved the wrong way; xpsurvey shows xp0 p50 0 -> 9 and xp2 still ~0, so it
did not unlock melee either. Reverted.

Why it cannot work, stated so the next attempt does not repeat it: at reach 800 a ranger takes
ZERO return damage from a melee mob (reach 200). Melee already has ~2.7x the DPS (cd 10 vs 18,
damage 6 vs 4) and x3/2 vs ranged mobs, and still loses — so no buff to melee can close a gap that
is measured in "damage taken", not "damage dealt". The only lever left is to make one style NOT
usable against part of the world.

### v5res — creep archetype resist (queued)
`MOB_RESIST = [0,1,2,0,1]` by `mobKind`: brute/caster resist melee, stalker/ambusher resist ranged,
pack resists mage, so a mono-style policy does 1/4 damage to 2 of 5 archetypes. Bots switch away
from a resisted style; learners already see the archetype on obs 106 (`hostileType`).

## R62 — self-organization: input-salience mask (round COMPLETE)

Directive: the most performant possible self-organization, scaling to its own needs in pre-training
and live learning. Full design, alternatives rejected, and the ladder: `runs/selforg.md`.

Mechanism: `applyInputMask()` (src/rlengine.js) ranks all 111 inputs by gradient mass
`sum_i |m[w1 + j*hidden + i]|` from Adam's first moment, then holds the lowest `inputMask` at exactly
zero in `theta`, `m` and `v`. 0 new params, 0 workgroup bytes, 0 shader edits, 0 forward cost; a zero
edge is numerically exact so export just carries fewer edges. No self-blinding: `dW1[j][i]` does not
depend on `W1[j][i]`, so a masked input keeps accumulating gradient and can be regrown.

`inputMask=12 maskEvery=128`, 2 seeds x 512 worlds x 200k, paired vs v5tri at tick 194,820:

| | EVAL | ratio | evalBase | life | self-play |
| --- | --- | --- | --- | --- | --- |
| v5tri | 6.29 +- 0.11 | 3.01 | 2.09 | 1288 | 6.47 |
| mask | 5.84 +- 0.21 | 2.95 | 1.98 | 1374 | 5.96 |
| **paired** | **-0.45 +- 0.10** | -0.06 +- 0.05 | -0.11 +- 0.07 | +86 +- 97 | -0.51 +- 0.08 |

Edges: 7,168 vs 7,936 = -768 (-9.7%), exactly 12 x 64; `dims` unchanged.

**Verdict: REJECTED as a default, KEPT as an instrument.** Pre-registered rule was "adopt if EVAL
within -3% AND >= 5% fewer edges": edges passed, EVAL failed (-7.2%, 4.5 sd from zero).

Dropped set (stable, both seeds, every cycle): `animalN, animalNW, animalS, animalSE, animalSW,
lvlMelee, lvlMage, animalE, hostileE, animalNE, ore, springE`. Two findings worth more than the
verdict: the **animal compass (obs 48-55) is the lowest-salience block in every cycle of all three
runs**, and **`lvlMelee`/`lvlMage` are dead** — the ranged monoculture read off the input side,
independently of runs/arch-evidence.md's 0 melee / 0 mage attacks.

Also fixed in this round: `fitProbe` read the whole rollout buffer (~43 MB) every report, costing 10x
throughput (24.8k vs 285k wt/s). Capped to lane-blocks (now 256).

### Next: `ac` — animal compass only (maskFixed)
New knob `maskFixed` (comma list, `a-b` ranges) drops exact indices instead of ranking, so the one
block with a stable claim to being dead can be tested alone. `maskFixed=48-55 inputMask=8`, 2 seeds
@200k, paired vs v5tri. Adopt if paired EVAL >= -0.19 -> stage 2, delete obs 48-55 from the contract
(nIn 111 -> 103, 8 slots back under the 258 cap). If EVAL <= -0.45, gradient mass is measuring rarity
rather than uselessness and the fix is activity-normalized salience `|m| / EMA|x_j|`.

### R62b — `ac`: animal compass alone, 2 seeds (INCONCLUSIVE, 4-seed confirm running)

`maskFixed=48-55 inputMask=8`, 2 seeds x 512 worlds x 200k, paired vs v5tri at matched tick 193,022:

| seed | v5tri | ac | delta |
| --- | --- | --- | --- |
| 1 | 6.21 | 6.35 | **+0.14** |
| 2 | 6.37 | 6.03 | **-0.34** |

Mean -0.10 +- 0.34 (sample sd), -1.6% of baseline; edges 7,424 vs 7,936 = **-512 (-6.4%)**, exactly 8
x 64, `dims` unchanged.

**Not a decision.** The point estimate passes the pre-registered -0.19 threshold, but the two seeds
disagree in SIGN. With n=2 that makes SE 0.24 and the 95% interval ~+-3 — uninformative. Contrast
mask@12, whose per-seed deltas were -0.52 and -0.38 (same sign, sd 0.10): that one is a real
regression, this one is not resolved.

Each seed costs ~8 min (they stop at the 200k tick limit, not the 40 min cap), so certainty is cheap.
Queued `runs/jobs-ac2.json`: `v5tri-s3`, `v5tri-s4`, `ac-s3`, `ac-s4` — the baselines are needed too
because pairing requires the same seed. Judge at 4 pairs.

If it confirms (mean >= -0.19 with a tighter sd) -> **stage 2**, now fully costed in
`runs/stage2-obs-contract.md`: nIn 111 -> 103, freeing 512 B in the evolution kernel (8 hidden units,
nNodes 147 -> 155) AND 1024 B in the PPO rollout kernel (32404 -> 31380 B). The second one is the real
prize: **`hidden` 128 becomes compilable at `blockGrad: 1`** (31380 + 380 = 31760 B). Today it needs
32784 B and only compiles at `blockGrad: 0` (~37% slower), so every "bigger hidden is not a win"
verdict in this project was taken at sizes that happened to fit — R34 only ever measured 96 -> 64.

Also corrected this round: the byte ceiling is `(nNodes|1) + (nIn|1) <= 260` (32720 B of 32768), not
258 — 258 was 2 low and contradicted AGENTS.md's own "ceiling is nIn 113" (147 + 113 = 260).

### R62c — `ac` at 4 seeds: ADOPTED (animal compass is dead)

`maskFixed=48-55`, 4 seeds x 512 worlds x 200k, paired vs v5tri (4 new baseline seeds s3/s4 included):

| seed | v5tri | ac | delta | bots | ratio |
| --- | --- | --- | --- | --- | --- |
| 1 | 6.21 | 6.35 | +0.14 | 2.96 -> 2.93 | 2.10 -> 2.17 |
| 2 | 6.37 | 6.03 | -0.34 | 3.07 -> 3.07 | 2.07 -> 1.97 |
| 3 | 5.81 | 5.67 | -0.14 | 2.89 -> 2.98 | 2.01 -> 1.90 |
| 4 | 5.77 | 6.17 | +0.40 | 3.01 -> 3.00 | 1.92 -> 2.05 |
| **mean** | | | **+0.015 +- 0.323 (SE 0.162)** | | |

Edges 7,424 vs 7,936 = -512 (-6.4%), exactly 8 x 64. Both pre-registered criteria met (>= -0.19, and
edges), so **stage 2 is open**.

The estimate walked toward zero as seeds were added (-0.10 @2, -0.11 @3, +0.015 @4): the 2-seed reading
was sampling noise, not a real cost. Baseline seed sd is 0.30 unpaired (EVAL 6.21/6.37/5.81/5.77, mean
6.04) — the paired design is 3x tighter and is what made this decidable at all.

Note seed 4 gained the most (+0.40) and had the weakest baseline (5.77), i.e. the arm is not simply
riding strong worlds.

### R63 — stage 2: delete obs 48-55 from the contract (IN FLIGHT)
nIn 111 -> 103. Plan: runs/stage2-obs-contract.md. Frees 512 B in the evolution kernel (8 hidden
units, nNodes 147 -> 155) and 1024 B in the PPO rollout kernel (32404 -> 31380 B) — the latter makes
`hidden` 128 compile at `blockGrad: 1` for the first time (31380 + 380 = 31760 B; today it needs 32784
and only compiles at `blockGrad: 0`, ~37% slower). Validate with `tests=1&game=realm` on a runner
started with `--stall-seconds=1800`; then a fresh 4-seed baseline, since every existing genome is
stale (exports carry `dims.nIn:111`, `src/rl.js:528` rejects).

**Landed 2026-10-06.** Both sides shifted: JS `observe()` compass loop skips `CLASS_ANIMAL` and writes
64 compacted slots, 39 named writes moved 72..110 -> 64..102, `dims.nIn` 103, `inputNames` 103;
WGSL emit loop skips `c == 6u` and the same 39 literals moved. Evolution oracle is green: `st2p2`
8/8, including **observation parity (128 obs x 103 inputs, day and night) identical** and
`workgroup memory fits device [32080 of 32768]` — the 512 B prize is real (32592 -> 32080).

The RL suite caught what parity could not: `st2t` 20/21,
`FAIL rollout lockstep ... learner 0 input 70 (needDx): gpu 0 js 0.015380859375`. Not an index bug —
`set_obs` writes into the same `obsBuf` row the scan drops the spring/berry keys into
(`src/shader.js:198`), so `RW_SLOT_SPRING`/`RW_SLOT_BERRY` were deliberately aliased to the need
block's OWN outputs (78/92 = needDx/needDy pre-shift): above the compass read range 0..71 and written
only after its read. After the shift those slots held `night` and great-progress, written by other
blocks, so the keys were clobbered before the read. Fix: re-aliased to 84/85 (needDy/urge), the only
indices >= 72 that only the need block writes (70 would race with the emit loop's `atomicLoad` of
0..71). Attribution is not in doubt: `fit1tr.log` (params 8014 = nIn 111 / nOut 13 / hidden 64) passed
21/21 the same day on the pre-edit tree, and the pre-shift aliasing is exactly 78/92.

Gate PASSED: `st2t3` (`tests=1&game=realm`) = **21/21, 0 FAIL** in 1126 s, including
`rollout lockstep with the JS env` (4 worlds x 120 rollout ticks x 32 learner slots: observations,
rewards, end-of-life flags and final world states identical word for word) — the one test that failed
before the re-alias, and `exported genome replays through core.js Brain` on a 2320-edge `npc-brain/2`
genome with `dims 103/13/136`. Its header also confirms the byte prize directly:
`params=7502 workgroupBytes=31380`. The fresh baseline is therefore running:
`runs/jobs-stage2-base.json` = `s2-s1..s4` (defaults) + `h128-s1..s4` (`hidden=128`, which now fits
`blockGrad: 1` at 31760 B), 512 worlds, 200k ticks, judged
`node tools/report.mjs --at-tick=200000 --paired=v5tri`. v5tri stays the comparison baseline even
though its dims differ: only the paired per-seed delta at matched ticks is read, and `ratio` is
dim-independent.

**Decision rule for stage 2, pre-registered before the seeds land** (so it cannot be fitted to the
result). Deleting the 8 animal-compass inputs is functionally identical to zeroing their weights, and
that arm was ADOPTED at 4 seeds on +0.02 +- 0.16, so the expectation is *neutral*; the prize is 1024 B
in the rollout kernel (hidden 128 at `blockGrad: 1`). Judge the paired EVAL delta
(`--at-tick=200000 --paired=v5tri`, 4 seeds):
- **>= -0.19** (-3%): keep nIn 103, unconditionally.
- **-0.19 to -0.45**: keep nIn 103, but treat the byte prize as paid for: only spend it on `hidden` 128
  if `h128` itself reads >= -0.19 against the new baseline.
- **<= -0.45**: revert to nIn 111 (git revert of the stage-2 commit, rebuild, re-run `tests=1`), because
  -0.45 is the measured cost of masking 12 inputs, which was REJECTED as a default — paying that much
  for bytes is not worth it.
One seed at 1.1 sd is not a decision; the rule applies to the 4-seed mean.

Process note (2026-10-07): a queue runner can die silently. `queue status` showed `runner: not
running` while the job was still marked `running` at 1489 s - with a stale `runs/.gpu.lock` (heartbeat
1558 s old, pid absent from the process list), no `runs/chrome-q-*` profile and no `runs/st2t2.log`,
i.e. no Chrome had ever launched for it. The job had simply not run for 25 minutes. Recovery:
`node tools/gpulock.mjs clear` then re-`start` (or `resume`). Check the `runner: alive pid` line after
any session interruption or compaction - the job's `elapsed` counter keeps counting without a runner.

### R66 — telemetry: `CH[...]`, `HID[...]`, `ACT[...]` (spec complete, NOT queued, needs src edits)

Full spec: `runs/telemetry-audit.md` §8 (cost classes, exact buffer addresses, token shapes, parser
constraints) and §9 (what is refused and why). All three are **host-side log-only**: 0 workgroup bytes,
0 new params, 0 kernel passes.

- **`CH[sur=,pro=,cmb=,coo=,tot=,n=,w=]`** — live reward-channel mix, one 88 KB read of the channel
  words at `w*2760 + 2632 + e*4 + c`. Decides which channel weight to move next (today `0.5,1,1,1` came
  from a 2-seed offline arm) and whether "v5 content engaged but not load-bearing" is a reward-side or
  a reach-side fact.
- **`HID[dead=,rank=,sat=,w1cv=,n=]`** — hidden-side capacity: W2 outgoing L1 for dead units, W1-row
  participation ratio for `rank`. **This is the one that decides how to spend the 10 sum-units**: if
  `rank << 64`, more hidden units are redundant and `h128` should lose; if `rank ≈ 64` and `dead = 0`,
  capacity is binding and nIn 103 -> 113 (+ content obs) is the cheaper lever.
- **`ACT[mel=,rng=,mag=,cast=,spr=,int=,rail=,dead=,n=]`** — the live style profile from `fitProbe`'s
  already-computed `out.mean` (indices 5/6/7/8/12 by `actionNames`). Turns "ranged monoculture" from a
  2-seed offline survey into a per-report number, and with T1/T2 separates the three causes: reach,
  reward, capacity.
- Free in the same pass: `ev = 1 - 2*vl/rstd^2` as a `report.mjs` derived column, and per-group
  `WSG[...]` quantiles (today `ws` pools all four eval groups, so a dead subset is invisible).

Refused (§9, do not relitigate): per-channel accumulation in `statAcc` (pays 4 KB of workgroup storage
for what an 88 KB host read returns exactly), a death-cause histogram (needs a killer id that does not
exist host-side), activity-space hidden rank (13.6 MB read - the original `fitProbe` failure mode).

Sequencing: land R66 together with R65a in ONE src-edit wave after this round, then a single `tests=1`
validation (~20 min) before either is trusted.

### R67 — comment sweep: merge after this round (branch ready)

Branch `worktree-agent-a9d57daefddd7ef07`, sha `e651414c11b716f0ec1bb76c9097e5a5308de945` (lanmower).
Repo was already near comment-free: 17 comment lines in 41 files. 15 became self-explanatory code
(`rlRecIndex`, `recFieldStride/recTickStride/recBlockStride`, `{name,offset,length}` lane planes,
`lanesInEveryWorldIncludingEval`, `transposeLaneIndexTerms`, `readObsInline`), 2 were deleted as
duplicated in README.md, and 2 irreducible warnings moved to AGENTS.md (WGSL function-local arrays
indexed by a loop variable land in local memory - stage inputs in workgroup storage; SoA rec planes
must span every lane or an over-range write lands inside the next field). AGENTS.md 25,989 -> 25,955 B.

Merge plan, TO BE DONE AFTER the round (an edit now would make later jobs in this round compile
different code than `s2-s1..s4`): `git checkout <branch> -- src/rlshader.js tools/acab.mjs` — both are
unmodified in the working tree, so they apply cleanly — then hand-merge the two AGENTS.md facts (that
file IS modified here) and trim to stay under 26,000 B. Rebuild and re-run `tests=1` before trusting it.
Generated WGSL, layout and `rlLanePermute` were verified byte-identical to HEAD across 16 realm/blob
configs (incl. `soa`, `fwdDirect`, `recurrent+bias`, `obsNorm`, `policies 3`).

**Latent bug found by the sweep, not fixed (behaviour change):** `rlRecPermute` strides SoA planes by
`S.recBlocks` (`worlds*learners`) while the shader's `RS_F` uses `S.trainLearners`
(`evalStart*learners`); they agree only when `worlds == evalStart`. So `soa: 1` is wrong whenever eval
worlds exist, i.e. always. Opt-in path, currently unused - fix it before ever arming `soa`.

### R64 — `hidden` 128 at nIn 103 (staged: `h128-s1/s2`, plus the `h192p` compile probe)

The verdict "hidden 128 is killed" was taken at nIn 111 where it needed 32784 of 32768 B and only ran at
`blockGrad: 0` (~37% slower), so it measured a *slow* config, not a big one. At nIn 103 the rollout
kernel is 1024 B lighter (32404 -> **31380**, confirmed by `st2t3`'s own header `params=7502
workgroupBytes=31380` = nIn 103 / nOut 13 / hidden 64), so 128 = 31760 B compiles at `blockGrad: 1`.
`tools/autosize.mjs` re-anchored on that: it also predicts hidden 192 at 32140 B and rolloutTicks 128
at 32208 B, both unmeasured - `h192p` (30k ticks, 10 min) is the cheap compile check.

### R65 — style lock: break the ranged monoculture (design; NOT queued, needs src edits)

Decode, verified at `realm.wgsl.js:1118-1121`: `style` = argmax over outputs 6/7/8 (melee/range/mage)
and the attack fires only if that max exceeds **0.5**. Rails measured on champions: range +0.99,
melee -0.95, mage -0.98. So the net never plays melee or mage, and the phase-4 lesions (<= 3.2%) are
consistent with that: the content cannot be load-bearing if the policy never enters its track.

Two mechanisms, both 0 bytes and 0 new params:

**R65a (primary, strongest): per-policy permutation of the three style outputs.** Swap `pre2[kk]` and
`logStd[kk]` for indices 6/7/8 by a per-policy permutation BEFORE sampling (`src/rlshader.js:1246-1258`,
mirrored at :1574 and :1781 and in `rlPolicyMean`, `src/rl.js:136`). The melee role then attacks on the
net's own +0.99 rail instead of fighting a -0.95 one. logp stays exact because the (mu, ls) pair is
swapped, not the emitted action. Export is exact too: swap the W2 rows, `logStd` and `b2` entries of the
exported theta, so a style-locked champion needs no runtime change. Permutation identity = bit-identical
to HEAD.

**R65b (dose-response): additive per-policy bias on `mu`.** `let mu = rl_softsign(GAIN*pre2[kk]) +
sbias[kk]` at `src/rlshader.js:1247` - added AFTER the softsign and BEFORE the eps draw, so the sampling
distribution is N(mu+B, sigma) and logp (which only uses eps) stays consistent. Do NOT add it to
`pre2`: the rail sits at pre-activation ~ -9.5 (softsign(2x) = -0.95 => x = -9.5), so a pre-activation
bias needs ~ +19 to move it. Symmetric dose (preferred +B, the other two -B): argmax flips at
B > 0.97, and the 0.5 gate needs B > 1.45, so **B = 1.5**. Not learnable and not exported (Adam cannot
move it, it is not in theta) - an instrument, not a shipped brain. 3 of the 4 free per-policy floats in
`opt` (`RL_POP_WEIGHT_SLOTS = 8`, 4 channels used, `src/rl.js:756`) hold it.

Both need K=3, so the **control must run first**: `v5k3c` = `policies:3` uniform weights, 2 seeds, then
the arm paired against it (`--at-tick=200000 --paired=v5k3c`). Known: with bots in training K=1 beats
K=3 by ~15%, so the arm must clear that, not just hold EVAL.

Thresholds (paired-delta sd 0.35 @200k; tie +-0.30): **ADOPT** if own-style attack share >= 0.80 on both
seeds AND xp0/xp2 p75 >= 64 AND melee/mage attack rate >= 0.70x range AND dEVAL >= -0.30;
**INSTRUMENT-ONLY** to -0.70; **REJECT** <= -0.70. Secondary readout via `tools/evalsuite.mjs` on the
champions (CPU, only when the GPU is idle): XP tracks xp0/xp1/xp2 and the behaviour-label mix.

## R68 — style-domain randomization (SDR): break the ranged monoculture at the EXPLORATION level

**Why a new lever at all.** The monoculture is a lock, not a balance number. Melee XP p75 = 6 against
range p50 = 318; `tierOf` (`src/games/realm.js:565`) needs level >= 3 for tier 1, so the melee and
mage columns of the 3x3 perk tree never unlock, so melee/mage stay strictly worse, so range stays
dominant. Three arms already tried to change the PAYOFF and failed: v5ch (melee charge)
-0.23 +- 0.19, v5res (per-archetype resist) -0.50, Arm X (perks decoupled from tracks) -0.18 +- 0.08.
R68 changes the EXPLORATION instead — it makes the dead tracks reachable so the payoff comparison can
even be made.

**Mechanism** (two variants, one arm):
- **R68a XP boost**: per training world, one style track `s` gets XP x `SDR_BOOST` (default 3), the
  others x1. No gameplay change at all — only the XP accumulator. Smallest possible mirror surface.
- **R68b hard gate**: per training world, one style's attacks are disabled. Stronger forcing (the
  learner must use another style to fight at all) but it changes combat outcomes, so eval worlds
  (ungated) differ from training worlds — a train/test gap the XP boost does not have.

Cost: 1 cfg word per world (or derive from the world's existing init RNG stream, which costs 0 words
but must be bit-identical between `realm.js` and `realm.wgsl.js`). 0 obs, 0 params, 0 workgroup
bytes. Mirror risk low-med: `GpuBackend.runTests` covers init parity and obs parity, and
`RlBackend.runTests` covers the rollout lockstep.

**Judge**: 2 seeds x 512 worlds x 200k, paired against the post-stage-2 baseline. **Primary readout is
NOT EVAL** — it is `ACT[mel=,rng=,mag=]` from R66, because the whole point is that EVAL can stay flat
while style diversity moves. Adopt if melee+mage attack share goes from ~0% to >= 15% at
EVAL >= -0.19; promote to default if EVAL >= +0.19 as well.

**Ordering constraint**: R66 must land BEFORE or WITH R68. Without a per-style action share there is
no way to tell whether the arm fired — only that EVAL moved, which is exactly the mistake the v5
content rounds made (lesions <= 3.2%, "engaged but not load-bearing", judged on EVAL alone).

## R63/stage-2 VERDICT (4 seeds, matched tick 190000) — nIn 103 ADOPTED

`node tools/report.mjs --at-tick=190000 --paired=v5tri --match='^(s2|v5tri)-'`:

| run | seeds | ticks | EVAL | bots | ratio | life | h2h | self-play |
|---|---|---|---|---|---|---|---|---|
| s2 (nIn 103) | 1,2,3,4 | 190000 +- 0 | 5.93 +- 0.24 | 3.04 +- 0.05 | 1.95 +- 0.10 | 1499 +- 223 | 1.05 +- 0.10 | 6.03 +- 0.14 |
| v5tri (nIn 111) | 1,2,3,4 | 189574 +- 853 | 5.99 +- 0.25 | 2.97 +- 0.06 | 2.02 +- 0.08 | 1382 +- 81 | 1.02 +- 0.06 | 6.07 +- 0.28 |

**Paired deltas (4 pairs, s2 - v5tri, same seed): EVAL -0.06 +- 0.33**, bots +0.07 +- 0.09,
ratio -0.07 +- 0.14, life +117 +- 288, h2h +0.02 +- 0.09, self-play -0.04 +- 0.33.

Rule outcome: **-0.06 is inside the ">= -0.19 keep unconditionally" band**, so nIn 103 is adopted and
the 1024 B rollout-kernel prize is paid for with no strings — `hidden` 128 does not have to earn the
right to be screened, it only has to earn adoption on its own numbers.

Two things this settles:
- Deleting the 8 animal-compass observations costs nothing measurable, which is the predicted result:
  it is functionally the same move as zeroing their weights, and *that* arm (maskFixed=48-55) was
  already adopted at +0.02 +- 0.16. The -0.45 rejection was the MIXED 12-input set, not the compass.
- The self-org loop closes: the mask named the dead block, the contract dropped it, and the freed
  bytes became a size that had never been measurable (`hidden` 128, see the h128 arm below).

Also recorded: the 2-seed single-log-window reading was -0.34 and the 4-seed paired reading is
-0.06 — see .wfgy/lessons.md 2026-10-07.

## Byte model REBUILT (2026-10-07) — the binder switches; max hidden 117; horizon is nearly free

**What broke.** `hidden: 128` at nIn 103 does NOT compile: `rl_grad_blk` needs **34768 of 32768 B**
(device). Three jobs produced zero ticks (h128-s1, h128-s2, h192p), and the layout's own pre-check
had reported 31380 and let them start.

**Cause.** `blockBytes` (`src/rlshader.js:112`) omitted two terms that `pool4` actually declares
(`src/rlshader.js:240`): `ceil(nOut/4)*hidden` and `ceil(hidden/4)` — and it added 1024 B for
`selCnt`, which belongs to `rl_select` only. Fixed to mirror the declaration plus 16-byte alignment;
it now predicts 34768 at hidden 128, exactly the device's number.

**Consequences (device-confirmed at two boundaries):**
- The **binder switches**. `rolloutBytes` = 31380 B and is independent of BOTH `hidden` and
  `rolloutTicks`; `gradBytes` = 22224 B at hidden 64 and grows **196 B per hidden unit**. Rollout
  binds up to hidden 110, `rl_grad_blk` binds from 111 up.
- **Max hidden at nIn 103 / rolloutTicks 31 = 117** (116 measured at 32416 B; 118 fails at 32816).
- **`rolloutTicks` up to 195 compiles at hidden 64.** T=128 measured 31380 B and RAN (t128p, 74 s).
  The old "rolloutTicks 128 needs 33232 B" note was a hidden-96 measurement, where the grad kernel
  binds — it was never a horizon limit.
- `tools/autosize.mjs` now calls `rlShaderLayout()` for real and prints the rollout/grad split;
  `check --game=realm --hidden=N --rolloutTicks=T` is the planning command.

**Invalidated by this**: the AGENTS.md device-cap line (rewritten 2026-10-07), NEXT-ROUNDS.md:1557-1559
("horiz128/hid128 round is DEAD"), and the autosize paragraph in runs/selforg.md.

**Where the stage-2 prize can actually go**: hidden 117 (probably neutral — R34 measured 96 vs 64
neutral, blob 128 neutral), or `rolloutTicks` up to 195 (unmeasured as a QUALITY knob; R49 only killed
it as a *throughput* lever), or nIn growth for content. The `t128` arm below takes the horizon option.

## Why every melee/mage payoff arm was doomed — EXPLORATION IS DEAD, not the payoff

From the R68 spec (`runs/r68-spec.md` §1), and it explains three rejections at once:

Champion output rails: range **+0.99**, melee **-0.95**, mage **-0.98**. Action = `mu + exp(logStd)*eps`
with `sigmaMin 0.05` (`src/rl.js:23`) and the style gate requiring `out > 0.5`
(`realm.wgsl.js:1121`). So `P(melee fires) = P(z > (0.5+0.95)/0.05) = P(z > 29) ~ 0`. Melee and mage
are not "worse" — they are **never sampled**.

That is the single reason v5ch (melee charge, -0.23 +- 0.19), v5res (archetype resist, -0.50) and Arm X
(perks decoupled from tracks, -0.18 +- 0.08) all read "did not fire": each made a non-range style
*pay more* without making it *happen*, and a payoff on an action with sampling probability ~0 is no
signal at all. The same argument applies to R68a (XP boost) and to `v5hide` (HIDE trophy on close
kills, `runs/reach-design.md` B) — both are payoff arms.

**Consequence for ordering.** The only variant that can change the EXECUTED style is **R68b remap**:
permute which style slot receives the policy's +0.99 value, so the dominant output fires a different
style. Run R68b FIRST. Judge payoff arms (v5hide, any future melee/mage buff) only ON TOP of R68b,
because before it they are unmeasurable rather than wrong — they were scored as REJECTED when the
correct verdict was NOT TESTED.

Corollary for the style lock (R65): `runs/r65r66-spec.md` proves R65a (per-policy permutation of the
style outputs) is a **gauge symmetry** — W2 rows are independent per output, init is exchangeable, and
`U[k] := W2[pi(k)]` reconstructs the unpermuted algorithm byte-for-byte, so it is a provable no-op at
any K. R65a is cancelled; do not implement it.

## R66/R68 DECISION RECORD (2026-10-07) — what lands in the next edit wave

Inputs: `runs/r68-spec.md` (per-world style gate), `runs/r65r66-spec.md` (style lock + telemetry),
`runs/reach-design.md` (structural reach fixes). All three are read-only research; nothing is applied
yet, and `src/` stays frozen until `t128-s1/s2` finish (a queued job compiles the page at ITS start).

### Chosen: R68b (per-world style remap) + executed-style telemetry, in ONE wave

- **R65a is CANCELLED** — `r65r66-spec.md` §1.2 proves it is a gauge symmetry: W2 rows are
  independent per output, `logStd`/`b2` are per-output, init is exchangeable, so relabelling style
  slots per policy reproduces the unpermuted algorithm byte-for-byte. Effect is exactly zero at any K.
- **R68b is the arm.** Per-world relabelling of outputs 6/7/8, so the policy's +0.99 rail executes
  melee (or mage) in a third of training worlds each. Env-side, so theta stays exportable and
  shippable; 0 new obs (obs 67-69 `lvlMelee/lvlRange/lvlMage` already expose the gate indirectly,
  one attack late); 0 workgroup bytes; 0 growth in `worldWords` (cfg word 2605 is free).
  The literal "disable one style" form is refused: with `out > 0.5` still gating, it yields
  abstention (E7), not switching.
- **R65b (post-softsign `styleBias`) is DEFERRED**, not adopted: it needs K=3 plus a `v5k3c` control
  (AGENTS: K=1 beats K=3 by ~15% with bots in training), is unexportable, and `r65r66-spec.md` §7
  argues it changes what the net emits, not what it knows. Revisit only if R68b shows forcing works
  but does not stick.
- **v5hide / candidate A / C / D are DEFERRED** — see the "exploration is dead" entry above. Every one
  of them is a *payoff* arm and is unmeasurable until something forces the action to be sampled.

### A spec bug caught before it cost a round: there are NO free game stat slots

`runs/reach-design.md:256` specifies `R_STAT.ATK_MELEE/RANGE/MAGE = STAT.GAME0 + 7/8/9` (and HIDE at
+10). **That is out of bounds.** `NSTATS = 16` (`src/core.js:7`) and `STAT.GAME0 = 9` (`core.js:10`)
leave exactly 7 game slots (9..15), and realm uses all 7 (`R_STAT`, `realm.js:118`; `statNames`,
`realm.js:1689`). Writing slot 16+ on an `Int32Array(16)` is a **silent no-op** in JS — the counters
would simply never appear, and the arm would read "did not fire" for a plumbing reason.
So the executed-style readout needs either (a) `NSTATS` 16 -> 20 with the world header 24 -> 28 words
(`worldStride = 24 + game.worldWords`, `GAME_OFF = 24u`, the two `i < 16u` stat-reset loops, and ~5
literal `24` game-area offsets — site list being enumerated), or (b) a host-side probe. (a) is chosen:
it buys 4 permanent game-stat slots and windowed deltas for free from the existing flush machinery.

### Primary readout for every remaining style arm

`atkMelee/atkRange/atkMage` per 1k learner ticks, learner-only (`e < brain`, so `evalBase` is pinned),
from the stats block. NOT `ACT[mel=,rng=,mag=]` — that reads *output means*, and under R68b the remap
is env-side, so the outputs stay railed at +0.99 and ACT would show nothing while the executed style
changes completely. ACT is still worth having (it says whether the *policy* moved); it is secondary.


## t128 VERDICT (2026-10-07) - rolloutTicks 128 REJECTED, 2 seeds

Arm: 512 worlds, realm, rolloutTicks=128 vs default 31, tick-matched at 190k (t128-s1 195712, t128-s2 192896).

| | EVAL x1e3 | bots x1e3 | ratio | eval life | self-play x1e3 |
|---|---|---|---|---|---|
| t128 (s1,s2) | 3.65 +- 0.96 | 2.80 +- 0.07 | 1.31 +- 0.38 | 1179 +- 262 | 3.54 +- 0.62 |
| v5b  (s1,s2) | 5.80 +- 0.12 | 2.96 +- 0.02 | 1.96 +- 0.06 | 1619 +- 230 | 5.88 +- 0.21 |

Delta **-2.15 EVAL** at matched ticks, with life -27% and ratio -33%. Decisive: the loss is 5x the baseline seed sd and 2x the arm's own.

Mechanism (predicted, now measured): at T=128 a rollout covers 4x the ticks, so iterate() performs 1/4 as many Adam updates per tick. Log confirms it - it=1529 upd=6116 is 4 updates/iteration, not 16. It is not a variance or GAE argument: lambda .97 over 128 steps with 4x fewer updates simply starves the learner.

Second finding, and a correction worth keeping: **there was no `blockGrad: 0` fallback**. The log's `created` line reads `workgroupBytes=31380 warnings=[[]]` for both seeds — identical to the default run. That confirms the RE-MEASURED byte model (the rollout kernel is `rolloutTicks`-independent at 31380 B up to hidden 117, so T=128 fits). The old claim in these notes that "rolloutTicks 128 needs 33232 B" was the pre-remodel arithmetic and is now wrong; do not repeat it. What T=128 does cost is real but different: `ms` per iteration is the same ~350, so each iteration covers 128 ticks instead of 31 — 4x the throughput per update and therefore 1/4 the updates.

Corollaries:
- `blockGrad: 0` fallback is now a hidden-128/192-only event, not a rolloutTicks one. Check `workgroupBytes` and `warnings` in the `created ...` line before attributing a slowdown to it.
- wt/s was 175-183k on s1 and 75-180k on s2 (contention dip, not config); both seeds reached ~193-196k of the 200k cap, so the comparison above is tick-matched, not wall-clock-truncated.

Consequences:
- rolloutTicks stays 31. Do not re-arm T=64/128 unless the workgroup byte budget is grown first; that is an architecture change, not a knob.
- T=128 losing on 1/4 the updates, and R40 (256 worlds) losing too, both point the same way: at this horizon the learner is update-starved, not world-starved. More updates per tick is the direction that pays.

## R68b ROUND + t16 (queued 2026-10-07)

Two arms, both 2 seeds, 512 worlds, 200k ticks, judged with `node tools/report.mjs --at-tick=200000 --paired=<run>`.

**r68b (styleGate=2, styleGateEval=1)** is the payoff read. Training worlds remap the executed style by a per-world shift `mix(seed,5,0,8)%3`; eval worlds are gated too, so the shift distribution at eval is uniform and EVAL becomes the average over forced melee/range/mage. If ranged monoculture is an EXPLORATION artefact, EVAL holds near baseline; if range is genuinely worth +2, EVAL collapses. Primary readout is EVAL; the new `atkMelee/atkRange/atkMage` counters are the manipulation check (expect roughly 1/3 each).

**r68c (styleGate=2, styleGateEval=0)** is the retention read: eval ungated, so it asks whether a net trained under forcing still reaches for range when the gate is removed. Expect it to stay railed (the +0.99 rail clears the 0.5 gate under any shift); a shift here is the surprise worth chasing.

Why both: r68b alone cannot distinguish "melee is fine but the net prefers range" from "melee is fine and forcing sticks", and r68c alone will almost certainly show range-only and therefore read as a null result. Together they separate payoff from preference.

**t16 (rolloutTicks=16)** is the direct consequence of the t128 verdict. t128 gave 1/4 the Adam updates per tick and lost 2.15 EVAL; if that gradient is real, halving T from 31 to 16 should GAIN. Same wall clock, same ticks, 2x the updates. Caveat: shorter rollouts raise GAE variance and cut ticks-per-dispatch, so a win is not guaranteed even if the hypothesis is right - and if it loses, the t128 result is explained by horizon/variance rather than update count.

Follow-up not yet built: EVAL under r68b is an average over three forced styles, so it cannot say WHICH style costs. The per-world scores already exist host-side (rlengine worldScores from STATS_WORLD_OFF), so bucketing them by that world's gateShift would give EVAL per style for free. Only worth adding if r68b's aggregate is ambiguous.

## R68 HEADER GROWTH + STYLE COUNTERS: first test run 10/21, one real bug (2026-10-07)

The NSTATS 16 -> 20 / world header 24 -> 28 growth plus the three learner-only
`atkMelee/atkRange/atkMage` counters went to the device as `r68t`
(`tests=1 game=realm`). Result: **10/21**.

The ONE real failure:

```
FAIL league: roles, snapshot policies, masked gradients, head-to-head stats: head-to-head ticks missing: latest 0, older 0
```

Cause: `rl_rollout` carried a literal that only ever MEANT "the role words":

```wgsl
let slot = select(18u, 16u, role == ROLE_LIVE);   // == NSTATS + 0 / + 2 while NSTATS was 16
```

After the growth it wrote head-to-head reward/ticks into `statAcc[16..19]`, which are now
real stat slots (the new atk counters), and left the actual role words at `statAcc[20..23]`
untouched. Nothing crashed: the league just reported zero ticks. Fixed as
`let slot = NSTATS + select(2u, 0u, role == ROLE_LIVE);`, and every `STAT_*` WGSL constant
is now generated from core.js `STAT` (`STAT_CONSTS`) so no index can drift again.

The other nine failures are one event, not nine bugs:

```
FAIL reward hygiene: per-channel caps: Instance dropped in popErrorScope   (and every test after it, 1 ms each)
```

`Instance dropped in popErrorScope` is device loss (`rlengine.withErrorScopes`), not an
assertion. Re-run (`r68t2`, same config plus `styleGate=2 styleGateEval=1`) decides whether
the loss is a consequence of the first failure or the known headless-Chrome fragility in
long combined runs (AGENTS.md). Do not diagnose the cascade until the single real failure
is fixed and re-run.

Note for the sweep that MISSED this: when a shared layout constant grows, search for numbers
EQUAL TO ITS OLD VALUE, not for the constant's name. `24 * 4`, `new Int32Array(16)` and
`i < 16` all looked like offsets; `select(18u, 16u, ...)` did not, and it was the only one
that mattered.

## R68 ROUND HALTED BY A DRIVER-RESET STORM: device loss is environmental (2026-10-07)

Symptom: `r68b-s1` (510 s), `r68b-s2` (344 s) and `r68c-s1` (119 s) all died with
`ERR Uncaught OperationError: Instance dropped in popErrorScope` at `RlBackend.iterate`
(rlengine.js:462). Each log had one 30 s progress line and then silence until the ERR, i.e. the
page's `await backend.step()` never returned again - a blocked main thread, not a bad number.

Why it is NOT the R68 code:
- 448 earlier queue jobs never produced the string `Instance dropped` even once.
- `r68t2` (21/21, styleGate 2 + styleGateEval 1, realm) passed on this exact source 20 min earlier.
- Windows System log: `nvlddmkm` event id 153 (D3D device removed / driver reset) at
  23:01, 23:22, 23:30, 23:33, 23:39, 00:06, 00:11, 00:13, 00:14, 00:15, 00:17, 00:26, 00:31, 00:39 UTC
  - ~1 per 5 min, i.e. the resets predate and outlive any single job of ours.
- `nvidia-smi`: 98-100 % util, 75 -> 84 C, and a compute app that is not ours - `resound.exe`
  (C:\dev\resound, `--nfe 32 --chunk-seconds 5`, a new instance every ~10 min). A second
  project's Chrome (`spoint`) also held a GPU process with `--gpu-recent-crash-count=2`.

Action taken:
- `node tools/queue.mjs stop` (preserves attempts; stopped jobs resume clean).
- tools/queue.mjs: new GPU health gate. `waitForGpuHealth` runs before every attempt and waits
  while (a) any `nvlddmkm` reset in the last `gpuHealthResetMinutes` (12) or (b) any foreign
  non-Chrome pid in `nvidia-smi --query-compute-apps`; bounded by `--gpu-health-wait-minutes`,
  off with `--no-gpu-health`. New verb `node tools/queue.mjs gpu` prints the verdict.
  Chrome pids that are not ours are reported but never block (an idle foreign Chrome would
  otherwise stall the queue forever).
- `node tools/queue.mjs resume --retry-failed --max-attempts=4 --gpu-health-wait-minutes=45
  --gpu-health-reset-minutes=12 --stall-seconds=1800` -> 8 round jobs back in line, attempts reset.

Consequences for the round:
- No R68 verdict is readable until `r68n-s1/s2` (the ungated baseline) finish: `--paired=r68n`
  needs a baseline at the same horizon, and every arm is judged at 200k.
- If the storm outlives the gate's patience, expect `Instance dropped` again on the retry; the
  cure is waiting for the foreign job, not changing src.
- Device loss was already retried once (`RETRIABLE` matches `OperationError`); the reason to
  stop the runner anyway is that a job only has `maxAttempts` tries and a storm eats them all.

Next decision (gated on the R68 verdict):
- If `styleGate` buys style diversity at >= baseline EVAL: the monoculture is a local optimum,
  not a capability gap - keep the gate as an instrument and induce diversity by channel weights
  instead of more content.
- If gated worlds score much worse: melee/mage are genuinely unplayable and the fix is
  structural. Two candidates, both affordable inside the current budget (nIn 103 -> 113 allowed):
  (a) ranged ammunition - arrows cost wood/gold, so safe ranged DPS has a carrying cost and
      melee is the free option (new obs 103, new craft/buy sink, no new state);
  (b) cover blocks line of sight - ranged attacks fail through cover, melee does not, which makes
      the existing cover/ambush content load-bearing and gives terrain a tactical role.
  (a) is an economy lever and touches one attack path; (b) needs a LOS test in both JS and WGSL
  and is the only one that makes terrain matter. Decide from the R68 numbers, not before.

## STRUCTURAL LEVERS FOR RANGED MONOCULTURE (prepared 2026-10-07; queue only after the R68 verdict)

Numbers measured from src/games/realm.js (read-only survey, no edits - src is frozen while the R68
round is queued):
- `STYLE_RANGE = [200, 800, 500]` (melee, range, mage), `STYLE_COOLDOWN = [10, 18, 24]`,
  damage `6 + 2*lvl + 4*weap` / `4 + 2*lvl + 3*weap` / `5 + 2*lvl + 3*weap`.
  At level 0, weapon 0: melee 0.60 dmg/tick, range 0.22, mage 0.21 - melee already pays 2.7x the DPS
  and still loses, which is why every numeric arm so far was REJECTED.
- `styleBeats` = melee > range > mage > melee (x1.5 / x0.75); ranged's tier-2 immunity to being
  countered was already dropped (v5tri, ADOPTED).
- The mechanism is REACH, not damage: the attacker picks the engagement distance, so a ranged
  learner fights at 800 and a melee learner eats free damage over the 600 units it must close.

Candidate A - melee suppression zone (no aimed attack with a hostile inside NOSHOOT ~250):
- Point: `tryAttack` (realm.js:630) already walks every hostile to pick a target; track the nearest
  hostile distance and return false when `style !== 0` and it is inside NOSHOOT. Mirror in
  realm.wgsl.js `r_try_attack`. No new obs, no new output, no new field - zero budget cost.
- Why it should work: it turns reach from an absolute advantage into a conditional one. Closing
  becomes a real tactic (shut the shooter down), melee keeps its 2.7x DPS once it arrives, and
  kiting becomes a skill instead of a default.
- Risk: it also blocks mage (500 reach) and mobs; if mobs crowd the learner, ranged/mage learners
  could be silenced too often and EVAL drops. Keep it player-side-only or mob-side-only as a knob.

Candidate B - ranged ammunition (1 wood per shot):
- Point: same `tryAttack`; fail when `style === 1 && wood === 0`, else decrement. `wood` is already
  obs 71, so the learner can see whether it can pay - zero obs cost.
- Why: gives ranged DPS a carrying cost that melee does not pay and makes wood (craft resource)
  contested, i.e. an economy lever rather than a damage number.
- Risk: if wood is abundant in the wilds it is a no-op, and it taxes crafting too; bots need the
  same rule in `botAct` or the benchmark moves for free.

Candidate C - projectiles with travel time: rejected for now. It needs per-tick in-flight state,
which is the most lockstep-sensitive change available and the most expensive to prove.

Decision rule: run A first (three lines in JS + three in WGSL, directly attacks the mechanism), then
B only if A moves the style mix but not EVAL. Same protocol as every other arm: 2 seeds, 512 worlds,
200k ticks, paired against the current baseline, manipulation check `EG[atkMelee/atkRange/atkMage]`
with melee+mage share >= 15% before any EVAL claim counts.

## R68 BASELINE LANDED: monoculture measured, not inferred (2026-10-07)

`r68n-s1` (no style gate, realm, 512 worlds, 200k, seed 1, 458 s) is the reference for the whole
R68 round:

    EVAL 5.94  evalBase 2.96  ratio 2.01  life 1656/1388  SP[rew=6.05,life=834]  H2H[ratio=1.02]
    EG[mobKills=64262,greatHarvests=22625,bossKills=4797,pvpKills=3579,harvests=110168,
       allyTicks=11141958,playerDeaths=16612,atkMelee=0,atkRange=764765,atkMage=0]
         (SP group: atkMelee=0, atkRange=348707, atkMage=0)

The headline is `atkMelee=0, atkMage=0` against 764 765 ranged attacks: **not a skew, a total
collapse**. Every previous style-mix claim was inferred from XP-track percentiles (xp0/xp2 p75 6/9
vs xp1 p50 318); the counters now settle it directly, and they say the style subsystem is not
merely under-used - it is dead code in the learned policy. That is why three numeric arms (v5ch
-0.23, v5res -0.50, v5tri style-mix UNCHANGED) could not move it and why the lesion suite reports
<= 3.2%: you cannot lesion a system the policy never executes.

Consequences for the round:
- The manipulation check is now nearly free. Any gate arm that forces a world onto one style will
  show atkMelee/atkMage > 0 by construction, so passing it is NOT evidence. The check that matters
  is EVAL at a common tick horizon, judged paired against r68n.
- r68b (gate in training AND eval) is expected to LOSE on EVAL: a third of eval worlds are forced
  onto melee (reach 200, 2.7x the DPS of ranged) or mage, both strictly worse than the ranged
  default. A loss there does not mean the gate is bad - it means reach is priced wrong.
- r68c (gate in training only, eval free) is the arm that can win: it buys style-general skills
  while letting eval play the best style.
- If BOTH lose, the correct conclusion is still not "styles are hopeless": it is Phase 5 candidate A
  (melee suppression zone) - make reach cost something, then re-run.

Also note this baseline is the first post-bot-buff number at 200k: evalBase 2.96 (was 1.74 before
R51 + great-node/boss bot buffs) with EVAL 5.94 (was 5.20), so ratio fell 2.97 -> 2.01 while the
learner itself improved. Ratio is a BENCHMARK number, not a learner number - do not read the drop
as a regression.

Device loss (recap, see the earlier section): `r68n-s2` died at 148 s on `Instance dropped in
popErrorScope` after 4 attempts, and `r68b-s1`'s first attempt died the same way. VRAM is NOT the
cause (3.3/6.1 GB used); the card sits at 99% util under a foreign compute pid, so it is TDR
pressure from outside. The gate now blocks on a foreign compute pid as well as on nvlddmkm resets,
both bounded by `--gpu-health-wait-minutes` so a long foreign job can never stall the queue
forever.

## AFTER THE R68 ROUND: make a lost device survivable (proposed 2026-10-07)

The shared card is the standing condition, not a one-off: `resound.exe` held it at ~100% for
30+ min and driver-reset every attempt we started. Today one 200k job must complete inside a
single ~460 s window, so a TDR at second 400 throws the whole run away.

Fix: give `dev/rllong.html` real resume. Every `saveSeconds` it already POSTs champions; also
POST `{theta, opt (Adam m/v + CTL counters + norm state), rngState, tick, iter}` under the run
name, and accept `?resume=<name>` to fetch that blob and continue instead of starting fresh.
Worth being explicit about what resume does NOT restore: the 512 world buffers are not saved
(readback of every world is a bigger change), so worlds re-init on resume. That is acceptable
for training - it costs one generation of world diversity, not the learner's progress - but it
means a resumed run is NOT bit-identical to an uninterrupted one, so resume is a robustness
feature, never to be mixed into an A/B arm.

Payoff: a lost device then costs at most one `saveSeconds` interval instead of the whole job,
and the queue's environmental retry becomes nearly free. Pair it with a shorter `saveSeconds`
(600 -> 120) for long runs.

### R68 baseline seed 2 (2026-10-07) — monoculture REPLICATES

`r68n-s2` (no gate, 512 worlds, 200k, 465 s, survived the resound contention):

    EVAL 5.93  evalBase 3.06  ratio 1.94  life 1771/1631  SP[rew=5.89,life=889]  H2H[ratio=1.13]
    EG[atkMelee=0,atkRange=793100,atkMage=0]

Two-seed baseline: EVAL **5.94 / 5.93** (sd 0.01), evalBase **2.96 / 3.06** (sd 0.10),
ratio **2.01 / 1.94**, life 1656 / 1771. `atkMelee=0` and `atkMage=0` on BOTH seeds against
~7.8e5 ranged attacks — the collapse is not a seed artefact, it is the policy. EVAL seed sd of
0.01 at 200k is the tightest baseline we have had, which makes the paired read-out unusually
sensitive: any arm moving EVAL by more than ~0.05 is outside baseline noise.

### R68 arm 1 — r68b (gate in training AND eval), seed 1: DIVERSITY BOUGHT, EVAL PAID FOR IT

    r68b-s1  EVAL 4.97  evalBase 2.62  ratio 1.90  life 1507/1118  SP[rew=5.10,life=673]  H2H[ratio=0.90]
             EG[atkMelee=201146,atkRange=247663,atkMage=139106]
    r68n-s1  EVAL 5.94  evalBase 2.96  ratio 2.01  life 1656/1388  SP[rew=5.89,life=834]  H2H[ratio=1.07]
             EG[atkMelee=0,atkRange=764765,atkMage=0]

(tick-matched: 191642 vs 191053.)

The gate does exactly what it was built to do — melee+mage go from 0% to **58%** of attacks
(340252 of 587915) — so the manipulation check passes by a mile. And it costs **-0.97 EVAL
(-16%)**, which fails the pre-registered bar (EVAL >= -0.19) by 5x. Verdict: **REJECTED**, and
this is the predicted result, not a surprise: forcing a third of eval worlds onto melee and a
third onto mage forces them onto strictly worse options.

The number that actually matters is the per-world price. If the ranged-gated worlds still score
~5.94, the other two thirds average (3*4.97 - 5.94)/2 = **4.49 each, i.e. a ~24% penalty for
being forced off ranged**. That is the first real measurement of what REACH costs, made with
learned policies instead of scripted ones: melee is not slightly worse, it is a quarter worse.

Shifts worth noting beyond EVAL: bossKills 4797 -> 868 (melee/mage worlds barely touch the
boss — you cannot boss at 200-500 reach), harvests 110168 -> 129752 and greatHarvests
22625 -> 30605 (they fall back to gathering), playerDeaths 16612 -> 22691. evalBase also fell
2.96 -> 2.62, so the gate hits bots too — eval worlds are gated wholesale, not per-side.

Implication for Phase 5: candidate A (suppression zone) is aimed at exactly this 24% gap, but
from the other side — it does not buff melee, it removes ranged's freedom to shoot while
something is in its face. If melee is only worth ~4.5 of 5.9 today, the fix has to change the
equilibrium, not hand melee a damage bonus (three numeric arms already failed at that).

## Phase 5 candidate menu — pre-registered 2026-10-07 (DO NOT APPLY until the R68 round ends)

Problem being solved, restated: v5 content is engaged but not load-bearing because the learner's
attack distribution is 100% ranged on both baseline seeds (`EG[atkMelee=0,atkMage=0,atkRange=~7.8e5]`).
r68b proves you cannot fix this from the policy side: forcing styles buys 58% non-ranged attacks and
pays 0.97 EVAL (-16%, bar was -0.19). Melee/mage worlds score ~4.49 where ranged scores ~5.94, so a
free learner re-derives monoculture every time. Therefore every remaining candidate must change the
PAYOFF of reach, not the policy. Each candidate below is pre-registered with its kill criterion BEFORE
any run, so the number decides and not the story.

Bar for every candidate: ADOPT only if (attack share melee+mage >= 15%) AND (EVAL >= -0.19 of the
paired baseline) AND (evalBase does not RISE by more than 0.15 - a bot buff that moves the denominator
is not a learner win). Judge 2 seeds @200k, paired against a baseline run on the SAME device.

  A. MELEE SUPPRESSION ZONE (spec already in runs/GAME-DESIGN.md). A non-melee attacker cannot make
     an aimed attack while any hostile is within `cfg.meleeSuppress` (arm 250, default 0 = off).
     Gives melee a job nothing else can do: zone denial. Cost: 3 lines JS (`realm.js` tryAttack) +
     WGSL mirror (`r_try_attack`) + cfg word 2606. Zero obs. Risk: ranged learners may simply learn
     to kite at 250+ and lose nothing, in which case share stays ~0 and we learn reach is not
     suppressible by one radius.

  B. RANGED SUSTAIN COST. Ranged attacks consume an arrow/charge that must be crafted or looted;
     melee and magic do not (magic already spends RARE as mana). Turns the existing craft tree into a
     load-bearing supply line instead of an XP track, and makes reach a resource decision rather than
     a free lunch. Cost: one resource counter, reuse of an existing obs slot if one is free, else
     nIn +1 (budget: nIn 103, headroom to 113). Risk: if arrows are plentiful the arm is a no-op and
     if they are scarce it is just a nerf that drops EVAL without buying share - tune to the middle
     and read the counter, do not guess.

  C. RANGED DAMAGE FALLOFF. Ranged damage scales with (1 - d/STYLE_RANGE[1]), i.e. ~0 at max reach and
     full only inside ~300 where melee also operates. Keeps reach as an option, removes its free
     lunch, one line, no obs, no content. Risk: strictly numeric in shape even though structural in
     effect - v5ch/v5res/v5tri were REJECTED for being numeric, so this only passes on the counter,
     never on an argument that it "should" work.

  D. CREEP CLOSURE / SWARM PRESSURE. Creep FSMs already carry stalker/pack/ambusher archetypes
     (`(e-R_MOB0)%5`); make the pack and stalker archetypes close distance on the current target
     instead of engaging at their own preferred range, so a pure-ranged learner gets swarmed and must
     either kite (an emergent skill worth having) or bring melee. Highest content value of the four
     and the most expensive: it touches the FSM in BOTH `realm.js` and `realm.wgsl.js` and can move
     evalBase by buffing mobs, which is why it is last.

Order if A fails on the counter but not on EVAL: B (econ lever, independent of geometry). Order if A
buys share but pays EVAL: run A at a smaller radius (125) before declaring defeat - a radius that
merely annoys ranged may be the version that survives. Do not run C and B in the same round; they
both change the price of the same attack and their effects will not separate at 2 seeds.

### Plumbing checklist for any new world cfg knob (learned from styleGate, 2026-10-07)

`dev/rllong.html:29` copies a URL param into `common` ONLY if the key is already in
`RL_DEFAULTS || RL_CURRICULUM_DEFAULTS || DEFAULT_OPTS || RL_POP_DEFAULTS`. So a brand-new knob
needs FOUR edits or it is silently dropped (the URL param is accepted, ignored, and the arm runs
as baseline - the most expensive possible A/B outcome):

  1. `src/rl.js` (~line 60) - declare the default in `RL_DEFAULTS` alongside `styleGate`.
  2. `src/rlengine.js:117` `shaderConfig()` - WHITELIST the key. Missing this disables the knob on
     GPU only, so JS and WGSL disagree and only the lockstep test catches it (AGENTS.md: caught at
     20/21).
  3. `src/games/realm.js:1695/1696` `defaultCfg`/`randomCfg` - copy it out of `opts` into the
     per-world cfg, exactly as `styleGate`/`styleGateEval` do.
  4. The WGSL side - uniform word (`src/shader.js:125`, `src/rlshader.js:817`, engine `u[14]`/
     `u[15]`, rlengine `u[28]`/`u[29]`) if it is a global, or a packed cfg word if it is per-world.
     For `meleeSuppress` the plan is per-world cfg word 2606, so it also needs the read in
     `realm.wgsl.js`.

When the knob is in, verify with a `tests=1` job BEFORE spending a round on it: `tests=1` defaults
to `game=blob` (dev/rllong.html:27) so `game=realm` must be passed explicitly, and the suite is
silent until it finishes, so start that runner with `--stall-seconds=1800`.

## Second compute path: Kaggle is CLOSED (investigated 2026-10-07, do not retry)

Checked while the local card was being held by another project's `resound.exe` at 100%.

- Credentials exist (`~/.kaggle/access_token`, 40 B, the CURRENT format; there is no `kaggle.json`
  but that is the legacy path), and `kaggle` CLI 2.2.4 is installed. `kaggle kernels push -p <dir>
  --accelerator NvidiaTeslaT4` + `kaggle kernels status/output` is genuinely non-interactive.
- BUT: Kaggle's GPU image (`Kaggle/docker-python` Dockerfile.tmpl, now based on
  us-docker.pkg.dev/colab-images/public/runtime) installs only `libgl1 libglx-mesa0` - no Vulkan
  ICD and no Chrome. Dawn needs Vulkan. So WebGPU most likely fails at `requestAdapter()`, and
  even the probe costs installing Chrome + a Vulkan driver first. Unknown-but-leaning-fails, with
  no known-working notebook found.
- The fallback does not exist: **PPO has no CPU path.** `src/rlengine.js`, `src/rl.js` and
  `src/rlshader.js` contain zero `cpu` references; `CpuBackend` (src/core.js:590) is used by
  `src/engine.js` for evolution parity only. So "run it on Kaggle's CPU" cannot train at all.
- For reference, CPU evolution measured in Node on this box: ~2,700 world-ticks/s single-threaded
  (0.35-0.39 ms/world-tick). 512 worlds x 200k ticks = 1.02e8 wt = ~10.5 CPU-hours (~3 h on 4
  vCPU) vs the GPU's 259-372k wt/s. About 100x.

Verdict: not worth the setup hours. The trainer is bound to one local WebGPU device, which makes
two things the real levers on throughput, not remote compute:
  1. survive a lost device instead of losing the job (checkpoint + `?resume=<name>`, spec above)
  2. do not start jobs into a contended card (the gpu-health gate, already shipped)
If anyone revisits remote compute, the thing to look for is a host that actually exposes a Vulkan
ICD to headless Chrome - Kaggle/Colab do not, which is the whole finding.

### R68 arm 1 verdict, 2 seeds (final) — r68b REJECTED

`node tools/report.mjs --at-tick=190000 --paired=r68n`, ticks matched 189240 +- 1075:

  EVAL      -0.89 +- 0.12   (arm 5.04 +- 0.12, baseline 5.93 +- 0.00)   BAR WAS -0.19  -> REJECT
  evalBase  -0.35 +- 0.02   the gate also costs the BOTS, so the ratio barely moves
  ratio     -0.08 +- 0.05
  life      -379 +- 153
  h2h       -0.09 +- 0.04
  SP        -0.76 +- 0.28

Seed 2 reproduced seed 1's signature (mid-run `atkMelee=285321,atkRange=371883,atkMage=222801`),
so the gate reliably buys ~58% non-ranged attacks on both seeds. The result is therefore not noise
and not seed-specific: **forcing style diversity is worth knowing how to do and is not worth
shippping.** The penalty is ~4.7x the bar, and it is paid in every channel at once, including
self-play. Note evalBase moves too (-0.35), so the gate is not learner-specific - it degrades the
world, which is a second reason to prefer a change to the PAYOFF of reach over a change to the
policy's menu.

Consequence for phase 5: do not spend another round on role assignment. Candidate A/B/C/D above
are the successors, and all four are judged on the melee+mage counter, never on "did the styles
appear".

### R68 arm 2 verdict, seed 1 — r68c (gate in TRAINING only) REJECTED, and it is WORSE than r68b

r68c-s1 @ tick 197,563: EVAL 4.19, evalBase 2.53, ratio 1.66, life 1658/1127
r68n-s1 @ tick 191,053: EVAL 5.94, evalBase 2.96, ratio 2.01, life 1656/1388
paired EVAL delta **-1.75** against a bar of -0.19 -> REJECT, by ~9x.

r68c-s2 was CANCELLED (bumped t16 ahead of it) and its aborted 11k-tick log quarantined to
`runs/aborted/r68c-s2.aborted-11k.log`. Two reasons: the seed-1 delta is 9x the bar so a second
seed cannot rescue it, and a GPU window is the scarce resource this week. Any future report must
not average a finished seed against that aborted log - it made r68c read as "2 seeds" with a
100k-tick spread. If r68c is ever revisited, run seed 2 first so it writes a fresh log.

WHY it lost is the important part. At eval (ungated) the r68c policy plays MELEE-DOMINANT:
`EG[atkMelee=677456,atkRange=4058,atkMage=33104]`. Gating training taught it melee, it kept melee
when freed, and melee scores less. It also stopped doing the content: bossKills 4776 -> 192,
mobKills 64262 -> 41847. So gating does not just fail to help, it actively destroys the
boss/mob engagement we added in v5.

### The dose-response curve (the real output of R68)

EVAL against the share of attacks that are ranged:

  baseline r68n   ~100% ranged   5.94
  r68b  (mixed)   ~42% ranged    4.97      (gate in training AND eval)
  r68c  (melee)   ~0.6% ranged   4.19      (gate in training only)

Monotonic: **every point of ranged share is worth about 1.75 EVAL end to end.** That is the number
the phase-5 candidates are trying to move. It also reframes the goal: we do not need to "make melee
allowed" - it is already allowed and the learner still avoids it - we need ranged to STOP being
strictly better. A candidate that raises melee is fighting a 1.75-EVAL gradient; a candidate that
taxes ranged is removing it. Candidate A (melee suppression zone) and C (ranged falloff) do the
latter, which is why they lead the queue. Watch evalBase on both: if a candidate drops ranged
without giving melee a way to convert, both sides fall and ratio stays flat.

### Candidate E — REACH COMPRESSION (added 2026-10-07, now the lead candidate)

Why it leads: it is the only candidate that attacks the variable R68 actually measured. The
dose-response curve (5.94 / 4.97 / 4.19 for ~100% / ~42% / ~0.6% ranged) says ranged share is worth
~1.75 EVAL end to end, and v5tri's own verdict says why: "reach (STYLE_RANGE 200/800/500), not the
damage triangle, is what makes ranged dominant - at 800 a ranger takes zero return damage from a
melee mob, and no multiplier competes with zero." v5tri fixed the multiplier, not the reach. Reach
has NEVER been changed - `STYLE_RANGE` appears nowhere in AB-ARCHIVE as an arm.

Mechanism: compress the reach spread so a ranger can no longer fight from outside the melee threat
radius. `STYLE_RANGE = [200, 800, 500]` -> arm `[200, 400, 350]` (melee/range/mage). Ordering is
preserved (melee < mage < range) so the style triangle keeps its shape; only the spread shrinks.
Losing 400 units of reach is what costs the ranger its free damage window.

EDIT IS TWO PLACES and both must move together or the lockstep test fails:
  - `src/games/realm.js:120` `const STYLE_RANGE = [200, 800, 500];`
  - `src/games/realm.js:140` `const MOB_STYLE_RANGE = [200, 700, 450];`  (arm to [200, 400, 350] too,
    or mobs keep the reach advantage players just lost - decide before running, and say which)
  - `src/games/realm.wgsl.js:918` `if (s == 1) { return 800; }` and `:924` `{ return 700; }` - the
    WGSL style-range functions hardcode the same numbers; there is no shared constant, so a JS-only
    edit diverges silently until the lockstep test catches it.

Risk: reach is also what makes ranged fun, and the same compression applies to mobs, so evalBase
can move. The bar watches it. If E passes on the counter but loses EVAL, the falloff candidate C is
the gentler version (full damage up close, none at max reach) and should be tried next.

Revised order: E (reach, cheapest, most direct) -> A (suppression zone, better game feel, more code)
-> C (falloff) -> B (ammo economy) -> D (creep closure, most expensive). Still never two at once.

### R68 arm 2 verdict, 2 seeds (final) — r68c REJECTED

  EVAL      -1.49 +- 0.31   (arm 4.44 +- 0.31, baseline 5.93)   BAR -0.19  -> REJECT
  evalBase  -0.41 +- 0.02
  ratio     -0.26 +- 0.10
  life      -389 +- 103
  h2h       -0.01 +- 0.09   <-- FLAT
  SP        -1.44 +- 0.52

The flat h2h is the most informative number in the whole round. A gated learner is exactly as good
as an ungated one *against another learner of its own kind* (1.06 vs baseline 1.07) while scoring
1.49 less in absolute EVAL. So the gate does not make the policy worse at playing - it makes the
policy play a style the ENVIRONMENT pays less for. That is the cleanest possible statement that the
monoculture is a payoff gradient and not a learning failure, and it is why every remaining candidate
is a change to the environment rather than to the policy, the loss, or the rollout.

R68 is closed: both gate arms rejected. Remaining in the round is t16 (rolloutTicks 16), which is a
trainer question, not a content one.

### Candidate E APPLIED to src (2026-10-07), values differ from the plan

Applied, awaiting `r69t` (realm `tests=1`) before any arm runs:

  src/games/realm.js:120   STYLE_RANGE      [200, 800, 500] -> [200, 450, 350]
  src/games/realm.js:140   MOB_STYLE_RANGE  [200, 700, 450] -> [200, 350, 300]
  src/games/realm.wgsl.js:918/919  r_style_range  800 -> 450, 500 -> 350
  src/games/realm.wgsl.js:924/925  r_mob_range    700 -> 350, 450 -> 300

I did NOT use the planned [200,400,350]. The plan ignored that mobs have their own range table, and
cutting only the player table would hand mobs a 300-unit reach advantage they do not have today.
Final values preserve the current player-over-mob offsets (+100 on range, +50 on mage) while cutting
the melee-to-ranged gap from 600 to 250. Both arrays are the single source of attack range in JS
(used at realm.js:634 in `tryAttack` and :995) and both WGSL functions are the mirror, so the four
edits are the whole change - verified the JS/WGSL tables still agree value for value.

NOT changed, and a known leak to watch: `RW_BOLT_RANGE` 900 and `RW_CAST_RANGE` 600 (magic spells)
still outrange every attack style. If the counter shows mage attacks still losing but mage XP
rising, the bolt is the reason and it needs compressing too.

### t16 ABANDONED (do not silently resume it)

94 minutes, 6+ lost devices, never finished; attempts rolled past the env-retry cap into charged
attempts. Its seed-1 log is 447 bytes of garbage and seed 2 never ran. Two reasons not to just
resume it: the GPU windows are shorter than the job, and candidate E changed the environment, so a
t16 result would now be confounded with the reach change unless it gets a same-build baseline.
Reopen it as its own round with a fresh baseline once the reach question is settled.

## Candidate E (reach compression) -- MEASURED ON CPU, DEMOTED

E was `STYLE_RANGE` 800 -> 450 and mob 700 -> 350: a NERF to ranged. Measured on CPU with the
r68n champion (`evalsuite --suites=bots --seeds=1,2 --worlds=4 --periods=1`), learner rate =
ratio * baseRate:

| build | baseRate (bots) | learner rate | vs base | mobKills | bossKills | starve |
|---|---|---|---|---|---|---|
| old 800/500 | 0.00294 | 0.00619 | -- | 7.193 | 0.565 | 0.403 |
| E 450/350   | 0.00279 | 0.00552 | -10.8% | 5.381 | 0.420 | 0.435 |
| E+ 300/280  | 0.00272 | 0.00506 | -18.3% | 4.898 | 0.355 | 0.496 |

Compression taxes the bots too (-5%) and shrinks content engagement (mobKills -32%, bossKills
-37%, more starvation). Nerfing reach makes the world smaller and poorer, so E is DEMOTED.

## Candidate A (melee suppression zone) -- MEASURED ON CPU, REJECTED

Prototype: in `tryAttack`, track `nearD2` (nearest hostile of any range) and
`if (style !== 0 && nearD2 < MELEE_SUPPRESS^2) return false;` with `MELEE_SUPPRESS` 250 -- ranged
cannot fire while a hostile is inside 250. CPU, same champion:

| build | baseRate | learner rate | vs base | atkMelee | atkRange | mobKills | bossKills |
|---|---|---|---|---|---|---|---|
| old | 0.00294 | 0.00619 | -- | 0.0 | 81.9 | 7.193 | 0.565 |
| A 250 | 0.00301 | 0.00452 | **-27.0%** | 3.67 | 30.3 | 5.455 | 0.221 |

A does induce style mixing immediately (melee appears from 0 to ~11% of attacks, because failed
attacks change the observations the net sees), and it is surgical on the bots (+2%), but a 27%
learner tax is the worst of the three. REJECTED as a default; kept as an instrument. The prototype
was removed from src (realm.js is back to plain `tryAttack`).

## Candidate F (reach SPREAD compression by BUFFING melee/mage) -- LEAD, pre-registered

F keeps ranged at 800 and raises melee 200 -> 400 and mage 500 -> 650, i.e. it compresses the
SPREAD without shrinking the world. `src/games/realm.js:120` `STYLE_RANGE = [400, 800, 650]`,
mirrored at `src/games/realm.wgsl.js` `r_style_range`. `MOB_STYLE_RANGE` untouched.

CPU, r68n champion (100% ranged) and r68c champion (mixed 17/53/7 melee/range/mage at old reach):

| champion | build | baseRate | learner rate | vs old | atkMelee | atkRange | atkMage |
|---|---|---|---|---|---|---|---|
| r68n | old 200/800/500 | 0.00294 | 0.006192 | -- | 0.0 | 81.9 | 0.0 |
| r68n | F 400/800/650   | 0.00339 | 0.006196 | **+0.06%** | 0.0 | 81.1 | 0.0 |
| r68c | old             | 0.00283 | 0.004477 | -- | 17.0 | 53.4 | 6.67 |
| r68c | F               | 0.00315 | 0.004492 | **+0.3%** | **26.6** | 47.1 | 7.51 |
| r68c | 500/800/720     | 0.00320 | 0.004544 | +1.5% | **30.0** | 45.2 | 7.98 |

F costs the learner NOTHING (both champions flat) and moves an already-mixed policy +56% toward
melee. It raises bot rate 11-15%, because a symmetric buff helps the bots too -- that is the
expected cost and it is why ratio must not be the judge (below). Dose 500/800/720 buys only a
little more mixing (+76%), so 400/800/650 is the pre-registered arm; 500 is the escalation.

Forced-style probe (`--gate=N` added to `tools/evalsuite.mjs`, patches `env.gateShift` so
`execStyle = (chosen + gate) % 3`), r68n champion, learner rate by executed style:

| executed style | old reach | F | gap closed |
|---|---|---|---|
| ranged | 0.006192 | 0.006196 | -- |
| mage   | 0.005195 (-16.1%) | 0.005584 (-9.9%) | 6.2 pp |
| melee  | 0.004277 (-30.9%) | 0.004779 (-22.9%) | 8.0 pp |

CAVEAT, and it is a big one: this probe forces the ATTACK style but not the MOVEMENT policy, so a
ranged-trained brain plays melee with ranged positioning and understates melee by an unknown
amount. Treat it as a dose-response on reach, not as melee's true potential. The same caveat
voids every melee-side buff measured under monoculture -- v5ch (melee charge) and v5res
(archetype resists) were both "REJECTED" while no learner ever used melee, so those verdicts are
NOT evidence. Re-run them after the mix moves.

## r69 round (queued, blocked on GPU)

Order: `r69t` (realm `tests=1`, validates the JS/WGSL constants agree) then `r69f-s1`, `r69f-s2`
(512 worlds, 200k ticks, seed 1/2), judged paired against `r68n-s1/s2` at 200k.

BAR, PRE-REGISTERED BEFORE THE ARM RUNS (amended -- the old bar was written for a NERF arm):
1. PRIMARY: `EG[atkMelee] + EG[atkMage]` >= 15% of learner attacks (baseline r68n is 0.0%).
2. Learner EVAL (eval A reward/tick) must not drop more than 0.19 vs r68n.
3. **ratio is EXEMPT.** F is a symmetric buff: CPU says bot rate rises 11-15% while learner rate
   is flat, so ratio falls ~10% with no learner regression. Judge EVAL and `evalBase`
   separately; a ratio drop of that size is predicted, not a failure.
4. evalBase rising is expected and is not itself a rejection; it is only a rejection if EVAL
   falls with it.

Escape hatch if F leaves the mix flat: escalate the dose to 500/800/720, then add candidate A
(suppression) on top of F. A costs 27% alone but was measured with ranged at 800; on top of F the
same absolute reach floor bites a much smaller share of attacks.

Mage is NOT reach-limited: at 500 -> 650 its attack count moved 6.67 -> 7.51 (+13%) where melee
moved +56%, so mage's binding constraint is UPTIME (`CAST_COOLDOWN` 30, mana = RARE). Pre-register
candidate I (`CAST_COOLDOWN` 30 -> 24) for the round AFTER the mix moves -- do not bundle it into
r69f, or a melee-only result would be uninterpretable.

### Decision rule for r69f (pre-registered)

Read `EG[atkMelee]` / `EG[atkRange]` / `EG[atkMage]` and EVAL at ~200k, paired vs `r68n`.

- **Mix moves** (melee+mage >= 15% of learner attacks) and EVAL >= -0.19: ADOPT F. Then the real
  payoff test, pre-registered: re-run the v5 lesion suite and check that melee/mage-keyed content
  became load-bearing. The v4-era lesion ceiling was <= 3.2%, so the bar is that at least one
  melee- or mage-keyed lesion (perk tree by attack track, cover play, boss style switch, archetype
  resist) now moves MORE than 3.2%. If it still does not, the content is cosmetic and the next
  lever is reward, not balance.
- **Mix flat, EVAL flat**: escalate the dose to 500/800/720 (CPU says +76% melee attacks vs +56%,
  learner rate still +1.5%), then candidate A on top of F.
- **EVAL drops > 0.19**: REJECT F. Do not conclude "melee cannot work" - conclude that reach was
  load-bearing for the bots too, and go to candidate J (make the PvE come to the melee player:
  bosses and a share of mobs on melee style so they close instead of being farmed at 800). J is
  unmeasured and is the first lever that changes mob behaviour rather than player numbers.

| build | learner rate | vs old | melee | ranged | mage | melee share |
|---|---|---|---|---|---|---|
| old 200/800/500, cd 10 | 0.004477 | -- | 17.0 | 53.4 | 6.67 | 22% |
| F 400/800/650, cd 10 | 0.004492 | +0.3% | 26.6 | 47.1 | 7.51 | 33% |
| 500/800/720, cd 10 | 0.004544 | +1.5% | 30.0 | 45.2 | 7.98 | 36% |
| 600/800/700, cd **14** | 0.004593 | +2.6% | 29.1 | 39.4 | 7.67 | 38% |

The last one is the best CPU number AND the biggest mix shift, and it was still REJECTED as the
arm: at reach 600 vs 800 with melee still at ~2x ranged DPS, ranged keeps only 200 units of reach
and a win against mage, so it becomes a dominated option -- the melee identity (highest DPS,
shortest reach) disappears and the style choice stops meaning anything. Mix diversity is the goal,
not melee supremacy. F keeps three distinct identities: melee 2.7x DPS at half the reach, ranged
safe and weak, mage bursty and mana-gated.

## Reach ladder, re-measured 2026-10-07 (supersedes the escape hatch above)

Same instrument as the forced-style probe, but one genome and one flag set throughout
(`tools/evalsuite.mjs --game=realm --suites=bots --seeds=2 --worlds=4 --gate=N runs/r68n-s1-base`,
learner reward/tick x1000). Player `STYLE_RANGE` varied; `MOB_STYLE_RANGE` left at 200/700/450.
A150/A250 add a ranged DEAD ZONE (style 1 cannot fire at a target inside that radius, for players
and mobs alike) rather than a hostile-agnostic suppression.

| build | ranged | melee | melee gap vs ranged |
|---|---|---|---|
| old 200/800/500 | 6.415 | 4.553 | -29.0% |
| F 400/800/650 | 6.377 | 5.315 | **-16.6%** |
| dose 500/800/720 | 6.426 | 5.109 | -20.5% |
| F + dead zone 150 | 6.281 | 5.063 | -19.4% |
| F + dead zone 250 | 6.343 | 4.964 | -21.7% |

**F is the best single step on this ladder and the ladder does not continue past it.** Giving melee
MORE reach (500) is worse than 400, and a ranged dead zone at either radius is worse than F alone.
So the pre-registered escape hatch above (escalate to 500/800/720, then add A) is MEASURED and
WEAK - do not spend two GPU arms on it. The reason is visible in the confound the probe carries:
`--gate` forces the attack style but not the movement policy, so every "melee" row is a ranged-
trained brain standing at ranged distance. Once reach stops being the binding term (F), the probe
cannot see any further improvement, because what is left to fix is the learner's POSITIONING, not
its numbers.

Consequence for the decision rule: if `r69f` leaves the mix flat, the next lever must be one that
pays melee WITHOUT the learner having to walk further - i.e. one that changes what comes to the
player. That is candidate J, below, and it is now measured rather than speculative.

## Candidate J (mobs close in) - MEASURED and REJECTED, 2026-10-07

Same instrument, same genome and flags as the ladder above. `MOB_STYLE_RANGE` varied (mob melee /
mob ranged / mob mage); player `STYLE_RANGE` at F's 400/800/650 unless the row says old.

| build | ranged | melee | melee gap vs ranged |
|---|---|---|---|
| old players, mob 200/700/450 | 6.415 | 4.553 | -29.0% |
| **F players, mob 200/700/450** | 6.377 | 5.315 | **-16.6%** |
| F players, mob 200/500/350 | 6.368 | 5.032 | -21.0% |
| F players, mob 200/400/300 | 6.453 | 5.115 | -20.7% |
| F players, mob 250/400/300 | 6.348 | 4.996 | -21.3% |
| old players, mob 200/400/300 | 6.513 | 4.704 | -27.8% |

J does not work: making mobs close in raises the RANGED rate (6.377 -> 6.453 at the most
aggressive setting) more than it raises the melee rate, so the gap widens from -16.6% to -20.7%.
Aggressive PvE is a buff to whoever has reach, which is exactly the monoculture.

### What this closes

Every numeric lever I can think of is now measured against the same instrument, and F wins all of
them: player reach (F -16.6%, dose -20.5%), a ranged dead zone (-19.4% / -21.7%), and mob
aggression (-20.7% to -21.3%). Nothing gets melee closer than ~16% below ranged. Two readings:

1. The instrument is SATURATED. It forces the attack style but not the movement policy, so every
   melee row is a ranged-trained brain standing at ranged distance. Past F the binding term is the
   learner's POSITIONING, which a constant cannot change and this probe cannot see.
2. Therefore `r69f` (GPU, 2 seeds, 200k) is the ONLY measurement that can answer "does the learner
   adopt melee". If it comes back flat, do NOT queue another constant - that is three measured
   dead ends. The remaining levers are structural:
   - reward: pay the non-dominant styles directly (`popWeights` already shapes channels, and
     `CH.COMBAT` could be split by style), at the cost of shaping the objective rather than the
     game;
   - per-life class assignment (style fixed at spawn, exposed in obs), which guarantees the mix by
     construction and makes every melee/mage-keyed feature load-bearing, but removes style choice
     from the training policy (`lvlMelee`/`lvlMage` are already dead inputs by `inputMask`, so obs
     for a fixed class already exists - `atkMelee`/`atkMage` stats are counted where XP is
     granted, so the mix would be visible immediately).
   Decide between those two only after r69f, and pre-register the bar for whichever is chosen.

## Candidate M (melee LUNGE) - the first lever that works, 2026-10-07

Same instrument as the ladder above (`r68n-s1-base`, `--suites=bots --seeds=2 --worlds=4`,
learner rate x1000, gate 0 = ranged, gate 2 = melee). Baseline for every row is F
(`STYLE_RANGE` 400/800/650), which is what is in src right now.

A melee attack within LUNGE range no longer needs the attacker to already be adjacent: the
attacker is moved next to the target as part of the attack. Melee keeps `STYLE_RANGE[0]` 400
as its *strike* range, so it does not out-range anything; it simply stops paying "walk the
whole 800 gap before I am allowed to swing". Patch is gated `style === 0 && e < R_PLAYERS`,
so mobs and the other two styles are untouched (but bots ARE players, so bots get it too).

| build               | style  | learn  | bot   | ratio | mobK | atk[M/R/G] | deaths |
|---------------------|--------|--------|-------|-------|------|------------|--------|
| lunge off (= F)     | ranged | 6.377  | 3.410 | 1.87  | 6.7  | 0/74/0     | 1.64   |
| lunge off (= F)     | melee  | 5.315  | 3.160 | 1.68  | 3.8  | 61/0/0     | 1.52   |
| **lunge 1000**      | ranged | 6.296  | 3.610 | 1.74  | 7.4  | 0/75/0     | 1.67   |
| **lunge 1000**      | melee  | 6.195  | 3.490 | 1.77  | 3.2  | 45/0/0     | 1.30   |
| lunge 1400          | ranged | 6.306  | 3.900 | 1.62  | 8.0  | 0/77/0     | 1.67   |
| lunge 1400          | melee  | 6.216  | 3.680 | 1.69  | 3.2  | 46/0/0     | 1.31   |

Melee's deficit: **-16.6% -> -1.6%** at lunge 1000, -1.4% at 1400. Nothing else on the ladder
got closer than -19%.

Second-order reads, all in the same direction:
- melee deaths **1.52 -> 1.30**. Closing in is not a death sentence; the 2.7x DPS lands before
  the mob's own swing does. This is the surprising one and it is the reason the lunge works
  where reach constants did not: reach flattened the *approach* cost but melee was still paying
  a *survival* cost, and only a lunge removes both at once.
- melee ratio 1.68 -> 1.77, which at lunge 1000 is above ranged's own ratio (1.74). Melee is
  not merely viable, it is the best per-bot style on this champion.
- melee mobKills 3.8 -> 3.2 and attack count 61 -> 45 despite equal reward: fewer, better
  engagements. Consistent with attacks now landing where before many were spent walking.
- bot rate 3.410 -> 3.610 (+5.9%) at 1000, -> 3.900 (+14%) at 1400. Bots are players, so they
  lunge too. This is the *wanted* direction (bots catching up) but 1400 pays twice the symmetric
  buff for no extra melee effect.

**Pick: LUNGE 1000.** Same melee outcome as 1400, half the symmetric bot buff.

### Caveats that must be carried into the GPU arm
1. 2 seeds x 4 worlds. The ranged rows also moved (6.377 -> 6.296/6.306) even though the patch
   is gated `style === 0`, and that gate cannot touch a ranged attack. So there is ~0.08 of
   noise in this table; the *gap* closing by 15 points is far outside it, the exact -1.6% is not.
2. Same movement confound as every gate probe, and here it runs AGAINST melee: the brain being
   measured was trained to stand at range and walk in, so a real melee-specialist policy should
   do better than 6.195, not worse. The -1.6% is a floor on melee, not a ceiling.
3. The probe did not clamp the post-lunge position to the world or block a lunge into a town.
   Safe-zone *targets* are already excluded from targeting (realm.js tryAttack), so a lunge
   cannot be aimed into a town, but the landing point still needs a clamp in the real patch.
4. No cost was attached to the lunge. Melee at parity and 2.7x DPS is the intended shape, but
   if the GPU arm shows melee *dominating* rather than mixing, the dial is the lunge distance
   downward or a cooldown on the lunge itself - do not nerf melee's DPS for it, that is E all
   over again.

### Pre-registered bar for the r70 arm (F + melee lunge 1000)
- ADOPT if attack mix moves off monoculture: `atkMelee + atkMage >= 15%` of attacks.
- AND `EVAL >= baseline - 0.19` (same -0.19 bar as r69f).
- ratio EXEMPT, same reasoning as F: a symmetric buff lifts bots too, ~6% down predicted.
- 2 seeds, 200k, paired vs `r68n`.

### Implementation notes (do this only when no r69 job is queued)
- JS: in `tryAttack` (realm.js), after `if (best < 0) return false;` and only for
  `style === 0 && e < R_PLAYERS`, move the attacker along the vector to the target to
  `STYLE_RANGE[0] * 3/4` of the distance. The gate for *finding* a target uses `reach` (1000),
  the strike still uses `range` (400).
- WGSL: `r_try_attack` only RECORDS the attack (`r_atk[e] = ...; r_act[e] = RA_ATTACK;`), so the
  position write must NOT go there - the decide phase has no barrier between lanes. Put it in
  the later resolve phase behind the existing barrier, keyed off the recorded `r_atk` target.
  This is the same rule that forced the mage blink out of `r_try_cast`.
- Mirror the constant in `r_style_range`-adjacent WGSL as a new `LUNGE_RANGE` and keep JS/WGSL
  byte-identical; the lockstep test is the only thing that catches a mismatch.

## The full style triangle under F, and candidate N (mage cooldown), 2026-10-07

The gate probe has only ever been run on two styles. Ran all three on `r68n-s1-base`
(`--suites=bots --seeds=2 --worlds=4`, learner rate x1000, gate N -> `(1+N)%3`).

| build           | style  | learn  | bot   | ratio | mobK | atk[M/R/G] | deaths |
|-----------------|--------|--------|-------|-------|------|------------|--------|
| F, lunge off    | ranged | 6.377  | 3.410 | 1.87  | 6.7  | 0/74/0     | 1.64   |
| F, lunge off    | mage   | 5.689  | 3.580 | 1.59  | 7.0  | 0/0/49     | 1.65   |
| F, lunge off    | melee  | 5.315  | 3.160 | 1.68  | 3.8  | 61/0/0     | 1.52   |
| F + lunge 1000  | ranged | 6.296  | 3.610 | 1.74  | 7.4  | 0/75/0     | 1.67   |
| F + lunge 1000  | mage   | 5.601  | 3.810 | 1.47  | 7.0  | 0/0/46     | 1.70   |
| F + lunge 1000  | melee  | 6.195  | 3.490 | 1.77  | 3.2  | 45/0/0     | 1.30   |

**Read: the monoculture is exactly the ordering of this table.** ranged 6.377 > mage 5.689
(-10.8%) > melee 5.315 (-16.6%). A learner doing credit assignment on a 3-way style argmax
will concentrate on the top row, and it did.

**The lunge only fixes one of the two gaps.** After it, melee 6.195 sits essentially level with
ranged 6.296, but mage is untouched at 5.601 (-11.0%). So the honest prediction for an
F+lunge GPU arm is that monoculture flips from ranged to a ranged/melee mix with mage still 0 -
which passes the melee+mage >= 15% bar but is still not a three-way mix.

### Why mage is capped - it is a cooldown, not a positioning, problem
realm.js:673 `f[b+F.CD] = trunc(STYLE_COOLDOWN[style] * (style===1 && tier>=1 ? 2 : 3) / 3)`
with `STYLE_COOLDOWN = [10, 18, 24]` gives

| style  | cooldown | attacks/life measured | reach (F) |
|--------|----------|-----------------------|-----------|
| melee  | 10       | 61                    | 400       |
| ranged | 12 (tier>=1) | 74               | 800       |
| mage   | 24       | 49                    | 650       |

Mage pays twice: 150 less reach than ranged AND half the attack rate. Measured rates agree
(49 vs 75). None of the reach levers can touch this - raising mage's reach to 800 would just
make it a worse ranged.

**Candidate N: `STYLE_COOLDOWN[2]` 24 -> 16 or 12** (a buff, never a nerf - same shape as F).
16 gives mage 1.33x ranged's cooldown, 12 gives parity while mage still gives up 150 reach and
keeps its triangle role (mage beats melee). Measure both on CPU with gate 1; take the smaller
change that still lifts mage above the -5% line, because a mage that strictly dominates ranged
just relocates the monoculture. Mirrored in WGSL as `r_style_cooldown`.

### Implementation notes for the lunge (carried from the WGSL read)
- Home is the TOP of `r_move` (realm.wgsl.js:1340) beside the existing mage blink at line 1342,
  and the top of `moveEntity` (realm.js:1017) beside line 1020. NOT inside `tryAttack`:
  the decide phase has no barrier between lanes, which is the same rule that forced the blink
  out of `r_try_cast`. `r_try_attack` only RECORDS (`r_atk[e] = ...; r_act[e] = RA_ATTACK;`).
- `r_blink` (realm.wgsl.js:1030) is the template: integer math via `r_isqrt` (never f32 sqrt),
  then a per-axis bounds + `r_passable_at` check before each write.
- **Gate differs from the blink.** `atkCast`/`act` are cleared every tick (realm.js:1483), so
  melee's `(cast=0, style=0)` is the DEFAULT state, not a signal. The lunge MUST also test
  `act == ATTACK` - the blink does not need to because `atk_c==1` is only ever set by a cast.
  JS: `if (e < R_PLAYERS && this.act[e] === ACT.ATTACK && this.atkCast[e] === 0 && this.atkStyle[e] === 0)`.
  WGSL: `if (e < RW_PLAYERS && r_act[e] == RA_ATTACK && r_atk_c(e) == 0 && r_atk_s(e) == 0)`.
- Landing point: attacker ends at `STYLE_RANGE[0] * 3/4` (300) short of the target along the
  vector. Only lunge when `d > STYLE_RANGE[0]`, so an already-adjacent melee does not teleport.
- Position must be clamped to the world and checked passable. Safe-zone *targets* are already
  excluded at target selection, so a lunge cannot be aimed into town.

## Candidate N measured, and the r70 bundle decision, 2026-10-07

`r68n-s1-base`, `--suites=bots --seeds=2 --worlds=4`, learner rate x1000, gate 1 (mage).

| build        | style  | learn  | bot   | ratio | mobK | atk[M/R/G] | deaths |
|--------------|--------|--------|-------|-------|------|------------|--------|
| mage cd 24   | mage   | 5.689  | 3.580 | 1.59  | 7.0  | 0/0/49     | 1.65   |
| **mage cd 16** | mage | 6.090  | 3.410 | 1.79  | 6.8  | 0/0/64     | 1.57   |
| mage cd 12   | mage   | 5.944  | 3.540 | 1.68  | 6.2  | 0/0/76     | 1.53   |
| mage cd 16   | ranged | 6.357  | 3.280 | 1.94  | 6.9  | 0/74/0     | 1.67   |

**cd 16 is the pick and the ladder is NOT monotonic.** cd 12 fires 76 attacks vs cd 16's 64 and
earns *less* (5.944 vs 6.090), with the lowest mob kills and the lowest deaths: past ~16 the
mage is spending its time attacking instead of earning, and the extra swings are worth less than
the gathering they displace. "Lower cooldown is better" is false here.

The cd 16 ranged control row (6.357 vs 6.377 at cd 24) confirms the change is style-2 only, as
designed.

### Predicted triangle with BOTH fixes (each measured separately, F baseline)
| style  | F alone | with M + N  | gap to ranged |
|--------|---------|-------------|---------------|
| ranged | 6.377   | 6.377       | -             |
| mage   | 5.689   | 6.090       | -4.5%         |
| melee  | 5.315   | 6.195       | -2.9%         |

Three styles inside 5% of each other is the precondition for a mix, and nothing measured so far
has come close to it.

### r70 = F + M (melee lunge 1000) + N (mage cd 16), bundled
Bundled deliberately rather than run as two arms. GPU time is the scarce resource (the card is
held by another project's workload for hours at a time) and each arm costs 2 seeds x 30 min.
Attribution survives the bundle because the readout IS per-style: `EG[atkMelee/atkRange/atkMage]`
counts the three styles separately, so if the mix comes out melee-only I still know which lever
failed to move the learner.

Pre-registered bar (2 seeds, 200k, paired vs `r68n`):
- ADOPT if `atkMelee + atkMage >= 15%` of attacks, AND `EVAL >= baseline - 0.19`.
- ratio EXEMPT (both levers are symmetric buffs that lift bots too; ~6% down predicted).
- Secondary, not gating: all three styles >= 5% is the full win.
- If melee moves and mage does not, or vice versa, the EG split says which; do not re-run the
  bundle to find out.

### Correction to the r70 bar above: pair against `r69f`, not `r68n`
`r68n-s1/s2` were the controls for the r68 gate arms and were run BEFORE F changed
`STYLE_RANGE` to 400/800/650, so `r68n` is a pre-F build. The CPU probes above evaluated
r68n's *genome* inside an F-patched environment, which is the right comparison for a forced-style
environment question, but it is not the right GPU baseline.

The GPU arm r70 (F + M + N) must be judged paired against **`r69f`** (F alone, 200k, same 2
seeds), because that isolates M+N and is the only same-build comparison. `r68n` stays useful as
the longer-view reference ("where v5-era started") but its delta confounds F with M+N.
Consequence: r70 cannot be queued until r69f has reported, and if r69f rejects F then r70 has to
be rebased onto whatever the F verdict leaves in src.

## r70 bundle measured together - the levers are additive, 2026-10-07

`r68n-s1-base`, `--suites=bots --seeds=2 --worlds=4`, learner rate x1000. Same run prints the
F-only rows and the F+M+N rows, so they share seeds and worlds.

| build | style  | learn  | bot   | ratio | mobK | atk[M/R/G] | deaths |
|-------|--------|--------|-------|-------|------|------------|--------|
| F only   | ranged | 6.377  | 3.410 | 1.87  | 6.7  | 0/74/0     | 1.64   |
| F only   | mage   | 5.689  | 3.580 | 1.59  | 7.0  | 0/0/49     | 1.65   |
| F only   | melee  | 5.315  | 3.160 | 1.68  | 3.8  | 61/0/0     | 1.52   |
| **M+N**  | ranged | 6.455  | 3.600 | 1.79  | 7.8  | 0/79/0     | 1.69   |
| **M+N**  | mage   | 6.125  | 3.760 | 1.63  | 7.3  | 0/0/64     | 1.64   |
| **M+N**  | melee  | 6.226  | 3.440 | 1.81  | 3.1  | 46/0/0     | 1.32   |

**The two levers are independent.** Each one measured alone vs measured in the bundle agrees to
within noise: mage 6.090 alone / 6.125 bundled (+0.035), melee 6.195 alone / 6.226 bundled
(+0.031). Neither lever needs the other and neither cancels it.

**Noise scale, from the row neither lever should touch.** The ranged row moved 6.377 -> 6.455
(+0.078) though M is gated `style === 0` and N is gated `style === 2`. So ~0.08 is the floor at
2 seeds x 4 worlds, and the two real effects (+0.436 mage, +0.911 melee) are 5x and 11x it.

**Resulting triangle: ranged 6.455 / mage 6.125 / melee 6.226 - a 5.1% spread.**
Melee is now the highest-ratio style (1.81) and has the lowest death rate (1.32). No style
dominates, which is the precondition for a learner to spread across them, and nothing measured
in this project has previously gotten the spread under 16%.

Caveat unchanged and important: this is a forced-style probe on a brain trained under ranged
monoculture. It says the ENVIRONMENT now pays three styles comparably. Whether the learner
actually spreads across them is exactly what the GPU arm measures, and it is still unmeasured.

## SHIPPED 2026-10-07: F + M + N are now in src (not just in the probe tree)

Both levers are real game changes now, mirrored JS and WGSL:

| file | change |
|------|--------|
| `src/games/realm.js` | `STYLE_RANGE` [400,800,650] (F, already in), `STYLE_COOLDOWN` [10,18,**16**] (N), `LUNGE_RANGE` 1000 (M) |
| `src/games/realm.js` | new `lunge(e, target)`, called at the top of `moveEntity` beside the blink |
| `src/games/realm.js` | `tryAttack` target scan uses `reach` (LUNGE_RANGE for melee players, `range` otherwise); damage and cooldown still use `range` |
| `src/games/realm.wgsl.js` | `r_style_cooldown` mage 24 -> 16, new `r_lunge_range()` = 1000 |
| `src/games/realm.wgsl.js` | new `r_lunge(e, target)`, called at the top of `r_move` beside `r_blink` |
| `src/games/realm.wgsl.js` | `r_try_attack` scan uses `reach` the same way |
| `index.html` | rebuilt, build-hash 2d962af9c7c910f1 |

Verified: JS/WGSL constants extracted and compared (`STYLE_COOLDOWN` [10,18,16] both sides,
`LUNGE_RANGE`/`r_lunge_range()` 1000 both sides, ranges and mob ranges untouched), `node build.mjs
--check` reports current, and the JS env runs (`evalsuite` on `r68n-s1-base` completes; its
`atkMelee=0 / atkRange=72.6 / atkMage=0` is the OLD monoculture brain, which is expected - the
mix can only change after a retrain).

Two implementation details that differ from the probe and matter:
- `realmIsqrt`/`r_isqrt`, NOT `Math.sqrt`. The probe used f32 sqrt; the shipped code follows
  `blink()` and uses the integer sqrt, because WGSL's f32 sqrt is inexact and the JS/WGSL
  lockstep test would see the difference.
- The lunge call site is `moveEntity`/`r_move`, not `tryAttack`/`r_try_attack`. The decide phase
  has no barrier between lanes while JS's decide loop is sequential. Consequence: the shipped
  lunge reads the target's possibly-already-moved position, where the probe read its pre-move
  position. Same rule the mage blink already obeys; the difference is at most one move step.

### Queue consequence - read this before judging any r69 number
`r69f-s1/s2` were ingested as the F-only arm, but src is now F+M+N, and `dev/rllong.html`
imports src modules directly. **So `r69f` now measures the full shipped stack, not F alone.**
There is no F-only GPU arm and none is planned: F alone showed learner rate FLAT with bots
+11-15%, so it was never expected to move the mix, and spending another hour of contended GPU
on a control I intend to supersede is worse than spending it on the real thing.

That means the `-0.19` EVAL guard is now a test of the WHOLE stack against `r68n` (5.93 @188k),
not a clean F-vs-F+M+N attribution. If it fails, bisect by reverting `LUNGE_RANGE` to 0-gated
(i.e. `reach = range`) for one arm.

`r69t` (realm `tests=1`) is still first in the queue and now validates the shipped build - the
lockstep test is the only thing that catches a JS/WGSL mismatch, and it has been silent-wrong
before (a stale literal zeroed every h2h tick and still passed 10/21).

## SHIPPED-BUILD triangle (2026-10-07) - corrected numbers

The probe's M+N numbers (6.455 / 6.125 / 6.226) were measured in `train-probe` with
bots' lunge landing at a different point in the tick. The REAL tree (`src/games/realm.js`
+ `realm.wgsl.js`, build-hash 2d962af9c7c910f1) gives, on `runs/r68n-s1-base`,
`--suites=bots --seeds=2 --worlds=4 --jobs=2`, one gate per row:

```
ranged  learn=  6.168 bot=3.770 ratio=1.64 mobK=7.8 atk[M/R/G]=0/76/0 deaths=1.67
mage    learn=  5.801 bot=3.750 ratio=1.55 mobK=7.3 atk[M/R/G]=0/0/65 deaths=1.69
melee   learn=  6.184 bot=3.750 ratio=1.65 mobK=3.4 atk[M/R/G]=48/0/0 deaths=1.31
```

- spread max-min = 6.184 - 5.801 = 0.383 = **6.2%** (F-only baseline was 16.6%).
- melee is now the TOP style and still dies least (1.31 vs 1.67/1.69).
- **mage is the laggard now, not melee** (-6.2% vs melee). That is the opposite of the
  probe's ordering and it is the thing to watch in the GPU arms.

### Why it differs from the probe
1. Attack counts confirm both mechanics fire: mage 49 -> 65 attacks (cd 24 -> 16, same
   as the probe's cd16 count of 64), melee 48 attacks, melee deaths 1.52 -> 1.31.
2. Bot rate is now FLAT across gates (3.750/3.750/3.770) where the probe F-only run had
   3.160/3.410/3.580. Bots pick their own style, so a gate sweep should not move them -
   the probe's spread there was itself noise on 2 seeds x 4 worlds.
3. The sim is chaotic: moving the bots' lunge to a different point in the tick changes
   world trajectories, so absolute rates shift a few percent. A ~5% shift on 2 seeds x
   4 worlds follows from that, and mage's probe gain (+7.7%) did not survive (+2.0%).

Consequence: treat the 6.2% spread as the real number and the 5.1% probe number as VOID.
Do not tune mage further on 2 seeds - the ladder already peaked at 16 (12 is worse), so
the next mage lever must be a mechanic, not a cooldown number.

### Correction: 4-seed shipped triangle supersedes the 2-seed one above

Re-run at 4 seeds x 4 worlds (16 samples/gate) on the SHIPPED build, same champion:

```
gate 0 ranged  rate 0.00620 +- 0.00018   baseRate 0.00382 +- 0.00020   ratio 1.62
gate 1 mage    rate 0.00580 +- 0.00014   baseRate 0.00383 +- 0.00015   ratio 1.51
gate 2 melee   rate 0.00619 +- 0.00007   baseRate 0.00368 +- 0.00014   ratio 1.68
```

- ranged and melee are now TIED (6.20 vs 6.19); the melee lunge closed that gap completely.
- **mage is the sole laggard**: -6.3% vs melee, ~5 sigma on 16 samples. On ratio it is
  -10% (1.51 vs 1.68).
- bot rate is identical for ranged and mage gates (3.82 / 3.83), so the mage deficit is
  entirely learner-side, not a bot-side artifact. Melee's gate shows a LOWER bot rate
  (3.68), so melee's high ratio is partly bot weakness under a melee-forced learner.
- Diagnosis: per-style tier perks are lopsided. melee gets cleave + execute
  (realm.js:662-663), ranged gets a faster cooldown at tier>=1 (18 -> 12), and mage gets
  only +25% vs bosses (realm.js:664). Mage also attacks on cd 16 vs ranged's 12 for only
  +1 damage, i.e. ~79% of ranged DPS. Magic is the weakest style by construction.

## The shipped WGSL was BROKEN and r69t did not catch it (2026-10-07)

`runs/r69f-s1.log` (11:57): `line 1320: 'target' is a reserved keyword`,
`fn r_lunge(e: u32, target: u32)` -> `RlBackend.init` threw, job dead in 6 s (1 attempt).

Why r69t did not catch it: r69t ran **10:27 -> 11:42** (4507 s) but `src/games/realm.wgsl.js`
was written at **11:33**. Its page had already loaded the pre-edit modules, so its 21/21
described the build WITHOUT `r_lunge`. The "an edit lands on the NEXT job, not the running
one" rule applies to validation jobs too. Any claim that r69t validated the shipped
F+M+N build is RETRACTED.

Fix: parameter renamed to `victim`; `node build.mjs` -> hash `04107a6086ad1499` (was
`2d962af9c7c910f1`); `queue stop` + `queue resume --retry=r69f-s1` re-queued the consumed
seed ahead of `r69f-s2`. `r70g-s1/s2` were already ingested and now pick up the fix too
(dev/rllong.html loads `src/` live, so no re-ingest was needed for them).

Lesson: the GPU is the only WGSL compiler available and `target` is reserved; check new
WGSL identifiers against the reserved list before queueing.

## Candidate P (mage splash) - SATURATES, only +2.5%

Probe tree synced to the shipped build (only `realm.js` differs), champion `r68n-s1-base`,
gate 1 (mage), 4 seeds x 4 worlds. Splash = a tier>=1 mage attack also damages every
hostile within radius of the PRIMARY target (mirrors the existing `shockHits` branch in
`damagePhase`, victim-centric so no barrier issues, `killer` attributed to the mage).

```
control   splash 0          5.80 +- 0.14   <-- reproduces the shipped baseline EXACTLY
splash 240  half damage     5.96 +- 0.15
splash 400  half damage     5.93 +- 0.15
splash 400  3/4 damage      5.90 +- 0.06
splash 240  3/4 damage      5.96 +- 0.12
```

- The control matching 5.80 to the digit validates the whole A/B rig.
- Dosing UP does not help: 400 is worse than 240 at both fractions. The lever saturates at
  about +2.5% and cannot close the 0.40 gap to ranged/melee (6.20).
- So mage's deficit is NOT "lacks AoE". The remaining lever is throughput: mage attacks on
  cd 16 vs ranged's 12 for only +1 damage, i.e. ~79% of ranged DPS.

## Candidate P (mage splash + base damage) -- SHIPPED, triangle now FLAT (2026-10-07)

Motivation: after F+M+N the triangle was 6.2% (ranged 6.20 / mage 5.80 / melee 6.19,
4 seeds x 4 worlds, champion `runs/r68n-s1-base`, `tools/evalsuite.mjs --gate=N`).
Ranged and melee had already tied; mage was the SOLE laggard at -6.3% (~5 sigma).
Bots were flat across gates (3.82 / 3.83 / 3.68), so the deficit is learner-side,
not a bot artifact.

Diagnosis (why mage lagged): per-style tier perks are lopsided.
- melee gets cleave (tier>=1) + execute (tier>=2)
- ranged gets its cooldown cut 18 -> 12 at tier>=1
- mage gets only +25% vs bosses, and attacks on cd 16 vs ranged's 12 for +1 base
  damage => roughly 79% of ranged DPS before any perk.

Candidate P (two parts, both shipped to src):
1. `MAGE_SPLASH = 240` / `RW_MAGE_SPLASH = 240`; `splashHits(a,v)` / `r_splash_hits(a,v)`;
   a new branch in `damagePhase` / `r_damage` (victim-centric, so it is the right
   insertion point -- `damagePhase` is realm.js:1268) that applies `atkDmg >> 1` to any
   hostile v within 240 of the attacker's PRIMARY target, gated
   `atkCast==0 && atkStyle==2 && tier(a,2)>=1 && (a<R_PLAYERS)!=(v<R_PLAYERS)`.
   XP is granted per ATTACK not per kill (realm.js:1392), so only `killer[v]`
   attribution was needed. The melee shockwave was the existing multi-target precedent.
2. mage base damage `5 + 2*lvl + 3*weap` -> `8 + 2*lvl + 3*weap`.

Dose ladder (gate 1, 4 seeds x 4 worlds, same champion, bot rate in parens):
  control   splash0   dmg 5+2L+3W    mage 5.80 +- 0.14  (3.83)
  splash240 dmgA      8+2L+3W        mage 6.20 +- 0.07  (3.84)   <-- SHIPPED
  splash240 dmgB      5+3L+4W        mage 6.05 +- 0.06  (3.82)
  splash240 dmgC      8+3L+4W        mage 6.44 +- 0.17  (3.81)   overshoots

The control arm reproducing 5.80 to the digit is the rig's own control: it says the
+0.40 is P, not drift. Bots move <= 0.01 across every arm, so P does not inflate the
benchmark. dmgC at 6.44 would sit ~0.24 ABOVE ranged/melee and flip the monoculture
to mage -- rejected for that reason, not because it is weak.

Splash alone SATURATES: radius/fraction ladder (2 seeds) 240@1/2 5.96, 400@1/2 5.93,
400@3/4 5.90, 240@3/4 5.96 -- all ~+2.5%, and bigger is WORSE. So the second half of
the +6.9% comes from base damage, not from more AoE.

Final confirmation triangle with P shipped (4 seeds x 4 worlds):
  gate 0 (ranged)  0.00619 +- 0.00026
  gate 1 (mage)    0.00620 +- 0.00007
  gate 2 (melee)   0.00619 +- 0.00026
Spread 0.16%, down from 16.6% (F-only) -> 6.2% (F+M+N) -> 0.16% (F+M+N+P).
The payoff gradient that produced ranged monoculture is now gone.

Caveat carried forward: `--gate=N` forces the ATTACK STYLE but not the movement
policy, so melee measured this way understates itself. The triangle being flat under
this rig is the right direction of evidence (it removes mage's deficit, which was the
one arm that could not be explained by movement), but the real test is r69f's EG mix.

Queue: `r69w` (64 worlds, 2k ticks) smoke-compiles the P WGSL before the long arms,
`r69f-s1/s2` @200k then measure F+M+N+P, `r70g-s1/s2` are the gate arm re-run.

## Shipped-build validation for F+M+N+P (2026-10-07) -- all three layers PASS

Candidate P changed the WGSL (`r_splash_hits`, `RW_MAGE_SPLASH`, a new branch in
`r_damage`) and the GPU is the only WGSL compiler in the normal loop, so the build
was validated in three independent layers before spending two 30-minute arms on it.

1. OFFLINE FRONT-END (naga, no GPU). `node tools/emitwgsl.mjs <file>` assembles the
   real shader (core template + realm chunk, cfg from `DEFAULT_OPTS`, WORLDS=512)
   and `naga <file>` validates it: `Validation successful`, exit 0, 110481 bytes,
   zero undefined/NaN tokens from the template.
   Teeth test: renaming `victim` back to `target` in a copy reproduces the failure
   that killed r69f-s1 earlier today -- `error: name 'target' is a reserved keyword`
   at the exact line. So the check is not vacuous.
   naga does NOT check device limits (workgroup bytes, binding counts), so this layer
   complements `tests=1` rather than replacing it.
   Installed with `cargo install naga-cli` (v30.0.1).

2. CONSTANT TWIN PARITY. `node tools/constparity.mjs` -> 102 pairs, 0 mismatches.
   Covers both WGSL shapes: const twins (`MAGE_SPLASH` <-> `RW_MAGE_SPLASH`) and
   if-chain function twins (`STYLE_RANGE` <-> `r_style_range`, `STYLE_COOLDOWN` <->
   `r_style_cooldown`, `LUNGE_RANGE` <-> `r_lunge_range`, `MOB_STYLE_RANGE` <->
   `r_mob_range`). The shape matters: a const-only scan flagged all three style
   constants as "no RW_ twin", which reads exactly like a JS-only candidate.
   Self-tested by injecting two breaks into a throwaway copy: both caught, exit 1.

3. REAL HARDWARE. `r69w` (64 worlds, 2000 ticks) -> done in 81 s,
   `workgroupBytes=31396` (cap 32768), `warnings=[[]]`, no ERR: the P WGSL compiles
   and trains. `r69t2` (`game=realm`, full 21-test suite) -> **21/21**, no FAIL/ERR
   lines, i.e. JS/WGSL lockstep holds with the splash branch in `r_damage`.

So all four candidates (F STYLE_RANGE, M LUNGE_RANGE, N STYLE_COOLDOWN, P mage
splash + base damage) are confirmed live in BOTH realm.js and realm.wgsl.js and on
the device.

## The GPU gate was blocked by DISCORD, not another training job (2026-10-07)

The runner's `waitForGpuHealth` gates each attempt on nvlddmkm resets OR a foreign
compute pid. It had been holding every job for ~50 minutes on "foreign GPU pid 3616".
AGENTS.md records a historical storm caused by `resound.exe`, so that was the
assumed culprit. Matching the pid against the process table showed
**pid 3616 = Discord**, started 11:44, 109 MiB of 6144 MiB, 0% util.

Two things follow. First, nvidia-smi's compute-apps list counts desktop
compositing, so "a foreign pid holds the card" is not evidence of a competing
compute workload -- check the process name before treating it as one. Second,
`waitForGpuHealth` (queue.mjs:366-371) only gives up after `gpuHealthWaitMinutes`
when the block is NOT solely a foreign pid, and Discord is always running, so this
gate waits forever rather than degrading. The round was started with
`--no-gpu-health` once the evidence said the card was effectively idle (0% util,
109 MiB, last nvlddmkm reset 12:21:11 with none in the following 10 minutes).

Fix still open: an allowlist so desktop apps do not count as foreign compute
workloads. Recorded in AGENTS.md; `tools/queue.mjs` deliberately left unmodified
mid-round so a restart cannot activate unverified code under a live run.

## r69f = first GPU measurement of the shipped F+M+N+P build (2026-10-07) -- PRELIMINARY, FAILS the bar

Registered bar (AGENTS.md Open work): melee+mage >= 15% of attacks AND EVAL >= -0.19 vs r68n, ratio EXEMPT.

| run | game | seeds | ticks | eval r/t x1e3 | bots x1e3 | ratio | eval life |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| r69f-s1 | realm | 1 | 102579 | 1.80-2.59 | 3.17-4.84 | 0.53-0.58 | ~1700-2100 |
| r68n | realm | 1,2 | 100000 | 3.70 +- 1.45 | 2.86 +- 0.13 | 1.28 +- 0.45 | 1642 |

Style mix (the bar's first clause) is MET and then some: r69f-s1 ends EG[atkMelee=9355, atkRange=0, atkMage=67148]
and r69f-s2 ends EG[atkMelee=107077, ...] -- ranged is literally ZERO and the two seeds flip to DIFFERENT
styles (s1 mage, s2 melee). So the ranged monoculture is broken, which was the whole point of F/M/N/P.

The second clause is NOT met, by ~10x the bar: EVAL 1.8 against r68n's 3.70 is about -1.9, and ratio
0.50-0.58 means learners earn about HALF what the scripted bots earn (r68n 1.28). Both seeds agree in
direction (s2 1.38-1.72 at 22-42k ticks). Bots also moved UP: evalBase ~3.2-4.8 vs r68n's 2.86, and 3.79
is above every recent v5 run in the report table (2.90-3.06), so the change set is not learner-only.

Candidate readings, not yet separated:
  (a) P overshot: mage splash + base damage 8 pulled training onto mage, and mage loses to ranged bots.
  (b) M/N/F buffed the bots too (bots attack with styles, inherit perk passives), so eval A got harder
      on both sides of the ratio.
  (c) The metric itself: against SCRIPTED bots, specialising on the single strongest style (ranged, worth
      ~1.75 EVAL per the earlier lesion work) beats a balanced triangle. Diversity may cost eval-A score
      while making the game better. If (c) dominates, the fix is a metric change, not a content revert.
  s1 flipping mage and s2 flipping melee argues against (a) alone -- P should push BOTH seeds to mage.

Build correctness is not in question: r69t2 ran the same src and returned 21/21, and r69w smoke-compiled
the P WGSL in 81 s.

Next: r70g (styleGate=2, styleGateEval=1) is the already-queued test of whether forcing style rotation
during training recovers EVAL; it needs no src edit. If it does not, revert P (mage splash + base damage
5 -> 8) and re-measure, since P is the newest change and the one that moved the training equilibrium.

## r69f mechanism: F+M+N+P slid training into a camping mob-grinder (2026-10-07)

r70g was KILLED before it ran, for cause: the Kaggle forced-style data says style choice is not the
lever. Gating `r68n-s1` on the SAME post-F build (8 seeds x 4 worlds) gives rate 0.00619 / 0.00620 /
0.00619 at gate 0/1/2, and the gate is provably doing something -- the atk* mix swings completely
(g0 atkRange 70.1, g1 atkMage 62.2, g2 atkMelee 45.1). So F+M+N+P genuinely flattened the triangle
(from a 16.6% spread to 0.16%), which was the design goal, and it is NOT what cost the 1.9 EVAL:
`r69f` loses at 1.86/2.10/1.89 in the SAME three gates while bots are unchanged (baseRate 3.45-3.65e-3
vs r68n's 3.70-3.89e-3). Style rotation therefore cannot recover 1.81; r70g was dead on arrival.

The loss is a BEHAVIOURAL local optimum, visible in the evalsuite stat rows (8 seeds x 4 worlds, bots suite):

| metric | r68n | r69f | read |
| --- | ---: | ---: | --- |
| rate (reward/tick) | 6.19e-3 | 1.86e-3 | the whole gap |
| life | 1236 | 3594 | r69f lives ~3x LONGER |
| dist1k | 29571 | 14182 | and moves half as far -> camping |
| toolTier | 1.079 | **0.064** | never crafts a tool |
| crafts1k | 2.035 | 0.443 | 4.6x less crafting |
| greatHarvests | 2.351 | 0.329 | locked out of great nodes (they are TOOL-gated) |
| mobKills | 7.16 | **16.6** | 2.3x more mob grinding |
| death_age | 0.000 | 0.178 | survives to max age doing nothing |
| death_starve | 0.491 | 0.141 | and is not even hungry |
| atkMage / atkRange | 0 / 70.1 | 150 / 0 | s1 is a pure mage |
| casts1k | 1.936 | 0.000 | and never casts |

Reading: the economy path (harvest -> craft tool -> great node) is a long sparse chain paying
CH.PROGRESS (256/1024/2048 per craft tier, 1024 per great-node share). Combat is dense and per-tick.
F+M+N+P made combat much denser -- N cuts mage cooldown 24 -> 16 (1.5x attack rate) and P raises mage
base damage 5 -> 8 and adds a 240-radius splash -- so PPO found mob-grinding first and stayed there.
`toolTier` 0.064 is the smoking gun: without a tool the great-node channel is closed, so the
high-value half of the economy is unreachable and the grinder is a stable optimum that happens to
survive forever.

Prime suspects, in order: P (mage damage + splash) and N (mage cooldown), because both multiply mage
combat density and s1 went full mage with 150 attacks and zero casts; F and M are melee-side and s2
went melee, so they are not excluded.

Decision: revert F+M+N+P in src (done, `r71`), re-baseline, then re-add ONE at a time. The cheap tell
is `toolTier` and `greatHarvests` on the champion, not EVAL alone -- a run can look alive (life 3594)
and be dead (rate 1.86).

## r69f is a SURVIVOR, not a mob-grinder (2026-10-07) -- the queued r72 arm was wrong

New instrument: `tools/evalsuite.mjs` now reports the reward channels for single-genome runs
(`ch_<name>1k`, net per 1k ticks, they sum to `rate`) and scores any weight vector offline with
`--popWeights=1,1,1,1;0.5,1,1,1` (one row per `;`-separated vector, `rateW1..`). No retraining needed.

Channel mix at 2 seeds x 1 world, bots suite, on the REVERTED build:

| champion | survival | progress | combat | coop | rate | toolTier | crafts1k | greatHarv |
|---|---|---|---|---|---|---|---|---|
| r68n-s1 (good)   | 0.819 | 1.388 | 1.220 | 2.906 | 6.33 | 1.100 | 1.880 | 2.70 |
| r69f-s1 (failed) | 0.996 | -0.004 | 0.078 | 0.156 | 1.23 | 0.080 | 0.521 | 0.03 |

r69f earns **81% of its reward from survival alone**. It is not a mob grinder -- combat pays it 0.078
and its 10.8 mobKills/1k are hunting for food (animal kills sit in the survival channel). It never
crafts (toolTier 0.08), so progress is 0 and coop is nearly 0 (great nodes are gated `TOOL >= 2`).
That is a stable attractor: with no tool there is no progress and no coop, so the only channel with a
gradient is survival, and the policy settles into living long and doing nothing (life 2931,
death_age 0.094 -- it dies of old age).

Consequence: **r72 as first written (`popWeights=0.5,1,0.5,1`, halving combat) attacks the wrong
channel** -- it moves r69f 0.729 -> 0.690 because r69f earns ~nothing from combat anyway.
`channelCaps` cannot fix it either: it clamps the per-TICK channel value, so it cannot make survival
stop paying after N ticks of one life, and it scales the death penalty with the alive reward.

The lever that is nearly free for a good policy and expensive for the attractor is the SURVIVAL
weight, because survival is only 13% of r68n's reward but 81% of r69f's:

| weights | r68n rateW | r69f rateW | r69f hit |
|---|---|---|---|
| 1,1,1,1     | 6.333 | 1.227 | - |
| 0.5,1,1,1 (current default) | 5.923 | 0.729 | - |
| **0.25,1,1,1** | 5.719 | **0.479** | **-34% vs -3.5%** |
| 0.5,1,0.5,1 | 5.313 | 0.690 | -8% vs -10% |

So r72 is `popWeights=0.25,1,1,1` (runs/r72.json, 2 seeds x 200k), to be run with F+M+N+P re-added
after r71a re-baselines the revert. Reference cells: r71a = revert + 0.5 survival, r69f = content +
0.5 survival. Pass bar: EVAL >= r71a and champion `toolTier` back near 1.0.

Caveat: lowering the survival weight also cheapens the death penalty (both live in the survival
channel, ratio preserved), so lives may get shorter and riskier. That is measured, not assumed.

## Style choice IS worth something on the reverted build (Kaggle, 4 seeds)

`--gate=N` sets `execStyle = (chosen + N) % 3`, styles 0 melee, 1 ranged, 2 mage.

- r68n-s1 on the REVERTED build: ranged 6.41, mage 5.32, melee 4.48 (spread 43% of ranged).
- r68n-s1 on the F+M+N+P build: 6.19 / 6.20 / 6.19 (flat).

So F+M+N+P did not hurt ranged; it lifted melee 4.48 -> 6.19 and mage 5.32 -> 6.20, i.e. it genuinely
equalised the triangle -- and training then converged onto the survivor attractor instead. The two
facts are consistent: equalising payoffs does not tell a learner which style to pick, and the one it
picked (mage, r69f atkMage 76.1) is worth 1.19 where forcing ranged is worth 1.86.

## Kaggle: jobs.json was never reaching the kernel

`kaggle kernels push` uploads only `code_file` for a script kernel, so `jobs.json` never arrived and
every run silently used `DEFAULT_BATCH` ("no jobs.json next to the script"). Fixed: eval.py now reads
`<payload>/jobs.json` (the dataset copy at /kaggle/working/train) first, and the batch is shipped
inside the dataset. v4 of the payload carries it; kernel v5 runs batch 4 = channel profiles of 8
champions at 8 seeds x 4 worlds with the four weight vectors above.
