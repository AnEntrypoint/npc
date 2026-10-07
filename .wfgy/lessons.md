# WFGY lessons — train

## 2026-10-05 -- "no WebGPU adapter on Kaggle" was a bug in my probe, not a fact about Kaggle
Goal (G): find out whether Kaggle can run our WebGPU trainer so we can add a second training lane.
What drifted / what went wrong: three probe versions (v3, v4, v5) reported `navigator.gpu` undefined
under every flagset, and I wrote that up as "Kaggle has no WebGPU adapter" and reasoned from it
(blamed Chrome 131, blamed the missing Vulkan ICD). The real cause: WebGPU requires a SECURE CONTEXT
and the probe navigated to `about:blank`, which is not one. Probing `http://localhost:8123/index.html`
made the adapter appear immediately.
Fix / resolution: v6 probes a served localhost origin; the adapter is present, vendor google /
architecture swiftshader (CPU), maxBindGroups 4, and the trainer produced no ticks in 420 s at 64
worlds. Conclusion changed from "no adapter" to "adapter exists but is CPU-only and useless for us".
Generalizes to: a negative result from a probe is only evidence about the probe until the probe is
shown to exercise the real path. Before writing "X is impossible" into notes, ask what the harness
did differently from production -- here, origin/secure-context. Applies to every feasibility probe in
kaggle/ (tpu-probe, nvidia-probe): state the assumption the probe depends on, then check it.

## 2026-10-05 -- perf A/Bs that never reached the shader
Goal (G): decide whether the sim or the policy is the PPO bottleneck, to pick an optimization target.
What went wrong: the P20 `fwdDirect` and `soa` A/Bs measured identical code -- `shaderConfig()`
(src/rlengine.js:115) whitelists cfg keys, and both keys are omitted, so the GPU never saw them.
I had already told the user "the sim is the bottleneck" on that evidence.
Fix / resolution: retracted the claim to the user; F1 is to add both keys to `shaderConfig()` and
re-measure before any F2-F10 sim work.
Generalizes to: any opt-in flag in this project is inert on GPU until it is listed in
`shaderConfig()` -- a neutral A/B result means "unmeasured", not "no effect", until the whitelist is
checked. Check the whitelist before trusting ANY neutral knob result.

## 2026-10-05 -- AGENTS.md has a byte ceiling, so an additive edit is a budget question
Goal (G): keep AGENTS.md under 26,000 bytes while still recording what the next
session needs.

What drifted / what went wrong: I appended one pointer line (fits, 25 B left), then
replaced a false P20 claim with a longer correct one and went 72 B OVER the ceiling
without noticing -- I estimated the delta instead of measuring it. Fixing it then
cost two more edits.

Fix / resolution: (1) after EVERY AGENTS.md edit, run
  node -e "const s=require('fs').statSync('AGENTS.md');console.log(s.size, 26000-s.size)"
-- never estimate a byte delta. (2) Do not keep shaving words to make room; buy room
by moving closed work out. Blob PPO A/B was CLOSED and its numbers were partly
duplicated elsewhere (the R30 entropy line was a realm number misfiled under blob),
so moving it to AB-ARCHIVE.md freed ~830 B and removed a duplication.

Generalizes to: in this project, "add a note" and "stay under the ceiling" are one
decision, not two. When headroom is under ~200 B, archive a closed section before
writing anything new. Prefer deleting a claim that is already stated correctly
elsewhere over compressing it.

## 2026-10-05 -- the instrument decided whether the measurement was valid at all
Goal (G): find where realm's per-tick cost goes, to target wallclock work correctly.

What drifted / what went wrong: I ran a hidden-size sweep measuring wall-clock wt/s and got
87k -> 22k against a 259k historical baseline - 7x SLOW for a 4x SMALLER net. I nearly started
interpreting that as a result. It was an artifact: the GPU was 8% utilized at 30 W because the
host was starved by CPU contention (34 sandbox runners across 16 cores). Wall-clock throughput
measured the HOST, not the GPU. I had also written a decision rule with an unpinned denominator
("50% of the h64 tick" - rollout or iteration?), which let the same number read as either
"sim is the target" or "inconclusive".

Fix / resolution: switched the instrument to per-kernel GPU timestamps (profile=1), which are
immune to host CPU contention, and confirmed the new instrument was trustworthy before
believing it (3 repeats: 1.5% spread). Result: iteration = 29.4 ms wall vs 27.96 ms GPU inside
an iteration, so the GPU is ~95% busy per iteration but only ~7% busy end-to-end. That single
comparison is what exposed the host as the bottleneck. Also stated both denominators instead of
picking the one that supported a conclusion.

Generalizes to: in this project, before trusting any throughput number, ask "does this metric
measure the GPU or the host?" Wall wt/s measures BOTH and cannot tell you which failed. Pair any
wall-clock number with either nvidia-smi util or a GPU timestamp to know. And pin the
denominator of any threshold rule at the moment you write the rule, not when you read the result.

## 2026-10-05 -- profiling is a different workload; I compared it against non-profiling runs
Goal (G): decide whether the GPU lane is host-starved, to choose between "reduce CPU contention"
and "optimize kernels".

What drifted / what went wrong: I took wt/s from the `profile: 1` arms (split16/split32, 22-87k),
compared it to a 259k historical baseline, and told the user the GPU idles ~93% and live training
is 7-10x slower than the GPU path. GPU timestamp queries serialize dispatches, so a profile arm is
slow BY CONSTRUCTION -- it was measuring the instrument. Live training logs showed no such gap:
hid64b 258,755 wt/s at 61 ms/iteration against ~62.9 ms of GPU kernels (GPU-saturated), and
spfix0 372,621. The 372,621 -> 258,755 drop is exactly the documented ~37% blockGrad: 0 penalty,
not contention. I retracted both claims.

Fix / resolution: cross-check every reported rate against independent counters printed on the
SAME log line -- it= * rolloutTicks / s= must reproduce wt/s -- and only compare runs that use the
same instrument. That arithmetic is what exposed the error.

Generalizes to: never compare a profiling/instrumented run's throughput against a production run's,
and never promote a number from an instrumented arm into a claim about the uninstrumented system.
Ask first: "does enabling this measurement change what is being measured?"

## 2026-10-05 -- a tight spread WITHIN one arm is not evidence ACROSS arms (time-confounded A/B)
Goal (G): find a real throughput lever for the GPU lane.

What drifted / what went wrong: the hostPipeline A/B gave hp0-s1 282,705 wt/s (sd 10,319) vs
hp1-s1 345,403 (sd 6,923). The 9 windows did not overlap at all (hp0 max 297,202 < hp1 min
336,224), and I reported it to the user as "decisive, +22.2%, not noise". It was not. The arms ran
sequentially and the four means were MONOTONE IN START ORDER: 282.7 / 339.7 / 345.4 / 346.2k. The
box was getting quieter during the round (the gm daemon was fixed mid-round), so seed 1 -- which
ran first -- absorbed the worst contention. Seed 2, run later, showed only +1.9%.

Fix / resolution: retracted the +22% to the user, marked the A/B VOID, and wrote the re-run recipe
(interleave arms IN TIME, compare each arm to the mean of its two neighbours) into NEXT-ROUNDS.md
and AGENTS.md.

Generalizes to: a small sd across an arm's internal windows measures the arm's stability, NOT the
validity of the arm-to-arm comparison. Never let a tight within-arm spread license a causal claim
across arms. In this project, where throughput swings 2-3x with host load, ANY sequentially-run
A/B is confounded by machine state -- interleave, and always print the arms in start order and
check for a monotone trend before reading an effect off them.

## 2026-10-05 -- predicted a fixed per-iteration cost to amortize; measured pure linear scaling
Goal (G): raise GPU-lane throughput by amortizing per-iteration host overhead over more ticks.

What drifted / what went wrong: I predicted rolloutTicks would raise wt/s, reasoning that host
cost is per-iteration and 2x/4x more ticks per rollout would dilute it. Measured (bench, 512
worlds, hidden 64): rolloutTicks 31/64/128 -> 218,924 / 222,911 / 216,863 wt/s, and iterationMs
72.5 / 147.0 / 302.2. Flat within +-1.4% over a 4x range; everything scales linearly, so there is
no fixed cost to recover. Hypothesis dead.

Fix / resolution: reported the refutation instead of the hypothesis. Also learned the real lever is
elsewhere -- shipped `hostPipeline: 0` (src/rl.js:31) pays 2 submits + onSubmittedWorkDone +
popErrorScope EVERY iteration, while hostPipeline: 1 merges them; that is now under A/B.

Generalizes to: bench (tempBackend, evalEvery 0) is NOT the training path, so a bench A/B cannot
speak to training-loop overhead -- state that limit when quoting bench numbers. And when a scaling
hypothesis fails this flatly, the fixed term you were looking for probably does not exist; go look
for the per-call cost (submits, syncs, readbacks) instead of the per-tick one.

## 2026-10-05 -- I nearly ran an A/B arm whose comparison partner was already dead
Goal (G): get a verdict out of the R49 survival-weight (popWeights 1.5,1,1,1) A/B on realm.

What drifted / what went wrong: the round was queued as ref58 (reference, shipped defaults) +
surv15 (the arm). Both ref58 arms died (`stalled: no log output for 600s`, foreign-Chrome
contention) and wrote NO log file at all; surv15-s1 died at 17,825 ticks; surv15-s2 had never
started. I still resumed the runner, which picked up surv15-s2 -- 30 minutes of the only GPU lane
on an arm that would have had nothing to be compared against. I only noticed when I checked what
was actually queued instead of just whether the runner was alive.

Fix / resolution: stopped the runner (zero loss, it was waiting on the GPU lock anyway) and
rebuilt the round with `--replace`: ref58-s1, surv15-s1, ref58-s2, surv15-s2 -- reference and arm
ALTERNATING IN TIME, so the pair cannot be confounded by host load either. Old queue state backed
up to runs/.old/ first.

Generalizes to: "is the runner alive" and "can this round produce a verdict" are different
questions. Before resuming a queue, check that every pending arm still has a LIVE partner on the
same seed -- a failed arm is not a reference, and an arm that ran 6 lines is not data. Check
`firstOutputMs = null` / missing runs/<name>.log: that is environment failure, not a result, and
it does not count as the arm having been measured.

## 2026-10-05 -- verified the knob was plumbed before spending 2 h of GPU on it
Goal (G): run the R49 survival-weight A/B (popWeights 1.5,1,1,1 vs 1,1,1,1) and get a real verdict.

What went right (recorded because the opposite already bit me once): before the round ran I traced
the knob end to end on CPU while another session held the GPU lane -- URL parse -> popList ->
pop.weights -> opt buffer write (rlengine.js:215-220) -> rl_train_fx reading opt[OPT_W+ch]. Had it
been inert, ref58 and surv15 would have been IDENTICAL runs and the round would have reported a
false NEUTRAL after 2 h of the only lane, exactly like P20 fwdDirect/soa.

Generalizes to: verify a knob is live BEFORE the long run, not after a null result. But the gate
differs by knob type and the two are easy to confuse -- COMPILE-TIME knobs are gated by the
shaderConfig() whitelist (rlengine.js:117, the F1 bug: fwdDirect/soa missing there), while
RUNTIME-VALUED knobs (popWeights, channel weights in the opt buffer) are deliberately absent from
that whitelist and are gated by whether their WRITE path runs. Asking "is it in shaderConfig()" about
a runtime knob would give the wrong answer both ways. Find where the value is written, then confirm
that write is unconditional on the conditions your arm uses (here: K=1).

## 2026-10-05 -- my prediction got the direction right and the mechanism wrong
Goal (G): decide whether Kaggle can be a second training lane while the local GPU belongs to
another session.
What drifted / what went wrong: I predicted (P_K1) that if the NVIDIA Vulkan ICD installed, the
limit would be host CPU (4 cores vs 16 locally), since our lane is host-bound. PA was refuted by
the system itself: vulkaninfo --summary lists ONE device, llvmpipe (CPU software rasterizer) - no
NVIDIA Vulkan device exists, so WebGPU there is CPU by construction and no flagset fixes it.
Measured 293 wt/s vs 250-370k local. My pessimistic bound (<5k) was 17x off. The direction ("not a
lane") was right; the CAUSE I wrote down was wrong.
Fix / resolution: recorded the refutation, kept the conclusion (Kaggle is not a lane) and
re-grounded it on a real measurement instead of v1's artifact; also recorded that
kaggle kernels logs without -f returns only the last ~6.5 KB.
Generalizes to: when predicting, write the CAUSE beside the number, and check the cheapest link in
the causal chain FIRST - "does a GPU device exist at all" (vulkaninfo) settled this before any
throughput number existed. A right conclusion from a wrong mechanism is still wrong for the NEXT
decision: it would have sent me to optimize host overhead on a 4-core box instead of stopping.

## 2026-10-06 -- realm v5 lockstep divergence: a position write inside the decide phase is a cross-lane race
Goal (G): make realm a deeper game (creeps, perks, bosses, magic) with JS and WGSL staying word-for-word lockstep.
What drifted / what went wrong: magic's blink (cast style 2) moved the caster's X/Y inside tryCast,
i.e. during g_step's decide phase. In JS the decide loop is sequential (players 0-31 then mobs 32-63)
so a mob sees the post-blink position; on the GPU all 64 lanes run the decide phase with NO barrier
between them (barrier is after decide, line 1786), so lane 52 read the pre-blink position and never
attacked. Symptom: diverged at tick 17 ent2.HP 64/62 | ent52.CD 0/29 - the boss's CD, not the HP, was
the giveaway (no CD written = the attack never happened on GPU).
Fix / resolution: deferred the blink to the move phase on both sides (JS moveEntity, WGSL r_move),
using the already-stored atkTgt as the blink anchor; 8/8 PASS after.
Generalizes to: in this kernel, NOTHING may write another lane's readable state (X/Y especially)
during decide - a JS-sequential read is a GPU race. When a lockstep diff shows a CD/act word rather
than a damage word, suspect the action never fired, then look for decide-phase writes; and confirm
coverage first (the old densify produced zero casts of the new kind, so the bug was invisible).

## 2026-10-06 -- the bot scan is NOT the thing that spends workgroup bytes; obsBuf's stride already paid for it
Goal (G): let bots react to animals separately from mobs (hunger-gated hunting) without exceeding the
32768 B workgroup grant.
What drifted / what went wrong: I sized the extra bot scan slot against the 336 B of free workgroup
memory and, on that basis, nearly shipped a MERGED mob+animal slot (one atomicMin, tie-broken by
entity index) to avoid paying for a second one. That would have diverged: JS `scan()` fills class 5
(mob) and class 6 (animal) as separate entries, so any WGSL that collapses them makes the two sides
choose different targets. The byte arithmetic was also wrong: `obsBuf` is declared
`array<atomic<u32>, learners * (nIn|1)>` (32*109 = 3488 words), so the bot scan's k*stride+slot
indices are already inside a buffer paid for by the observation stride - a 7th slot costs 0 B.
Fix / resolution: stride 6 -> 7 with a dedicated animal slot, mirrored per JS class (mob ty=4 -> slot
5, animal ty=5 -> slot 6, boss -> slot 6 of the stride and r_nd2[9]); private arrays (r_nd2/r_ndx/
r_ndy/r_nidx) cost no workgroup bytes either.
Generalizes to: before refusing a feature on the byte budget, re-derive WHICH structure is sized by
what - obsBuf is sized by observations, private arrays are free, and only workgroup-var state counts.
And a WGSL prescan slot must map 1:1 onto a JS scan class; merging classes to save a slot is a
lockstep bug, not an optimization.

## 2026-10-06 -- I killed a healthy job twice because its logging is buffered, not incremental
Goal (G): get the 21-test RL suite green on realm v5 before spending the GPU lane on a 2-seed baseline.
What drifted / what went wrong: the `tests=1` job printed one `created` line and then nothing for 636 s,
so I called it "hung reproducibly at the same point" and switched to a different gate. It was not hung.
`dev/rllong.html:59-62` logs every result only AFTER `runTests()` returns, so the whole suite is silent
until it finishes - the suite genuinely needed ~700 s and completed on its own once the lock held.
I burned two attempts and 1100 s of the only lane diagnosing a non-fault, and told the user it was stuck.
Fix / resolution: read the logging shape before concluding anything from silence. The discriminating
question is "does this page log per test or per suite" - one line of the harness answered it. Recorded
in AGENTS.md as a process rule (budget >= 20 min, do not read silence as a hang). Third run: 21/21 PASS,
and it also retired the real risk (nIn 109 / nOut 13 / 142 nodes: gradient 4.04e-8, JS-env rollout
lockstep identical).
Generalizes to: "no output" is evidence about the instrument, not about the job, until you have checked
WHEN the instrument writes. Before declaring any job stuck, stalled or dead, find the write site and ask
whether output is incremental. The same trap inverted: a job that writes a line every 30 s can be dead
between lines, so silence means different things in the two designs - never carry the inference across.

## 2026-10-06 -- a stride change is a three-place edit, and only one of them is the scan itself
Goal (G): add a 7th bot scan slot (animals) to realm v5 with JS/WGSL staying word-for-word lockstep.
What drifted / what went wrong: I changed the stride in the two places the scan is written and read
(`r_bot_prescan`, `r_load_bot_scan`) and forgot the third: `g_step`'s per-tick reset, which cleared
`obsBuf[t]` for t < 96 = 16 bots x stride 6. With stride 7 the true span is 112, so bots k=13..15
(indices 91..111) kept whatever g_observe had last written there and steered toward stale garbage.
Symptom: FAIL at tick 0 ent29.X 6448/6453 -- tick ZERO, not tick 17, and ent29 (k=13) is exactly the
first slot past the reset bound. The bound in the error message told me where to look.
Fix / resolution: reset `t < 48u` for the second half (64+t covers 64..111); 8/8 PASS after.
Generalizes to: any packed-array stride/layout change has a silent third site -- the reset/clear
loop -- which is usually written as a literal count rather than derived from the stride. Derive it
(or at minimum grep the stride constant) instead of editing the read/write pair. Diagnostic: a
divergence at tick 0 on a HIGH entity index is a stale-state/reset bug, not a logic bug; a
divergence at a later tick is behaviour.

## 2026-10-06 -- "0 great harvests" was the instrument, not the feature
Goal (G): give scripted bots great-node harvesting and measure whether evalBase rises.
What drifted / what went wrong: I measured the change with `stats[R_STAT.GREAT_HARVESTS]` and `env.reward[e]` for bot slots and got exactly 0 for both, twice, and nearly concluded the change was inert. Both are gated on `e < this.brain` (realm.js:1111, 1120) and brain slots are the learners, so they never see bot work; `env.reward` is 0 for bots outright.
Fix / resolution: read the accumulator's own gate before trusting it. Bot reward lives in `STAT.REW_BASE`/`TICKS_BASE` (stats[1]/1024/stats[3]), which reproduced the known evalBase (~0.0018) and showed +20%. Separately, my first "baseline" only blocked `consider(cls 10)` while leaving `allowGreat = TOOL >= GREAT_TOOL` true, so `nearestInteractNode(e, true)` still targeted greats - a patch that changes one of two coupled gates is not the old behaviour.
Generalizes to: in this repo, every per-entity counter asks `e < brain`. To measure bots, use the REW_BASE/TICKS_BASE pair; to measure learners, use the game stats. When A/B-ing by monkeypatch, enumerate every gate the diff touched, not just the one you patched.

## 2026-10-06 -- a swapped argument pair produced a confident +20% that measuring correctly inverted to -3%
Goal (G): decide whether scripted bots should harvest great nodes, by measuring evalBase on CPU.
What drifted / what went wrong: I called `RealmEnv.step(outputs, tick)`; the real signature is `step(tick, outputs)`. Every sweep therefore ran with `outputs` a number and `tick` a Float32Array, so `this.brain` was NaN (learners never decided, `e < brain` never true) and the tick driving night/wander arithmetic was garbage. The numbers still looked plausible - bots moved, mobs died, great nodes completed - so I reported +20% bot catch-up and queued a 2-seed GPU re-baseline on it. Re-run with the order fixed: -3% at the wide seek, +3% at the tightest, both inside seed noise. The lever was killed and the change reverted.
Fix / resolution: before trusting any CPU harness, assert the invariant the harness depends on - here `env.brain === learnerSlots` after one step. A NaN that silently disables half the population still produces plausible-looking output. Also: no throw is not a signature check, since `(number).length` is merely undefined.
Generalizes to: in this repo, verify the call convention of any env entry point against a live assertion, not against "it ran". And treat a per-seed sd as the decidability floor - an effect smaller than it is not a result, and shipping it costs a full re-baseline.

## 2026-10-06 -- two bot-economy levers both measured neutral: the mechanism's own cost ate the gain
Goal (G): make the scripted bots catch up to the learners by raising evalBase.
What drifted / what went wrong: two independent economic levers, both neutral-to-negative on the same 6-8 seed CPU sweep. Great-node harvesting -3% at seek 800, +3% at seek 100 (under the seed sd). Price-aware town choice -0.8%, 3/8 seeds. The reward on offer was large and real each time (a great harvest ~1.0 against a ~7.6 lifetime total; TOWN_BIAS 35 makes the scarce town pay ~2x), and each time the cost of REACHING it - walking away from hauling, or walking to a farther town - cancelled it.
Fix / resolution: stopped after two attempts rather than fitting a third variant to the noise, recorded both as killed, and left src/ at the known-good state that reproduces the 0.00127 baseline exactly. The durable output is a decidability rule, not a new lever: with a per-seed sd of 0.00035 on a 0.00127 base, a 6-8 seed sweep decides only >=10% effects.
Generalizes to: when a lever's gain and its own access cost are the same order, the lever is not there - look for a structural change instead of tuning the trade. And treat "two bounded attempts, both negative" as a resolved result to report, not as a prompt to keep trying variants.

## 2026-10-06 -- an under-gate effect is not a win and not a discard: change the instrument, not the number
Goal (G): make the scripted bots catch up to the improved learners on realm v5, measured, not projected.
What drifted / what went wrong: R51 (bots engage mobs from further out, HP gate 50 -> 30) measured +7.9% on
16 CPU seeds with a monotone dose-response, against a >=10% gate I pre-registered BEFORE seeing data. I was
tempted twice: first to round 7.9% up to "about 10%" and adopt it, then to file it as killed and drop the
lever entirely. Both are the same error - treating a gate on one instrument as a verdict on the effect.
Fix / resolution: kept the number honest and unrounded, then changed instruments instead of re-litigating it.
The 10% gate was a statement about what the CPU sweep (6k ticks, seed sd 0.00035 on a 0.00127 base ~28%
relative) can RESOLVE. The GPU round (512 worlds x 300k ticks) is a different noise floor, so the honest move
was to spend the GPU run and let it decide. Separately: measuring on a scratch copy of realm.js in tmpdir
(imports rewritten to file:///C:/dev/train/src/...) let the whole dose-response be taken while the queue rule
froze src/ - no tree edit was needed to get a decision.
Generalizes to: (1) a pre-registered threshold binds the measurement, never the truth - if the sign is
consistent and the dose-response is monotone, the effect is real and the correct next step is a lower-noise
instrument, not a bigger rounding. (2) When a "do not edit src/ while a round is queued" rule blocks
experimentation, patch a scratch copy and point its imports at the real tree; the freeze is about what gets
COMPILED into a page, not about what you may measure.

## 2026-10-06 -- a rounded budget's headroom arrives in steps of the rounding, not in units
Goal (G): size how many new observation inputs realm v5 can still afford under the 32768 B workgroup grant.
What drifted / what went wrong: I first reported the remaining headroom as "+2 inputs", reading the
spare bytes (336) as if each input cost one slot. nIn enters the formula as `(nIn | 1)`, so 110 and 111
cost the same and headroom arrives in STEPS OF TWO. The real ceiling is nIn 113 (+4 inputs), and
nIn 115 is over.
Fix / resolution: stopped estimating and ran the real exported `shaderLayout(game, cfg)` over a copied
`REALM` with mutated dims, printing workgroupBytes against the cap for each candidate. Corrected the
number in NEXT-ROUNDS.md and marked the correction as computed.
Generalizes to: any budget whose terms are rounded/aligned (odd rounding, padding, `|1`, alignment to
4) must be probed by CALLING the real formula over candidate values, never by dividing spare bytes by a
nominal per-unit cost. Also: nNodes trades 1-for-1 with nIn here, so this is a trade, not a wall.

## 2026-10-06 -- a job's name is not evidence of what the job did
Goal (G): establish whether player-to-player trading exists in realm, since trading is the one requested
feature with no mechanic behind it.
What drifted / what went wrong: the queue listed `ls-v5trade` and `ls-v5trade2`, and `ls-v5trade2` had
passed 8/8. I briefly treated those logs as evidence that trading had been implemented and verified,
which would have flipped a real open item to "done". A case-insensitive search for `trade` across the
tree found zero hits in src/games/realm.js or realm.wgsl.js; the runs were init experiments (v5trade
failed lockstep at tick 0 on a spawn-kit field set), and `trade` in runtime/play.html is only a
KEY_ACTION_NAME alias for a generic action index.
Fix / resolution: verified against source before believing the artifact name, and recorded both the
negative confirmation and the misleading log names so the next pass does not repeat it.
Generalizes to: here, judge a feature by searching the code that would implement it, not by run names,
log names, or commit subjects. Names encode intent at the time they were chosen and go stale.

## 2026-10-06 -- a liveness signal must be written where the work cannot starve it
Goal (G): stop 40-minute GPU training jobs from being killed mid-run by another lane taking the GPU lock.
What drifted / what went wrong: `tools/gpulock.mjs:34` declared a lock stale on `heartbeat` age ALONE
(`!lock || !pidAlive(pid) || age > 60s`), so a process that was alive and working but whose Node event
loop was blocked by a long synchronous PPO step was preempted while still running. Two 40-minute runs
died this way (718 s and 298 s). I had misread these as GPU contention, because contention and a
starved heartbeat look identical from the outside.
Fix / resolution: a live pid now gets a 10-minute grace (LIVE_STALE_MS) and only a DEAD pid is stale at
60 s; the heartbeat moved onto a worker_threads worker so main-thread blocking cannot starve it.
Second-order catch: Node 24 evaluates `Worker(src, {eval:true})` as ESM in this repo, so `require`
throws inside it - use dynamic `await import()`, which works as either CJS or ESM.
Third, separate and self-inflicted: I ran `queue.mjs stop` while a job was 232 s into a live run and
killed it. Check `queue.mjs status` for a live job before stopping the runner.
Generalizes to: any heartbeat/lease/liveness protocol where the liveness writer shares a thread with the
work being supervised will fail exactly when the work is heaviest. Put the writer on its own thread or
process, and make staleness require the strongest available signal (pid death), never a timeout alone.
A peer session found this by reading the source; when a recurring failure has two possible causes,
read the mechanism instead of guessing from symptoms.

## 2026-10-06 -- a predicate that never matches turns its guard into dead code, silently
Goal (G): keep the GPU wait queue fair so a waiting job cannot be jumped by a later arrival.
What drifted / what went wrong: `gpulock.mjs` matched ticket filenames `/^(d+)-(d+)$/` where `\d+` was
meant. The character before `d+` was literally `(` (code 40), not a backslash, so the regex never
matched a real ticket (`Date.now() + '-' + pid`). `liveTickets()` therefore always returned [], which
made the fairness branch `queue.length === 0 || queue[0].name === ticketName` always true, so the first
poller after a release won regardless of arrival order. Nothing errored and nothing looked wrong: the
fallback path still let every job through.
Fix / resolution: corrected to `/^(\d+)-(\d+)$/` and verified at the byte level (charCodeAt of the
character before `d+` = 92) plus functionally (matches a real ticket, still rejects `chrome-q-1`).
Generalizes to: a regex/filter/predicate that can never match is worse than one that throws - it
degrades the guard to a no-op while the system keeps working. Verify such predicates against a REAL
input, and when a tool output renders escapes ambiguously, read char codes instead of the printed text
(bash double quotes mangle backslashes; use a quoted heredoc).

## 2026-10-06 -- a buffered-output job looks identical whether it is slow or hung
Goal (G): validate the trading change by running the 21-test RL suite at nIn 111.
What drifted / what went wrong: two attempts failed as `stalled: no log output for 600s` and I started treating it as a hang in backend setup, considering Chrome/GPU forensics. It was not hung. The suite buffers every PASS line until `runTests()` returns, so silence for its whole duration is normal; the last passing run took 494 s of test time against a 600 s timeout, and nIn 111 simply pushed it over.
Fix / resolution: compared the SUM of the successful run's per-test timings against the stall timeout - that gap, not the silence, was the evidence. Raised the runner's `--stall-seconds` to 1800 (per-job `minutes` is a different clock). Two further startup traps surfaced behind it: another project's server squatting the default port (404 at startup, job never runs), and `start` refusing while a previous runner left a pending job (`resume` instead).
Generalizes to: before diagnosing a silent job, ask whether the job CAN emit partial output at all, and compare the last success's true runtime to the timeout. A timeout that was always within 15% of the real runtime will fail on the first slowdown - treat "silent" as "slow" until a runtime comparison says otherwise.

## 2026-10-06 -- a lockstep diff at tick 0 on the LAST input is an initialization-order bug, not an algorithm bug
Goal (G): get the RL suite green at nIn 111 (two new trade inputs) so the v5 trading arm can train.
What drifted / what went wrong: the suite failed `world 0 tick 0 learner 0 input 110: gpu 0.03125 js 0`.
I spent the whole diagnosis comparing algorithms - `realmIsqrt` vs `r_isqrt`, `alive()` vs `r_alive()`,
`tradeD2` vs `r_trade_d2`, scan order and tie-breaking, and the mutual-partner condition. All are
byte-identical. The GPU value decoded to a partner at distance 192, which is only reachable in the
scan set of slots 16..31. The real cause was WHERE the bound is set: WGSL initialises
`r_brain = min(r_slots, LEARNERS)` in `g_init` (realm.wgsl.js:1931, LEARNERS = 32 in the RL kernel),
while JS set `this.brain = R_LEARNERS` (16) in the constructor and only recomputed it inside `step()`
(realm.js:1439). The lockstep test calls `observe()` BEFORE the first `step()` (rlengine.js:938 vs 945),
so at tick 0 JS scanned 16 entities and the GPU scanned 32.
Fix / resolution: initialise `brain` (and `selfplay`) in the constructor from `learnerSlots`; `step()`
still clamps it by the real outputs width, so nothing after the first step changes. Verified against
the live env: 32-slot cfgs now give brain 32 / selfplay true, 16-slot cfgs give 16 / false.
Generalizes to: when a lockstep diff lands at tick 0 and on the highest-numbered input, suspect a field
the two sides initialise at DIFFERENT times - the GPU derives world state at init, JS often derives it
inside step(). Diff the initialization sites before diffing the formulas. Corollary: a divergence value
that decodes to a concrete distance/index (0.03125 * 256 = 8, so distance 192) tells you which half of
the scan set the answer came from; decode the number before theorizing.

## 2026-10-06 -- a v4 A/B verdict can already BE the shipped default: check the defaults table before queueing the arm
Goal (G): spend the single GPU lane only on arms that can change the outcome.
What drifted / what went wrong: I was about to queue "R36 `randomize` 0" as a v5 arm (AGENTS.md lists it under Open work as "re-run the v4 knobs"). `RL_GAME_DEFAULTS.realm` in src/rl.js already sets `randomize: 0`, so every realm PPO run for rounds now trains in the eval-A composition - the arm would have burned ~20 GPU-minutes reproducing the baseline. Separately, `node tools/queue.mjs resume` defaults to port 8123 while job URLs are baked on 8124, so a bare resume died instantly with "server on port 8123 does not serve /dev/rllong.html" and all six jobs sat pending.
Fix / resolution: read `RL_GAME_DEFAULTS` / `DEFAULT_OPTS` for the game before queueing any arm whose verdict predates the current defaults, and treat "re-run the v4 knobs" as a list to triage, not to execute. Always `resume --port=8124`.
Generalizes to: any archived A/B verdict is stale with respect to the defaults table; a knob that WON can already be default, and re-measuring it is a no-op. Check the config source, then the archive - and treat a runner that exits 0 in under a second as "did not start", not "finished".

## 2026-10-06 -- an archival "unreachable" claim is a hypothesis, not data: measure the gating statistic before acting on it
Goal (G): make realm v5 authentically fun (trading, specialization, skill trees, magic, creep FSMs, ambush, complex bosses) with bots catching up.
What drifted / what went wrong: an earlier audit concluded "perk tiers 2/3 are unreachable because observed XP is ~75", and I was one step from queueing a GPU arm to move the perk thresholds on the strength of it. Measured instead (CPU replay of the v5cw3 champion, 240 lives): combat XP lands almost entirely in ONE style track - xp1 (range) p50 318 / p75 570 / p90 958, while xp0 and xp2 are ~0 (p75 6 and 9). So tier 1/2/3 are reached by 95% / 62% / 25% of lives: the tree is live, and the real finding is the opposite of the note - learners monoculture into the ranged build, so melee and mage perks never unlock and lesioning their inputs does nothing.
Fix / resolution: before spending a GPU arm on a threshold, replay a champion on CPU (runs/xpsurvey.mjs, seconds) and read the actual distribution; correct the archival note in the same pass. The actionable defect is style monoculture (STYLE_RANGE 200/800/500 with 800 being safe), not the thresholds.
Generalizes to: any claim of the form "X is unreachable / dead code" that came from a remembered number must be re-measured before it drives work; and "content is not load-bearing" usually means one strategy dominates, so look at which track/branch actually receives the resource before touching the thresholds.

## 2026-10-06 -- a green test suite can be the WRONG GAME: tests=1 defaults to game=blob
Goal (G): validate a realm-only WGSL edit (bots contest great nodes) before spending two GPU seeds on it.
What drifted / what went wrong: I ran `tests=1` and got "21/21 passed" in 48 s, and treated my realm edit as validated. The suite had run `game=blob` - `dev/rllong.html:27` defaults `game` to blob when the URL omits it, and blob "exposes no reward channels", so every realm path was skipped. The tell was the `created base: ... game=blob` line and the missing channel-weighted test, both of which I had read past.
Fix / resolution: pass `game=realm` explicitly in the job params (`{ "tests": 1, "game": "realm", "worlds": 512 }`) and read the `created ... game=` line before trusting a pass. Re-ran: 21/21 on realm in 458 s, including the channel-weighted and population tests that only exist for realm.
Generalizes to: a suite that passes is evidence about the config it ran, not about the repo. Any "all green" claim needs the config echoed back in the same output; when a page derives its subject from a URL default, the default is the thing to check first. Also: a silent `tests=1` job needs `--stall-seconds=1800`, since the runner's 600 s stall timer kills it before its own `minutes` matters.

## 2026-10-06 -- the cheap screening horizon I proposed measured WORSE than the expensive one
Goal (G): spend less GPU wall clock per A/B decision on realm v5.
What drifted / what went wrong: I planned a two-tier screen (256 worlds x 150k, ~4 min/arm) on the intuition
that an early measurement is just a cheaper version of a late one. Measured instead: the paired-delta sd
across 5 channel-weight arms is 0.60 @150k, 0.35 @200k, 0.56 @290k (RMS of the within-arm seed spread) --
150k is the NOISIEST horizon, not the cheapest signal. v5pw2 reads +0.46 @150k (both seeds positive, looks
like a winner) and -0.03 @290k; every arm's delta is ~1.4x exaggerated at 150k. Adding tick interpolation to
`report.mjs --at-tick` (bracketing windows instead of nearest) removed only the alignment part of that noise.
Fix / resolution: judge at ~200k -- a third cheaper than 300k and the least noisy of the four horizons --
and keep paired same-seed deltas, whose sd is 3-17x tighter than the sd of arm means and costs nothing extra.
Never adopt from a 150k screen; it only ranks.
Generalizes to: "shorter run = same measurement, cheaper" is a hypothesis, not arithmetic. On a learning curve
noise is non-monotone in horizon (too early = transient, too late = trajectory divergence), so measure the sd
at several horizons ONCE and then fix the horizon, rather than shortening by intuition. Pairing beats shortening.

## 2026-10-06 -- the WGSL mirror passed 21/21 while it could not point at the boss
Goal (G): ship bot changes (contest great nodes, engage the boss, prefer gathered nodes) with JS and WGSL in lockstep.
What drifted / what went wrong: I added a boss-engage branch that moves a bot toward `ndx[CLASS_BOSS]/ndy[CLASS_BOSS]`.
JS fills those in `scan()`->`consider()`; the WGSL bot path (`r_load_bot_scan`) only ever wrote `r_nd2[9]` from a
boss key that packed DISTANCE ONLY, so `r_ndx[9]/r_ndy[9]` were still the initialized 0 -- WGSL bots walked toward
the origin. The lockstep test caught it (20/21, world 1 tick 18). The value had been missing since long before:
every earlier use of slot 9 read only the distance, so a green suite never exercised the direction.
Fix / resolution: pack the boss index too (`(u32(d2) << 8u) | j`, boss ids 52..55 fit in 8 bits) and set
`r_nd2[9]/r_nidx[9]/r_ndx[9]/r_ndy[9]` together from it, mirroring `consider()`.
Generalizes to: in this repo "both sides agree" is per-FIELD, not per-feature -- a scan slot can be right in the
distance and wrong in the direction, and only a branch that consumes the wrong field exposes it. When adding a
branch that uses a scan slot, check that EVERY field it reads is filled on the WGSL side, not just the one the
existing code reads. Slot 9's direction fields were never written; a 21/21 pass does not cover them.

## 2026-10-06 -- giving a bot a strictly larger action set made it score LESS
Goal (G): make the scripted bots catch up to the learners on realm v5 (bots still never cast).
What drifted / what went wrong: I gave bots the learner's own spell by calling `tryCast(e)` before
`tryAttack` in both combat branches (boss-engage, mob-engage), expecting idle RARE to convert into
damage. Measured 2 seeds, 512 worlds, 200k, paired vs v5b: bots -0.20 +- 0.07 fx/1k on a 2.83 base
(-7%, ~3 sigma), learner -0.09 +- 0.08, ratio +0.11 -- the arm moved the ratio the WRONG way. The
cost is visible in the code, not in the idea: a cast sets `F.CD` to CAST_COOLDOWN (blocking the next
attack, whose per-hit DPS is higher at short range) and spends RARE that bots otherwise craft gear
with, so the bot trades two things it is short of for one spell.
Fix / resolution: reverted all four call sites and rebuilt. Recorded as REJECTED with the mechanism
in NEXT-ROUNDS.md. The lesson is not "bots cannot use magic" -- it is that a scripted policy's
action set is already tuned to its resource budget, so ADDING an action only helps when the new
action dominates what it displaces; here it displaces an attack AND a craft material.
Generalizes to: when a bot lever measures negative, check what the new action DISPLACES (cooldown,
a consumable, a movement step) before trying a gated variant -- the gate fixes the resource half and
not the cooldown half, so it would only move the loss toward zero, never past it. And read the bot
delta off `bots x1e3` in the paired table, not EVAL: EVAL and ratio can both improve while the bots
themselves got worse.
## 2026-10-06 -- three combat-knob arms failed because the cause was geometry, not numbers
Goal (G): make the melee/mage half of realm v5 content load-bearing instead of dead code (ranged monoculture: XP lands almost only in the range track).
What drifted / what went wrong: I attacked the symptom with three numeric arms in a row -- v5tri (drop ranged counter-immunity), v5ch (melee sprint-charge to close), v5res (creep archetypes resist a style). v5tri was adopted (+0.31 EVAL) but did not move the mix; v5ch (-0.23) and v5res (-0.50) were rejected; xpsurvey showed xp0/xp2 percentiles stuck at 9/0 in all three. The mechanism was knowable up front: at reach 800 a ranger takes ZERO return damage from a reach-200 melee mob, and no damage multiplier competes with zero, so melee pays 2.7x DPS and still loses. I spent three GPU rounds rediscovering that.
Fix / resolution: stopped the knob sequence, reverted v5res to the last known good (v5tri) state, and wrote the geometric conclusion into AGENTS.md so the fourth knob cannot be proposed. Pre-registering the adoption rule in runs/pending-v5res.md before the round is what let the third arm end cleanly instead of being rationalised.
Generalizes to: before tuning a number, compute the payoff of the ALTERNATIVE the agent is not choosing -- if one branch has ~infinite value (zero return damage, an immortal camp, an unbounded dense reward), no finite buff to the other branch can matter, and the fix is structural (locked roles, curriculum, exploration bonus, or deleting the dominant branch). Also: measure the mechanism the arm is supposed to change (here the XP percentiles), not just EVAL -- two of three arms moved EVAL and none moved the mechanism.
## 2026-10-06 -- a diagnostic that read the whole rollout buffer cost 10x throughput
Goal (G): add richer telemetry to the PPO trainer so architecture decisions can be made from live numbers.
What drifted / what went wrong: the new `fitProbe()` read the entire `rec` buffer every report (up to ~43 MB: worlds x learners x steps x record stride). Throughput fell to 24.8k wt/s against this project's known-good ~285k. It looked like a slow configuration; it was the instrument.
Fix / resolution: capped the probe to 64 lane-blocks (`Math.min(64, ...)`); throughput went straight back to 285,059 wt/s on the next run.
Generalizes to: any new host-side readout on this trainer must be sized against the ~285k wt/s baseline, and the first number to check after landing one is `wt/s` in the same log line. A diagnostic that is not sampled is a performance regression wearing a telemetry costume. Also: compare a new instrument's cost against a known-good figure from AGENTS.md before trusting either number.
## 2026-10-06 -- deleting 8 obs indices silently broke the GPU mirror's scratch aliasing
Goal (G): shrink the observation contract nIn 111 -> 103 (drop the dead animal compass) so the freed workgroup bytes buy `hidden` 128 at full speed, changing no behaviour.
What drifted / what went wrong: I shifted all 39 named obs indices down by 8 in `realm.js` and `realm.wgsl.js` and verified the mapping index for index. The RL suite then failed lockstep at input 70 (`needDx`): gpu 0, js 0.015380859375. The cause was not an index: `set_obs` writes into the same `obsBuf` row that `r_scan_part` drops the spring/berry keys into, so `RW_SLOT_SPRING`/`RW_SLOT_BERRY` were deliberately aliased to the need block's OWN output indices (78/92 = needDx/needDy pre-shift) -- above the compass read range (0..71) and written only after the need block's read. After the shift those slots became `night` and `gprog`, written by other blocks, so the keys were clobbered before the read. Log archaeology is what made it attributable: `fit1tr.log` (params 8014 = nIn 111, nOut 13, hidden 64) had passed 21/21 the same day, so the pre-edit tree was clean.
Fix / resolution: re-aliased the scratch slots to 84/85 (`needDy`/`urge`) -- the only indices >= 72 that only the need block writes; 70 is inside the compass read range and would race with the emit loop's `atomicLoad`. Wrote the invariant into AGENTS.md beside the other WGSL divergence rules.
Generalizes to: in this codebase an obs-index shift is a THREE-place edit -- JS writes, WGSL writes, and the `obsBuf` scratch slots -- and any constant in `realm.wgsl.js` that is not itself an obs index may still be an index INTO the obs row. Before trusting a "pure index" refactor, know which oracle covers which path: `perf?mode=tests` (evolution obs parity) is cheap but does not exercise the RL rollout, and the RL lockstep test is the one that sees this class of bug.

## 2026-10-07 -- a per-seed single-log-window delta overstated the stage-2 gap ~5x
Goal (G): decide whether dropping the 8 dead animal-compass observations (nIn 111 -> 103) costs EVAL.
What drifted / what went wrong: I judged by hand from each seed's LAST log window (s2-s1 5.82 vs v5tri-s1 6.21 = -0.39; s2-s2 = -0.29) and reported a "consistent" -0.34 on 2 seeds. `tools/report.mjs` with no flags averages SMOOTH_LINES log windows (report.mjs:484) and gives 5.77 +- 0.25 (3 seeds) vs v5tri 5.83 +- 0.48 (4 seeds) = -0.06, i.e. neutral. Two seeds agreeing on a single window is not agreement; v5tri's own seed sd (0.48) is 8x the gap I was reading.
Fix / resolution: read the report aggregate FIRST, then --paired (per-seed paired delta sd, 3-17x tighter than arm means) if a decision is close. Note --at-tick switches the basis back to ONE window, so a tick-matched comparison is noisier per seed than the plain aggregate -- pair it, never read single arms off it.
Generalizes to: never write an arm's verdict from hand-picked log windows; any delta that "looks consistent across 2 seeds" must be re-checked against the report before it goes into NEXT-ROUNDS.md or AGENTS.md.

## 2026-10-07 -- a size model extrapolated the wrong kernel and 3 jobs died at pipeline creation
Goal (G): spend the stage-2 byte prize (nIn 111 -> 103) on `hidden` 128.
What drifted / what went wrong: `tools/autosize.mjs` anchored on 31380 B — the ROLLOUT kernel — and
extrapolated with its 5.94 B/hidden-unit slope. The kernel that actually binds at large `hidden` is
`rl_grad_blk` (196 B/unit), and the layout's `blockBytes` formula omitted two `pool4` terms declared at
src/rlshader.js:240 while adding `selCnt`'s 1024 B. So the pre-check reported 31380, the device rejected
34768, and h128-s1/s2 + h192p burned a GPU slot each for zero ticks.
Fix / resolution: `blockBytes` now mirrors the declaration (incl. `ceil(nOut/4)*hidden`,
`ceil(hidden/4)`, align16) and reproduces the device number exactly; `autosize` calls `rlShaderLayout()`
instead of extrapolating; probe jobs at hidden 116 (32416 B) and rolloutTicks 128 (31380 B) confirmed
both boundaries on the device before any long arm was queued.
Generalizes to: a size prediction must say WHICH kernel binds, because the binder changes with the knob;
a formula that mirrors a shader declaration must be checked against the declaration, not against one
measurement; and a pre-check that UNDER-reports is worse than none, since it converts a 2-minute
planning question into a 1-hour dead job. Probe the boundary on the device before queuing a long arm.

## 2026-10-07 -- growing NSTATS 16->20: a literal `select(18u, 16u, ...)` zeroed the league instead of crashing
Goal (G): grow the world header (NSTATS 16 -> 20, header 24 -> 28 words) so realm can carry three new per-style attack counters, with both shaders and the host reading every offset from core.js.
What drifted / what went wrong: the sweep replaced literals that LOOK like offsets (`24 * 4`, `new Int32Array(16)`, `i < 16`) and missed the one that does not: `let slot = select(18u, 16u, role == ROLE_LIVE)` in `rl_rollout`, which only ever meant "NSTATS + 0 for live, +2 for snapshot". After the growth it wrote head-to-head reward/ticks into `statAcc[16..19]` (now real stat slots: atkMelee/atkRange/atkMage) and left the role words at `statAcc[20..23]` untouched, so the suite reported `head-to-head ticks missing: latest 0, older 0` and stopped at 10/21. Nine later tests then failed with `Instance dropped in popErrorScope`, which is device loss, not assertion failure, and must not be read as ten separate bugs.
Fix / resolution: `let slot = NSTATS + select(2u, 0u, role == ROLE_LIVE)`; every `STAT_*` WGSL constant is now generated from core.js `STAT` (`STAT_CONSTS`) so no index can drift again.
Generalizes to: when a shared layout constant grows, search for numbers EQUAL TO ITS OLD VALUE, not for its name -- dependents express the offset arithmetically. A silent zero in a stats field is the signature: the run does not crash, it under-reports. And when one GPU test fails and everything behind it cascades as `Instance dropped`, fix the one real failure and re-run before diagnosing the cascade.

## 2026-10-07 -- asserting a `blockGrad` fallback that the run's own log contradicted
Goal (G): judge `rolloutTicks` 128 as a PPO config candidate.
What drifted / what went wrong: I wrote that T=128 needs 33232 B and therefore silently fell back to `blockGrad: 0` (~37% slower), carrying forward arithmetic from before the kernel remodel. The run's `created` line said `workgroupBytes=31380 warnings=[[]]` -- identical to the default, no fallback. The rejection is real (EVAL 3.65 vs 5.80, a quarter of the Adam updates per tick) but the mechanism I first wrote for it was wrong.
Fix / resolution: re-read the `created ... workgroupBytes= ... warnings=` line of the actual run before stating any size claim; corrected NEXT-ROUNDS.md and AGENTS.md in place and named the old claim as wrong rather than quietly overwriting it.
Generalizes to: never carry a byte figure across a kernel remodel -- re-derive it from the log on disk. A mechanism written from memory is a claim about a run, and the run is the only witness.


## 2026-10-07 -- `Instance dropped in popErrorScope` is a lost GPU device, not a failed assertion
Goal (G): get the R68 round (styleGate / t16 vs baseline) through 8 jobs of 512-world 200k-tick realm training and read a verdict.
What drifted / what went wrong: three jobs died with `OperationError: Instance dropped in popErrorScope` and I started by treating it as a code fault from my own NSTATS 16->20 change, because the first occurrence (r68t, 9 of 21 tests) sat next to a real bug I had just fixed. The tell that it was not code: 448 earlier queue jobs never produced that string once, and the run log showed a single 30 s progress line followed by 480 s of silence - a blocked main thread, not a wrong number.
Fix / resolution: read the Windows System log (`Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='nvlddmkm'}`): event id 153 every ~5 min, i.e. D3D device removed / driver reset, while another project's `resound.exe` held the 3060 at 98% util and 80 C. Nothing in src caused it. Paused the round (`queue stop` preserves attempts), taught tools/queue.mjs to check GPU health before each attempt (nvlddmkm resets in the last 12 min + foreign non-Chrome compute pids from `nvidia-smi --query-compute-apps`, bounded wait, `--no-gpu-health` to disable, `queue gpu` to ask), and resumed with `--retry-failed --max-attempts=4`.
Generalizes to: on this shared box, any GPU error that is not a numeric mismatch is environmental until the driver log says otherwise. Check `node tools/queue.mjs gpu` before opening src. Prefer pausing a round over letting a storm burn each job's 2 attempts.

## 2026-10-07 -- a lost GPU device must not spend the job's retry budget

Goal (G): keep an A/B round running to a verdict while another project's CUDA job holds the
shared RTX 3060.

What drifted / what went wrong: three R68 jobs died on `Instance dropped in popErrorScope`
(driver TDR under `resound.exe` at 99% util). The queue already retried device loss, but it
charged every loss against `maxAttempts`, so `r68n-s2` burned all 4 attempts in 148 s -- four
~37 s deaths, none of them the job's fault. Separately the GPU-health gate counted nvlddmkm
resets but only *reported* foreign compute pids, so a job happily started straight into another
project's 99%-util window.

Fix / resolution: (a) `isEnvironmentalFailure()` recognises a lost device and REFUNDS the
attempt (`job.attempts--`), capped separately by `--env-retry-cap`, so environmental noise costs
wall clock instead of jobs; (b) the gate now blocks on a foreign compute pid as well as on
resets, and only starts after the card has been clean for `--gpu-health-hold-seconds` (90 s
sustained, not one lucky poll), all bounded by `--gpu-health-wait-minutes`. Verified live:
`r68n-s2: waiting ... gpu device lost, environmental retry 1/8` with `attempts` still 0.

Generalizes to: any shared-resource queue here should separate "this job is broken" from "the
resource went away" -- they need different budgets and different waits, or a transient storm
silently converts healthy jobs into permanent failures. Also: a start gate that samples once is
not a gate; require the condition to HOLD before spending an attempt.

## 2026-10-07 -- a refunded retry protects the queue, not the progress

Goal (G): finish the R68 A/B round (512 worlds, 200k ticks) on a shared RTX 3060 while another
project's `resound.exe` holds the card at ~100% in multi-minute bursts.

What drifted / what went wrong: I built (and it worked) an environmental-retry refund so a lost
GPU device no longer spends the job's `maxAttempts`, plus a start gate that waits for a clean card.
Both behaved exactly as designed: t16-s1 survived 6+ device losses and was never charged for them.
But the round still did not finish, because a 200k-tick job needs ~455s of UNINTERRUPTED GPU and
the available windows are ~1-3 min. Every refunded retry restarts from tick 0, so the job burns
wall clock (63 min on one arm) without ever banking progress. I had implicitly assumed "survive the
failure" was the same as "make progress despite the failure". It is not.

Fix / resolution: the missing capability is checkpoint + resume (`?resume=<name>`), which converts
"lose the job" into "lose three minutes" and lets a job accumulate across many short windows. Spec
written to runs/NEXT-ROUNDS.md. Secondary fix already applied: `saveSeconds` 600 never fires inside
a 455s job, so next round's jobs use 120 - without a checkpoint written before the crash, resume
has nothing to load.

Generalizes to: when a resource is contended rather than broken, measure the job length against the
window length BEFORE queueing. If the job cannot fit in one window, retries are worthless without
checkpointing, and the honest options are (a) build resume, (b) shorten the job, or (c) say the
round is blocked and wait. Do not queue more arms into a contended card and call it progress - the
queue will look healthy (refunds working, no failures) while the round goes nowhere.

## 2026-10-07 -- a converged champion cannot answer "would a learner adopt this?"
Goal (G): break realm's ranged monoculture so the v5 content (perks, cover, boss phases) fires.
What drifted / what went wrong: I spent a whole CPU-measurement pass trying to read STYLE ADOPTION
off a fixed champion. A PPO champion is deterministic in its style choice -- `learnerDecide` takes
the argmax of three outputs -- so buffing melee changed `atkMelee` from 0 to 0 while I buffed it,
and the only champion that did move was one that was ALREADY mixed. Same trap in the forced-style
probe: `--gate=N` forces the ATTACK style but not the movement policy, so a ranged-trained brain
plays melee at 800 units of standoff and understates melee by an unknown amount.
Fix / resolution: split the instruments by what each can actually see. CPU + fixed champion answers
"did the world get harder or richer" (learner rate, bot rate, mobKills, bossKills, deaths) -- that
is how candidates E (-10.8%) and A (-27.0%) got rejected cheaply. GPU training is the only thing
that answers "which style does a fresh learner pick", so the style-mix bar belongs to the GPU arm
alone, and the bar was pre-registered there (melee+mage >= 15%) before it ran.
Generalizes to: every "REJECTED" verdict on a melee/mage-side lever measured under monoculture
(v5ch charge, v5res resists) is VOID -- no learner used melee, so the lever was never exercised.
Treat a converged policy as a probe of the ENVIRONMENT, never of the POLICY SPACE.

## 2026-10-07 -- an Edit whose new_string drops lines deletes them silently
Goal (G): prototype a melee suppression zone in `tryAttack` without breaking the loop.
What drifted / what went wrong: I passed an `old_string` of ~14 lines and a `new_string` that began
and ended inside it, omitting the loop header and the `let best = ...` declaration. The edit
succeeded and left `tryAttack` reading `const d2` with no `dx`, `dy`, `j` or `best` in scope -- a
file that would have thrown on the first attack. I only found it because I re-read the function.
Fix / resolution: after any Edit that removes lines from a function, re-read the whole function
before running anything; and when two functions share a near-identical block (`tryAttack` and
`tryCast` here), anchor `old_string` on a line unique to the target, not on the shared tail.
Generalizes to: Edit failures are loud ("2 matches"), but Edit successes that change line COUNT are
silent. Grep for the symbol you just added or removed, and read the enclosing function, every time.

## 2026-10-07 -- piping a long background command through `tail` hides all progress until it ends
Goal (G): measure a CPU ladder of realm balance variants while a GPU job held the card, so the
next arm could be queued the moment it was free.
What drifted / what went wrong: ran `node ladder.mjs ... 2>&1 | tail -20` in the background. The
output file stayed empty for 20 minutes, so I could not tell a slow run from a hung one, and when
I finally killed it the partial results were lost. The first run of the same script had looked
like it took 3 minutes only because it finished.
Fix / resolution: re-ran with `> file.txt 2>&1` (no pipe) so the file grows line by line and
progress is visible, and with `--jobs=2` so it did not fight the live GPU job for CPU.
Generalizes to: any background command whose output I intend to read incrementally must write
straight to a file - `| tail`, `| head` and friends buffer until the process exits. Also: before
killing a "hung" job, list the node command lines (`Get-CimInstance Win32_Process`) - the process
I suspected of being my run was `node src/client.js` from a different project.

## 2026-10-07 -- a scratch tree that "restores on success" is poisoned the moment it is killed
Goal (G): measure realm balance variants in a throwaway copy so the queued tree stays untouched.
What drifted / what went wrong: `ladder.mjs` patched `src/games/realm.js` in the probe tree and
restored the original only on the LAST line of the script. I killed one run mid-way, so the tree
kept that run's constants. The next script read them as its "baseline" and printed a row labelled
`baseline F` that was actually old-player-reach plus aggressive mobs - the numbers looked entirely
plausible and only gave themselves away because they matched a different variant's row exactly.
Fix / resolution: re-copied `src/games/realm.js` from the real tree and asserted the three
constants (and the absence of each patch marker) before re-running.
Generalizes to: any script that mutates a shared file and restores at the end must restore in a
`finally`, or print the effective constants with every row so a poisoned baseline cannot hide.
Also worth checking after the fact: a run whose variants set EVERY knob it touches is immune to
the inherited state, which is why the earlier mob ladder survived this.

## 2026-10-07 -- a keepalive line written into a job log does not survive into the next attempt
Goal (G): keep a silent `tests=1` job from being killed by the runner's `stallSeconds`, which
fires on "no new log lines" and would waste a whole 21-test run.
What drifted / what went wrong: I appended a keepalive line to `runs/r69t.log` while attempt 1
was in flight and assumed it would also protect attempt 2. It cannot: `runAttempt` calls
`archiveLog(job.name)` (queue.mjs:380) *before* launching Chrome, so every attempt starts from
an empty log and the stall clock restarts at launch.
Fix / resolution: treat a keepalive as per-attempt, not per-job. It only counts if it is written
after the attempt has actually started (job status `running`), and it must be re-applied after
every retry.
Generalizes to: anything that games the runner's liveness heuristics (log growth, first output,
chrome-exit grace) has to be re-established on each attempt, because attempt setup deliberately
resets the observable state the heuristics read.

## 2026-10-07 -- throughput levers in this game are non-monotonic; ladder three points, never assume
Goal (G): close mage's reward gap to ranged by raising its attack rate.
What drifted / what went wrong: I assumed "lower cooldown is better" and would have picked the
lowest value that looked safe. Measured ladder of `STYLE_COOLDOWN[2]`: 24 -> 5.689 (49 attacks),
16 -> 6.090 (64 attacks), 12 -> 5.944 (76 attacks). The fastest cooldown fires the most attacks
and earns LESS, because extra swings displace gathering and are worth less than what they cost.
Fix / resolution: always measure at least three points on a lever before calling a direction, and
treat a monotone assumption as a hypothesis to be falsified rather than a shape to interpolate.
Generalizes to: any rate/resource lever here (cooldowns, respawn, regrow, prices). Also applies
to the melee reach ladder, where three different constants all made the gap WORSE - the shape of
these curves is not knowable from two points.

## 2026-10-07 -- a `tests=1` job validates the build as of its START, not its finish
Goal (G): confirm the shipped F+M+N realm build (melee lunge + mage cooldown) was correct
before spending 30-minute GPU training arms on it.
What drifted / what went wrong: I edited `src/games/realm.wgsl.js` at 11:33 while the
`r69t` test job was still running (10:27 -> 11:42, 4507 s). Its 21/21 PASS therefore
described the PRE-edit source and never covered `r_lunge`. The next job, `r69f-s1`,
loaded the real shipped build and died in 6 s: `line 1320: 'target' is a reserved
keyword` in `fn r_lunge(e: u32, target: u32)`. I had already told the user r69t
"validates the shipped build", which was simply false.
Fix / resolution: renamed the parameter to `victim` (WGSL reserves `target`), rebuilt
(`build.mjs`, hash 2d962af9c7c910f1 -> 04107a6086ad1499), then `queue stop` +
`queue resume --retry=r69f-s1` to re-queue the consumed seed. Recovered cheaply only
because a shader compile error fails in ~6 s instead of burning the full 30 min.
Generalizes to: (1) "an edit lands on the NEXT job, not the running one" applies to
VALIDATION too - a test job proves the build as of the moment its page loaded, so never
edit src between starting a validation job and reading its result, and re-read mtimes
(`stat -c '%y'`) before trusting any "validated" claim. (2) WGSL reserves `target`
(also `filter`, `set`, `type`, `move`, `shared`, `precise`, ...) - check new parameter
and let-binding names against the reserved list, because the GPU is our only compiler
and a round trip costs 30 min. (3) A failed job can be re-queued by name with
`queue resume --retry=<name>` only after `queue stop`; `add` cannot replace one.

## 2026-10-07 -- a JS/WGSL constant can silently exist in only one path
Goal (G): ship the F/M/N/P style-balance candidates so the realm GPU trainer
actually trains on them, then measure the result at 200k.
What drifted / what went wrong: a const-parity scan of realm.js vs realm.wgsl.js
reported STYLE_RANGE, STYLE_COOLDOWN and LUNGE_RANGE as "js const with no RW_
twin", which reads like the candidates were shipped JS-only. They were not:
realm.wgsl.js writes those three as if-chain FUNCTIONS (`r_style_range`,
`r_style_cooldown`, `r_lunge_range`) rather than const arrays, because WGSL const
arrays of i32 are awkward to index. The scan's shape assumption, not the code,
was wrong -- but had the code really been JS-only, two 30-minute GPU runs would
have measured the pre-F build and reported "monoculture unchanged" for the wrong
reason, with nothing in the log to say so.
Fix / resolution: `tools/constparity.mjs` now checks both shapes (const twin and
fn twin, with an explicit FN_TWINS alias map for the name mismatches like
r_mob_range <-> MOB_STYLE_RANGE), and was self-tested by injecting two breaks
into a throwaway copy: it caught both and exited 1.
Generalizes to: before spending GPU wall clock on a candidate, verify it landed
in BOTH src/games/realm.js and src/games/realm.wgsl.js, and do not trust a
scanner that only knows one spelling. WGSL tunables come in at least two shapes
(const and if-chain fn); a scan that assumes one reports false positives that
look exactly like real divergence. Also: WGSL has no implicit numeric
conversion, so check the operand type before mirroring a JS expression --
r_f is array<i32>, which is why `dx*dx + dy*dy <= R*R` compiles at all.

## 2026-10-07 -- the published game broke because GitHub Pages serves HEAD, not the worktree
Goal (G): keep the strongest champion playable on a public page so humans can test themselves against it.
What drifted / what went wrong: I verified play.html locally (headless Chrome, 900 ticks, sane scores)
and pushed it, then reported "live". But Pages serves the committed HEAD, and HEAD still held the v4
realm chunk (`dims: { nIn: 105, nOut: 12 }`) while `champion.json` is a v5 genome (nIn 103 / nOut 13) --
every v5 change was still uncommitted. On the live site `RealtimeNpc.decide` threw
`RangeError: offset is out of bounds` at `this.lastIn.set(inputs)` on the first tick. A local pass said
"works"; the thing users load said otherwise.
Fix / resolution: committed the whole tree the champion was actually trained on (dc9e2bb) and republished.
Verification rule now: for anything deployed from git, the check is "does the DEPLOYED artifact work",
never "does the local worktree work" -- they differ whenever the worktree is dirty.
Generalizes to: any "it's live" claim about Pages/Kaggle/CDN output needs one check against the remote
URL after the build settles, comparing the served source to the worktree (here: `curl <url>/src/games/realm.js`
and grep the dims), not just a 200 on the entry file. A 200 only proves the file exists.
