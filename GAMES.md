# Adding a game

The trainer (`src/core.js`, `src/shader.js`, `src/engine.js`, `src/app.js`) never changes per game. A game is one plugin: a JS `Env` that is the real game logic, plus an optional WGSL port of the same logic so it trains on the GPU.

## Real-time rule
Training is the same fixed-step simulation the shipped game runs, only stepped faster. The Env has no wall-clock dependence, advances exactly one tick per `step`, and declares `tickHz`. The shipped game steps it with an accumulator at `tickHz`; the trainer steps it as fast as the hardware allows. NPC senses, actions and per-tick mechanics are never simplified for training. Domain randomization (`randomCfg`) only varies world parameters between training worlds.

## Files
- `src/games/<id>.js` exports a `GameDef` and the Env class.
- `src/games/<id>.wgsl.js` exports the WGSL chunk string (optional; without it the game trains on the CPU backend only).
- Register in `src/games/index.js`. `node build.mjs` inlines everything into `index.html`.

## GameDef
```
id, name, tickHz
dims { nIn, nOut, nNodes }        nNodes <= 256 and > nIn+nOut
agents, learners                  learners are agents 0..learners-1; agents <= 64 (one GPU thread each)
maxAge                            lifetime ticks before a learner is replaced
rewardChannels (optional)         names of per-source reward channels; env.rewardCh Int32Array(slots*channels) holds the tick's fixed-point values, env.reward stays their sum, WGSL exposes g_reward_channel(a, c) after g_step
worldWords                        u32 words of per-world game state
workgroupBytes                    workgroup memory the chunk declares (the kernel adds its own share)
inputNames[], actionNames[], statNames[<=7]
defaultCfg(), randomCfg(seed)     per-world config; randomCfg must be integer arithmetic on mix(seed,...)
createEnv(seed, cfg, isEval, stats) -> Env
packEnv(env) -> Uint32Array(worldWords)          identical layout to the WGSL g_save
snapshotFromWords(words, seed), validateWords(words), densify(env, rng), render(ctx, snapshot, w, h)
wgsl
```

## Env
```
observe(a, out Float32Array(nIn), tick)
step(actions Int32Array(learners), outputs Float32Array(learners*nOut), tick)
reward Int32Array(learners)     fixed point x1024 for the tick just stepped
dead Uint8Array(learners)
respawn(a, tick)
snapshot()
```
`outputs` hold the raw output-node activations (softsign, -1..1); `actions` is the argmax with epsilon exploration. Use whichever suits the game (continuous games read `outputs`). Scripted agents (the baseline) live inside the Env; add their reward to `stats[STAT.REW_BASE]` and one to `stats[STAT.TICKS_BASE]` per scripted agent-tick. Game counters go to `stats[STAT.GAME0 + i]` in `statNames` order. `render` must be self-contained (the runtime server ships it to browsers via `toString()`).

## Determinism rules (needed for exact CPU/GPU parity)
- Integer state and integer math only: positions, velocities, masses, timers. Use `>>` for arithmetic shifts, `Math.trunc(a / b)` for division (WGSL truncates toward zero), integer square root by correction loop.
- Randomness only through `mix(seed, tick, index, salt)` (`mix4` in WGSL); never `Math.random`.
- Brain outputs enter the game through a quantizing step (for example `floor(fround(o0 - o1) * 128)`); floats never persist in game state. Observations should be dyadic (multiples of a power of two) so CPU and GPU produce identical inputs.
- Cross-entity effects are resolved victim-centric or node-centric (each entity scans all others and writes only its own fields) or with integer atomics; ties break by lowest index. Never depend on thread order.

## WGSL chunk contract
The kernel calls these uniformly (barriers allowed inside); thread `t` is entity `t`, 64 threads per workgroup, one workgroup per world:
```
g_init(wi, seed, is_eval, randomize, t)   generate the world and g_save it
g_load(wi, t) / g_save(wi, t)             workgroup memory <-> world buffer (game words start at GAME_OFF)
g_observe(t, tick)                        learners write inputs with set_obs(a, i, v)
g_step(t, tick)                           advance one tick; call set_reward(a, fx), set_dead(a), stat_add(idx, v)
g_respawn(a, tick)                        called serially by thread 0, ascending a, for dead learners
```
Kernel-provided: `NIN NOUT NODES LEARNERS AGENTS`, `w_seed`, `action_of(a)`, `out_of(a, k)`, `game_word(wi, i)`, `game_set(wi, i, v)`, `stat_add`, `STAT_*`, `mix4`, `pcg`, `rf`, `rn`, `bf`, `fb`. Names in the chunk must not collide with kernel names; prefix them per game.

## PPO trainer compatibility
The same chunk trains with `RlBackend` (src/rlshader.js prelude provides the same kernel-provided names; `out_of(a, k)` returns the sampled action clamped to [-1, 1] during training and mu during eval, `action_of` returns 0). Optional self-play support: `GameDef.maxLearners` (kernel `LEARNERS` becomes this value), WGSL `g_learner_slots() -> u32` (active learner slots of the loaded world; slots at or above it are scripted), and `g_init(..., randomize == 2u)` = default cfg with every slot a learner. Games without `g_learner_slots` get a default returning `LEARNERS`. Optional reward channels: `GameDef.rewardChannels` (names) plus WGSL `g_reward_channel(a, c) -> i32` (this tick's per-channel reward in fixed point, valid right after `g_step`) let the PPO option `channelCaps` clamp every channel per tick before the total is used and let a PPO population (`policies` > 1) train each policy on its own weighted sum of the channels (the channels should therefore be separable pressures, e.g. survival / progress / combat / cooperation; at most 8, and `g_reward_channel` must be callable for every learner slot right after `g_step`). The trainer's curriculum re-initialises worlds through `g_init` (randomize 0 = default cfg with bots, 2 = self-play). Otherwise nothing extra is required from a game: `g_observe`, `g_step`, `g_respawn`, `set_reward`, `set_dead`, `stat_add` behave exactly as under evolution. Requirements that matter for PPO: rewards must be dense-ish and bounded (they are divided by 1024 and auto-scaled), `g_observe` must be side-effect free (the rollout calls it once more before respawning a character truncated at `maxAge` and once at the end of a rollout), workgroup use of the chunk plus the kernel's own share must fit `maxComputeWorkgroupStorageSize` (on realm that share is 17,064 B in the evolution kernel and about 16 KB in the PPO rollout kernel, against a 32,768 B grant), and node ids of the dense genome (`nIn + nOut + hidden`) must stay below 256. `RlBackend.runTests()` covers rollout determinism, forward/GAE/gradient/Adam parity against src/rl.js and export replay through `core.js` `Brain`.

## Verification (product code, `GpuBackend.runTests()`, also the Run tests button)
`runTests()` records one result per check, eight of them (seven without `densify`):
1. workgroup memory fits the device grant
2. world init identical to the JS env, word for word
3. brain forward/learn/structural parity between GPU and `Brain`
4. env lockstep parity: same actions on the JS env and the GPU kernel, packed state compared every tick, natural start
5. the same lockstep on a dense scenario from `densify` that forces interactions (only when the game defines `densify`)
6. observation parity, GPU vs JS
7. invariants after training passes
8. statistical equivalence of a full GPU run with the CPU backend

A port is done when every check passes. Budget: the kernel needs `layout.workgroupBytes` (shown in the device panel) within `maxComputeWorkgroupStorageSize` (32 KB on most desktop GPUs, 16 KB is the spec floor).

## Design lessons (realm v4)
- Make progression the gate, not a bonus: group harvest nodes need tool tier 2, boss damage scales with weapon tier, the wilds (over 1800 units from any town) halve the food and water period unless the character carries a tool tier. Champions only discover crafting when every fixed cost is discoverable: a spawn kit (3 wood, 3 ore, 10 gold) makes tier 1 one button away, and crafting, buying and selling are executed by priority when the button is pressed and the action is possible, otherwise a saturated always-interact policy blocks them for ever (v4 first attempt: interact 0.97, craft -0.9, buy never).
- Rewards for the same materials must rank craft above sale (sale gold pays x8, craft tier pays 256 / 1024 / 2048) or the policy sells the kit at spawn; per-life first-time rewards must stay below the death penalty or suicide for a new kit pays (kit rewards sum to 0.75 against -1).
- Repeatable sources get diminishing repeats (halving per repeat, floor 1/8) and caps (mob gold, sale gold, PvP bounty, boss loot); gold itself is granted but not rewarded, so loot cannot be farmed for reward.
- Structural shortages are not difficulty: 32 players draining 1 water per 16 ticks against 32 springs regrowing every 700 ticks starves everyone by tick 1000 no matter how good the policy is. Keep supply about 1.3x demand and let travel and contention supply the scarcity.

## Design lessons (realm v3)
- A score-maximising evolved policy finds the cheapest reward source. Audit champions in Node (`tools/replay.mjs`: action shares, distance, safe-zone time, cause of death, harvests by type) before blaming the trainer. Typical exploits: attacking from an invulnerable zone, an infinite free resource point, a per-tick survival reward that dwarfs everything else, repeatable xp.
- Make needs local and rate-limited (springs and berries regrow slower than one visitor consumes them), give every resource a cooldown that also blocks the previous harvester, and never let a hub supply a need for free.
- Cap repeatable rewards: xp pays only until the level cap, repeated group rewards halve per repeat (`diminish`), first-time achievements pay once (`milestone`). Heavy-tailed rewards make the archive select lucky lineages.
- Use potential-based shaping only for discovery (distance to the needed resource while thirsty or hungry, clamped, telescoping so it cannot be farmed) and expose the same thing as input compasses so a 60-edge brain can wire it.
- Give the brain the information that a mechanic needs (boss resistance one-hot, great node progress and bearing, safe zone flag, recent hit) and keep the scripted teacher blind to cooperative content so it stays a baseline.
- Any nondeterministic-looking cross-entity write in a phase is a race on the GPU: precompute per-entity values (`sharePhase`) after a barrier and consume them in the next phase.
- `densify` must place entities outside safe zones, next to nodes and bosses with tiny hp, or the parity test silently stops covering combat, harvest and boss loot.
- Frozen copies of an older game version live in `dev/legacy/` (imported by `dev/ab.html`, `variants=label:game=realm31`) so two rule sets can be trained interleaved in one page; they are not part of the build.
