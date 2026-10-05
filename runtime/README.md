# runtime

## Real-time fidelity

An NPC at runtime is the training brain, not an approximation: `RealtimeNpc` in `npc.js` is a self-contained copy of `Brain` in `src/core.js` (softsign leaky recurrence, novelty gate where skipped ticks repeat the last action, epsilon exploration, online ABCD Hebbian plasticity on every nonzero reward, prune/grow every 32 lifetime ticks). Plasticity stays on in deployment. The per-tick RNG streams use the same `mix(seed, tick, agent, 1|3)` keys as `WorldRunner`, so given the same env seed a `RealtimeNpc` reproduces training bit for bit. A 6000-tick lockstep run of `WorldRunner` against `RealtimeNpc` on the `blob` game (24 learners, 683 deaths, several prune events) matched on env state, weights, edge lists, activations, `rewAvg` and `ops` at every tick.

The game loop is the same fixed-step `Env` as training, driven by `FixedStepLoop` at `game.tickHz` (10 Hz if the game declares none).

## npc.js

Dependency-free ES module. To drop it into a plain HTML game, strip the `export ` prefixes (as `build.mjs` does); it then defines `window.Npc`.

```js
import { RealtimeNpc, FixedStepLoop, REWARD_SCALE } from './npc.js';
const npc = new RealtimeNpc(genomeJson, { seed, agent: slotIndex });
const loop = new FixedStepLoop({ hz: game.tickHz, step: (tick) => {
  env.observe(slotIndex, obs, tick);
  const action = npc.decide(obs, tick);
  env.step(actions, outputs, tick);
  npc.reward(env.reward[slotIndex] / REWARD_SCALE, tick, env.dead[slotIndex] !== 0);
} });
loop.start();
```

- `new RealtimeNpc(genomeJson, {seed, agent, maxEdges, learning})`: dims come from the genome (`npc-brain/3` adds an optional per-node `bias` and an optional per-node `leak` vector that overrides the global leak, which is how PPO-trained recurrent policies with hidden-to-hidden edges deploy; `npc-brain/2`; `npc-brain/1` means the legacy arena 12/8/64 layout). `maxEdges` defaults to 256, the trainer default; pass the trainer's value if you changed it, since it caps structural growth.
- `decide(inputs, tick)` returns the argmax action (with epsilon exploration); `npc.outputs` holds the output-node activations for envs that read analog outputs (blob reads `outputs`, not `actions`).
- `reward(r, tick, died)` applies plasticity, counts the lifetime tick and runs `structural` every 32 lifetime ticks unless the agent died this tick.
- `rebirth()` reloads the original genome and resets lifetime state (call it when the env respawns the agent). `toJSON(gameId)` exports the current, adapted brain.
- `FixedStepLoop({hz, step, render, speed, maxCatchUp, now})` is an accumulator loop; `advance(nowMs)` pumps it manually, `start()/stop()` self-schedule (requestAnimationFrame in browsers, setTimeout in Node).

### Streaming AC — online actor-critic (opt-in)

`new RealtimeNpc(genome, { learnMode: 'ac', ac: {...} })` replaces the Hebbian rule with **Stream AC(λ) + AdaptiveObGD** (arXiv 2410.14606, Elsayed/Vasan/Mahmood): one update per tick, no replay, no batches, no BPTT. This is the mode for an NPC that has to keep improving while it plays with real players; `learnMode` defaults to `'hebbian'`, so nothing else changes.

- Policy: Gaussian over the output nodes — the mean is the output activation, `sigma` is per output, and `outputs[k]` is the *sample* (`mean + sigma*z`), so analog envs act on the sample and discrete envs take its argmax. Critic: `v = wv . act + bias` over every node.
- Credit assignment: the policy score is backpropagated exactly within the tick (seed `g[out] = z/sigma`, then `g[src] += w·phi[dst]·g[dst]` down the node order through a CSR-by-destination index), so every edge gets real policy gradient, not just output-adjacent ones. One eligibility trace per parameter, `e ← γλ·e + d·g[dst]`, where `d` is the recurrent direct sensitivity of the edge's target to that parameter (`d ← leak·d + (1-leak)·gain/(1+|x|)^2 · presyn`, presyn taken from the previous activations for recurrent sources). O(edges) per tick — no RTRL matrix, no truncated window.
- Step size: ObGD rescales the step so the L1 norm of one update stays under `1/kappa`; AdaptiveObGD additionally divides by the debiased RMS of `δ·e` (`beta2` .999). `delta` is divided by its own running RMS, so the reward scale does not matter and `lr` barely matters once the cap binds.
- Options (`AC_DEFAULTS`): `lr` 1, `kappa` 2, `gamma` .997, `lamda` .97, `beta2` .999, `eps` 1e-8, `sigma` .05, `valueCoef` .5, `rewardScale` 1, `normDelta` 1, `adaptive` 1; `fromMeta` 1, `sigmaFromMeta` 0, `acCritic` 1 (below). Per-output `sigma` may be an array. `sigma` is the exploration the deployed NPC actually uses, so it is deliberately much smaller than the PPO trainer's `logStd` (see Measured).
- The critic is trained by semi-gradient TD on its own head only: its gradient is deliberately kept out of the shared weights (`valueCoef` now only scales the critic's share of the step budget). Letting it into the trunk destroyed learning in the bandit probe, so the trunk is pure policy gradient. Structural plasticity, epsilon exploration and the novelty gate are inert in this mode, and `sigma` is fixed, so there is no entropy coefficient to tune.
- Warm start from a PPO export: `meta.valueWeights`/`meta.valueBias` become the critic (rescaled by `1/REWARD_SCALE`), or the previous session's `acValue` if the genome carries one; `ac:{acCritic:0}` starts it at zero instead, and `ac:{fromMeta:0}` also skips everything else. `meta.logStd` is only used with `ac:{sigmaFromMeta:1}` — by default `sigma` stays at the deployment value. Traces reset on `rebirth()`, and `toJSON()` writes the adapted critic as `acValue` (nNodes+1 floats), which re-seeds the next session. `runtime/server.mjs --ac [--ac-opt kappa=4,sigma=0.05]` runs a whole lobby this way.
- Measured (blob, 24 learners, reward/tick/learner x1e3, `tools/acab.mjs`): a bandit with reward delayed 4 ticks is solved (delays 0 and 4 converge, delay 8 collapses onto the wrong sign because a constant reward leaves no advantage). On **converged PPO champions** (50k ticks, 2 champions) frozen .623/.647, AC at the champion's own training `sigma` (.48) .589/.606, .25 .613/.626, .10 .639/.661, **.05 .669/.658**, .02 .642/.654, .05 with `acCritic:0` .648/.660 — so the deployment sigma must be far below the training one, and warm-starting the critic is worth a little. On **untrained random brains** (35k ticks, 2 seeds) frozen -4.43, Hebbian -4.46, AC sigma .05 -4.38, **AC sigma .25 -4.17** with ~2 fewer deaths per learner — there the larger sigma wins by 6% over frozen and over the Hebbian rule. Rule of thumb: `sigma` .05 keeps improving a trained champion without damaging it; start an untrained brain at ~.25 (`ac:{sigma:0.25}`). Every run: no non-finite weights, |delta| ~.6 after normalisation, per-window weight drift decaying — it settles, it does not diverge. On **realm** (dense 11k-edge champion, 20k ticks, n=1) frozen 4.79 vs AC sigma .05 4.85 with fewer deaths and steady but non-growing weight drift: neutral-to-slightly-positive and stable, not an adoption at one seed. Realm's mean |delta| is only ~.15 of its RMS because its reward is spiky, so most updates are small and the rare large one does the work — that is what the RMS normaliser is for.

## server.mjs

```
node runtime/server.mjs --champions champions.json [--game blob] [--port 8080] [--seed 1234] [--hz N] [--speed 1] [--max-edges 256] [--ac [--ac-opt k=v,...]]
```

Hosts any `GameDef` registered in `src/games/index.js` (default `arena` if registered, otherwise the first game). Learner slots `0..learners-1` are driven by `RealtimeNpc` instances built from the champions file (a single genome, an array, `{champions:[...]}` or `{genomes:[...]}`; genomes are assigned round-robin; genomes whose `game` field mismatches are rejected; with no file, 8 random genomes are used). Scripted agents run inside the env. A connecting human takes over the highest free learner slot (the NPC in it is suspended, `env.respawn` is called) and is released on disconnect. Serves `/` and `/play.html`, `/npc.js`, `/render.js` (the game's `render` function source, when it has one), `/health`, and the WebSocket at `/ws`.

Protocol: client sends `{"t":"act","a":<index>,"o":[nOut floats]}` repeatedly while keys are held; each message lasts 2 ticks. `a` is the last-pressed action index (discrete games read `actions`), `o` is the multi-hot output vector (analog games read `outputs`). Without `o` the server derives a one-hot vector from `a`. Server sends `{"t":"hello","slot","game":{id,name,tickHz,actionNames,inputNames,palette,hasRender,...}}` once, then one `{"t":"snap","tick",...env snapshot fields...,"agents":[...+human],"scores":[...]}` per tick (typed arrays become plain arrays).

## play.html

Open `http://localhost:8080/`. Arrows move (mapped by action name: up/down/left/right or `-y/+y/-x/+x`), space is `use` or `boost`, F `attack`, T `trade`, digits 0..9 pick an action index directly. `grid` snapshots are drawn by the page (tile word: low 4 bits type indexes `game.palette`, bits 4..15 nonzero dims the tile as on cooldown); any other snapshot kind is drawn by the game's own `render(ctx, snap, w, h)`, which must be self-contained because it is shipped to the browser via `toString()`. The HUD lists every numeric field of your agent except `x`, `y`, `face`, `evo`, `alive`, `human` and `r`, plus your score and the tick.

## Plugging a new game into trainer and runtime

1. Write `src/games/<id>.js` exporting a `GameDef` (see SPEC.md): `dims`, `agents`, `learners`, `maxAge`, `tickHz`, `inputNames`, `actionNames`, `statNames`, `defaultCfg/randomCfg`, `createEnv`, and register it in `src/games/index.js`. Keep the Env a pure fixed-step simulation; scripted agents live inside it.
2. Train it in the trainer UI, export champions (`npc-brain/2` JSON carries `game` and `dims`).
3. `node runtime/server.mjs --game <id> --champions champions.json`. The server needs nothing else from the game: humans override a learner slot's `actions[a]` and `outputs[a*nOut..]` before `env.step`, so any game whose env reads those arrays is playable. For a game whose keys do not fit the default name mapping in `play.html`, name its actions `up/down/left/right/use/attack/trade` or `+x/-x/+y/-y/boost`, or extend `KEY_ACTION_NAMES`.
4. To ship NPCs inside the game itself, embed `npc.js` and use the `RealtimeNpc` + `FixedStepLoop` pattern above with the game's own observe/step.
