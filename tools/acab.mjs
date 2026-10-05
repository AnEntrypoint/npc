// Online-learning A/B on the JS env: frozen vs Hebbian vs streaming AC.
// usage: node tools/acab.mjs [--game blob] [--champ runs/<name>-base | --random] [--modes frozen,heb,ac] [--ac sigma=0.05] [--ticks 50000] [--windows 5] [--seeds 11,12]
import { GAMES } from '../src/games/index.js';
import { RealtimeNpc, REWARD_SCALE } from '../runtime/npc.js';
import { randomGenome, genomeToJSON, Rng, mix } from '../src/core.js';
import fs from 'node:fs';

function parseArgs(argv) {
  const out = { game: 'blob', champ: null, random: false, modes: 'frozen,heb,ac', ac: null, ticks: 50000, windows: 5, seeds: '11,12', massStat: 12 };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    if (a === '--game') out.game = next();
    else if (a === '--champ') out.champ = next();
    else if (a === '--random') out.random = true;
    else if (a === '--modes') out.modes = next();
    else if (a === '--ac') {
      out.ac = {};
      for (const part of String(next()).split(',')) {
        const kv = part.split('=');
        if (kv.length === 2 && kv[0].trim()) out.ac[kv[0].trim()] = Number(kv[1]);
      }
    }
    else if (a === '--ticks') out.ticks = Number(next());
    else if (a === '--windows') out.windows = Number(next());
    else if (a === '--seeds') out.seeds = next();
    else if (a === '--mass-stat') out.massStat = Number(next());
  }
  return out;
}

function genomeFor(opts, game, seed) {
  if (opts.champ) return JSON.parse(fs.readFileSync(opts.champ, 'utf8'));
  return genomeToJSON(randomGenome(new Rng(mix(seed, 7, 0, 77)), 256, game.dims), game);
}

function run(opts, game, genome, mode, seed) {
  const stats = new Int32Array(16);
  const env = game.createEnv(seed, game.defaultCfg(), false, stats);
  const learners = game.learners;
  const maxAge = game.maxAge;
  const npcs = [];
  for (let a = 0; a < learners; a++) {
    npcs.push(new RealtimeNpc(genome, {
      seed,
      agent: a,
      learning: mode !== 'frozen',
      learnMode: mode === 'ac' ? 'ac' : undefined,
      ac: mode === 'ac' ? opts.ac : undefined
    }));
  }
  const obs = new Float32Array(game.dims.nIn);
  const actions = new Int32Array(learners);
  const outputs = new Float32Array(learners * game.dims.nOut);
  const per = opts.ticks / opts.windows;
  const w0 = npcs[0].w.slice();
  const rows = [];
  let massPrev = stats[opts.massStat];
  let rew = 0, deaths = 0, dsum = 0, dn = 0;
  for (let t = 0; t < opts.ticks; t++) {
    for (let a = 0; a < learners; a++) {
      env.observe(a, obs, t);
      actions[a] = npcs[a].decide(obs, t);
      for (let k = 0; k < game.dims.nOut; k++) outputs[a * game.dims.nOut + k] = npcs[a].outputs[k];
    }
    env.step(actions, outputs, t);
    for (let a = 0; a < learners; a++) {
      const r = env.reward[a] / REWARD_SCALE;
      rew += r;
      if (a === 0 && npcs[0].ac) { dsum += Math.abs(npcs[0].ac.lastDelta || 0); dn++; }
      const died = env.dead[a] !== 0 || npcs[a].lifeTicks + 1 >= maxAge;
      npcs[a].reward(r, t, died);
      if (died) {
        deaths++;
        env.respawn(a, t);
        npcs[a].rebirth();
      }
    }
    if ((t + 1) % per === 0) {
      let dw = 0, bad = 0;
      const w = npcs[0].w;
      for (let i = 0; i < w.length; i++) {
        dw += Math.abs(w[i] - w0[i]);
        if (!Number.isFinite(w[i])) bad++;
        w0[i] = w[i];
      }
      const gained = stats[opts.massStat] - massPrev;
      massPrev = stats[opts.massStat];
      rows.push({
        t: t + 1,
        rew: +(rew / learners / per * 1000).toFixed(3),
        gain: +(gained / learners).toFixed(1),
        deaths: +(deaths / learners).toFixed(2),
        dw: +dw.toFixed(2),
        dbar: dn ? +(dsum / dn).toFixed(2) : 0,
        bad
      });
      rew = 0; deaths = 0; dsum = 0; dn = 0;
    }
  }
  const mean = rows.reduce((s, r) => s + r.rew, 0) / rows.length;
  return { mode, opts: opts.ac, seed, mean: +mean.toFixed(3), rows };
}

const opts = parseArgs(process.argv.slice(2));
const game = GAMES[opts.game];
const seeds = String(opts.seeds).split(',').map(Number);
const out = [];
for (const seed of seeds) {
  const genome = genomeFor(opts, game, seed);
  for (const mode of opts.modes.split(',')) out.push(run(opts, game, genome, mode, seed));
}
console.log(JSON.stringify({ game: game.id, champ: opts.champ, ticks: opts.ticks, out }, null, 1));
