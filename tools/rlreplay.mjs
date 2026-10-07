import fs from 'node:fs';
import { GAMES } from '../src/games/index.js';
import { Brain, Rng, mix, evoConfig, genomeFromJSON, STAT, NSTATS, REWARD_SCALE } from '../src/core.js';

function parseArgs(argv) {
  const opts = { game: 'blob', worlds: 32, periods: 2, seed: 777, teacher: false, files: [] };
  for (const arg of argv) {
    const m = /^--([a-zA-Z]+)(?:=(.*))?$/.exec(arg);
    if (!m) { opts.files.push(arg); continue; }
    if (m[1] === 'teacher') opts.teacher = true;
    else if (m[1] === 'game') opts.game = m[2];
    else if (m[1] in opts) opts[m[1]] = Number(m[2]);
    else throw new Error('unknown flag --' + m[1]);
  }
  return opts;
}

function evaluate(game, genome, worldCount, periods, seed, useTeacher) {
  const dims = genome ? genome.dims : game.dims;
  const nOut = dims.nOut;
  const learners = game.learners;
  const total = new Float64Array(16);
  let deaths = 0;
  let learnerTicks = 0;
  let learnerReward = 0;
  const perWorld = [];
  for (let w = 0; w < worldCount; w++) {
    const worldSeed = mix(seed, w, 0, 9);
    const stats = new Int32Array(NSTATS);
    const env = game.createEnv(worldSeed, game.defaultCfg(), true, stats);
    env.difficulty = 100;
    const brains = [];
    if (!useTeacher) {
      const parsed = genomeFromJSON(genome, genome.edges.length);
      for (let a = 0; a < learners; a++) brains.push(new Brain(parsed, genome.edges.length, dims, evoConfig(null)));
    }
    const obs = new Float32Array(game.dims.nIn);
    const actions = new Int32Array(learners);
    const outputs = new Float32Array(learners * nOut);
    const life = new Int32Array(learners);
    let worldReward = 0;
    let worldTicks = 0;
    const ticks = periods * game.maxAge;
    for (let tick = 0; tick < ticks; tick++) {
      if (tick > 0 && tick % game.maxAge === 0) {
        for (let a = 0; a < game.agents; a++) env.respawn(a, tick);
        if (!useTeacher) for (const brain of brains) brain.load(genomeFromJSON(genome, genome.edges.length));
        life.fill(0);
      }
      for (let a = 0; a < learners; a++) {
        if (useTeacher) {
          const out = new Float32Array(nOut);
          game.teacher(env, a, tick, out);
          for (let k = 0; k < nOut; k++) outputs[a * nOut + k] = out[k];
        } else {
          env.observe(a, obs, tick);
          actions[a] = brains[a].step(obs, new Rng(mix(worldSeed, tick, a, 1)));
          for (let k = 0; k < nOut; k++) outputs[a * nOut + k] = brains[a].act[dims.nIn + k];
        }
      }
      env.step(actions, outputs, tick);
      const respawn = [];
      for (let a = 0; a < learners; a++) {
        worldReward += env.reward[a];
        worldTicks++;
        life[a]++;
        if (env.dead[a] !== 0 || life[a] >= game.maxAge) {
          deaths++;
          respawn.push(a);
        }
      }
      for (const a of respawn) {
        env.respawn(a, tick);
        if (!useTeacher) brains[a].load(genomeFromJSON(genome, genome.edges.length));
        life[a] = 0;
      }
    }
    learnerReward += worldReward;
    learnerTicks += worldTicks;
    for (let i = 0; i < NSTATS; i++) total[i] += stats[i];
    perWorld.push(worldReward / REWARD_SCALE / worldTicks);
  }
  const mean = perWorld.reduce((s, v) => s + v, 0) / perWorld.length;
  const std = Math.sqrt(perWorld.reduce((s, v) => s + (v - mean) * (v - mean), 0) / Math.max(1, perWorld.length - 1));
  return {
    learnerRate: learnerReward / REWARD_SCALE / learnerTicks,
    standardError: std / Math.sqrt(perWorld.length),
    baseRate: total[STAT.REW_BASE] / REWARD_SCALE / Math.max(1, total[STAT.TICKS_BASE]),
    meanLife: learnerTicks / Math.max(1, deaths),
    game: Object.fromEntries(game.statNames.map((name, i) => [name, total[STAT.GAME0 + i]]))
  };
}

const opts = parseArgs(process.argv.slice(2));
const game = GAMES[opts.game];
const format = (r) => Object.entries({ learnerRate: r.learnerRate.toFixed(5), stderr: r.standardError.toFixed(5), baseRate: r.baseRate.toFixed(5), ratio: (r.learnerRate / r.baseRate).toFixed(2), meanLife: r.meanLife.toFixed(0) }).map(([k, v]) => k + '=' + v).join(' ') + ' game=' + JSON.stringify(r.game);
if (opts.teacher) console.log('teacher-as-learner ' + format(evaluate(game, null, opts.worlds, opts.periods, opts.seed, true)));
for (const file of opts.files) {
  const parsed = JSON.parse(fs.readFileSync(file, 'utf8'));
  const list = Array.isArray(parsed) ? parsed : [parsed];
  console.log(file + ' (' + list[0].edges.length + ' edges) ' + format(evaluate(game, list[0], opts.worlds, opts.periods, opts.seed, false)));
}
