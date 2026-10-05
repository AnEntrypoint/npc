import { writeFile, mkdir } from 'node:fs/promises';
import { dirname } from 'node:path';
import { Brain, Rng, mix, makeGenome, genomeToJSON, NSTATS, STAT, REWARD_SCALE, PARAM_DEFAULT } from '../src/core.js';
import { GAMES } from '../src/games/index.js';

const args = Object.fromEntries(process.argv.slice(2).reduce((pairs, token, i, all) => (token.startsWith('--') ? pairs.concat([[token.slice(2), all[i + 1]]]) : pairs), []));
const game = GAMES[args.game || 'realm'];
const seeds = Number(args.seeds || 8);
const ticks = Number(args.ticks || 3000);
const stride = Number(args.stride || 3);
const maxEdges = Number(args.edges || 256);
const epochs = Number(args.epochs || 4);
const gain = Number(args.gain || 2);
const outPath = args.out || 'runs/clone-' + game.id + '.json';
const { nIn, nOut, nNodes } = game.dims;
const hidden = Number(args.hidden || nNodes - nIn - nOut);
if (!game.teacher) throw new Error('game ' + game.id + ' has no teacher');

const softsign = (x) => x / (1 + Math.abs(x));

function collect() {
  const inputs = [];
  const targets = [];
  const previous = [];
  for (let s = 0; s < seeds; s++) {
    const seed = mix(9001, s, 0, 9);
    const cfg = s % 2 === 0 ? game.defaultCfg() : game.randomCfg(seed);
    const env = game.createEnv(seed, cfg, false, new Int32Array(NSTATS));
    const obs = new Float32Array(nIn);
    const target = new Float32Array(nOut);
    const actions = new Int32Array(game.learners);
    const outputs = new Float32Array(game.learners * nOut);
    for (let a = 0; a < game.learners; a++) previous[a] = null;
    for (let tick = 0; tick < ticks; tick++) {
      for (let a = 0; a < game.learners; a++) {
        env.observe(a, obs, tick);
        game.teacher(env, a, tick, target);
        if (previous[a] && tick % stride === 0) {
          inputs.push(previous[a]);
          targets.push(Float32Array.from(target));
        }
        previous[a] = Float32Array.from(obs);
        let best = 0;
        for (let k = 0; k < nOut; k++) {
          outputs[a * nOut + k] = target[k];
          if (target[k] > target[best]) best = k;
        }
        actions[a] = best;
      }
      env.step(actions, outputs, tick);
      for (let a = 0; a < game.learners; a++) if (env.dead[a]) { env.respawn(a, tick); previous[a] = null; }
    }
  }
  return { inputs, targets };
}

function initWeights(rng) {
  const w1 = new Float32Array(hidden * nIn);
  const w2 = new Float32Array(nOut * hidden);
  for (let i = 0; i < w1.length; i++) w1[i] = rng.normal() * Math.sqrt(1 / nIn);
  for (let i = 0; i < w2.length; i++) w2[i] = rng.normal() * Math.sqrt(1 / hidden);
  return { w1, w2 };
}

function trainEpochs(model, masks, data, count, lr) {
  const { w1, w2 } = model;
  const n = data.inputs.length;
  const order = Int32Array.from({ length: n }, (_, i) => i);
  const batch = 128;
  const m1 = new Float32Array(w1.length), v1 = new Float32Array(w1.length), m2 = new Float32Array(w2.length), v2 = new Float32Array(w2.length);
  const g1 = new Float32Array(w1.length), g2 = new Float32Array(w2.length);
  const h = new Float32Array(hidden), y = new Float32Array(nOut), dz2 = new Float32Array(nOut), dh = new Float32Array(hidden);
  const outputWeight = Float32Array.from({ length: nOut }, (_, k) => (k < 4 ? 1 : 3));
  const rng = new Rng(77);
  let step = 0;
  let loss = 0;
  for (let epoch = 0; epoch < count; epoch++) {
    for (let i = n - 1; i > 0; i--) { const j = Math.floor(rng.next() * (i + 1)); const t = order[i]; order[i] = order[j]; order[j] = t; }
    loss = 0;
    for (let start = 0; start + batch <= n; start += batch) {
      g1.fill(0);
      g2.fill(0);
      for (let s = start; s < start + batch; s++) {
        const x = data.inputs[order[s]];
        const t = data.targets[order[s]];
        for (let j = 0; j < hidden; j++) {
          let z = 0;
          const base = j * nIn;
          for (let i = 0; i < nIn; i++) z += w1[base + i] * x[i];
          h[j] = softsign(gain * z);
        }
        dh.fill(0);
        for (let k = 0; k < nOut; k++) {
          let z = 0;
          const base = k * hidden;
          for (let j = 0; j < hidden; j++) z += w2[base + j] * h[j];
          y[k] = softsign(gain * z);
          const err = y[k] - t[k];
          loss += outputWeight[k] * err * err;
          const d = (2 * outputWeight[k] * err * gain) / ((1 + Math.abs(gain * z)) ** 2);
          dz2[k] = d;
          for (let j = 0; j < hidden; j++) {
            g2[base + j] += d * h[j];
            dh[j] += d * w2[base + j];
          }
        }
        for (let j = 0; j < hidden; j++) {
          let z = 0;
          const base = j * nIn;
          for (let i = 0; i < nIn; i++) z += w1[base + i] * x[i];
          const d = (dh[j] * gain) / ((1 + Math.abs(gain * z)) ** 2);
          if (d !== 0) for (let i = 0; i < nIn; i++) g1[base + i] += d * x[i];
        }
      }
      step++;
      const b1 = 1 - 0.9 ** step, b2 = 1 - 0.999 ** step;
      const update = (w, g, m, v, mask) => {
        for (let i = 0; i < w.length; i++) {
          if (!mask[i]) { w[i] = 0; continue; }
          const grad = g[i] / batch;
          m[i] = 0.9 * m[i] + 0.1 * grad;
          v[i] = 0.999 * v[i] + 0.001 * grad * grad;
          w[i] = Math.max(-4, Math.min(4, w[i] - (lr * (m[i] / b1)) / (Math.sqrt(v[i] / b2) + 1e-8)));
        }
      };
      update(w1, g1, m1, v1, masks.m1);
      update(w2, g2, m2, v2, masks.m2);
    }
    loss /= Math.floor(n / batch) * batch;
  }
  return loss;
}

function fitReport(model, data) {
  const { w1, w2 } = model;
  const h = new Float32Array(hidden);
  const mean = new Float64Array(nOut), sse = new Float64Array(nOut), sst = new Float64Array(nOut);
  for (const t of data.targets) for (let k = 0; k < nOut; k++) mean[k] += t[k] / data.targets.length;
  data.inputs.forEach((x, n) => {
    for (let j = 0; j < hidden; j++) { let z = 0; for (let i = 0; i < nIn; i++) z += w1[j * nIn + i] * x[i]; h[j] = softsign(gain * z); }
    for (let k = 0; k < nOut; k++) {
      let z = 0;
      for (let j = 0; j < hidden; j++) z += w2[k * hidden + j] * h[j];
      const err = softsign(gain * z) - data.targets[n][k];
      sse[k] += err * err;
      sst[k] += (data.targets[n][k] - mean[k]) ** 2;
    }
  });
  return Array.from(sse, (v, k) => game.actionNames[k] + ' r2=' + (1 - v / Math.max(sst[k], 1e-9)).toFixed(2) + ' (mean ' + mean[k].toFixed(3) + ')').join(' | ');
}

function pruneTo(model, masks, keep) {
  const entries = [];
  for (let i = 0; i < model.w1.length; i++) if (masks.m1[i]) entries.push([Math.abs(model.w1[i]), 1, i]);
  for (let i = 0; i < model.w2.length; i++) if (masks.m2[i]) entries.push([Math.abs(model.w2[i]), 2, i]);
  entries.sort((a, b) => b[0] - a[0]);
  masks.m1.fill(0);
  masks.m2.fill(0);
  for (let e = 0; e < Math.min(keep, entries.length); e++) (entries[e][1] === 1 ? masks.m1 : masks.m2)[entries[e][2]] = 1;
}

function toGenome(model, masks) {
  const genome = makeGenome(maxEdges);
  PARAM_DEFAULT.forEach((v, i) => { genome.p[i] = v; });
  Object.assign(genome.p, { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 4, 7: 0, 8: 0.05, 9: 0, 10: 0, 11: 0, 12: gain, 13: 0.02, 14: 0.2, 15: 0, 16: 1, 19: 1 });
  let n = 0;
  for (let j = 0; j < hidden; j++) for (let i = 0; i < nIn; i++) if (masks.m1[j * nIn + i] && n < maxEdges) { genome.pk[n] = (i | ((nIn + nOut + j) << 8)) >>> 0; genome.w[n++] = model.w1[j * nIn + i]; }
  for (let k = 0; k < nOut; k++) for (let j = 0; j < hidden; j++) if (masks.m2[k * hidden + j] && n < maxEdges) { genome.pk[n] = ((nIn + nOut + j) | ((nIn + k) << 8)) >>> 0; genome.w[n++] = model.w2[k * hidden + j]; }
  genome.n = n;
  genome.fit = 0;
  return genome;
}

function evaluate(driver) {
  let learnerReward = 0, learnerTicks = 0, baseReward = 0, baseTicks = 0, deaths = 0;
  for (let s = 0; s < 3; s++) {
    const seed = mix(4242, s, 0, 9);
    const stats = new Int32Array(NSTATS);
    const env = game.createEnv(seed, game.defaultCfg(), false, stats);
    const obs = new Float32Array(nIn);
    const target = new Float32Array(nOut);
    const actions = new Int32Array(game.learners);
    const outputs = new Float32Array(game.learners * nOut);
    const brains = driver.brain ? Array.from({ length: game.learners }, () => new Brain(driver.brain, maxEdges, game.dims)) : null;
    for (let tick = 0; tick < 4000; tick++) {
      for (let a = 0; a < game.learners; a++) {
        env.observe(a, obs, tick);
        if (brains) {
          actions[a] = brains[a].step(obs, new Rng(mix(seed, tick, a, 1)));
          for (let k = 0; k < nOut; k++) outputs[a * nOut + k] = brains[a].act[nIn + k];
        } else {
          game.teacher(env, a, tick, target);
          for (let k = 0; k < nOut; k++) outputs[a * nOut + k] = target[k];
        }
      }
      env.step(actions, outputs, tick);
      for (let a = 0; a < game.learners; a++) {
        learnerReward += env.reward[a];
        learnerTicks++;
        if (env.dead[a]) { deaths++; env.respawn(a, tick); if (brains) brains[a].load(driver.brain); }
      }
    }
    baseReward += stats[STAT.REW_BASE];
    baseTicks += stats[STAT.TICKS_BASE];
  }
  return { learnerRate: learnerReward / REWARD_SCALE / learnerTicks, baseRate: baseReward / REWARD_SCALE / Math.max(1, baseTicks), deaths };
}

const started = Date.now();
const data = collect();
console.log('samples', data.inputs.length, 'collect s', ((Date.now() - started) / 1000).toFixed(1));
const teacherScore = evaluate({});
console.log('teacher-driven learners', JSON.stringify(teacherScore));
const model = initWeights(new Rng(5));
const masks = { m1: new Uint8Array(model.w1.length).fill(1), m2: new Uint8Array(model.w2.length).fill(1) };
let loss = trainEpochs(model, masks, data, epochs, 3e-3);
console.log('dense loss', loss.toFixed(4));
console.log(fitReport(model, data));
const dense = model.w1.length + model.w2.length;
const rounds = 5;
for (let r = 1; r <= rounds; r++) {
  const keep = Math.round(dense * (maxEdges / dense) ** (r / rounds));
  pruneTo(model, masks, keep);
  loss = trainEpochs(model, masks, data, Math.max(2, epochs >> 1), 1.5e-3);
  console.log('pruned to', keep, 'loss', loss.toFixed(4));
}
console.log('pruned fit', fitReport(model, data));
const genome = toGenome(model, masks);
const cloned = evaluate({ brain: genome });
console.log('cloned brain', genome.n, 'edges', JSON.stringify(cloned), 'total s', ((Date.now() - started) / 1000).toFixed(1));
await mkdir(dirname(outPath), { recursive: true });
await writeFile(outPath, JSON.stringify([genomeToJSON(genome, game, { source: 'clone', teacher: teacherScore, cloned })]));
console.log('wrote', outPath);
