export const NPARAMS = 20;
export const MIN_EDGES = 30;
export const GRAVE_SLOTS = 4;
export const ARCHIVE_SIZE = 64;
export const STRUCT_PERIOD = 32;
export const REWARD_SCALE = 1024;
export const NSTATS = 16;
export const NODE_BITS = 8;
export const NODE_MASK = 255;
export const STAT = { REW_EVO: 0, REW_BASE: 1, TICKS_EVO: 2, TICKS_BASE: 3, OPS_EVO: 4, BIRTHS: 5, DEATHS: 6, LIFE_FIT: 7, EDGES: 8, GAME0: 9 };
export const GAME_STAT_SLOTS = 7;
export const PARAM_NAMES = ['eta', 'A', 'B', 'C', 'D', 'decay', 'wmax', 'pruneT', 'growP', 'gateT', 'gateMax', 'eps', 'gain', 'baseRate', 'growW', 'leak', 'inScale', 'reserved17', 'reserved18', 'mutMult'];
export const PARAM_LO = [0, -1, -1, -1, -1, 0, 0.5, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0.25, 0, 0, 0.1];
export const PARAM_HI = [0.5, 1, 1, 1, 1, 0.01, 4, 0.2, 1, 2, 16, 0.2, 4, 0.2, 1, 0.9, 4, 1, 1, 4];
export const PARAM_DEFAULT = [0.05, 0.3, 0, 0, 0, 0.0005, 2, 0.02, 0.3, 0.1, 4, 0.03, 1.5, 0.02, 0.2, 0.3, 1, 0, 0, 1];
export const NEG_INF_FIT = -1e30;
export const VALID_FIT = -1e29;
export const ARCHIVE_DECAY_PER_TICK = 0.0005;
export const RATE_EMA_ALPHA = 0.002;
export const LIFE_WEIGHT = 0.35;
export const EVO_DEFAULTS = { lifeWeight: 0.6, bias: 0, plast: 1, freshFrac: 0.05, eliteFrac: 0.25, eliteCount: 8, tournament: 3, wStep: 0.05, perturbP: 0.1, crossP: 0, cloneFrac: 0.15, nicheCap: 8, nicheFrac: 0.03, archiveDecay: ARCHIVE_DECAY_PER_TICK };

export function evoConfig(opts) {
  const out = {};
  for (const key in EVO_DEFAULTS) out[key] = opts && opts[key] !== undefined ? Number(opts[key]) : EVO_DEFAULTS[key];
  return out;
}

export function pcg(v) {
  v = v >>> 0;
  const s = (Math.imul(v, 747796405) + 2891336453) >>> 0;
  const w = Math.imul(((s >>> ((s >>> 28) + 4)) ^ s) >>> 0, 277803737) >>> 0;
  return ((w >>> 22) ^ w) >>> 0;
}

export function mix(a, b, c, d) {
  let h = pcg(a >>> 0);
  h = pcg((h + (b >>> 0)) >>> 0);
  h = pcg((h + (c >>> 0)) >>> 0);
  h = pcg((h + (d >>> 0)) >>> 0);
  return h;
}

export class Rng {
  constructor(state) { this.s = state >>> 0; }
  next() { this.s = pcg(this.s); return (this.s >>> 8) / 16777216; }
  normal() { const a = this.next(); const b = this.next(); const c = this.next(); return (a + b + c - 1.5) * 2; }
}

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

export function hiddenStart(dims) { return dims.nIn + dims.nOut; }

export function validateGame(game) {
  const d = game.dims;
  if (d.nNodes > 256) throw new Error('nNodes must be <= 256');
  if (d.nNodes <= d.nIn + d.nOut) throw new Error('nNodes must exceed nIn+nOut to leave hidden units');
  if (game.learners > game.agents) throw new Error('learners must be <= agents');
  if (game.statNames.length > GAME_STAT_SLOTS) throw new Error('at most ' + GAME_STAT_SLOTS + ' game stats');
}

export function makeGenome(maxEdges) {
  return { fit: NEG_INF_FIT, niche: 0, n: 0, p: new Float32Array(NPARAMS), pk: new Uint32Array(maxEdges), w: new Float32Array(maxEdges) };
}

export function copyGenome(src, maxEdges) {
  const g = makeGenome(maxEdges);
  g.fit = src.fit;
  g.niche = src.niche || 0;
  g.n = src.n;
  g.p.set(src.p);
  g.pk.set(src.pk.subarray(0, src.n));
  g.w.set(src.w.subarray(0, src.n));
  if (src.bias) g.bias = Float32Array.from(src.bias);
  if (src.leak) g.leak = Float32Array.from(src.leak);
  if (src.inputNorm) g.inputNorm = src.inputNorm;
  return g;
}

function hasBias(bias) {
  if (!bias) return false;
  for (let i = 0; i < bias.length; i++) if (bias[i] !== 0) return true;
  return false;
}

export function foldInputNorm(W1, b1, mean, std) {
  const nHidden = W1.length;
  const nIn = mean.length;
  const outW = [];
  const outB = new Array(nHidden);
  for (let j = 0; j < nHidden; j++) {
    const row = new Array(nIn);
    let shift = b1 ? b1[j] : 0;
    for (let i = 0; i < nIn; i++) {
      row[i] = W1[j][i] / std[i];
      shift -= (W1[j][i] * mean[i]) / std[i];
    }
    outW.push(row);
    outB[j] = shift;
  }
  return { W1: outW, b1: outB };
}

function randomEdge(rng, dims) {
  const srcInput = rng.next() < 0.6;
  const sr = rng.next();
  const dstOutput = rng.next() < 0.5;
  const dr = rng.next();
  const wr = rng.normal();
  const hid0 = hiddenStart(dims);
  const src = srcInput ? Math.floor(sr * dims.nIn) : hid0 + Math.floor(sr * (dims.nNodes - hid0));
  const dst = dstOutput ? dims.nIn + Math.floor(dr * dims.nOut) : hid0 + Math.floor(dr * (dims.nNodes - hid0));
  return { pk: (src | (dst << NODE_BITS)) >>> 0, w: wr * 0.5 };
}

export function randomGenome(rng, maxEdges, dims) {
  const g = makeGenome(maxEdges);
  for (let i = 0; i < NPARAMS; i++) {
    const noise = rng.normal();
    g.p[i] = clamp(PARAM_DEFAULT[i] + noise * 0.25 * (PARAM_HI[i] - PARAM_LO[i]), PARAM_LO[i], PARAM_HI[i]);
  }
  g.n = 60;
  for (let e = 0; e < 60; e++) {
    const edge = randomEdge(rng, dims);
    g.pk[e] = edge.pk;
    g.w[e] = clamp(edge.w, -g.p[6], g.p[6]);
  }
  return g;
}

export function mutateGenome(g, rng, mutation, maxEdges, dims, evo) {
  const wStep = evo ? evo.wStep : EVO_DEFAULTS.wStep;
  const perturbP = evo ? evo.perturbP : EVO_DEFAULTS.perturbP;
  const rateNoise = rng.normal();
  g.p[19] = clamp(g.p[19] * Math.exp(0.2 * rateNoise), 0.1, 4);
  const m = mutation * g.p[19];
  for (let i = 0; i < NPARAMS - 1; i++) {
    const noise = rng.normal();
    const applies = rng.next() < 0.5;
    if (applies) g.p[i] = clamp(g.p[i] + noise * m * 0.05 * (PARAM_HI[i] - PARAM_LO[i]), PARAM_LO[i], PARAM_HI[i]);
  }
  const wmax = g.p[6];
  let e = 0;
  while (e < g.n) {
    const removeRoll = rng.next();
    const noise = rng.normal();
    const perturbs = rng.next() < perturbP;
    if (removeRoll < m * 0.02 && g.n > MIN_EDGES) {
      g.n--;
      g.pk[e] = g.pk[g.n];
      g.w[e] = g.w[g.n];
      continue;
    }
    g.w[e] = clamp(g.w[e] + (perturbs ? noise * m * wStep : 0), -wmax, wmax);
    e++;
  }
  const addRolls = [rng.next() < m * 0.3, rng.next() < m * 0.1];
  for (const adds of addRolls) {
    const edge = randomEdge(rng, dims);
    if (adds && g.n < maxEdges) {
      g.pk[g.n] = edge.pk;
      g.w[g.n] = clamp(edge.w, -wmax, wmax);
      g.n++;
    }
  }
  return g;
}

export function genomeToJSON(g, game, meta) {
  const edges = [];
  for (let e = 0; e < g.n; e++) edges.push([g.pk[e] & NODE_MASK, (g.pk[e] >>> NODE_BITS) & NODE_MASK, g.w[e]]);
  const out = { format: 'npc-brain/2', game: game ? game.id : null, dims: game ? game.dims : null, params: Array.from(g.p), edges, meta: Object.assign({ fitness: g.fit }, meta || {}) };
  if (hasBias(g.bias) || g.leak) out.format = 'npc-brain/3';
  if (hasBias(g.bias)) out.bias = Array.from(g.bias);
  if (g.leak) out.leak = Array.from(g.leak);
  if (out.format === 'npc-brain/3' && g.inputNorm) out.inputNorm = g.inputNorm;
  return out;
}

export function genomeFromJSON(json, maxEdges) {
  if (!json || (json.format !== 'npc-brain/3' && json.format !== 'npc-brain/2' && json.format !== 'npc-brain/1')) throw new Error('unsupported genome format');
  const legacy = json.format === 'npc-brain/1';
  const g = makeGenome(maxEdges);
  for (let i = 0; i < NPARAMS; i++) g.p[i] = json.params[i] === undefined ? PARAM_DEFAULT[i] : json.params[i];
  const count = Math.min(json.edges.length, maxEdges);
  for (let e = 0; e < count; e++) {
    const [src, dst, weight] = json.edges[e];
    g.pk[e] = ((src & NODE_MASK) | ((dst & NODE_MASK) << NODE_BITS)) >>> 0;
    g.w[e] = weight;
  }
  g.n = count;
  g.fit = json.meta && typeof json.meta.fitness === 'number' ? json.meta.fitness : 0;
  g.legacy = legacy;
  if (json.format === 'npc-brain/3') {
    if (Array.isArray(json.bias)) {
      if (json.dims && json.bias.length !== json.dims.nNodes) throw new Error('bias length must equal dims.nNodes');
      g.bias = Float32Array.from(json.bias);
    }
    if (Array.isArray(json.leak)) {
      if (json.dims && json.leak.length !== json.dims.nNodes) throw new Error('leak length must equal dims.nNodes');
      g.leak = Float32Array.from(json.leak);
    }
    if (json.inputNorm) g.inputNorm = json.inputNorm;
  }
  return g;
}

export class Brain {
  constructor(genome, maxEdges, dims, evo) {
    this.maxEdges = maxEdges;
    this.dims = dims;
    this.evo = evo || evoConfig(null);
    this.biasNode = this.evo.bias ? dims.nNodes - 1 : -1;
    this.hid0 = hiddenStart(dims);
    this.pk = new Uint32Array(maxEdges);
    this.w = new Float32Array(maxEdges);
    this.p = new Float32Array(NPARAMS);
    this.act = new Float32Array(dims.nNodes);
    this.acc = new Float32Array(dims.nNodes);
    this.lastIn = new Float32Array(dims.nIn);
    this.hist = new Float32Array(dims.nOut);
    this.bias = new Float32Array(dims.nNodes);
    this.hasBias = false;
    this.leakVec = new Float32Array(dims.nNodes);
    this.hasLeakVec = false;
    this.load(genome);
  }

  load(genome) {
    this.n = genome.n;
    this.p.set(genome.p);
    this.pk.set(genome.pk.subarray(0, genome.n));
    this.w.set(genome.w.subarray(0, genome.n));
    this.hasBias = hasBias(genome.bias);
    if (this.hasBias) this.bias.set(genome.bias);
    else this.bias.fill(0);
    this.hasLeakVec = Boolean(genome.leak);
    if (this.hasLeakVec) this.leakVec.set(genome.leak);
    this.inputNorm = genome.inputNorm || null;
    this.act.fill(0);
    this.lastIn.fill(0);
    this.hist.fill(0);
    this.skip = 1000;
    this.rewAvg = 0;
    this.ops = 0;
    this.lastAct = 0;
    this.score = 0;
    this.parentFit = genome.fit > VALID_FIT ? genome.fit : 0;
  }

  step(inputs, rng) {
    const { nIn, nOut, nNodes } = this.dims;
    const p = this.p;
    const act = this.act;
    let diff = 0;
    for (let i = 0; i < nIn; i++) {
      act[i] = inputs[i] * p[16];
      diff += Math.abs(inputs[i] - this.lastIn[i]);
    }
    if (this.biasNode >= 0) act[this.biasNode] = 1;
    if (this.skip < Math.floor(p[10]) && diff < p[9]) {
      this.skip++;
      return this.lastAct;
    }
    this.skip = 0;
    this.lastIn.set(inputs);
    const acc = this.acc;
    acc.fill(0);
    for (let e = 0; e < this.n; e++) {
      const packed = this.pk[e];
      acc[(packed >>> NODE_BITS) & NODE_MASK] += this.w[e] * act[packed & NODE_MASK];
    }
    const gain = p[12];
    const leak = p[15];
    const lastNode = this.biasNode >= 0 ? nNodes - 1 : nNodes;
    const bias = this.bias;
    const leakVec = this.hasLeakVec ? this.leakVec : null;
    for (let j = nIn; j < lastNode; j++) {
      const x = this.hasBias ? gain * (acc[j] + bias[j]) : gain * acc[j];
      const lk = leakVec ? leakVec[j] : leak;
      act[j] = lk * act[j] + (1 - lk) * (x / (1 + Math.abs(x)));
    }
    this.ops += this.n;
    let best = nIn;
    for (let j = nIn + 1; j < nIn + nOut; j++) if (act[j] > act[best]) best = j;
    let action = best - nIn;
    const exploreRoll = rng.next();
    const exploreAction = Math.floor(rng.next() * nOut);
    if (exploreRoll < p[11]) action = exploreAction;
    this.lastAct = action;
    return action;
  }

  learn(reward) {
    const p = this.p;
    const advantage = reward - this.rewAvg;
    this.rewAvg += p[13] * (reward - this.rewAvg);
    this.score += reward;
    if (reward === 0) return;
    if (this.evo.plast === 0) return;
    const eta = p[0] * this.evo.plast, A = p[1], B = p[2], C = p[3], D = p[4], decay = p[5], wmax = p[6];
    for (let e = 0; e < this.n; e++) {
      const packed = this.pk[e];
      const pre = this.act[packed & NODE_MASK];
      const post = this.act[(packed >>> NODE_BITS) & NODE_MASK];
      const dw = eta * advantage * (A * pre * post + B * pre + C * post + D);
      this.w[e] = clamp(this.w[e] * (1 - decay) + dw, -wmax, wmax);
    }
    this.ops += this.n;
  }

  structural(rng) {
    const { nIn, nNodes } = this.dims;
    const p = this.p;
    const pruneT = p[7];
    let e = 0;
    while (e < this.n) {
      if (Math.abs(this.w[e]) < pruneT && this.n > MIN_EDGES) {
        this.n--;
        this.pk[e] = this.pk[this.n];
        this.w[e] = this.w[this.n];
      } else e++;
    }
    const growRoll = rng.next();
    let found = false;
    let src = 0, dst = 0;
    for (let attempt = 0; attempt < 4; attempt++) {
      const rs = rng.next();
      const rd = rng.next();
      const s = Math.floor(rs * nNodes);
      const d = nIn + Math.floor(rd * (nNodes - nIn));
      if (!found && Math.abs(this.act[s]) > 0.25 && Math.abs(this.act[d]) > 0.25) {
        found = true;
        src = s;
        dst = d;
      }
    }
    const weightRoll = rng.next();
    if (growRoll < p[8] && found && this.n < this.maxEdges) {
      this.pk[this.n] = (src | (dst << NODE_BITS)) >>> 0;
      this.w[this.n] = clamp((weightRoll * 2 - 1) * p[14], -p[6], p[6]);
      this.n++;
    }
    this.ops += this.n;
  }

  toGenome(fitness) {
    const g = makeGenome(this.maxEdges);
    g.fit = fitness;
    g.n = this.n;
    g.p.set(this.p);
    g.pk.set(this.pk.subarray(0, this.n));
    g.w.set(this.w.subarray(0, this.n));
    if (this.hasBias) g.bias = Float32Array.from(this.bias);
    if (this.hasLeakVec) g.leak = Float32Array.from(this.leakVec);
    if (this.inputNorm) g.inputNorm = this.inputNorm;
    return g;
  }
}

export class WorldRunner {
  constructor(game, index, seed, isEval, opts) {
    this.game = game;
    this.index = index;
    this.seed = seed >>> 0;
    this.isEval = isEval;
    this.maxEdges = opts.maxEdges;
    this.evo = evoConfig(opts);
    this.dims = game.dims;
    this.stats = new Int32Array(NSTATS);
    const cfg = opts.randomize && !isEval ? game.randomCfg(this.seed) : game.defaultCfg();
    this.env = game.createEnv(this.seed, cfg, isEval, this.stats);
    this.brains = [];
    for (let a = 0; a < game.learners; a++) this.brains.push(new Brain(randomGenome(new Rng(mix(this.seed, 0, a, 6)), opts.maxEdges, this.dims), opts.maxEdges, this.dims, this.evo));
    this.lifeTicks = new Int32Array(game.learners);
    this.rateEma = 0;
    this.prevOps = new Float64Array(game.learners);
    this.obs = new Float32Array(this.dims.nIn);
    this.actions = new Int32Array(game.learners);
    this.outputs = new Float32Array(game.learners * this.dims.nOut);
    this.graveyard = [];
    for (let s = 0; s < GRAVE_SLOTS; s++) this.graveyard.push({ fit: NEG_INF_FIT, genome: makeGenome(opts.maxEdges) });
  }

  parentGenome(archive, a, rng) {
    if (this.isEval) {
      if (archive.count === 0) return randomGenome(rng, this.maxEdges, this.dims);
      return copyGenome(archive.genomes[Math.min(a % GRAVE_SLOTS, archive.count - 1)], this.maxEdges);
    }
    const evo = this.evo;
    const roll = rng.next();
    if (archive.count === 0 || roll < evo.freshFrac) return randomGenome(rng, this.maxEdges, this.dims);
    const pick = () => {
      const eliteRoll = rng.next();
      if (eliteRoll < evo.eliteFrac) return Math.floor(rng.next() * Math.min(evo.eliteCount, archive.count));
      let best = archive.count;
      for (let k = 0; k < evo.tournament; k++) best = Math.min(best, Math.floor(rng.next() * archive.count));
      return best;
    };
    const child = copyGenome(archive.genomes[pick()], this.maxEdges);
    const crossRoll = rng.next();
    if (evo.crossP > 0 && crossRoll < evo.crossP) {
      const mate = archive.genomes[pick()];
      for (let e = 0; e < mate.n && child.n < this.maxEdges; e++) {
        if (rng.next() < 0.5) {
          child.pk[child.n] = mate.pk[e];
          child.w[child.n] = clamp(mate.w[e], -child.p[6], child.p[6]);
          child.n++;
        }
      }
    }
    return child;
  }

  rebirth(a, tick, ctx) {
    const rng = new Rng(mix(this.seed, tick, a, 2));
    const genome = this.parentGenome(ctx.archive, a, rng);
    const cloneRoll = this.evo.cloneFrac > 0 ? rng.next() : 1;
    if (!this.isEval && cloneRoll >= this.evo.cloneFrac) mutateGenome(genome, rng, ctx.mutation, this.maxEdges, this.dims, this.evo);
    this.brains[a].load(genome);
    this.prevOps[a] = 0;
    this.lifeTicks[a] = 0;
    this.stats[STAT.BIRTHS]++;
  }

  loadFromArchive(a, tick, archive) {
    const rng = new Rng(mix(this.seed, tick, a, 2));
    this.brains[a].load(this.parentGenome(archive, a, rng));
    this.prevOps[a] = 0;
    this.lifeTicks[a] = 0;
  }

  step(tick, ctx) {
    const game = this.game;
    const nOut = this.dims.nOut;
    const stats = this.stats;
    this.env.difficulty = this.isEval ? 100 : ctx.difficulty;
    for (let a = 0; a < game.learners; a++) {
      this.env.observe(a, this.obs, tick);
      const brain = this.brains[a];
      this.actions[a] = brain.step(this.obs, new Rng(mix(this.seed, tick, a, 1)));
      brain.hist[this.actions[a]]++;
      for (let k = 0; k < nOut; k++) this.outputs[a * nOut + k] = brain.act[this.dims.nIn + k];
    }
    this.env.step(this.actions, this.outputs, tick);
    const dead = [];
    let tickReward = 0;
    for (let a = 0; a < game.learners; a++) tickReward += this.env.reward[a];
    this.rateEma += RATE_EMA_ALPHA * (tickReward / REWARD_SCALE / game.learners - this.rateEma);
    for (let a = 0; a < game.learners; a++) {
      const brain = this.brains[a];
      const rewardFx = this.env.reward[a];
      brain.learn(rewardFx / REWARD_SCALE);
      stats[STAT.REW_EVO] += rewardFx;
      stats[STAT.TICKS_EVO]++;
      this.lifeTicks[a]++;
      const died = this.env.dead[a] !== 0 || this.lifeTicks[a] >= game.maxAge;
      if (!died && this.lifeTicks[a] % STRUCT_PERIOD === 0) brain.structural(new Rng(mix(this.seed, tick, a, 3)));
      stats[STAT.OPS_EVO] += brain.ops - this.prevOps[a];
      this.prevOps[a] = brain.ops;
      if (died) dead.push(a);
    }
    if (dead.length > 0) this.handleDeaths(dead, tick, ctx);
  }

  handleDeaths(dead, tick, ctx) {
    const stats = this.stats;
    const fitness = new Map();
    for (const a of dead) {
      const brain = this.brains[a];
      const raw = brain.score - this.rateEma * this.lifeTicks[a] - (ctx.penalty * brain.ops) / 1000;
      fitness.set(a, this.evo.lifeWeight * raw + (1 - this.evo.lifeWeight) * brain.parentFit);
      stats[STAT.DEATHS]++;
      stats[STAT.LIFE_FIT] += Math.floor(raw * REWARD_SCALE);
    }
    if (!this.isEval) {
      for (const [a, fit] of fitness) {
        let minSlot = 0;
        for (let s = 1; s < GRAVE_SLOTS; s++) if (this.graveyard[s].fit < this.graveyard[minSlot].fit) minSlot = s;
        if (fit > this.graveyard[minSlot].fit) {
          this.graveyard[minSlot].fit = fit;
          this.graveyard[minSlot].genome = this.brains[a].toGenome(fit);
          this.graveyard[minSlot].genome.niche = nicheMask(this.brains[a].hist, this.lifeTicks[a], this.evo.nicheFrac);
        }
      }
    }
    for (const a of dead) {
      this.env.respawn(a, tick);
      this.rebirth(a, tick, ctx);
    }
  }

  snapshot() {
    return this.env.snapshot();
  }
}

export class StatsAccumulator {
  constructor() {
    this.total = { train: new Float64Array(NSTATS), eval: new Float64Array(NSTATS) };
    this.window = { train: new Float64Array(NSTATS), eval: new Float64Array(NSTATS) };
    this.edgeSum = { train: 0, eval: 0 };
    this.evoCount = { train: 1, eval: 1 };
  }

  add(group, delta, learnerAgents) {
    for (let i = 0; i < NSTATS; i++) {
      if (i === STAT.EDGES) continue;
      this.total[group][i] += delta[i];
      this.window[group][i] += delta[i];
    }
    this.edgeSum[group] = delta[STAT.EDGES];
    this.evoCount[group] = Math.max(1, learnerAgents);
  }

  summary(group, game) {
    const w = this.window[group];
    const t = this.total[group];
    const evoTicks = Math.max(1, w[STAT.TICKS_EVO]);
    const baseTicks = Math.max(1, w[STAT.TICKS_BASE]);
    const out = {
      rewRate: w[STAT.REW_EVO] / REWARD_SCALE / evoTicks,
      baseRate: w[STAT.REW_BASE] / REWARD_SCALE / baseTicks,
      opsPerTick: w[STAT.OPS_EVO] / evoTicks,
      meanLife: w[STAT.TICKS_EVO] / Math.max(1, w[STAT.DEATHS]),
      edgesMean: this.edgeSum[group] / this.evoCount[group],
      births: t[STAT.BIRTHS],
      deaths: t[STAT.DEATHS],
      lifeFit: w[STAT.DEATHS] > 0 ? w[STAT.LIFE_FIT] / REWARD_SCALE / w[STAT.DEATHS] : 0,
      game: {}
    };
    for (let i = 0; i < game.statNames.length; i++) out.game[game.statNames[i]] = t[STAT.GAME0 + i];
    return out;
  }

  clearWindow() {
    this.window.train.fill(0);
    this.window.eval.fill(0);
  }
}

export const DEFAULT_OPTS = { ...EVO_DEFAULTS, difficultyAdaptive: 0, adaptLife: 2000, adaptStep: 1, worlds: 16, islands: 2, maxEdges: 256, ticksPerDispatch: 0, mutation: 1, penaltyStart: 0, penaltyEnd: 0, penaltyRampTicks: 200000, randomize: true, evalFraction: 0.125, seed: 12345, migrateEvery: 8, migrateCount: 4, selectEvery: 1, evalEvery: 1, difficultyStart: 100, difficultyEnd: 100, difficultyRampTicks: 300000 };

export function penaltyAt(opts, tick) {
  return opts.penaltyStart + (opts.penaltyEnd - opts.penaltyStart) * Math.min(1, tick / Math.max(1, opts.penaltyRampTicks));
}

export function difficultyAt(opts, tick) {
  return Math.round(opts.difficultyStart + (opts.difficultyEnd - opts.difficultyStart) * Math.min(1, tick / Math.max(1, opts.difficultyRampTicks)));
}

export function evalWorldCount(opts) {
  return Math.min(opts.worlds - opts.islands, Math.max(opts.islands, Math.round(opts.worlds * opts.evalFraction)));
}

export function nicheMask(hist, lifeTicks, frac) {
  let mask = 0;
  const life = Math.max(lifeTicks, 1);
  for (let k = 0; k < hist.length && k < 32; k++) if (hist[k] >= frac * life) mask |= 1 << k;
  return mask >>> 0;
}

export function mergeIntoArchive(archive, candidates, migrants, decay, nicheCap) {
  const pool = [];
  for (let i = 0; i < archive.count; i++) pool.push({ fit: archive.genomes[i].fit - (decay || 0), genome: archive.genomes[i] });
  for (const c of candidates) if (c.fit > VALID_FIT) pool.push(c);
  for (const m of migrants) pool.push({ fit: m.fit, genome: m });
  pool.sort((x, y) => y.fit - x.fit);
  let kept = pool;
  if (nicheCap > 0) {
    const cellCounts = new Map();
    kept = pool.filter((c) => {
      const cell = c.genome.niche || 0;
      const used = cellCounts.get(cell) || 0;
      if (used >= nicheCap) return false;
      cellCounts.set(cell, used + 1);
      return true;
    });
  }
  kept = kept.slice(0, ARCHIVE_SIZE);
  archive.genomes = kept.map((c) => {
    c.genome.fit = c.fit;
    return c.genome;
  });
  archive.count = kept.length;
}

export class CpuBackend {
  static async create(opts) {
    return new CpuBackend(Object.assign({}, DEFAULT_OPTS, opts));
  }

  constructor(opts) {
    validateGame(opts.game);
    this.opts = opts;
    this.game = opts.game;
    this.kind = 'cpu';
    this.stats = new StatsAccumulator();
    this.tick = 0;
    this.passes = 0;
    this.buildWorlds();
    this.info = { kind: 'cpu', adapter: 'JavaScript', limits: {}, features: [], worlds: opts.worlds, islands: opts.islands, maxEdges: opts.maxEdges, game: this.game.id };
  }

  buildWorlds() {
    const o = this.opts;
    this.islandSize = Math.floor(o.worlds / o.islands);
    this.evalStart = o.worlds - evalWorldCount(o);
    this.archives = [];
    for (let i = 0; i < o.islands; i++) this.archives.push({ count: 0, genomes: [] });
    this.worlds = [];
    this.worldScores = new Float32Array(o.worlds);
    this.worldRew = new Float64Array(o.worlds);
    this.worldTicks = new Float64Array(o.worlds);
    for (let w = 0; w < o.worlds; w++) this.worlds.push(new WorldRunner(this.game, w, mix(o.seed, w, 0, 9), w >= this.evalStart, o));
  }

  islandOf(w) {
    return Math.min(this.opts.islands - 1, Math.floor(w / this.islandSize));
  }

  setParams(partial) {
    Object.assign(this.opts, partial);
  }

  async step() {
    const start = performance.now();
    const o = this.opts;
    const ticks = o.ticksPerDispatch > 0 ? o.ticksPerDispatch : 4;
    const evalPeriod = this.game.maxAge * o.evalEvery;
    if (o.evalEvery > 0 && Math.floor((this.tick + ticks) / evalPeriod) > Math.floor(this.tick / evalPeriod)) this.refreshEvalWorlds();
    for (let k = 0; k < ticks; k++) {
      const penalty = penaltyAt(o, this.tick);
      for (let w = 0; w < o.worlds; w++) this.worlds[w].step(this.tick, { penalty, mutation: o.mutation, difficulty: this.currentDifficulty(), archive: this.archives[this.islandOf(w)] });
      this.tick++;
    }
    this.lastPassTicks = ticks;
    this.collectStats();
    this.passes++;
    if (this.passes % o.selectEvery === 0) this.select();
    return { ticks, ms: performance.now() - start };
  }

  currentDifficulty() {
    return this.opts.difficultyAdaptive && this.adaptiveLevel !== undefined ? Math.round(this.adaptiveLevel) : difficultyAt(this.opts, this.tick);
  }

  refreshEvalWorlds() {
    for (let w = this.evalStart; w < this.opts.worlds; w++) {
      const world = this.worlds[w];
      const archive = this.archives[this.islandOf(w)];
      for (let a = 0; a < this.game.learners; a++) {
        world.loadFromArchive(a, this.tick, archive);
        world.env.respawn(a, this.tick);
      }
      for (let a = this.game.learners; a < this.game.agents; a++) world.env.respawn(a, this.tick);
    }
  }

  collectStats() {
    const delta = { train: new Float64Array(NSTATS), eval: new Float64Array(NSTATS) };
    let trainWorlds = 0;
    let evalWorlds = 0;
    for (let w = 0; w < this.opts.worlds; w++) {
      const world = this.worlds[w];
      const group = world.isEval ? 'eval' : 'train';
      for (let i = 0; i < NSTATS; i++) if (i !== STAT.EDGES) delta[group][i] += world.stats[i];
      let edges = 0;
      for (let a = 0; a < this.game.learners; a++) edges += world.brains[a].n;
      delta[group][STAT.EDGES] += edges;
      this.worldRew[w] += world.stats[STAT.REW_EVO];
      this.worldTicks[w] += world.stats[STAT.TICKS_EVO];
      world.stats.fill(0);
      if (world.isEval) evalWorlds++;
      else trainWorlds++;
    }
    if (this.opts.difficultyAdaptive) {
      if (this.adaptiveLevel === undefined) this.adaptiveLevel = this.opts.difficultyStart;
      const deaths = delta.train[STAT.DEATHS];
      if (deaths > 0 && delta.train[STAT.TICKS_EVO] / deaths > this.opts.adaptLife) this.adaptiveLevel = Math.min(this.opts.difficultyEnd, this.adaptiveLevel + this.opts.adaptStep);
    }
    this.stats.add('train', delta.train, trainWorlds * this.game.learners);
    this.stats.add('eval', delta.eval, evalWorlds * this.game.learners);
  }

  select() {
    const o = this.opts;
    const migrating = o.migrateEvery > 0 && this.passes % o.migrateEvery === 0;
    const previousTops = this.archives.map((a) => a.genomes.slice(0, o.migrateCount).map((g) => copyGenome(g, o.maxEdges)));
    for (let island = 0; island < o.islands; island++) {
      const candidates = [];
      for (let w = island * this.islandSize; w < (island + 1) * this.islandSize && w < this.evalStart; w++) {
        for (const slot of this.worlds[w].graveyard) {
          if (slot.fit > VALID_FIT) candidates.push({ fit: slot.fit, genome: slot.genome });
          slot.fit = NEG_INF_FIT;
        }
      }
      const migrants = migrating ? previousTops[(island + o.islands - 1) % o.islands] : [];
      mergeIntoArchive(this.archives[island], candidates, migrants, o.archiveDecay * this.lastPassTicks, o.nicheCap);
    }
  }

  async readStats() {
    const train = this.stats.summary('train', this.game);
    const evalSummary = this.stats.summary('eval', this.game);
    this.stats.clearWindow();
    for (let w = 0; w < this.opts.worlds; w++) this.worldScores[w] = this.worldRew[w] / REWARD_SCALE / Math.max(1, this.worldTicks[w]);
    this.worldRew.fill(0);
    this.worldTicks.fill(0);
    let best = -Infinity;
    let sum = 0;
    let count = 0;
    for (const archive of this.archives) {
      if (archive.count > 0) best = Math.max(best, archive.genomes[0].fit);
      for (const g of archive.genomes) {
        sum += g.fit;
        count++;
      }
    }
    return { tick: this.tick, passes: this.passes, train, eval: evalSummary, archive: { best: count ? best : 0, mean: count ? sum / count : 0, count }, worldScores: this.worldScores };
  }

  async snapshot(worldIndex) {
    return this.worlds[worldIndex].snapshot();
  }

  async exportChampions(n) {
    const all = [];
    for (const archive of this.archives) for (const g of archive.genomes) all.push(g);
    all.sort((a, b) => b.fit - a.fit);
    return all.slice(0, n).map((g) => genomeToJSON(g, this.game));
  }

  async importGenomes(jsonArray) {
    const genomes = jsonArray.map((j) => genomeFromJSON(j, this.opts.maxEdges));
    for (const archive of this.archives) {
      const copies = genomes.map((g) => copyGenome(g, this.opts.maxEdges));
      mergeIntoArchive(archive, copies.map((g) => ({ fit: g.fit, genome: g })), []);
    }
  }

  async checkpoint() {
    const o = Object.assign({}, this.opts);
    delete o.game;
    return { kind: 'cpu', game: this.game.id, tick: this.tick, passes: this.passes, opts: o, archives: this.archives.map((a) => a.genomes.map((g) => genomeToJSON(g, this.game))) };
  }

  async restore(obj) {
    this.tick = obj.tick;
    this.passes = obj.passes;
    this.archives = obj.archives.map((list) => {
      const genomes = list.map((j) => genomeFromJSON(j, this.opts.maxEdges));
      return { count: genomes.length, genomes };
    });
  }

  async dispose() {}
}
