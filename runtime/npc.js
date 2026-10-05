const NPARAMS = 20;
const MIN_EDGES = 30;
const STRUCT_PERIOD = 32;
const NODE_BITS = 8;
const NODE_MASK = 255;
const DEFAULT_MAX_EDGES = 256;
const LEGACY_DIMS = { nIn: 12, nOut: 8, nNodes: 64 };
const PARAM_DEFAULT = [0.05, 0.3, 0, 0, 0, 0.0005, 2, 0.02, 0.3, 0.1, 4, 0.03, 1.5, 0.02, 0.2, 0.3, 1, 0, 0, 1];
const REWARD_SCALE = 1024;
const AC_DEFAULTS = { lr: 1, kappa: 2, gamma: 0.997, lamda: 0.97, beta2: 0.999, eps: 1e-8, sigma: 0.05, valueCoef: 0.5, rewardScale: 1, normDelta: 1, adaptive: 1 };
const TWO_PI = 6.283185307179586;

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
}

function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

export class RealtimeNpc {
  constructor(genomeJson, options) {
    const opts = options || {};
    if (!genomeJson || (genomeJson.format !== 'npc-brain/3' && genomeJson.format !== 'npc-brain/2' && genomeJson.format !== 'npc-brain/1')) throw new Error('unsupported genome format');
    this.dims = genomeJson.dims || LEGACY_DIMS;
    this.source = genomeJson;
    this.seed = (opts.seed === undefined ? 1 : opts.seed) >>> 0;
    this.agent = opts.agent || 0;
    this.learning = opts.learning !== false;
    const declaredCap = genomeJson.meta && genomeJson.meta.maxEdges ? genomeJson.meta.maxEdges : 0;
    this.maxEdges = Math.max(opts.maxEdges || DEFAULT_MAX_EDGES, genomeJson.edges.length, declaredCap);
    const { nIn, nNodes } = this.dims;
    this.pk = new Uint32Array(this.maxEdges);
    this.w = new Float32Array(this.maxEdges);
    this.p = new Float32Array(NPARAMS);
    this.act = new Float32Array(nNodes);
    this.acc = new Float32Array(nNodes);
    this.bias = new Float32Array(nNodes);
    this.hasBias = false;
    this.leakVec = new Float32Array(nNodes);
    this.hasLeakVec = false;
    this.lastIn = new Float32Array(nIn);
    this.outputs = new Float32Array(this.dims.nOut);
    this.phi = null;
    this.actPrev = null;
    this.learnMode = opts.learnMode === 'ac' && opts.learning !== false ? 'ac' : 'hebbian';
    this.rebirth();
    this.ac = null;
    if (this.learnMode === 'ac') {
      this.phi = new Float32Array(this.dims.nNodes);
      this.actPrev = new Float32Array(this.dims.nNodes);
      this.ac = new StreamingAc(this, opts.ac);
      if (!(opts.ac && opts.ac.fromMeta === 0)) this.initFromMeta(genomeJson.meta, opts.ac);
    }
  }

  initFromMeta(meta, acOpts) {
    if (!meta || acOpts && acOpts.fromMeta === 0) return;
    const { nIn, nOut, nNodes } = this.dims;
    if (acOpts && acOpts.sigmaFromMeta && meta.logStd && meta.logStd.length === nOut) this.ac.setSigma(meta.logStd.map((x) => Math.exp(x)));
    if (acOpts && acOpts.acCritic === 0) return;
    const prior = this.source.acValue;
    if (prior && prior.length === nNodes + 1) {
      this.ac.wv.set(prior);
      return;
    }
    const vw = meta.valueWeights;
    if (!vw || meta.valueBias === undefined) return;
    const off = vw.length === nNodes ? 0 : vw.length === nNodes - nIn ? nIn : vw.length === nNodes - nIn - nOut ? nIn : -1;
    if (off < 0 || off + vw.length > nNodes) return;
    for (let j = 0; j < vw.length; j++) this.ac.wv[off + j] = vw[j] / REWARD_SCALE;
    this.ac.wv[nNodes] = meta.valueBias / REWARD_SCALE;
  }

  rebirth() {
    const json = this.source;
    this.n = json.edges.length;
    for (let i = 0; i < NPARAMS; i++) this.p[i] = json.params[i] === undefined ? PARAM_DEFAULT[i] : json.params[i];
    for (let e = 0; e < this.n; e++) {
      const edge = json.edges[e];
      this.pk[e] = ((edge[0] & NODE_MASK) | ((edge[1] & NODE_MASK) << NODE_BITS)) >>> 0;
      this.w[e] = edge[2];
    }
    const srcBias = json.format === 'npc-brain/3' ? json.bias : null;
    if (srcBias && srcBias.length !== this.dims.nNodes) throw new Error('bias length must equal dims.nNodes');
    this.bias.fill(0);
    this.hasBias = false;
    if (srcBias) for (let j = 0; j < srcBias.length; j++) {
      this.bias[j] = srcBias[j];
      if (this.bias[j] !== 0) this.hasBias = true;
    }
    const srcLeak = json.format === 'npc-brain/3' ? json.leak : null;
    if (srcLeak && srcLeak.length !== this.dims.nNodes) throw new Error('leak length must equal dims.nNodes');
    this.hasLeakVec = Boolean(srcLeak);
    if (srcLeak) this.leakVec.set(srcLeak);
    this.act.fill(0);
    this.lastIn.fill(0);
    this.skip = 1000;
    this.rewAvg = 0;
    this.ops = 0;
    this.lastAct = 0;
    this.score = 0;
    this.lifeTicks = 0;
    if (this.ac) this.ac.resetTraces();
  }

  decide(inputs, tick) {
    const { nIn, nOut, nNodes } = this.dims;
    const p = this.p;
    const act = this.act;
    let diff = 0;
    for (let i = 0; i < nIn; i++) {
      act[i] = inputs[i] * p[16];
      diff += Math.abs(inputs[i] - this.lastIn[i]);
    }
    if (this.skip < Math.floor(p[10]) && diff < p[9]) {
      this.skip++;
      if (!this.ac) this.publishOutputs();
      return this.lastAct;
    }
    this.skip = 0;
    this.lastIn.set(inputs);
    if (this.ac) this.actPrev.set(act);
    const acc = this.acc;
    acc.fill(0);
    for (let e = 0; e < this.n; e++) {
      const packed = this.pk[e];
      acc[(packed >>> NODE_BITS) & NODE_MASK] += this.w[e] * act[packed & NODE_MASK];
    }
    const gain = p[12];
    const leak = p[15];
    const leakVec = this.hasLeakVec ? this.leakVec : null;
    const phi = this.phi;
    for (let j = nIn; j < nNodes; j++) {
      const x = this.hasBias ? gain * (acc[j] + this.bias[j]) : gain * acc[j];
      const lk = leakVec ? leakVec[j] : leak;
      const ax = 1 + (x < 0 ? -x : x);
      act[j] = lk * act[j] + (1 - lk) * (x / ax);
      if (phi) phi[j] = ((1 - lk) * gain) / (ax * ax);
    }
    this.ops += this.n;
    if (this.ac) {
      this.lastAct = this.ac.tick(tick);
      return this.lastAct;
    }
    let best = nIn;
    for (let j = nIn + 1; j < nIn + nOut; j++) if (act[j] > act[best]) best = j;
    let action = best - nIn;
    const rng = new Rng(mix(this.seed, tick, this.agent, 1));
    const exploreRoll = rng.next();
    const exploreAction = Math.floor(rng.next() * nOut);
    if (exploreRoll < p[11]) action = exploreAction;
    this.lastAct = action;
    this.publishOutputs();
    return action;
  }

  publishOutputs() {
    for (let k = 0; k < this.dims.nOut; k++) this.outputs[k] = this.act[this.dims.nIn + k];
  }

  reward(r, tick, died) {
    this.lifeTicks++;
    if (this.ac) {
      this.ac.observe(r, died);
      return;
    }
    if (this.learning) this.learn(r);
    if (this.learning && !died && this.lifeTicks % STRUCT_PERIOD === 0) this.structural(new Rng(mix(this.seed, tick, this.agent, 3)));
  }

  learn(reward) {
    const p = this.p;
    const advantage = reward - this.rewAvg;
    this.rewAvg += p[13] * (reward - this.rewAvg);
    this.score += reward;
    if (reward === 0) return;
    const eta = p[0], A = p[1], B = p[2], C = p[3], D = p[4], decay = p[5], wmax = p[6];
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

  toJSON(game) {
    const edges = [];
    for (let e = 0; e < this.n; e++) edges.push([this.pk[e] & NODE_MASK, (this.pk[e] >>> NODE_BITS) & NODE_MASK, this.w[e]]);
    const out = { format: 'npc-brain/2', game: game || this.source.game || null, dims: this.dims, params: Array.from(this.p), edges, meta: Object.assign({}, this.source.meta, { fitness: this.score }) };
    if (this.hasBias || this.hasLeakVec) out.format = 'npc-brain/3';
    if (this.hasBias) out.bias = Array.from(this.bias);
    if (this.hasLeakVec) out.leak = Array.from(this.leakVec);
    if (out.format === 'npc-brain/3' && this.source.inputNorm) out.inputNorm = this.source.inputNorm;
    if (this.ac) out.acValue = Array.from(this.ac.wv);
    return out;
  }
}

export const NpcBrain = RealtimeNpc;

export class StreamingAc {
  constructor(npc, options) {
    const o = Object.assign({}, AC_DEFAULTS, options || {});
    this.npc = npc;
    this.lr = o.lr;
    this.kappa = o.kappa;
    this.gamma = o.gamma;
    this.beta = o.gamma * o.lamda;
    this.beta2 = o.beta2;
    this.eps = o.eps;
    this.sigma = o.sigma;
    this.valueCoef = o.valueCoef;
    this.rewardScale = o.rewardScale;
    this.normDelta = o.normDelta !== 0;
    this.adaptive = o.adaptive !== 0;
    const m = npc.dims.nNodes;
    this.d = new Float32Array(npc.maxEdges);
    this.e = new Float32Array(npc.maxEdges);
    this.v = new Float32Array(npc.maxEdges);
    this.wv = new Float32Array(m + 1);
    this.ev = new Float32Array(m + 1);
    this.vv = new Float32Array(m + 1);
    this.z = new Float32Array(npc.dims.nOut);
    this.sig = new Float32Array(npc.dims.nOut);
    this.invSig = new Float32Array(npc.dims.nOut);
    this.g = new Float32Array(m);
    this.dstArr = new Int32Array(npc.maxEdges);
    this.dstStart = new Int32Array(m + 1);
    this.dstCursor = new Int32Array(m + 1);
    this.dstIdx = new Int32Array(npc.maxEdges);
    this.csrDirty = true;
    this.setSigma(o.sigma);
    this.counter = 1;
    this.seen = 0;
    this.drms = 0;
    this.drmsN = 0;
    this.debias = 1;
    this.resetTraces();
  }

  setSigma(sigma) {
    const k = this.npc.dims.nOut;
    const arr = Array.isArray(sigma) ? sigma : null;
    for (let i = 0; i < k; i++) {
      const s = arr ? sigma[i] : sigma;
      const v = s > 1e-4 ? s : 1e-4;
      this.sig[i] = v;
      this.invSig[i] = 1 / v;
    }
  }

  buildCsr() {
    const npc = this.npc;
    const m = npc.dims.nNodes;
    const start = this.dstStart;
    start.fill(0);
    for (let i = 0; i < npc.n; i++) {
      const dst = (npc.pk[i] >>> NODE_BITS) & NODE_MASK;
      this.dstArr[i] = dst;
      start[dst + 1]++;
    }
    for (let j = 0; j < m; j++) start[j + 1] += start[j];
    this.dstCursor.set(start.subarray(0, m));
    for (let i = 0; i < npc.n; i++) this.dstIdx[this.dstCursor[this.dstArr[i]]++] = i;
    this.csrDirty = false;
  }

  resetTraces() {
    this.d.fill(0);
    this.e.fill(0);
    this.ev.fill(0);
    this.g.fill(0);
    this.csrDirty = true;
    this.pending = false;
    this.pendingR = 0;
    this.pendingDied = 0;
    this.prevValue = 0;
  }

  observe(reward, died) {
    if (!this.pending) return;
    this.pendingR += reward;
    if (died) this.pendingDied = 1;
  }

  value() {
    const act = this.npc.act;
    const m = this.npc.dims.nNodes;
    const wv = this.wv;
    let v = wv[m];
    for (let j = 0; j < m; j++) v += wv[j] * act[j];
    return v;
  }

  tick(tick) {
    const npc = this.npc;
    const { nIn, nOut, nNodes } = npc.dims;
    const act = npc.act;
    const prev = npc.actPrev;
    const phi = npc.phi;
    const value = this.value();
    if (this.pending) this.update(this.pendingR * this.rewardScale, this.pendingDied, value);
    const rng = new Rng(mix(npc.seed, tick, npc.agent, 1));
    const out = npc.outputs;
    const z = this.z;
    for (let k = 0; k < nOut; k += 2) {
      const u1 = 1 - rng.next();
      const ang = TWO_PI * rng.next();
      const mag = Math.sqrt(-2 * Math.log(u1));
      z[k] = mag * Math.cos(ang);
      if (k + 1 < nOut) z[k + 1] = mag * Math.sin(ang);
    }
    let best = 0;
    for (let k = 0; k < nOut; k++) {
      const a = act[nIn + k] + this.sig[k] * z[k];
      out[k] = a;
      if (a > out[best]) best = k;
    }
    const leak = npc.p[15];
    const leakVec = npc.hasLeakVec ? npc.leakVec : null;
    const beta = this.beta;
    const cv = this.valueCoef;
    const inv = this.invSig;
    const pk = npc.pk;
    const d = this.d;
    const e = this.e;
    const g = this.g;
    g.fill(0);
    for (let k = 0; k < nOut; k++) g[nIn + k] = z[k] * inv[k];
    if (this.csrDirty) this.buildCsr();
    const dstStart = this.dstStart;
    const dstIdx = this.dstIdx;
    const dstArr = this.dstArr;
    const w = npc.w;
    for (let j = nNodes - 1; j >= nIn; j--) {
      const gj = g[j];
      if (gj === 0) continue;
      for (let c = dstStart[j]; c < dstStart[j + 1]; c++) {
        const i = dstIdx[c];
        g[pk[i] & NODE_MASK] += w[i] * phi[dstArr[i]] * gj;
      }
    }
    for (let i = 0; i < npc.n; i++) {
      const packed = pk[i];
      const src = packed & NODE_MASK;
      const dst = dstArr[i];
      const lk = leakVec ? leakVec[dst] : leak;
      const pre = src < nIn ? act[src] : prev[src];
      const di = lk * d[i] + phi[dst] * pre;
      d[i] = di;
      e[i] = beta * e[i] + di * g[dst];
    }
    const ev = this.ev;
    for (let j = 0; j < nNodes; j++) ev[j] = beta * ev[j] + cv * act[j];
    ev[nNodes] = beta * ev[nNodes] + cv;
    this.prevValue = value;
    this.pending = true;
    this.pendingR = 0;
    this.pendingDied = 0;
    npc.ops += npc.n * 5 + nNodes;
    return best;
  }

  update(reward, died, nextValue) {
    let delta = reward + (died ? 0 : this.gamma * nextValue) - this.prevValue;
    if (this.normDelta) {
      this.drmsN++;
      const rate = this.drmsN < 1000 ? 1 / this.drmsN : 0.001;
      const rms = Math.sqrt(this.drms) + 1e-12;
      this.drms += (delta * delta - this.drms) * rate;
      delta = delta / rms;
      if (delta > 10) delta = 10;
      else if (delta < -10) delta = -10;
    }
    const abs = delta < 0 ? -delta : delta;
    const dbar = abs < 1 ? 1 : abs;
    this.lastDelta = delta;
    const npc = this.npc;
    const m = this.wv.length;
    this.counter++;
    this.debias = 1 / (1 - Math.pow(this.beta2, this.counter));
    let zsum = this.traceSum(this.e, this.v, npc.n, delta);
    zsum += this.traceSum(this.ev, this.vv, m, delta);
    this.seen++;
    if (this.seen === 1) {
      if (died) this.resetTraces();
      return;
    }
    const dot = dbar * zsum * this.lr * this.kappa;
    const stepSize = dot > 1 ? this.lr / dot : this.lr;
    const upd = stepSize * delta;
    this.applyGroup(npc.w, this.e, this.v, npc.n, upd, npc.p[6]);
    this.applyGroup(this.wv, this.ev, this.vv, m, upd, 0);
    if (died) this.resetTraces();
  }

  traceSum(e, vArr, len, delta) {
    const eps = this.eps;
    let z = 0;
    if (!this.adaptive) {
      for (let i = 0; i < len; i++) z += e[i] < 0 ? -e[i] : e[i];
      return z;
    }
    const b2 = this.beta2;
    const om = 1 - b2;
    const dd = delta * delta;
    const db = this.debias;
    for (let i = 0; i < len; i++) {
      const ei = e[i];
      const nv = b2 * vArr[i] + om * dd * ei * ei;
      vArr[i] = nv;
      z += (ei < 0 ? -ei : ei) / (Math.sqrt(nv * db) + eps);
    }
    return z;
  }

  applyGroup(theta, e, vArr, len, upd, wmax) {
    const eps = this.eps;
    const db = this.debias;
    for (let i = 0; i < len; i++) {
      const ei = e[i];
      if (ei === 0) continue;
      const denom = this.adaptive ? Math.sqrt(vArr[i] * db) + eps : 1;
      let w = theta[i] + (upd * ei) / denom;
      if (wmax > 0) w = w < -wmax ? -wmax : w > wmax ? wmax : w;
      theta[i] = w;
    }
  }
}

export class FixedStepLoop {
  constructor(options) {
    this.hz = options.hz;
    this.step = options.step;
    this.render = options.render || null;
    this.now = options.now || (() => performance.now());
    this.maxCatchUp = options.maxCatchUp || 5;
    this.speed = options.speed || 1;
    this.tick = options.startTick || 0;
    this.accumulatorMs = 0;
    this.lastMs = null;
    this.handle = null;
    this.running = false;
  }

  get periodMs() {
    return 1000 / this.hz;
  }

  advance(nowMs) {
    if (this.lastMs === null) this.lastMs = nowMs;
    this.accumulatorMs += (nowMs - this.lastMs) * this.speed;
    this.lastMs = nowMs;
    let stepped = 0;
    while (this.accumulatorMs >= this.periodMs && stepped < this.maxCatchUp) {
      this.step(this.tick++);
      this.accumulatorMs -= this.periodMs;
      stepped++;
    }
    if (stepped === this.maxCatchUp) this.accumulatorMs = 0;
    if (this.render && stepped > 0) this.render(this.tick, this.accumulatorMs / this.periodMs);
    return stepped;
  }

  start() {
    if (this.running) return;
    this.running = true;
    this.lastMs = null;
    const pump = () => {
      if (!this.running) return;
      this.advance(this.now());
      this.handle = typeof requestAnimationFrame === 'function' ? requestAnimationFrame(pump) : setTimeout(pump, Math.max(1, Math.floor(this.periodMs / 4)));
    };
    pump();
  }

  stop() {
    this.running = false;
    if (this.handle === null) return;
    if (typeof cancelAnimationFrame === 'function') cancelAnimationFrame(this.handle);
    else clearTimeout(this.handle);
    this.handle = null;
  }
}

export const SENSOR_LAYOUT = ['bias=1', 'energy/200', 'hp/100', 'min(gold,5)/5', 'food gradient dx', 'food gradient dy', 'ore gradient dx', 'ore gradient dy', 'agent in facing tile', 'standing on ready food', 'standing on ready ore', 'damaged last tick'];

export function sensorsFromWorld(state) {
  const inputs = new Float32Array(12);
  inputs[0] = 1;
  inputs[1] = state.energy / 200;
  inputs[2] = state.hp / 100;
  inputs[3] = Math.min(state.gold, 5) / 5;
  inputs[4] = state.foodDir ? state.foodDir[0] : 0;
  inputs[5] = state.foodDir ? state.foodDir[1] : 0;
  inputs[6] = state.oreDir ? state.oreDir[0] : 0;
  inputs[7] = state.oreDir ? state.oreDir[1] : 0;
  inputs[8] = state.agentAhead ? 1 : 0;
  inputs[9] = state.onReadyFood ? 1 : 0;
  inputs[10] = state.onReadyOre ? 1 : 0;
  inputs[11] = state.damagedLastTick ? 1 : 0;
  return inputs;
}

export { REWARD_SCALE };

if (typeof window !== 'undefined') window.Npc = { RealtimeNpc, NpcBrain, StreamingAc, FixedStepLoop, Rng, mix, pcg, sensorsFromWorld, SENSOR_LAYOUT, REWARD_SCALE, AC_DEFAULTS };
