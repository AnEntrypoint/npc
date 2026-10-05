import { NPARAMS, Rng, mix, foldInputNorm } from '../../src/core.js';

export const RL_GAIN = 2;
export const RL_HALF_LOG_2PI = 0.9189385332046727;
export const RL_ADAM = { beta1: 0.9, beta2: 0.999, epsilon: 1e-8 };
export const RL_STAT_SLOTS = 8;
export const RL_DEFAULTS = {
  hidden: 96,
  rolloutTicks: 31,
  gamma: 0.997,
  lambda: 0.97,
  clip: 0.2,
  lr: 1e-3,
  lrEnd: 1e-3,
  lrDecayIterations: 2000,
  entropy: 0.001,
  valueCoef: 0.5,
  epochs: 1,
  minibatches: 4,
  maxGrad: 1,
  sigmaInit: 0.5,
  sigmaMin: 0.05,
  sigmaMax: 1,
  weightMax: 4,
  rewardScale: 4,
  adaptScale: 1,
  lifeCapScale: 1,
  gradGroups: 256,
  tileEntries: 16,
  league: 1,
  leagueFraction: 0.5,
  leagueSlotFraction: 0.5,
  poolSize: 8,
  snapshotEvery: 40,
  leaguePeriod: 8,
  latestBias: 0.4,
  leagueEvalFraction: 0.03,
  referenceAge: 120,
  bias: 0,
  recurrent: 0,
  obsNorm: 0,
  recurrentInit: 0.4,
  leakInit: 0.3,
  leakTauMax: 16,
  learnLeak: 1,
  rewardClip: 1e9,
  valueClip: 1e9,
  blockGrad: 1,
  obsNormFloor: 0.1,
  obsNormCap: 8000000
};

export function rlLayout(nIn, nOut, hidden, features) {
  const f = features === true ? { bias: true } : features || {};
  const bias = !!f.bias;
  const recurrent = !!f.recurrent;
  const obsNorm = !!f.obsNorm;
  const w1 = 0;
  const w2 = w1 + nIn * hidden;
  const logStd = w2 + nOut * hidden;
  const valueW = logStd + nOut;
  const valueB = valueW + hidden;
  let next = valueB + 1;
  const b1 = bias ? next : -1;
  if (bias) next += hidden;
  const b2 = bias ? next : -1;
  if (bias) next += nOut;
  const wr = recurrent ? next : -1;
  if (recurrent) next += hidden * hidden;
  const leak = recurrent ? next : -1;
  if (recurrent) next += hidden;
  const learn = next;
  const normMean = obsNorm ? next : -1;
  if (obsNorm) next += nIn;
  const normIstd = obsNorm ? next : -1;
  if (obsNorm) next += nIn;
  return { nIn, nOut, hidden, w1, w2, logStd, valueW, valueB, b1, b2, wr, leak, normMean, normIstd, bias, recurrent, obsNorm, learn, count: next, partialStride: next + RL_STAT_SLOTS, recStride: nOut + 8, entries: 0 };
}

export const RL_LEAK_MAX = 0.98;
export const RL_OBS_CLIP = 10;

export function rlRecordLayout(nOut) {
  return { action: 0, logProb: nOut, value: nOut + 1, reward: nOut + 2, flag: nOut + 3, truncValue: nOut + 4, prevValid: nOut + 5, advantage: nOut + 6, returns: nOut + 7, stride: nOut + 8 };
}

export function rlSoftsign(x) {
  return x / (1 + Math.abs(x));
}

export function rlSlope(h) {
  const s = 1 - Math.abs(h);
  return RL_GAIN * s * s;
}

export function rlHiddenActivations(theta, L, x, out) {
  const { nIn, hidden } = L;
  for (let i = 0; i < hidden; i++) {
    let acc = 0;
    for (let j = 0; j < nIn; j++) acc += theta[L.w1 + j * hidden + i] * x[j];
    if (L.bias) acc += theta[L.b1 + i];
    out[i] = rlSoftsign(RL_GAIN * acc);
  }
  return out;
}

export function rlPolicyStep(theta, L, x, hPrev, hOut) {
  const { nIn, hidden } = L;
  for (let i = 0; i < hidden; i++) {
    let acc = 0;
    for (let j = 0; j < nIn; j++) acc += theta[L.w1 + j * hidden + i] * x[j];
    if (L.bias) acc += theta[L.b1 + i];
    if (L.recurrent) for (let m = 0; m < hidden; m++) acc += theta[L.wr + m * hidden + i] * hPrev[m];
    const s = rlSoftsign(RL_GAIN * acc);
    if (L.recurrent) {
      const lk = theta[L.leak + i];
      hOut[i] = lk * hPrev[i] + (1 - lk) * s;
    } else hOut[i] = s;
  }
  return hOut;
}

export function rlPolicyMean(theta, L, h, out) {
  const { nOut, hidden } = L;
  for (let k = 0; k < nOut; k++) {
    let acc = 0;
    for (let i = 0; i < hidden; i++) acc += theta[L.w2 + k * hidden + i] * h[i];
    if (L.bias) acc += theta[L.b2 + k];
    out[k] = rlSoftsign(RL_GAIN * acc);
  }
  return out;
}

export function rlValueOf(theta, L, h) {
  let acc = theta[L.valueB];
  for (let i = 0; i < L.hidden; i++) acc += theta[L.valueW + i] * h[i];
  return acc;
}

export function rlLogProb(theta, L, action, mu) {
  let total = 0;
  for (let k = 0; k < L.nOut; k++) {
    const ls = theta[L.logStd + k];
    const eps = (action[k] - mu[k]) / Math.exp(ls);
    total += -0.5 * eps * eps - ls - RL_HALF_LOG_2PI;
  }
  return total;
}

export function rlLeakInit(L, opts) {
  const o = Object.assign({}, RL_DEFAULTS, opts);
  const out = new Float64Array(L.hidden);
  for (let i = 0; i < L.hidden; i++) {
    if (o.leakTauMax > 1) {
      const frac = L.hidden > 1 ? i / (L.hidden - 1) : 0;
      out[i] = 1 - 1 / Math.pow(o.leakTauMax, frac);
    } else out[i] = o.leakInit;
  }
  return out;
}

export function rlInitTheta(L, seed, opts) {
  const o = Object.assign({}, RL_DEFAULTS, opts);
  const rng = new Rng(seed >>> 0);
  const theta = new Float32Array(L.count);
  const w1Bound = Math.sqrt(3 / L.nIn);
  for (let p = L.w1; p < L.w2; p++) theta[p] = (rng.next() * 2 - 1) * w1Bound;
  const w2Bound = 0.1 * Math.sqrt(3 / L.hidden);
  for (let p = L.w2; p < L.logStd; p++) theta[p] = (rng.next() * 2 - 1) * w2Bound;
  for (let k = 0; k < L.nOut; k++) theta[L.logStd + k] = Math.log(o.sigmaInit);
  if (L.recurrent) {
    const wrBound = o.recurrentInit * Math.sqrt(3 / L.hidden);
    for (let p = L.wr; p < L.leak; p++) theta[p] = (rng.next() * 2 - 1) * wrBound;
    theta.set(rlLeakInit(L, o), L.leak);
  }
  if (L.obsNorm) {
    for (let j = 0; j < L.nIn; j++) {
      theta[L.normMean + j] = 0;
      theta[L.normIstd + j] = 1;
    }
  }
  return theta;
}

export function rlGae(trajectory, hp) {
  const { steps, reward, flag, value, truncValue, boot } = trajectory;
  const clipR = hp.rewardClip === undefined ? Infinity : hp.rewardClip;
  const clipV = hp.valueClip === undefined ? Infinity : hp.valueClip;
  const advantage = new Float64Array(steps);
  const returns = new Float64Array(steps);
  let carry = 0;
  for (let t = steps - 1; t >= 0; t--) {
    const r = Math.max(-clipR, Math.min(clipR, reward[t] * hp.rewardScale));
    let next = t === steps - 1 ? boot : value[t + 1];
    let ends = false;
    if (flag[t] === 1) {
      next = 0;
      ends = true;
    } else if (flag[t] === 2) {
      next = truncValue[t];
      ends = true;
    }
    const delta = r + hp.gamma * next - value[t];
    carry = ends ? delta : delta + hp.gamma * hp.lambda * carry;
    advantage[t] = carry;
    returns[t] = Math.max(-clipV, Math.min(clipV, carry + value[t]));
  }
  return { advantage, returns };
}

export function rlMinibatchOf(learner, salt, minibatches) {
  return mix(salt >>> 0, learner >>> 0, 7, 9) % minibatches;
}

export function rlLossAndGrad(theta, L, data, hp, wantGrad) {
  const { nIn, nOut, hidden } = L;
  const H = hidden;
  const R = rlRecordLayout(nOut);
  const { steps, learners, obs, rec, hInit, normMean, normInvStd, selected, divisor } = data;
  const rc = L.recurrent;
  const grad = wantGrad ? new Float64Array(L.count) : null;
  const stats = { policy: 0, value: 0, entropy: 0, clipFrac: 0, kl: 0, samples: 0 };
  const zeros = (n) => new Float64Array(n);
  const hs = [];
  const ss = [];
  const ef = [];
  const dhr = [];
  for (let j = 0; j <= steps; j++) {
    hs.push(zeros(H));
    ss.push(zeros(H));
    ef.push(zeros(H));
    dhr.push(zeros(H));
  }
  const dpres = [];
  for (let j = 0; j < steps; j++) dpres.push(zeros(nOut));
  const dvs = zeros(steps);
  const x = zeros(nIn);
  const mu = zeros(nOut);
  const valid = zeros(steps + 1);
  for (let b = 0; b < learners; b++) {
    if (selected && !selected(b)) continue;
    const xrow = (j) => {
      for (let c = 0; c < nIn; c++) x[c] = obs[(b * (steps + 1) + j) * nIn + c];
      return x;
    };
    for (let j = 0; j < steps; j++) valid[j] = rec[(b * steps + j) * R.stride + R.prevValid] > 0.5 ? 1 : 0;
    valid[steps] = 1;
    for (let j = 0; j <= steps; j++) dhr[j].fill(0);
    const eff = (j) => {
      const e = ef[j];
      for (let i = 0; i < H; i++) e[i] = valid[j] ? hs[j][i] : 0;
      return e;
    };
    for (let j = 0; j <= steps; j++) {
      if (rc && j === 0) {
        for (let i = 0; i < H; i++) {
          hs[0][i] = hInit ? hInit[b * H + i] : 0;
          ss[0][i] = 0;
        }
        continue;
      }
      xrow(j);
      const prev = rc ? eff(j - 1) : null;
      for (let i = 0; i < H; i++) {
        let acc = 0;
        for (let c = 0; c < nIn; c++) acc += theta[L.w1 + c * H + i] * x[c];
        if (L.bias) acc += theta[L.b1 + i];
        if (rc) for (let m = 0; m < H; m++) acc += theta[L.wr + m * H + i] * prev[m];
        const s = rlSoftsign(RL_GAIN * acc);
        ss[j][i] = s;
        hs[j][i] = rc ? theta[L.leak + i] * prev[i] + (1 - theta[L.leak + i]) * s : s;
      }
    }
    for (let j = 0; j < steps; j++) {
      const base = (b * steps + j) * R.stride;
      const hj = eff(j);
      rlPolicyMean(theta, L, hj, mu);
      const action = rec.subarray(base + R.action, base + R.action + nOut);
      const logProbNew = rlLogProb(theta, L, action, mu);
      const logRatio = Math.max(-20, Math.min(20, logProbNew - rec[base + R.logProb]));
      const ratio = Math.exp(logRatio);
      const A = (rec[base + R.advantage] - normMean) * normInvStd;
      const unclipped = ratio * A;
      const clipped = Math.max(1 - hp.clip, Math.min(1 + hp.clip, ratio)) * A;
      const active = unclipped <= clipped;
      const dLogProb = active ? -A * ratio : 0;
      stats.policy += -Math.min(unclipped, clipped);
      stats.clipFrac += active ? 0 : 1;
      stats.kl += ratio - 1 - logRatio;
      stats.samples += 1;
      for (let k = 0; k < nOut; k++) {
        const ls = theta[L.logStd + k];
        const sigma = Math.exp(ls);
        const eps = (action[k] - mu[k]) / sigma;
        stats.entropy += ls + RL_HALF_LOG_2PI + 0.5;
        if (grad) {
          grad[L.logStd + k] += dLogProb * (eps * eps - 1) - hp.entropy;
          dpres[j][k] = dLogProb * (eps / sigma) * rlSlope(mu[k]);
          if (L.bias) grad[L.b2 + k] += dpres[j][k];
          for (let i = 0; i < H; i++) grad[L.w2 + k * H + i] += dpres[j][k] * hj[i];
        }
      }
      const v = rlValueOf(theta, L, hs[j + 1]);
      const err = v - rec[base + R.returns];
      stats.value += 0.5 * err * err;
      dvs[j] = hp.valueCoef * err;
      if (grad) {
        grad[L.valueB] += dvs[j];
        for (let i = 0; i < H; i++) grad[L.valueW + i] += dvs[j] * hs[j + 1][i];
      }
    }
    if (!grad) continue;
    const dzNext = zeros(H);
    const lowest = rc ? 1 : 0;
    for (let j = steps; j >= lowest; j--) {
      const dz = zeros(H);
      xrow(j);
      for (let i = 0; i < H; i++) {
        let d = 0;
        if (j >= 1) d += dvs[j - 1] * theta[L.valueW + i];
        if (j < steps && valid[j]) {
          let e = 0;
          for (let k = 0; k < nOut; k++) e += dpres[j][k] * theta[L.w2 + k * H + i];
          if (rc) {
            e += theta[L.leak + i] * dhr[j + 1][i];
            for (let m = 0; m < H; m++) e += theta[L.wr + i * H + m] * dzNext[m];
          }
          d += e;
        }
        dhr[j][i] = d;
        dz[i] = d * (rc ? 1 - theta[L.leak + i] : 1) * rlSlope(ss[j][i]);
      }
      const prev = rc ? eff(j - 1) : null;
      for (let i = 0; i < H; i++) {
        if (L.bias) grad[L.b1 + i] += dz[i];
        for (let c = 0; c < nIn; c++) grad[L.w1 + c * H + i] += dz[i] * x[c];
        if (rc) {
          for (let m = 0; m < H; m++) grad[L.wr + m * H + i] += prev[m] * dz[i];
          grad[L.leak + i] += dhr[j][i] * (prev[i] - ss[j][i]);
        }
      }
      dzNext.set(dz);
    }
  }
  const total = stats.policy + hp.valueCoef * stats.value - hp.entropy * stats.entropy;
  if (grad) for (let p = 0; p < L.count; p++) grad[p] /= divisor;
  return { grad, loss: total / divisor, stats };
}

export function rlTrained(L, p, learnLeak) {
  if (p >= L.learn) return false;
  if (L.recurrent && !learnLeak && p >= L.leak) return false;
  return true;
}

export function rlAdamStep(theta, grad, m, v, step, hp, L) {
  let normSq = 0;
  for (let p = 0; p < L.learn; p++) if (rlTrained(L, p, hp.learnLeak)) normSq += grad[p] * grad[p];
  const norm = Math.sqrt(normSq);
  const scale = Math.min(1, hp.maxGrad / (norm + 1e-6));
  const c1 = 1 - Math.pow(RL_ADAM.beta1, step);
  const c2 = 1 - Math.pow(RL_ADAM.beta2, step);
  const lsLo = Math.log(hp.sigmaMin);
  const lsHi = Math.log(hp.sigmaMax);
  for (let p = 0; p < L.learn; p++) {
    if (!rlTrained(L, p, hp.learnLeak)) continue;
    const g = grad[p] * scale;
    m[p] = RL_ADAM.beta1 * m[p] + (1 - RL_ADAM.beta1) * g;
    v[p] = RL_ADAM.beta2 * v[p] + (1 - RL_ADAM.beta2) * g * g;
    let value = theta[p] - (hp.lr * (m[p] / c1)) / (Math.sqrt(v[p] / c2) + RL_ADAM.epsilon);
    if (p >= L.logStd && p < L.valueW) value = Math.max(lsLo, Math.min(lsHi, value));
    else if (L.recurrent && p >= L.leak) value = Math.max(0, Math.min(RL_LEAK_MAX, value));
    else if (p < L.logStd || p > L.valueB) value = Math.max(-hp.weightMax, Math.min(hp.weightMax, value));
    theta[p] = value;
  }
  return norm;
}

export function rlGenomeParams(hp, wmax) {
  const params = new Array(NPARAMS).fill(0);
  params[6] = wmax === undefined ? hp.weightMax : wmax;
  params[12] = RL_GAIN;
  params[13] = 0.02;
  params[16] = 1;
  params[19] = 1;
  return params;
}

export function rlNormOf(theta, L) {
  const mean = new Array(L.nIn);
  const std = new Array(L.nIn);
  for (let j = 0; j < L.nIn; j++) {
    mean[j] = L.obsNorm ? theta[L.normMean + j] : 0;
    std[j] = L.obsNorm ? 1 / theta[L.normIstd + j] : 1;
  }
  return { mean, std };
}

export function rlThetaToGenome(theta, L, game, hp, meta) {
  const { nIn, nOut, hidden } = L;
  const hid0 = nIn + nOut;
  const norm = rlNormOf(theta, L);
  const W1 = [];
  for (let i = 0; i < hidden; i++) {
    const row = new Array(nIn);
    for (let j = 0; j < nIn; j++) row[j] = theta[L.w1 + j * hidden + i];
    W1.push(row);
  }
  const b1 = new Array(hidden);
  for (let i = 0; i < hidden; i++) b1[i] = L.bias ? theta[L.b1 + i] : 0;
  const folded = L.obsNorm ? foldInputNorm(W1, b1, norm.mean, norm.std) : { W1, b1 };
  let wmax = hp.weightMax;
  const edges = [];
  for (let j = 0; j < nIn; j++) {
    for (let i = 0; i < hidden; i++) {
      const w = folded.W1[i][j];
      wmax = Math.max(wmax, Math.abs(w));
      edges.push([j, hid0 + i, w]);
    }
  }
  for (let k = 0; k < nOut; k++) for (let i = 0; i < hidden; i++) edges.push([hid0 + i, nIn + k, theta[L.w2 + k * hidden + i]]);
  if (L.recurrent) for (let m = 0; m < hidden; m++) for (let i = 0; i < hidden; i++) edges.push([hid0 + m, hid0 + i, theta[L.wr + m * hidden + i]]);
  const extras = {
    trainer: 'ppo',
    hidden,
    recurrent: L.recurrent,
    logStd: Array.from(theta.subarray(L.logStd, L.logStd + nOut)),
    valueWeights: Array.from(theta.subarray(L.valueW, L.valueW + hidden)),
    valueBias: theta[L.valueB]
  };
  const json = { format: 'npc-brain/2', game: game ? game.id : null, dims: { nIn, nOut, nNodes: hid0 + hidden }, params: rlGenomeParams(hp, wmax * 1.0001), edges, meta: Object.assign(extras, meta || {}) };
  const withBias = L.bias || L.obsNorm;
  if (withBias || L.recurrent) json.format = 'npc-brain/3';
  if (withBias) {
    const bias = new Array(hid0 + hidden).fill(0);
    for (let k = 0; k < nOut; k++) bias[nIn + k] = L.bias ? theta[L.b2 + k] : 0;
    for (let i = 0; i < hidden; i++) bias[hid0 + i] = folded.b1[i];
    json.bias = bias;
  }
  if (L.recurrent) {
    const leak = new Array(hid0 + hidden).fill(0);
    for (let i = 0; i < hidden; i++) leak[hid0 + i] = theta[L.leak + i];
    json.leak = leak;
  }
  if (L.obsNorm) json.inputNorm = { mean: norm.mean, std: norm.std, folded: true };
  return json;
}

export function rlGenomeToTheta(json, L, hp, seed) {
  const { nIn, nOut, hidden } = L;
  const dims = json.dims;
  if (!dims || dims.nIn !== nIn || dims.nOut !== nOut) throw new Error('genome dims do not match the game');
  const hid0 = nIn + nOut;
  const sourceHidden = dims.nNodes - hid0;
  if (sourceHidden !== hidden) throw new Error('genome has ' + sourceHidden + ' hidden units, trainer is configured for ' + hidden);
  const theta = rlInitTheta(L, seed, hp);
  theta.fill(0, L.w1, L.logStd);
  if (L.recurrent) theta.fill(0, L.wr, L.leak);
  for (const [src, dst, w] of json.edges) {
    if (src < nIn && dst >= hid0) theta[L.w1 + src * hidden + (dst - hid0)] = w;
    else if (src >= hid0 && dst >= nIn && dst < hid0) theta[L.w2 + (dst - nIn) * hidden + (src - hid0)] = w;
    else if (src >= hid0 && dst >= hid0) {
      if (!L.recurrent) throw new Error('genome has hidden-to-hidden edges but the trainer layout is feed-forward');
      theta[L.wr + (src - hid0) * hidden + (dst - hid0)] = w;
    }
  }
  if (Array.isArray(json.bias)) {
    if (json.bias.length !== dims.nNodes) throw new Error('bias length must equal dims.nNodes');
    if (L.bias) {
      for (let k = 0; k < nOut; k++) theta[L.b2 + k] = json.bias[nIn + k];
      for (let i = 0; i < hidden; i++) theta[L.b1 + i] = json.bias[hid0 + i];
    } else {
      for (let j = nIn; j < json.bias.length; j++) if (json.bias[j] !== 0) throw new Error('genome carries bias but the trainer layout has none');
    }
  }
  if (Array.isArray(json.leak)) {
    if (json.leak.length !== dims.nNodes) throw new Error('leak length must equal dims.nNodes');
    if (L.recurrent) for (let i = 0; i < hidden; i++) theta[L.leak + i] = json.leak[hid0 + i];
  }
  const norm = json.inputNorm;
  if (L.obsNorm && norm && norm.folded && Array.isArray(norm.mean) && norm.mean.length === nIn) {
    for (let j = 0; j < nIn; j++) {
      theta[L.normMean + j] = norm.mean[j];
      theta[L.normIstd + j] = 1 / norm.std[j];
    }
    for (let i = 0; i < hidden; i++) {
      let shift = 0;
      for (let j = 0; j < nIn; j++) {
        const w = theta[L.w1 + j * hidden + i];
        shift += w * norm.mean[j];
        theta[L.w1 + j * hidden + i] = w * norm.std[j];
      }
      if (L.bias) theta[L.b1 + i] += shift;
    }
  }
  const meta = json.meta || {};
  if (Array.isArray(meta.logStd) && meta.logStd.length === nOut) for (let k = 0; k < nOut; k++) theta[L.logStd + k] = meta.logStd[k];
  else for (let k = 0; k < nOut; k++) theta[L.logStd + k] = Math.log(hp.sigmaInit);
  if (Array.isArray(meta.valueWeights) && meta.valueWeights.length === hidden) for (let i = 0; i < hidden; i++) theta[L.valueW + i] = meta.valueWeights[i];
  if (typeof meta.valueBias === 'number') theta[L.valueB] = meta.valueBias;
  return theta;
}

export function rlGradientCheck(seed, features) {
  const f = features === true ? { bias: true } : features || {};
  const nIn = 7, nOut = 3, hidden = 5, steps = 6, learners = 3;
  const L = rlLayout(nIn, nOut, hidden, f);
  const R = rlRecordLayout(nOut);
  const hp = Object.assign({}, RL_DEFAULTS, { clip: 0.2, entropy: 0.01, valueCoef: 0.5 });
  const rng = new Rng(seed >>> 0);
  const theta = new Float64Array(L.count);
  for (let p = 0; p < L.count; p++) theta[p] = (rng.next() * 2 - 1) * 0.8;
  for (let k = 0; k < nOut; k++) theta[L.logStd + k] = Math.log(0.4 + rng.next() * 0.3);
  if (L.recurrent) for (let i = 0; i < hidden; i++) theta[L.leak + i] = rng.next() * 0.9;
  const obs = new Float64Array(learners * (steps + 1) * nIn);
  for (let i = 0; i < obs.length; i++) obs[i] = rng.next() < 0.3 ? 0 : rng.next();
  const hInit = new Float64Array(learners * hidden);
  for (let i = 0; i < hInit.length; i++) hInit[i] = (rng.next() * 2 - 1) * 0.7;
  const rec = new Float64Array(learners * steps * R.stride);
  const hprev = new Float64Array(hidden);
  const hnext = new Float64Array(hidden);
  const mu = new Float64Array(nOut);
  const x = new Float64Array(nIn);
  for (let b = 0; b < learners; b++) {
    const row = (j) => {
      for (let c = 0; c < nIn; c++) x[c] = obs[(b * (steps + 1) + j) * nIn + c];
      return x;
    };
    if (L.recurrent) hprev.set(hInit.subarray(b * hidden, (b + 1) * hidden));
    else rlHiddenActivations(theta, L, row(0), hprev);
    for (let t = 0; t < steps; t++) {
      const base = (b * steps + t) * R.stride;
      const valid = rng.next() < 0.8 ? 1 : 0;
      if (!valid) hprev.fill(0);
      rlPolicyMean(theta, L, hprev, mu);
      for (let k = 0; k < nOut; k++) rec[base + R.action + k] = mu[k] + Math.exp(theta[L.logStd + k]) * (rng.next() * 2 - 1) * 1.5;
      rec[base + R.prevValid] = valid;
      rec[base + R.logProb] = rlLogProb(theta, L, rec.subarray(base, base + nOut), mu) + (rng.next() - 0.5) * 1.2;
      rec[base + R.advantage] = (rng.next() - 0.5) * 3;
      rec[base + R.returns] = (rng.next() - 0.5) * 2;
      rlPolicyStep(theta, L, row(t + 1), hprev, hnext);
      hprev.set(hnext);
    }
  }
  const data = { steps, learners, obs, rec, hInit: L.recurrent ? hInit : null, normMean: 0.1, normInvStd: 0.9, selected: null, divisor: learners * steps };
  const analytic = rlLossAndGrad(theta, L, data, hp, true);
  let worst = 0;
  let worstIndex = 0;
  const eps = 1e-6;
  for (let p = 0; p < L.learn; p++) {
    const saved = theta[p];
    theta[p] = saved + eps;
    const up = rlLossAndGrad(theta, L, data, hp, false).loss;
    theta[p] = saved - eps;
    const down = rlLossAndGrad(theta, L, data, hp, false).loss;
    theta[p] = saved;
    const numeric = (up - down) / (2 * eps);
    const err = Math.abs(numeric - analytic.grad[p]) / Math.max(1e-3, Math.abs(numeric) + Math.abs(analytic.grad[p]));
    if (err > worst) {
      worst = err;
      worstIndex = p;
    }
  }
  return { worst, worstIndex, params: L.learn, loss: analytic.loss };
}

export function rlNormMerge(state, s1, s2, nb, prev, opts) {
  const nIn = state.mean.length;
  const cap = opts && opts.cap !== undefined ? opts.cap : 8000000;
  const floor = opts && opts.floor !== undefined ? opts.floor : 0.1;
  const mean = new Float64Array(nIn);
  const variance = new Float64Array(nIn);
  const istd = new Float64Array(nIn);
  for (let f = 0; f < nIn; f++) {
    const sd0 = 1 / prev.istd[f];
    const mn = s1[f] / nb;
    const vn = Math.max(s2[f] / nb - mn * mn, 0);
    const mb = prev.mean[f] + mn * sd0;
    const vb = vn * sd0 * sd0;
    let rm = mb;
    let rv = vb;
    if (state.count > 0) {
      const tot = state.count + nb;
      const delta = mb - state.mean[f];
      rm = state.mean[f] + (delta * nb) / tot;
      rv = (state.count * state.variance[f] + nb * vb + (delta * delta * state.count * nb) / tot) / tot;
    }
    mean[f] = rm;
    variance[f] = rv;
    istd[f] = 1 / Math.max(Math.sqrt(rv), floor);
  }
  return { count: Math.min(state.count + nb, cap), mean, variance, istd };
}

export const RL_CURRICULUM_DEFAULTS = {
  curriculum: 0,
  curriculumMetric: 0,
  curriculumDifficultyStart: 30,
  curriculumDifficultyStep: 5,
  curriculumLifeUp: 0.2,
  curriculumLifeDown: 0.05,
  curriculumPatience: 2,
  curriculumSmoothing: 0.5,
  curriculumSelfPlay: 0,
  curriculumSelfPlayStart: 0.15,
  curriculumSelfPlayEnd: 0.9,
  curriculumRatioLow: 0.5,
  curriculumRatioHigh: 3
};

export class RlCurriculum {
  constructor(opts, maxAge) {
    this.o = Object.assign({}, RL_CURRICULUM_DEFAULTS, opts);
    this.maxAge = maxAge;
    this.difficulty = Math.max(0, Math.min(100, this.o.curriculumDifficultyStart));
    this.selfPlay = this.o.curriculumSelfPlay ? this.o.curriculumSelfPlayStart : 0;
    this.life = null;
    this.progress = 0;
    this.ups = 0;
    this.downs = 0;
    this.reports = 0;
    this.events = 0;
  }

  configure(opts) {
    Object.assign(this.o, opts);
  }

  update(sample) {
    const o = this.o;
    const raw = o.curriculumMetric === 1 ? sample.trainLife : sample.evalLife;
    let note = '';
    this.reports++;
    if (Number.isFinite(raw) && raw > 0) {
      this.life = this.life === null ? raw : this.life + o.curriculumSmoothing * (raw - this.life);
      const fraction = this.life / this.maxAge;
      if (fraction >= o.curriculumLifeUp) {
        this.ups++;
        this.downs = 0;
      } else if (fraction <= o.curriculumLifeDown) {
        this.downs++;
        this.ups = 0;
      } else {
        this.ups = 0;
        this.downs = 0;
      }
      if (this.ups >= o.curriculumPatience) {
        this.difficulty = Math.min(100, this.difficulty + o.curriculumDifficultyStep);
        this.ups = 0;
        this.events++;
        note = 'difficulty up';
      } else if (this.downs >= o.curriculumPatience) {
        this.difficulty = Math.max(0, this.difficulty - o.curriculumDifficultyStep);
        this.downs = 0;
        this.events++;
        note = 'difficulty down';
      }
    }
    const before = this.selfPlay;
    if (o.curriculumSelfPlay && Number.isFinite(sample.evalRate) && sample.baseRate > 0) {
      const ratio = sample.evalRate / sample.baseRate;
      const target = Math.max(0, Math.min(1, (ratio - o.curriculumRatioLow) / Math.max(1e-6, o.curriculumRatioHigh - o.curriculumRatioLow)));
      this.progress = this.progress + o.curriculumSmoothing * (target - this.progress);
      this.selfPlay = o.curriculumSelfPlayStart + (o.curriculumSelfPlayEnd - o.curriculumSelfPlayStart) * this.progress;
    }
    return { difficulty: this.difficulty, selfPlay: this.selfPlay, selfPlayChanged: Math.abs(this.selfPlay - before) > 0.02, note };
  }

  state() {
    return { difficulty: this.difficulty, selfPlay: this.selfPlay, life: this.life, progress: this.progress, ups: this.ups, downs: this.downs, reports: this.reports, events: this.events };
  }

  restore(state) {
    if (state) Object.assign(this, state);
  }
}
