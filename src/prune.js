import { PARAM_DEFAULT } from './core.js';

const NPARAMS = 20;
const FLAG_LIFE_START = 2;

export function softsign(x) {
  return x / (1 + Math.abs(x));
}

export function netFromGenome(json) {
  if (!json || !/^npc-brain\/[23]$/.test(json.format)) throw new Error('prune needs an npc-brain/2 or npc-brain/3 genome');
  const { nIn, nOut, nNodes } = json.dims;
  const params = Array.from({ length: NPARAMS }, (_, i) => (json.params[i] === undefined ? PARAM_DEFAULT[i] : json.params[i]));
  if (params[9] > 0 && Math.floor(params[10]) >= 1) throw new Error('genomes with a novelty gate (gateT > 0) cannot be distilled');
  const kept = json.edges.filter((edge) => edge[1] >= nIn);
  const explicitLeak = Array.isArray(json.leak);
  const net = {
    nIn, nOut, nNodes, gain: params[12], inScale: params[16], params,
    src: Int32Array.from(kept, (edge) => edge[0]),
    dst: Int32Array.from(kept, (edge) => edge[1]),
    w: Float64Array.from(kept, (edge) => edge[2]),
    bias: new Float64Array(nNodes),
    leak: new Float64Array(nNodes),
    orig: Int32Array.from({ length: nNodes }, (_, i) => i),
    explicitLeak, game: json.game || null, meta: json.meta || {}, inputNorm: json.inputNorm || null
  };
  if (Array.isArray(json.bias)) net.bias.set(json.bias);
  for (let j = nIn; j < nNodes; j++) net.leak[j] = explicitLeak ? json.leak[j] : params[15];
  return net;
}

export function cloneNet(net) {
  return Object.assign({}, net, { params: Array.from(net.params), src: Int32Array.from(net.src), dst: Int32Array.from(net.dst), w: Float64Array.from(net.w), bias: Float64Array.from(net.bias), leak: Float64Array.from(net.leak), orig: Int32Array.from(net.orig) });
}

export function hiddenCount(net) {
  return net.nNodes - net.nIn - net.nOut;
}

export function edgeGroupOf(net, e) {
  const hid0 = net.nIn + net.nOut;
  const s = net.src[e];
  const d = net.dst[e];
  const toHidden = d >= hid0;
  if (s < net.nIn) return toHidden ? 0 : 1;
  if (s >= hid0) return toHidden ? 2 : 3;
  return toHidden ? 4 : 5;
}

export const EDGE_GROUP_NAMES = ['in>hid', 'in>out', 'hid>hid', 'hid>out', 'out>hid', 'out>out'];

export function groupCounts(net) {
  const counts = new Array(EDGE_GROUP_NAMES.length).fill(0);
  for (let e = 0; e < net.w.length; e++) counts[edgeGroupOf(net, e)]++;
  return counts;
}

export function isFeedForward(net) {
  const hid0 = net.nIn + net.nOut;
  for (let j = net.nIn; j < net.nNodes; j++) if (net.leak[j] !== 0) return false;
  for (let e = 0; e < net.w.length; e++) {
    const s = net.src[e];
    const d = net.dst[e];
    if (s >= net.nIn && s < hid0) return false;
    if (s >= hid0 && d >= hid0) return false;
  }
  return true;
}

function sortEdgesByDestination(net) {
  const n = net.w.length;
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => net.dst[a] - net.dst[b] || net.src[a] - net.src[b]);
  const src = new Int32Array(n);
  const dst = new Int32Array(n);
  const w = new Float64Array(n);
  order.forEach((from, to) => { src[to] = net.src[from]; dst[to] = net.dst[from]; w[to] = net.w[from]; });
  net.src = src;
  net.dst = dst;
  net.w = w;
  return order;
}

export function compactNet(net) {
  const { nIn, nOut, nNodes } = net;
  const hid0 = nIn + nOut;
  const edgeCount = net.w.length;
  const fromInput = new Uint8Array(nNodes);
  for (let i = 0; i < nIn; i++) fromInput[i] = 1;
  for (let changed = true; changed;) {
    changed = false;
    for (let e = 0; e < edgeCount; e++) if (fromInput[net.src[e]] && !fromInput[net.dst[e]]) { fromInput[net.dst[e]] = 1; changed = true; }
  }
  const toOutput = new Uint8Array(nNodes);
  for (let k = 0; k < nOut; k++) toOutput[nIn + k] = 1;
  for (let changed = true; changed;) {
    changed = false;
    for (let e = 0; e < edgeCount; e++) if (toOutput[net.dst[e]] && !toOutput[net.src[e]]) { toOutput[net.src[e]] = 1; changed = true; }
  }
  const steady = new Float64Array(nNodes);
  const constantNodes = [];
  for (let j = nIn; j < nNodes; j++) if (!fromInput[j]) constantNodes.push(j);
  if (constantNodes.length > 0) {
    const acc = new Float64Array(nNodes);
    for (let iteration = 0; iteration < 600; iteration++) {
      for (const j of constantNodes) acc[j] = net.bias[j];
      for (let e = 0; e < edgeCount; e++) if (!fromInput[net.src[e]] && !fromInput[net.dst[e]]) acc[net.dst[e]] += net.w[e] * steady[net.src[e]];
      for (const j of constantNodes) steady[j] = net.leak[j] * steady[j] + (1 - net.leak[j]) * softsign(net.gain * acc[j]);
    }
  }
  const newId = new Int32Array(nNodes).fill(-1);
  for (let j = 0; j < hid0; j++) newId[j] = j;
  let nextId = hid0;
  for (let j = hid0; j < nNodes; j++) if (fromInput[j] && toOutput[j]) newId[j] = nextId++;
  const bias = new Float64Array(nextId);
  const leak = new Float64Array(nextId);
  const orig = new Int32Array(nextId);
  for (let j = 0; j < nNodes; j++) {
    if (newId[j] < 0) continue;
    bias[newId[j]] = net.bias[j];
    leak[newId[j]] = net.leak[j];
    orig[newId[j]] = net.orig[j];
  }
  const keep = [];
  for (let e = 0; e < edgeCount; e++) {
    const s = net.src[e];
    const d = net.dst[e];
    if (newId[d] < 0) continue;
    if (newId[s] >= 0) { keep.push(e); continue; }
    if (s >= hid0 && !fromInput[s] && toOutput[s]) bias[newId[d]] += net.w[e] * steady[s] / net.gain;
  }
  const out = Object.assign({}, net, {
    nNodes: nextId, bias, leak, orig,
    src: Int32Array.from(keep, (e) => newId[net.src[e]]),
    dst: Int32Array.from(keep, (e) => newId[net.dst[e]]),
    w: Float64Array.from(keep, (e) => net.w[e])
  });
  sortEdgesByDestination(out);
  return out;
}

export function keepEdges(net, flags) {
  const keep = [];
  for (let e = 0; e < net.w.length; e++) if (flags[e]) keep.push(e);
  return Object.assign({}, net, { src: Int32Array.from(keep, (e) => net.src[e]), dst: Int32Array.from(keep, (e) => net.dst[e]), w: Float64Array.from(keep, (e) => net.w[e]) });
}

export function pruneEdgesByScore(net, targetEdges, scores, scope) {
  const total = net.w.length;
  const target = Math.min(total, Math.max(1, targetEdges));
  const flags = new Uint8Array(total);
  const byScore = (ids) => ids.sort((a, b) => scores[b] - scores[a] || a - b);
  if (scope === 'layer') {
    const groups = EDGE_GROUP_NAMES.map(() => []);
    for (let e = 0; e < total; e++) groups[edgeGroupOf(net, e)].push(e);
    const exact = groups.map((ids) => (ids.length * target) / total);
    const quota = exact.map((x, g) => Math.min(groups[g].length, Math.floor(x)));
    let remaining = target - quota.reduce((s, v) => s + v, 0);
    const remainder = exact.map((x, g) => ({ g, r: x - Math.floor(x) })).sort((a, b) => b.r - a.r);
    for (const { g } of remainder) { if (remaining <= 0) break; if (quota[g] < groups[g].length) { quota[g]++; remaining--; } }
    groups.forEach((ids, g) => { byScore(ids).slice(0, quota[g]).forEach((e) => { flags[e] = 1; }); });
  } else {
    byScore(Array.from({ length: total }, (_, i) => i)).slice(0, target).forEach((e) => { flags[e] = 1; });
  }
  return compactNet(keepEdges(net, flags));
}

export function pruneUnitsByScore(net, unitsToKeep, unitScores) {
  const hid0 = net.nIn + net.nOut;
  const ids = [];
  for (let j = hid0; j < net.nNodes; j++) ids.push(j);
  ids.sort((a, b) => unitScores[b] - unitScores[a] || a - b);
  const alive = new Uint8Array(net.nNodes).fill(1);
  ids.slice(Math.max(0, unitsToKeep)).forEach((j) => { alive[j] = 0; });
  const flags = new Uint8Array(net.w.length);
  for (let e = 0; e < net.w.length; e++) flags[e] = alive[net.src[e]] && alive[net.dst[e]] ? 1 : 0;
  return compactNet(keepEdges(net, flags));
}

export function unitScores(net, nodeRms) {
  const scores = new Float64Array(net.nNodes);
  for (let e = 0; e < net.w.length; e++) scores[net.src[e]] += net.w[e] * net.w[e];
  for (let j = 0; j < net.nNodes; j++) scores[j] = Math.sqrt(scores[j]) * (nodeRms ? nodeRms[j] : 1);
  return scores;
}

export function edgeScores(net, method, nodeRms, fisher) {
  const scores = new Float64Array(net.w.length);
  for (let e = 0; e < scores.length; e++) {
    const magnitude = Math.abs(net.w[e]);
    if (method === 'act') scores[e] = magnitude * (nodeRms ? nodeRms[net.src[e]] : 1);
    else if (method === 'fisher') scores[e] = fisher ? fisher[e] : magnitude;
    else scores[e] = magnitude;
  }
  return scores;
}

export function buildCandidateNet(net, teacherNet) {
  const studentIdOf = new Int32Array(teacherNet.nNodes).fill(-1);
  for (let j = 0; j < net.nNodes; j++) studentIdOf[net.orig[j]] = j;
  const activeWeight = new Map();
  for (let e = 0; e < net.w.length; e++) activeWeight.set(net.src[e] * net.nNodes + net.dst[e], net.w[e]);
  const src = [];
  const dst = [];
  const w = [];
  const active = [];
  for (let e = 0; e < teacherNet.w.length; e++) {
    const s = studentIdOf[teacherNet.src[e]];
    const d = studentIdOf[teacherNet.dst[e]];
    if (s < 0 || d < 0 || d < net.nIn) continue;
    const key = s * net.nNodes + d;
    src.push(s);
    dst.push(d);
    w.push(activeWeight.has(key) ? activeWeight.get(key) : 0);
    active.push(activeWeight.has(key) ? 1 : 0);
  }
  const candidate = Object.assign({}, net, { src: Int32Array.from(src), dst: Int32Array.from(dst), w: Float64Array.from(w) });
  return { candidate, active: Uint8Array.from(active) };
}

export function rewireNet(candidate, active, growScores, dropScores, swaps) {
  const activeIds = [];
  const inactiveIds = [];
  for (let e = 0; e < candidate.w.length; e++) (active[e] ? activeIds : inactiveIds).push(e);
  const count = Math.min(swaps, activeIds.length, inactiveIds.length);
  activeIds.sort((a, b) => dropScores[a] - dropScores[b] || a - b);
  inactiveIds.sort((a, b) => growScores[b] - growScores[a] || a - b);
  const flags = new Uint8Array(candidate.w.length);
  for (const e of activeIds.slice(count)) flags[e] = 1;
  for (const e of inactiveIds.slice(0, count)) flags[e] = 1;
  return compactNet(keepEdges(candidate, flags));
}

export function genomeFromNet(net, extraMeta) {
  const edges = [];
  for (let e = 0; e < net.w.length; e++) edges.push([net.src[e], net.dst[e], Math.fround(net.w[e])]);
  const meta = {};
  for (const key of Object.keys(net.meta || {})) if (!['logStd', 'valueWeights', 'valueBias', 'fitness', 'best'].includes(key)) meta[key] = net.meta[key];
  Object.assign(meta, { trainer: 'prune' }, extraMeta || {});
  const out = { format: 'npc-brain/2', game: net.game, dims: { nIn: net.nIn, nOut: net.nOut, nNodes: net.nNodes }, params: Array.from(net.params), edges, meta };
  let hasBias = false;
  for (let j = 0; j < net.nNodes; j++) if (net.bias[j] !== 0) hasBias = true;
  if (hasBias || net.explicitLeak) out.format = 'npc-brain/3';
  if (hasBias) out.bias = Array.from(net.bias, Math.fround);
  if (net.explicitLeak) out.leak = Array.from(net.leak, Math.fround);
  if (out.format === 'npc-brain/3' && net.inputNorm) out.inputNorm = net.inputNorm;
  return out;
}

export function lossSpecFor(game, weights) {
  const moveWeight = weights && weights.move !== undefined ? weights.move : 2;
  const thresholdWeight = weights && weights.threshold !== undefined ? weights.threshold : 1;
  const nOut = game.dims.nOut;
  const names = game.actionNames || [];
  const spec = { weights: new Float64Array(nOut).fill(1), combos: [], thresholds: [], moves: [] };
  const index = (name) => names.indexOf(name);
  for (const [plus, minus] of [['+x', '-x'], ['+y', '-y']]) {
    const a = index(plus);
    const b = index(minus);
    if (a < 0 || b < 0) continue;
    const coef = new Float64Array(nOut);
    coef[a] = 1;
    coef[b] = -1;
    spec.combos.push({ coef, weight: moveWeight });
    spec.moves.push([a, b]);
  }
  const moveOutputs = new Set(spec.moves.flat());
  if (spec.moves.length > 0) names.forEach((name, k) => { if (!moveOutputs.has(k)) spec.thresholds.push({ k, thr: 0.5, weight: thresholdWeight, margin: 0.3 }); });
  return spec;
}

function lossAndDerivative(y, t, yOff, tOff, spec, dy) {
  const { weights, combos, thresholds } = spec;
  const n = weights.length;
  let loss = 0;
  for (let k = 0; k < n; k++) {
    const err = y[yOff + k] - t[tOff + k];
    loss += weights[k] * err * err;
    dy[k] = 2 * weights[k] * err;
  }
  for (const { coef, weight } of combos) {
    let d = 0;
    for (let k = 0; k < n; k++) if (coef[k] !== 0) d += coef[k] * (y[yOff + k] - t[tOff + k]);
    loss += weight * d * d;
    for (let k = 0; k < n; k++) if (coef[k] !== 0) dy[k] += 2 * weight * d * coef[k];
  }
  for (const { k, thr, weight, margin } of thresholds) {
    const target = t[tOff + k];
    const side = target > thr ? 1 : -1;
    const need = 0.5 * Math.min(margin, Math.abs(target - thr));
    const v = need - side * (y[yOff + k] - thr);
    if (v > 0) {
      loss += weight * v * v;
      dy[k] -= 2 * weight * v * side;
    }
  }
  return loss;
}

export function newAccumulator(net, spec, wantStats) {
  const stateCount = net.nNodes - net.nIn;
  return {
    grad: new Float64Array(net.w.length), gradBias: new Float64Array(stateCount), loss: 0, count: 0,
    sqErr: new Float64Array(net.nOut), sideAgree: new Float64Array(spec.thresholds.length), positiveTeacher: new Float64Array(spec.thresholds.length), positiveBoth: new Float64Array(spec.thresholds.length), moveCos: 0, moveMagErr: 0, moveCount: 0,
    sumSq: wantStats ? new Float64Array(stateCount) : null, inputSumSq: wantStats ? new Float64Array(net.nIn) : null, statSamples: 0
  };
}

export function nodeRmsFrom(acc, net) {
  const rms = new Float64Array(net.nNodes);
  const samples = Math.max(1, acc.statSamples);
  for (let i = 0; i < net.nIn; i++) rms[i] = Math.sqrt(acc.inputSumSq[i] / samples);
  for (let j = 0; j < net.nNodes - net.nIn; j++) rms[net.nIn + j] = Math.sqrt(acc.sumSq[j] / samples);
  return rms;
}

export function mergeAccumulator(into, from) {
  for (let i = 0; i < into.grad.length; i++) into.grad[i] += from.grad[i];
  for (let i = 0; i < into.gradBias.length; i++) into.gradBias[i] += from.gradBias[i];
  for (let i = 0; i < into.sqErr.length; i++) into.sqErr[i] += from.sqErr[i];
  for (let i = 0; i < into.sideAgree.length; i++) {
    into.sideAgree[i] += from.sideAgree[i];
    into.positiveTeacher[i] += from.positiveTeacher[i];
    into.positiveBoth[i] += from.positiveBoth[i];
  }
  into.loss += from.loss;
  into.count += from.count;
  into.moveCos += from.moveCos;
  into.moveMagErr += from.moveMagErr;
  into.moveCount += from.moveCount;
  if (from.sumSq && into.sumSq) for (let i = 0; i < into.sumSq.length; i++) into.sumSq[i] += from.sumSq[i];
  if (from.inputSumSq && into.inputSumSq) for (let i = 0; i < into.inputSumSq.length; i++) into.inputSumSq[i] += from.inputSumSq[i];
  into.statSamples += from.statSamples;
}

export class ChunkEngine {
  constructor(data) {
    this.data = data;
  }

  configure(net, spec, segmentLength) {
    this.nIn = net.nIn;
    this.nOut = net.nOut;
    this.nState = net.nNodes - net.nIn;
    this.edgeCount = net.w.length;
    this.src = net.src;
    this.dstLocal = Int32Array.from(net.dst, (d) => d - net.nIn);
    this.gain = net.gain;
    this.inScale = net.inScale;
    this.leak = net.leak.subarray(net.nIn);
    this.orig = net.orig;
    this.spec = spec;
    this.feedForward = isFeedForward(net);
    this.segmentLength = segmentLength;
    this.burn = this.feedForward ? 1 : 0;
    const steps = segmentLength + this.burn;
    this.states = new Float64Array((steps + 1) * this.nState);
    this.dfac = new Float64Array(steps * this.nState);
    this.gradState = new Float64Array(steps * this.nState);
    this.accum = new Float64Array(this.nState);
    this.carry = new Float64Array(this.nState);
    this.nextCarry = new Float64Array(this.nState);
    this.dy = new Float64Array(this.nOut);
    this.yBuf = new Float64Array(this.nOut);
    return this;
  }

  setWeights(w, bias) {
    this.w = w;
    this.bias = bias.subarray(this.nIn);
  }

  initialState(chunk, lifeStart, out) {
    out.fill(0);
    const data = this.data;
    if (lifeStart || this.feedForward || !data.h0) return;
    const base = chunk * data.teacherStateCount;
    for (let j = 0; j < this.nState; j++) out[j] = data.h0[base + this.orig[this.nIn + j] - this.nIn];
  }

  runItem(chunk, offset, length, acc, wantGrad) {
    const data = this.data;
    const chunkStart = data.chunkStart[chunk];
    const chunkLength = data.chunkLength[chunk];
    const lifeStart = (data.flags[chunkStart] & FLAG_LIFE_START) !== 0;
    const nState = this.nState;
    const state = new Float64Array(nState);
    let first = chunkStart + offset;
    let remaining = length;
    let burnSteps = 0;
    if (this.feedForward) {
      if (offset > 0) { first -= 1; remaining += 1; burnSteps = 1; }
      else if (!lifeStart) { burnSteps = 1; }
    } else this.initialState(chunk, lifeStart && offset === 0, state);
    const segmentMax = this.segmentLength + this.burn;
    let carriedState = state;
    let stepIndex = 0;
    while (remaining > 0) {
      const steps = Math.min(segmentMax, remaining);
      carriedState = this.segment(first + stepIndex, steps, carriedState, stepIndex === 0 ? burnSteps : 0, acc, wantGrad);
      stepIndex += steps;
      remaining -= steps;
    }
  }

  segment(firstSample, steps, stateIn, burnSteps, acc, wantGrad) {
    const { nIn, nOut, nState, edgeCount, src, dstLocal, gain, inScale, leak, states, dfac, gradState, accum, spec } = this;
    const data = this.data;
    const w = this.w;
    const bias = this.bias;
    const obs = data.obs;
    states.set(stateIn, 0);
    for (let k = 0; k < steps; k++) {
      const xBase = (firstSample + k) * nIn;
      const prev = k * nState;
      const next = (k + 1) * nState;
      accum.fill(0);
      for (let e = 0; e < edgeCount; e++) {
        const s = src[e];
        accum[dstLocal[e]] += w[e] * (s < nIn ? obs[xBase + s] * inScale : states[prev + s - nIn]);
      }
      for (let j = 0; j < nState; j++) {
        const x = gain * (accum[j] + bias[j]);
        const lk = leak[j];
        const denom = 1 + Math.abs(x);
        states[next + j] = lk * states[prev + j] + (1 - lk) * (x / denom);
        dfac[k * nState + j] = ((1 - lk) * gain) / (denom * denom);
      }
      if (acc.inputSumSq) {
        for (let i = 0; i < nIn; i++) acc.inputSumSq[i] += obs[xBase + i] * obs[xBase + i] * inScale * inScale;
        for (let j = 0; j < nState; j++) acc.sumSq[j] += states[next + j] * states[next + j];
        acc.statSamples++;
      }
    }
    gradState.fill(0, 0, steps * nState);
    const dy = this.dy;
    for (let k = burnSteps; k < steps; k++) {
      const sample = firstSample + k;
      const teacherBase = sample * nOut;
      const yBase = (k + 1) * nState;
      acc.loss += lossAndDerivative(states, data.tout, yBase, teacherBase, spec, dy);
      acc.count++;
      for (let o = 0; o < nOut; o++) {
        const err = states[yBase + o] - data.tout[teacherBase + o];
        acc.sqErr[o] += err * err;
        gradState[k * nState + o] += dy[o];
      }
      spec.thresholds.forEach(({ k: ok, thr }, idx) => {
        const target = data.tout[teacherBase + ok];
        const studentOn = states[yBase + ok] > thr;
        const teacherOn = target > thr;
        if (studentOn === teacherOn) acc.sideAgree[idx]++;
        if (teacherOn) { acc.positiveTeacher[idx]++; if (studentOn) acc.positiveBoth[idx]++; }
      });
      if (spec.moves.length === 2) {
        const [[ax, bx], [ay, by]] = spec.moves;
        const studentX = states[yBase + ax] - states[yBase + bx];
        const studentY = states[yBase + ay] - states[yBase + by];
        const teacherX = data.tout[teacherBase + ax] - data.tout[teacherBase + bx];
        const teacherY = data.tout[teacherBase + ay] - data.tout[teacherBase + by];
        const studentNorm = Math.hypot(studentX, studentY);
        const teacherNorm = Math.hypot(teacherX, teacherY);
        acc.moveMagErr += Math.abs(studentNorm - teacherNorm);
        if (teacherNorm > 0.1 && studentNorm > 1e-9) { acc.moveCos += (studentX * teacherX + studentY * teacherY) / (studentNorm * teacherNorm); acc.moveCount++; }
      }
    }
    if (wantGrad) this.backward(firstSample, steps, acc);
    return states.slice(steps * nState, (steps + 1) * nState);
  }

  backward(firstSample, steps, acc) {
    const { nIn, nState, edgeCount, src, dstLocal, inScale, leak, states, dfac, gradState, carry, nextCarry } = this;
    const obs = this.data.obs;
    const w = this.w;
    carry.fill(0);
    for (let k = steps - 1; k >= 0; k--) {
      const xBase = (firstSample + k) * nIn;
      const prev = k * nState;
      nextCarry.fill(0);
      for (let j = 0; j < nState; j++) {
        const total = gradState[k * nState + j] + carry[j];
        gradState[k * nState + j] = total * dfac[k * nState + j];
        acc.gradBias[j] += gradState[k * nState + j];
        nextCarry[j] = leak[j] * total;
      }
      for (let e = 0; e < edgeCount; e++) {
        const g = gradState[k * nState + dstLocal[e]];
        if (g === 0) continue;
        const s = src[e];
        if (s < nIn) acc.grad[e] += g * obs[xBase + s] * inScale;
        else {
          acc.grad[e] += g * states[prev + s - nIn];
          nextCarry[s - nIn] += w[e] * g;
        }
      }
      carry.set(nextCarry);
    }
  }
}

export class Adam {
  constructor(size) {
    this.m = new Float64Array(size);
    this.v = new Float64Array(size);
    this.t = 0;
  }

  step(params, grad, scale, lr) {
    this.t++;
    const b1 = 0.9;
    const b2 = 0.999;
    const c1 = 1 - Math.pow(b1, this.t);
    const c2 = 1 - Math.pow(b2, this.t);
    for (let i = 0; i < params.length; i++) {
      const g = grad[i] * scale;
      this.m[i] = b1 * this.m[i] + (1 - b1) * g;
      this.v[i] = b2 * this.v[i] + (1 - b2) * g * g;
      params[i] -= (lr * (this.m[i] / c1)) / (Math.sqrt(this.v[i] / c2) + 1e-8);
    }
  }
}

export function forwardStep(net, inputs, state, nextState) {
  const { nIn, nNodes, src, dst, w, gain, inScale, leak, bias } = net;
  const acc = new Float64Array(nNodes);
  for (let e = 0; e < w.length; e++) {
    const s = src[e];
    acc[dst[e]] += w[e] * (s < nIn ? inputs[s] * inScale : state[s]);
  }
  for (let j = nIn; j < nNodes; j++) {
    const x = gain * (acc[j] + bias[j]);
    nextState[j] = leak[j] * state[j] + (1 - leak[j]) * softsign(x);
  }
}

