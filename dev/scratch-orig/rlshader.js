import { rlLayout, RL_GAIN, RL_LEAK_MAX, RL_OBS_CLIP } from './rl.js';

export const RL_BINDINGS = { theta: 0, world: 1, obs: 2, rec: 3, lane: 4, partials: 5, opt: 6, statsOut: 7, uniforms: 8 };
export const RL_ENTRIES = ['rl_init_world', 'rl_reinit_world', 'rl_rollout', 'rl_gae', 'rl_advstats', 'rl_grad', 'rl_grad_seq', 'rl_select', 'rl_grad_blk', 'rl_reduce_grad', 'rl_adam', 'rl_rescale', 'rl_normstats', 'rl_normapply', 'rl_reduce_stats'];
export const RL_UNIFORM_WORDS = 32;
export const RL_CTL = { STEP: 0, SCALE: 1, ADV_MEAN: 2, ADV_INV_STD: 3, GRAD_NORM: 4, POLICY: 5, VALUE: 6, ENTROPY: 7, CLIP: 8, KL: 9, RET_STD: 10, ADV_STD: 11, ITER: 12, STEPS: 13, RET_MEAN: 14, ACTIVE: 15, NCOUNT: 16, SIZE: 32 };
export const RL_LANE = { LIFE: 0, SCORE: 1, VALID: 2, BOOT: 3, ROLE: 4, BASE: 5, HEADER: 8 };
export const RL_ROLE = { INACTIVE: 0, LIVE: 1, SNAPSHOT: 2 };
export const RL_STATS = { GROUPS: 4, ROLE_WORDS: 4, ROLE_OFFSET: 64, WORLD_OFFSET: 72 };

export const RL_NORM_GROUPS = 64;

export function rlFeatures(cfg) {
  return { bias: !!(cfg.bias || cfg.obsNorm), recurrent: !!cfg.recurrent, obsNorm: !!cfg.obsNorm };
}

export function rlShaderLayout(game, cfg) {
  const { nIn, nOut } = game.dims;
  const learners = game.maxLearners || game.learners;
  const hidden = cfg.hidden;
  const features = rlFeatures(cfg);
  const L = rlLayout(nIn, nOut, hidden, features);
  const lanes = Math.max(1, Math.floor(64 / learners));
  const trainWorlds = cfg.evalStart;
  const trainLearners = trainWorlds * learners;
  const tile = cfg.tileEntries;
  const entriesPerLearner = cfg.rolloutTicks + 1;
  const tilesPerLearner = Math.ceil(entriesPerLearner / tile);
  const partFloats = lanes * (nOut + 1);
  const hiddenBuffers = features.recurrent ? 2 : 1;
  const laneStride = RL_LANE.HEADER + hiddenBuffers * hidden + nIn + partFloats;
  const accumulators = Math.ceil(L.learn / 256);
  const obsStride = nIn | 1;
  const hiddenStride = hidden | 1;
  const poolSize = cfg.poolSize || 0;
  const obsLearnerStride = entriesPerLearner * nIn + (features.recurrent ? hidden : 0);
  const normSlices = Math.max(1, Math.floor(256 / nIn));
  const normPartFloats = features.obsNorm ? RL_NORM_GROUPS * normSlices * 2 * nIn : 0;
  const blockHalves = hidden <= 128 ? 2 : 1;
  const blockThreads = hidden * blockHalves;
  const blockGrad = Boolean(cfg.blockGrad) && blockThreads <= 256;
  const groups = blockGrad ? Math.max(1, Math.min(cfg.gradGroups, Math.ceil((trainLearners / Math.max(1, cfg.minibatches)) / 16 * 1.5) + 2)) : cfg.gradGroups;
  const blockStride = 4 * 16 * entriesPerLearner * hidden + 2 * 16 * cfg.rolloutTicks * nOut + 2 * 16 * cfg.rolloutTicks;
  const seqScratch = blockGrad ? blockStride : features.recurrent ? 2 * entriesPerLearner * hidden : 0;
  const scratchBase = (groups + 1) * L.partialStride + normPartFloats;
  const selBase = scratchBase + groups * seqScratch;
  const rolloutBytes = (learners * (obsStride + nOut + 6) + 2 * game.agents + 20 + 4 + 3) * 4 + game.workgroupBytes;
  const T = cfg.rolloutTicks;
  const tileBytes = (tile * (obsStride + 2 * hiddenStride + 3 * nOut + 2) + tile * 8 + tile + 256 + 1) * 4;
  const seqBytes = (2 * (obsStride + 4 * hiddenStride) + 2 * T * nOut + 3 * (T + 1) + T * 8 + 8) * 4;
  const blockBytes = ((entriesPerLearner * 4 + 2 * hidden * 4 + nIn * 4 + nOut * 4 + 4) * 16) + (16 + 16 + 1) * 4 + 256 * 4;
  const gradBytes = blockGrad ? blockBytes : features.recurrent ? seqBytes : tileBytes;
  return {
    nIn, nOut, hidden, learners, agents: game.agents, params: L, features, lanes, trainWorlds, trainLearners, worlds: cfg.worlds, tile, entriesPerLearner, tilesPerLearner,
    evalBStart: cfg.evalBStart, evalCStart: cfg.evalCStart, poolSize, hiddenBuffers, obsLearnerStride, normSlices, normPartFloats, seqScratch, scratchBase,
    laneStride, partFloats, accumulators, obsStride, hiddenStride, rolloutTicks: cfg.rolloutTicks, gradGroups: groups, blockGrad, blockHalves, blockThreads, selBase,
    worldStride: 24 + game.worldWords, recStride: L.recStride, partialStride: L.partialStride,
    obsFloats: trainLearners * obsLearnerStride, recFloats: trainLearners * cfg.rolloutTicks * L.recStride,
    thetaFloats: (1 + poolSize) * L.count,
    laneFloats: cfg.worlds * learners * laneStride, optFloats: 2 * L.count + RL_CTL.SIZE + (features.obsNorm ? 2 * nIn : 0), partialFloats: selBase + (blockGrad ? trainLearners + 1 : 0),
    statsFloats: RL_STATS.WORLD_OFFSET + cfg.worlds * 2,
    denseEdges: hidden * (nIn + nOut) + (features.recurrent ? hidden * hidden : 0), workgroupBytes: Math.max(rolloutBytes, gradBytes), rolloutBytes, gradBytes
  };
}

function blockGemm(o) {
  const r4 = Math.ceil(o.rows / 4);
  const nu = Math.ceil((r4 * o.quads) / o.threads);
  return `
  {
    let ns = BS * ${o.nj}u;
    var acc: array<vec4<f32>, ${nu * 4}>;
    var ur: array<u32, ${nu}>;
    var uq: array<u32, ${nu}>;
    for (var u = 0u; u < ${nu}u; u++) {
      let uu = tid + BT * u;
      ur[u] = uu / QH;
      uq[u] = uu - ur[u] * QH;
    }
    for (var u = 0u; u < ${nu * 4}u; u++) { acc[u] = vec4<f32>(0.0); }
    for (var n0 = 0u; n0 < ns; n0 += 16u) {
      for (var idx = tid; idx < 16u * ${r4 * 4}u; idx += BT) {
        let nn = idx / ${r4 * 4}u;
        let r = idx - nn * ${r4 * 4}u;
        let n = n0 + nn;
        var v = 0.0;
        if (n < ns && r < ${o.rows}u) {
          let s = n / ${o.nj}u;
          let j = ${o.jlo}u + (n - s * ${o.nj}u);
          if (s < nsel) { v = ${o.a}; }
        }
        pset(GA4 * 4u + idx, v);
      }
      for (var idx = tid; idx < 16u * QH; idx += BT) {
        let nn = idx / QH;
        let q = idx - nn * QH;
        let n = n0 + nn;
        var v4 = vec4<f32>(0.0);
        if (n < ns) {
          let s = n / ${o.nj}u;
          let j = ${o.jlo}u + (n - s * ${o.nj}u);
          if (s < nsel) {
            for (var cc = 0u; cc < 4u; cc++) {
              let i = q * 4u + cc;
              if (i < HID) { v4[cc] = ${o.b}; }
            }
          }
        }
        pool4[GB4 + idx] = v4;
      }
      workgroupBarrier();
      for (var nn = 0u; nn < 16u; nn++) {
        for (var u = 0u; u < ${nu}u; u++) {
          if (tid + BT * u < ${r4}u * QH) {
            let a4 = pool4[GA4 + nn * ${r4}u + ur[u]];
            let b4 = pool4[GB4 + nn * QH + uq[u]];
            acc[u * 4u] += a4.x * b4;
            acc[u * 4u + 1u] += a4.y * b4;
            acc[u * 4u + 2u] += a4.z * b4;
            acc[u * 4u + 3u] += a4.w * b4;
          }
        }
      }
      workgroupBarrier();
    }
    for (var u = 0u; u < ${nu}u; u++) {
      if (tid + BT * u < ${r4}u * QH) {
        for (var tr = 0u; tr < 4u; tr++) {
          let r = ur[u] * 4u + tr;
          if (r < ${o.rows}u) {
            for (var cc = 0u; cc < 4u; cc++) {
              let i = uq[u] * 4u + cc;
              if (i < HID) { partials[gp + ${o.out} + r * HID + i] += acc[u * 4u + tr][cc]; }
            }
          }
        }
      }
    }
  }
`;
}

function blockKernelSource(S, F) {
  const nj = S.entriesPerLearner;
  const gemms = [];
  const j0 = F.recurrent ? 1 : 0;
  gemms.push(blockGemm({ rows: S.nIn, quads: Math.ceil(S.hidden / 4), threads: S.blockThreads, nj: nj - j0, jlo: j0, out: 'OFF_W1', a: 'rObs[sOb[s] + j * NIN + r]', b: 'partials[scr + DZ_OFF + (s * TE + j) * HID + i]' }));
  if (F.recurrent) gemms.push(blockGemm({ rows: S.hidden, quads: Math.ceil(S.hidden / 4), threads: S.blockThreads, nj: S.rolloutTicks, jlo: 1, out: 'OFF_WR', a: 'partials[scr + HB_OFF + (s * TE + j - 1u) * HID + r] * pv(j - 1u, s)', b: 'partials[scr + DZ_OFF + (s * TE + j) * HID + i]' }));
  gemms.push(blockGemm({ rows: S.nOut, quads: Math.ceil(S.hidden / 4), threads: S.blockThreads, nj: S.rolloutTicks, jlo: 0, out: 'OFF_W2', a: 'partials[scr + DP_OFF + (s * ROLL + j) * NOUT + r]', b: 'partials[scr + HB_OFF + (s * TE + j) * HID + i] * pv(j, s)' }));
  gemms.push(blockGemm({ rows: 1, quads: Math.ceil(S.hidden / 4), threads: S.blockThreads, nj: S.rolloutTicks, jlo: 0, out: 'OFF_WV', a: 'partials[scr + DV_OFF + s * ROLL + j]', b: 'partials[scr + HB_OFF + (s * TE + j + 1u) * HID + i]' }));
  return `
const BS: u32 = 16u;
const BH: u32 = ${S.blockHalves}u;
const BT: u32 = ${S.blockThreads}u;
const SBQ: u32 = ${16 / S.blockHalves}u;
const NQ4: u32 = ${16 / S.blockHalves / 4}u;
const QH: u32 = ${Math.ceil(S.hidden / 4)}u;
const P_VALID: u32 = 0u;
const P_PP: u32 = ${nj * 4}u;
const P_XS: u32 = ${nj * 4 + 2 * S.hidden * 4}u;
const P_DPS: u32 = ${nj * 4 + 2 * S.hidden * 4 + S.nIn * 4}u;
const P_DVV: u32 = ${nj * 4 + 2 * S.hidden * 4 + S.nIn * 4 + S.nOut * 4}u;
const GA4: u32 = ${nj * 4}u;
const GB4: u32 = ${nj * 4 + 16 * Math.ceil(Math.max(S.nIn, S.hidden) / 4)}u;
const KQ: u32 = ${Math.ceil(S.nOut / 4)}u;
const P_W2T: u32 = ${nj * 4 + 2 * S.hidden * 4 + S.nIn * 4 + S.nOut * 4 + 4}u;
const P_WV: u32 = ${nj * 4 + 2 * S.hidden * 4 + S.nIn * 4 + S.nOut * 4 + 4 + Math.ceil(S.nOut / 4) * S.hidden}u;
const SB_OFF: u32 = 0u;
const HB_OFF: u32 = ${16 * nj * S.hidden}u;
const DZ_OFF: u32 = ${2 * 16 * nj * S.hidden}u;
const DH_OFF: u32 = ${3 * 16 * nj * S.hidden}u;
const DP_OFF: u32 = ${4 * 16 * nj * S.hidden}u;
const MU_OFF: u32 = ${4 * 16 * nj * S.hidden + 16 * S.rolloutTicks * S.nOut}u;
const DV_OFF: u32 = ${4 * 16 * nj * S.hidden + 2 * 16 * S.rolloutTicks * S.nOut}u;
const GW_OFF: u32 = ${4 * 16 * nj * S.hidden + 2 * 16 * S.rolloutTicks * S.nOut + 16 * S.rolloutTicks}u;
const SEL_BASE: u32 = ${S.selBase}u;

var<workgroup> pool4: array<vec4<f32>, ${nj * 4 + 2 * S.hidden * 4 + S.nIn * 4 + S.nOut * 4 + 4 + Math.ceil(S.nOut / 4) * S.hidden + Math.ceil(S.hidden / 4)}>;
var<workgroup> sOb: array<u32, 16>;
var<workgroup> sSel: array<u32, 16>;
var<workgroup> blkCount: u32;

fn pf(i: u32) -> f32 { return pool4[i >> 2u][i & 3u]; }
fn pset(i: u32, v: f32) { pool4[i >> 2u][i & 3u] = v; }
fn pv(j: u32, s: u32) -> f32 { return pool4[P_VALID + j * 4u + (s >> 2u)][s & 3u]; }

var<workgroup> selCnt: array<u32, 256>;

@compute @workgroup_size(256)
fn rl_select(@builtin(local_invocation_index) tid: u32) {
  let step = u32(opt[CTL + C_STEP]);
  let mb = step % U.minibatches;
  let salt = step / U.minibatches;
  let chunk = (TRAIN_LEARNERS + 255u) / 256u;
  let lo = tid * chunk;
  let hi = min(lo + chunk, TRAIN_LEARNERS);
  var cnt = 0u;
  for (var b = lo; b < hi; b++) {
    if (mix4(salt, b, 7u, 9u) % U.minibatches == mb && lane[b * LS + LS_ROLE] == 1.0) { cnt++; }
  }
  selCnt[tid] = cnt;
  workgroupBarrier();
  var before = 0u;
  for (var t = 0u; t < tid; t++) { before += selCnt[t]; }
  var at = before;
  for (var b = lo; b < hi; b++) {
    if (mix4(salt, b, 7u, 9u) % U.minibatches == mb && lane[b * LS + LS_ROLE] == 1.0) {
      partials[SEL_BASE + 1u + at] = f32(b);
      at++;
    }
  }
  if (tid == 255u) { partials[SEL_BASE] = f32(at); }
}

@compute @workgroup_size(${S.blockThreads})
fn rl_grad_blk(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let g = wg.x;
  let gp = g * PST;
  let scr = SCR_BASE + g * SCR_STRIDE;
  let hh = tid / HID;
  let ii = tid - hh * HID;
  let advMean = opt[CTL + C_ADV_MEAN];
  let advInv = opt[CTL + C_ADV_INV];
  for (var p = tid; p < LEARN; p += BT) { partials[gp + p] = 0.0; }
  var sPol = 0.0;
  var sVal = 0.0;
  var sEnt = 0.0;
  var sClip = 0.0;
  var sKl = 0.0;
  var sN = 0.0;
  if (tid == 0u) { blkCount = u32(partials[SEL_BASE]); }
  storageBarrier();
  workgroupBarrier();
  let total = workgroupUniformLoad(&blkCount);
  let nblocks = (total + 15u) / 16u;
  let j0 = ${F.recurrent ? 1 : 0}u;
  for (var blk = g; blk < nblocks; blk += GROUPS) {
    let nsel = min(16u, total - blk * 16u);
    if (tid < 16u) {
      var b = 0u;
      if (tid < nsel) { b = u32(partials[SEL_BASE + 1u + blk * 16u + tid]); }
      sSel[tid] = b;
      sOb[tid] = b * OBS_LS;
    }
    workgroupBarrier();
    for (var idx = tid; idx < 16u * TE; idx += BT) {
      let s = idx / TE;
      let j = idx - s * TE;
      var v = 0.0;
      if (s < nsel) {
        v = 1.0;
        if (j < ROLL && rRec[(sSel[s] * ROLL + j) * REC + R_PREV] <= 0.5) { v = 0.0; }
      }
      pool4[P_VALID + j * 4u + (s >> 2u)][s & 3u] = v;
    }
    workgroupBarrier();
    if (RECUR == 1u) {
      for (var idx = tid; idx < 16u * HID; idx += BT) {
        let s = idx / HID;
        let i = idx - s * HID;
        var h0 = 0.0;
        if (s < nsel) { h0 = rObs[sOb[s] + OBS_HINIT + i]; }
        partials[scr + HB_OFF + (s * TE) * HID + i] = h0;
        pool4[P_PP + i * 4u + (s >> 2u)][s & 3u] = h0 * pv(0u, s);
      }
    }
    storageBarrier();
    workgroupBarrier();
    for (var j = j0; j <= ROLL; j++) {
      for (var idx = tid; idx < NIN * 16u; idx += BT) {
        let s = idx / NIN;
        let c = idx - s * NIN;
        var v = 0.0;
        if (s < nsel) { v = rObs[sOb[s] + j * NIN + c]; }
        pool4[P_XS + c * 4u + (s >> 2u)][s & 3u] = v;
      }
      workgroupBarrier();
      if (hh < BH) {
        let cur = (j & 1u) * HID * 4u;
        let prv = ((j + 1u) & 1u) * HID * 4u;
        var z: array<vec4<f32>, NQ4>;
        for (var qv = 0u; qv < NQ4; qv++) { z[qv] = vec4<f32>(0.0); }
        for (var c = 0u; c < NIN; c++) {
          let w = theta[OFF_W1 + c * HID + ii];
          for (var qv = 0u; qv < NQ4; qv++) { z[qv] += w * pool4[P_XS + c * 4u + hh * NQ4 + qv]; }
        }
        if (RECUR == 1u) {
          for (var m = 0u; m < HID; m++) {
            let w = theta[OFF_WR + m * HID + ii];
            for (var qv = 0u; qv < NQ4; qv++) { z[qv] += w * pool4[P_PP + prv + m * 4u + hh * NQ4 + qv]; }
          }
        }
        var bz = 0.0;
        if (BIAS == 1u) { bz = theta[OFF_B1 + ii]; }
        var lk = 0.0;
        if (RECUR == 1u) { lk = theta[OFF_LEAK + ii]; }
        for (var qv = 0u; qv < NQ4; qv++) {
          let zz = GAIN * (z[qv] + vec4<f32>(bz));
          let sv = zz / (vec4<f32>(1.0) + abs(zz));
          var hn = sv;
          if (RECUR == 1u) {
            let hp = pool4[P_PP + prv + ii * 4u + hh * NQ4 + qv];
            hn = lk * hp + (1.0 - lk) * sv;
          }
          let vq = pool4[P_VALID + j * 4u + hh * NQ4 + qv];
          if (RECUR == 1u) { pool4[P_PP + cur + ii * 4u + hh * NQ4 + qv] = hn * vq; }
          for (var l = 0u; l < 4u; l++) {
            let s = (hh * NQ4 + qv) * 4u + l;
            partials[scr + SB_OFF + (s * TE + j) * HID + ii] = sv[l];
            partials[scr + HB_OFF + (s * TE + j) * HID + ii] = hn[l];
          }
        }
      }
      workgroupBarrier();
    }
    storageBarrier();
    workgroupBarrier();
    for (var idx = tid; idx < KQ * HID; idx += BT) {
      let i = idx / KQ;
      let kq = idx - i * KQ;
      var w4 = vec4<f32>(0.0);
      for (var cc = 0u; cc < 4u; cc++) {
        let k = kq * 4u + cc;
        if (k < NOUT) { w4[cc] = theta[OFF_W2 + k * HID + i]; }
      }
      pool4[P_W2T + i * KQ + kq] = w4;
    }
    for (var i = tid; i < HID; i += BT) { pool4[P_WV + (i >> 2u)][i & 3u] = theta[OFF_WV + i]; }
    workgroupBarrier();
    for (var idx = tid; idx < 16u * ROLL; idx += BT) {
      let s = idx / ROLL;
      let j = idx - s * ROLL;
      let ob = (s * ROLL + j) * NOUT;
      if (s >= nsel) {
        for (var k = 0u; k < NOUT; k++) { partials[scr + DP_OFF + ob + k] = 0.0; }
        partials[scr + DV_OFF + s * ROLL + j] = 0.0;
        partials[scr + GW_OFF + s * ROLL + j] = 0.0;
        continue;
      }
      let row = scr + HB_OFF + (s * TE + j) * HID;
      let ok = pv(j, s) > 0.5;
      var pre4: array<vec4<f32>, KQ>;
      for (var kq = 0u; kq < KQ; kq++) { pre4[kq] = vec4<f32>(0.0); }
      if (BIAS == 1u) {
        for (var k = 0u; k < NOUT; k++) { pre4[k >> 2u][k & 3u] = theta[OFF_B2 + k]; }
      }
      var vv = theta[OFF_BV];
      for (var i = 0u; i < HID; i++) {
        vv += pool4[P_WV + (i >> 2u)][i & 3u] * partials[row + HID + i];
        if (ok) {
          let hj = partials[row + i];
          for (var kq = 0u; kq < KQ; kq++) { pre4[kq] += hj * pool4[P_W2T + i * KQ + kq]; }
        }
      }
      let rb = (sSel[s] * ROLL + j) * REC;
      var logpNew = 0.0;
      var entropy = 0.0;
      for (var k = 0u; k < NOUT; k++) {
        let mu = rl_softsign(GAIN * pre4[k >> 2u][k & 3u]);
        partials[scr + MU_OFF + ob + k] = mu;
        let ls = theta[OFF_LS + k];
        let eps = (rRec[rb + R_ACTION + k] - mu) / exp(ls);
        logpNew += -0.5 * eps * eps - ls - HALF_LOG_2PI;
        entropy += ls + HALF_LOG_2PI + 0.5;
      }
      let logRatio = clamp(logpNew - rRec[rb + R_LOGP], -20.0, 20.0);
      let ratio = exp(logRatio);
      let adv = (rRec[rb + R_ADV] - advMean) * advInv;
      let unclipped = ratio * adv;
      let clipped = clamp(ratio, 1.0 - U.clip, 1.0 + U.clip) * adv;
      let pushes = unclipped <= clipped;
      let gw = select(0.0, -adv * ratio, pushes);
      sPol += -min(unclipped, clipped);
      sClip += select(1.0, 0.0, pushes);
      sKl += ratio - 1.0 - logRatio;
      sEnt += entropy;
      sN += 1.0;
      let err = vv - rRec[rb + R_RET];
      sVal += 0.5 * err * err;
      partials[scr + DV_OFF + s * ROLL + j] = U.valueCoef * err;
      partials[scr + GW_OFF + s * ROLL + j] = gw;
      for (var k = 0u; k < NOUT; k++) {
        let mu = partials[scr + MU_OFF + ob + k];
        let sigma = exp(theta[OFF_LS + k]);
        let eps = (rRec[rb + R_ACTION + k] - mu) / sigma;
        partials[scr + DP_OFF + ob + k] = gw * (eps / sigma) * rl_slope(mu);
      }
    }
    storageBarrier();
    workgroupBarrier();
    var dhN: array<vec4<f32>, NQ4>;
    for (var qv = 0u; qv < NQ4; qv++) { dhN[qv] = vec4<f32>(0.0); }
    for (var jj = 0u; jj <= ROLL - j0; jj++) {
      let j = ROLL - jj;
      let cur = (j & 1u) * HID * 4u;
      let nxt = ((j + 1u) & 1u) * HID * 4u;
      if (j < ROLL) {
        for (var idx = tid; idx < NOUT * 16u; idx += BT) {
          let k = idx / 16u;
          let s = idx - k * 16u;
          pool4[P_DPS + k * 4u + (s >> 2u)][s & 3u] = partials[scr + DP_OFF + (s * ROLL + j) * NOUT + k];
        }
      }
      for (var s = tid; s < 16u; s += BT) {
        var v = 0.0;
        if (j >= 1u) { v = partials[scr + DV_OFF + s * ROLL + j - 1u]; }
        pool4[P_DVV + (s >> 2u)][s & 3u] = v;
      }
      workgroupBarrier();
      if (hh < BH) {
        var lk = 0.0;
        if (RECUR == 1u) { lk = theta[OFF_LEAK + ii]; }
        for (var qv = 0u; qv < NQ4; qv++) {
          var d = theta[OFF_WV + ii] * pool4[P_DVV + hh * NQ4 + qv];
          if (j < ROLL) {
            var e = vec4<f32>(0.0);
            for (var k = 0u; k < NOUT; k++) { e += theta[OFF_W2 + k * HID + ii] * pool4[P_DPS + k * 4u + hh * NQ4 + qv]; }
            if (RECUR == 1u) { e += lk * dhN[qv]; }
            d += e * pool4[P_VALID + j * 4u + hh * NQ4 + qv];
          }
          dhN[qv] = d;
        }
        if (RECUR == 1u && j < ROLL) {
          var e2: array<vec4<f32>, NQ4>;
          for (var qv = 0u; qv < NQ4; qv++) { e2[qv] = vec4<f32>(0.0); }
          for (var m = 0u; m < HID; m++) {
            let w = theta[OFF_WR + ii * HID + m];
            for (var qv = 0u; qv < NQ4; qv++) { e2[qv] += w * pool4[P_PP + nxt + m * 4u + hh * NQ4 + qv]; }
          }
          for (var qv = 0u; qv < NQ4; qv++) { dhN[qv] += e2[qv] * pool4[P_VALID + j * 4u + hh * NQ4 + qv]; }
        }
        for (var qv = 0u; qv < NQ4; qv++) {
          var s4 = vec4<f32>(0.0);
          for (var l = 0u; l < 4u; l++) {
            let s = (hh * NQ4 + qv) * 4u + l;
            s4[l] = partials[scr + SB_OFF + (s * TE + j) * HID + ii];
          }
          let one = vec4<f32>(1.0) - abs(s4);
          var dz = dhN[qv] * (GAIN * one * one);
          if (RECUR == 1u) { dz = dz * (1.0 - lk); }
          pool4[P_PP + cur + ii * 4u + hh * NQ4 + qv] = dz;
          for (var l = 0u; l < 4u; l++) {
            let s = (hh * NQ4 + qv) * 4u + l;
            partials[scr + DZ_OFF + (s * TE + j) * HID + ii] = dz[l];
            partials[scr + DH_OFF + (s * TE + j) * HID + ii] = dhN[qv][l];
          }
        }
      }
      workgroupBarrier();
    }
    storageBarrier();
    workgroupBarrier();
${gemms.join('\n')}
    if (tid < HID) {
      var sb = 0.0;
      var sl = 0.0;
      for (var s = 0u; s < nsel; s++) {
        for (var j = j0; j <= ROLL; j++) {
          let q = scr + (s * TE + j) * HID + tid;
          sb += partials[q + DZ_OFF];
          if (RECUR == 1u && j >= 1u) {
            let heff = partials[scr + HB_OFF + (s * TE + j - 1u) * HID + tid] * pv(j - 1u, s);
            sl += partials[q + DH_OFF] * (heff - partials[q + SB_OFF]);
          }
        }
      }
      if (BIAS == 1u) { partials[gp + OFF_B1 + tid] += sb; }
      if (RECUR == 1u && LEARN_LEAK == 1u) { partials[gp + OFF_LEAK + tid] += sl; }
    }
    if (tid < NOUT) {
      let sigma = exp(theta[OFF_LS + tid]);
      var sls = 0.0;
      var sb2 = 0.0;
      for (var s = 0u; s < nsel; s++) {
        for (var j = 0u; j < ROLL; j++) {
          let ob = (s * ROLL + j) * NOUT + tid;
          let eps = (rRec[(sSel[s] * ROLL + j) * REC + R_ACTION + tid] - partials[scr + MU_OFF + ob]) / sigma;
          sls += partials[scr + GW_OFF + s * ROLL + j] * (eps * eps - 1.0) - U.entropy;
          sb2 += partials[scr + DP_OFF + ob];
        }
      }
      partials[gp + OFF_LS + tid] += sls;
      if (BIAS == 1u) { partials[gp + OFF_B2 + tid] += sb2; }
    }
    if (tid == NOUT) {
      var sv = 0.0;
      for (var s = 0u; s < nsel; s++) { for (var j = 0u; j < ROLL; j++) { sv += partials[scr + DV_OFF + s * ROLL + j]; } }
      partials[gp + OFF_BV] += sv;
    }
    storageBarrier();
    workgroupBarrier();
  }
  pool4[tid * 2u][0] = sPol;
  pool4[tid * 2u][1] = sVal;
  pool4[tid * 2u][2] = sEnt;
  pool4[tid * 2u][3] = sClip;
  pool4[tid * 2u + 1u][0] = sKl;
  pool4[tid * 2u + 1u][1] = sN;
  workgroupBarrier();
  if (tid < 6u) {
    var acc6 = 0.0;
    for (var t = 0u; t < BT; t++) { acc6 += pool4[t * 2u + (tid >> 2u)][tid & 3u]; }
    partials[gp + PCOUNT + tid] = acc6;
  }
}
`;
}

export function buildRlShader(game, cfg) {
  const S = rlShaderLayout(game, cfg);
  const P = S.params;
  const lit = (x) => (Number.isInteger(x) ? x + '.0' : String(x));
  const maxAge = game.maxAge;
  const trainAge = Math.max(1, Math.round(game.maxAge * cfg.lifeCapScale));
  const defaultSlots = /fn\s+g_learner_slots\s*\(/.test(game.wgsl) ? '' : 'fn g_learner_slots() -> u32 { return LEARNERS; }';
  const channels = /fn\s+g_reward_channel\s*\(/.test(game.wgsl) && Array.isArray(cfg.channelCaps) ? cfg.channelCaps : null;
  const channelBody = channels
    ? 'var total = 0;\n' + channels.map((cap, ch) => '  total += clamp(g_reward_channel(a, ' + ch + 'u), ' + -Math.round(cap * 1024) + ', ' + Math.round(cap * 1024) + ');').join('\n') + '\n  return total;'
    : 'return rewardBuf[a];';
  const F = S.features;
  const normFloor = lit(cfg.obsNormFloor === undefined ? 0.1 : cfg.obsNormFloor);
  const normCap = lit(Math.max(1, cfg.obsNormCap === undefined ? 8000000 : cfg.obsNormCap));
  return `
const NIN: u32 = ${S.nIn}u;
const NOUT: u32 = ${S.nOut}u;
const HID: u32 = ${S.hidden}u;
const OSTR: u32 = ${S.obsStride}u;
const HSS: u32 = ${S.hiddenStride}u;
const LEARNERS: u32 = ${S.learners}u;
const AGENTS: u32 = ${S.agents}u;
const WORLDS: u32 = ${S.worlds}u;
const EVAL_START: u32 = ${S.trainWorlds}u;
const EVAL_B_START: u32 = ${S.evalBStart}u;
const EVAL_C_START: u32 = ${S.evalCStart}u;
const TRAIN_LEARNERS: u32 = ${S.trainLearners}u;
const ROLL: u32 = ${S.rolloutTicks}u;
const QL: u32 = ${S.lanes}u;
const MAX_AGE: f32 = ${maxAge}.0;
const TRAIN_AGE: f32 = ${trainAge}.0;
const WSTRIDE: u32 = ${S.worldStride}u;
const STATS_OFF: u32 = 8u;
const ROLE_STATS_WORD: u32 = 3u;
const GAME_OFF: u32 = 24u;
const GAIN: f32 = ${lit(RL_GAIN)};
const BIAS: u32 = ${F.bias ? 1 : 0}u;
const RECUR: u32 = ${F.recurrent ? 1 : 0}u;
const NORM: u32 = ${F.obsNorm ? 1 : 0}u;
const OFF_B1: u32 = ${Math.max(0, P.b1)}u;
const OFF_B2: u32 = ${Math.max(0, P.b2)}u;
const OFF_WR: u32 = ${Math.max(0, P.wr)}u;
const OFF_LEAK: u32 = ${Math.max(0, P.leak)}u;
const OFF_NM: u32 = ${Math.max(0, P.normMean)}u;
const OFF_NI: u32 = ${Math.max(0, P.normIstd)}u;
const LEARN: u32 = ${P.learn}u;
const LEARN_LEAK: u32 = ${F.recurrent && cfg.learnLeak ? 1 : 0}u;
const LEAK_MAX: f32 = ${lit(RL_LEAK_MAX)};
const OBS_CLIP: f32 = ${lit(RL_OBS_CLIP)};
const OBS_LS: u32 = ${S.obsLearnerStride}u;
const OBS_HINIT: u32 = ${S.entriesPerLearner * S.nIn}u;
const HBUFS: u32 = ${S.hiddenBuffers}u;
const REWARD_CLIP: f32 = ${lit(cfg.rewardClip === undefined ? 1e9 : cfg.rewardClip)};
const VALUE_CLIP: f32 = ${lit(cfg.valueClip === undefined ? 1e9 : cfg.valueClip)};
const NORM_GROUPS: u32 = ${RL_NORM_GROUPS}u;
const NORM_SLICES: u32 = ${S.normSlices}u;
const NORM_PART: u32 = ${(S.gradGroups + 1) * S.partialStride}u;
const NORM_OPT: u32 = ${2 * P.count + RL_CTL.SIZE}u;
const NORM_FLOOR: f32 = ${normFloor};
const NORM_CAP: f32 = ${normCap};
const SCR_BASE: u32 = ${S.scratchBase}u;
const SCR_STRIDE: u32 = ${S.seqScratch}u;
const TE: u32 = ${S.entriesPerLearner}u;
const C_NCOUNT: u32 = ${RL_CTL.NCOUNT}u;
const HALF_LOG_2PI: f32 = 0.9189385332046727;
const OFF_W1: u32 = ${P.w1}u;
const OFF_W2: u32 = ${P.w2}u;
const OFF_LS: u32 = ${P.logStd}u;
const OFF_WV: u32 = ${P.valueW}u;
const OFF_BV: u32 = ${P.valueB}u;
const PCOUNT: u32 = ${P.count}u;
const PST: u32 = ${P.partialStride}u;
const REC: u32 = ${P.recStride}u;
const R_ACTION: u32 = 0u;
const R_LOGP: u32 = ${S.nOut}u;
const R_VALUE: u32 = ${S.nOut + 1}u;
const R_REWARD: u32 = ${S.nOut + 2}u;
const R_FLAG: u32 = ${S.nOut + 3}u;
const R_TRUNC: u32 = ${S.nOut + 4}u;
const R_PREV: u32 = ${S.nOut + 5}u;
const R_ADV: u32 = ${S.nOut + 6}u;
const R_RET: u32 = ${S.nOut + 7}u;
const LS: u32 = ${S.laneStride}u;
const LS_LIFE: u32 = 0u;
const LS_SCORE: u32 = 1u;
const LS_VALID: u32 = 2u;
const LS_BOOT: u32 = 3u;
const LS_ROLE: u32 = ${RL_LANE.ROLE}u;
const LS_BASE: u32 = ${RL_LANE.BASE}u;
const LS_H0: u32 = ${RL_LANE.HEADER}u;
const LS_XLAST: u32 = ${RL_LANE.HEADER + S.hiddenBuffers * S.hidden}u;
const LS_PART: u32 = ${RL_LANE.HEADER + S.hiddenBuffers * S.hidden + S.nIn}u;
const ROLE_LIVE: u32 = 1u;
const ROLE_SNAPSHOT: u32 = 2u;
const GROUPS: u32 = ${S.gradGroups}u;
const TL: u32 = ${S.tile}u;
const NT: u32 = ${S.tilesPerLearner}u;
const XS: u32 = ${S.obsStride}u;
const ACCN: u32 = ${S.accumulators}u;
const DENSE_EDGES: i32 = ${S.denseEdges};
const CTL: u32 = ${2 * P.count}u;
const C_STEP: u32 = ${RL_CTL.STEP}u;
const C_SCALE: u32 = ${RL_CTL.SCALE}u;
const C_ADV_MEAN: u32 = ${RL_CTL.ADV_MEAN}u;
const C_ADV_INV: u32 = ${RL_CTL.ADV_INV_STD}u;
const C_GRAD_NORM: u32 = ${RL_CTL.GRAD_NORM}u;
const C_POLICY: u32 = ${RL_CTL.POLICY}u;
const C_VALUE: u32 = ${RL_CTL.VALUE}u;
const C_ENTROPY: u32 = ${RL_CTL.ENTROPY}u;
const C_CLIP: u32 = ${RL_CTL.CLIP}u;
const C_KL: u32 = ${RL_CTL.KL}u;
const C_RET_STD: u32 = ${RL_CTL.RET_STD}u;
const C_ADV_STD: u32 = ${RL_CTL.ADV_STD}u;
const C_ITER: u32 = ${RL_CTL.ITER}u;
const C_STEPS: u32 = ${RL_CTL.STEPS}u;
const C_RET_MEAN: u32 = ${RL_CTL.RET_MEAN}u;
const C_ACTIVE: u32 = ${RL_CTL.ACTIVE}u;
const STATS_ROLE_OFF: u32 = ${RL_STATS.ROLE_OFFSET}u;
const STATS_WORLD_OFF: u32 = ${RL_STATS.WORLD_OFFSET}u;
const NEG_INF: f32 = -1e30;
const REWARD_SCALE: f32 = 1024.0;
const STAT_REW_EVO: u32 = 0u;
const STAT_REW_BASE: u32 = 1u;
const STAT_TICKS_EVO: u32 = 2u;
const STAT_TICKS_BASE: u32 = 3u;
const STAT_OPS_EVO: u32 = 4u;
const STAT_BIRTHS: u32 = 5u;
const STAT_DEATHS: u32 = 6u;
const STAT_LIFE_FIT: u32 = 7u;
const STAT_EDGES: u32 = 8u;
const STAT_GAME0: u32 = 9u;
const EVAL_RESET: u32 = 1u;

struct Uniforms {
  ticks: u32,
  tick0: u32,
  flags: u32,
  evalStart: u32,
  seedBase: u32,
  randomize: u32,
  difficulty: u32,
  minibatches: u32,
  gamma: f32,
  lambda: f32,
  clip: f32,
  entropy: f32,
  valueCoef: f32,
  lr: f32,
  maxGrad: f32,
  sigmaMin: f32,
  sigmaMax: f32,
  weightMax: f32,
  adaptScale: f32,
  leagueFraction: f32,
  slotFraction: f32,
  latestBias: f32,
  poolValid: u32,
  leagueEpoch: u32,
  refSnap: u32,
  leagueOn: u32,
  spOn: u32,
  spFraction: f32,
  pad2: u32,
  pad3: u32,
  pad4: u32,
  pad5: u32,
}

@group(0) @binding(0) var<storage, read_write> theta: array<f32>;
@group(0) @binding(1) var<storage, read_write> world: array<u32>;
@group(0) @binding(2) var<storage, read_write> rObs: array<f32>;
@group(0) @binding(3) var<storage, read_write> rRec: array<f32>;
@group(0) @binding(4) var<storage, read_write> lane: array<f32>;
@group(0) @binding(5) var<storage, read_write> partials: array<f32>;
@group(0) @binding(6) var<storage, read_write> opt: array<f32>;
@group(0) @binding(7) var<storage, read_write> statsOut: array<i32>;
@group(0) @binding(8) var<uniform> U: Uniforms;

var<workgroup> obsBuf: array<atomic<u32>, ${S.learners * S.obsStride}>;
var<workgroup> outBuf: array<f32, ${S.learners * S.nOut}>;
var<workgroup> rewardBuf: array<i32, ${S.agents}>;
var<workgroup> deadBuf: array<u32, ${S.agents}>;
var<workgroup> statAcc: array<atomic<i32>, 20>;
var<workgroup> deadCount: atomic<u32>;
var<workgroup> truncCount: atomic<u32>;
var<workgroup> ndShared: u32;
var<workgroup> ntShared: u32;
var<workgroup> lifeSh: array<f32, ${S.learners}>;
var<workgroup> scoreSh: array<f32, ${S.learners}>;
var<workgroup> validSh: array<u32, ${S.learners}>;
var<workgroup> endedSh: array<u32, ${S.learners}>;
var<workgroup> roleSh: array<u32, ${S.learners}>;
var<workgroup> baseSh: array<u32, ${S.learners}>;
var<workgroup> w_seed: u32;
var<workgroup> w_index: u32;
var<workgroup> w_eval: u32;

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

fn mix4(a: u32, b: u32, c: u32, d: u32) -> u32 {
  var h = pcg(a);
  h = pcg(h + b);
  h = pcg(h + c);
  h = pcg(h + d);
  return h;
}

fn rf(st: ptr<function, u32>) -> f32 {
  *st = pcg(*st);
  return f32(*st >> 8u) / 16777216.0;
}

fn rn(st: ptr<function, u32>) -> f32 {
  let a = rf(st);
  let b = rf(st);
  let c = rf(st);
  return (a + b + c - 1.5) * 2.0;
}

fn rl_unit(h: u32) -> f32 { return f32(h >> 8u) / 16777216.0; }

fn rl_gauss(st: ptr<function, u32>) -> f32 {
  let u1 = 1.0 - rf(st);
  let u2 = rf(st);
  return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

fn rl_softsign(x: f32) -> f32 { return x / (1.0 + abs(x)); }
fn rl_slope(h: f32) -> f32 { let s = 1.0 - abs(h); return GAIN * s * s; }

fn world_difficulty() -> i32 { return select(i32(U.difficulty), 100, w_eval == 1u); }
fn action_of(a: u32) -> u32 { return 0u; }
fn out_of(a: u32, k: u32) -> f32 { return outBuf[a * NOUT + k]; }
fn set_obs(a: u32, i: u32, v: f32) { atomicStore(&obsBuf[a * OSTR + i], bitcast<u32>(v)); }
fn obs_get(a: u32, i: u32) -> f32 { return bitcast<f32>(atomicLoad(&obsBuf[a * OSTR + i])); }
fn set_reward(a: u32, fx: i32) { rewardBuf[a] = fx; }
fn set_dead(a: u32) { deadBuf[a] = 1u; }
fn stat_add(idx: u32, v: i32) { atomicAdd(&statAcc[idx], v); }
fn game_word(wi: u32, i: u32) -> u32 { return world[wi * WSTRIDE + GAME_OFF + i]; }
fn game_set(wi: u32, i: u32, v: u32) { world[wi * WSTRIDE + GAME_OFF + i] = v; }
fn bf(v: u32) -> f32 { return bitcast<f32>(v); }
fn fb(v: f32) -> u32 { return bitcast<u32>(v); }

${game.wgsl}

${defaultSlots}

fn rl_load_header(wi: u32) {
  w_seed = world[wi * WSTRIDE];
  w_eval = select(0u, 1u, wi >= EVAL_START);
  w_index = wi;
}

fn rl_stat_flush(wi: u32) {
  for (var i = 0u; i < 16u; i++) {
    let idx = wi * WSTRIDE + STATS_OFF + i;
    if (i == STAT_EDGES) {
      world[idx] = bitcast<u32>(atomicLoad(&statAcc[i]));
    } else {
      world[idx] = bitcast<u32>(bitcast<i32>(world[idx]) + atomicLoad(&statAcc[i]));
    }
  }
  for (var i = 0u; i < 4u; i++) {
    let idx = wi * WSTRIDE + ROLE_STATS_WORD + i;
    world[idx] = bitcast<u32>(bitcast<i32>(world[idx]) + atomicLoad(&statAcc[16u + i]));
  }
}

fn rl_zero_hidden(lb: u32) {
  for (var i = 0u; i < HBUFS * HID; i++) { lane[lb + LS_H0 + i] = 0.0; }
}

fn hread(lb: u32, tick: u32) -> u32 {
  if (RECUR == 0u) { return lb + LS_H0; }
  return lb + LS_H0 + (tick & 1u) * HID;
}

fn hwrite(lb: u32, tick: u32) -> u32 {
  if (RECUR == 0u) { return lb + LS_H0; }
  return lb + LS_H0 + ((tick + 1u) & 1u) * HID;
}

fn rl_norm(tb: u32, j: u32, raw: f32) -> f32 {
  if (NORM == 0u) { return raw; }
  return clamp((raw - theta[tb + OFF_NM + j]) * theta[tb + OFF_NI + j], -OBS_CLIP, OBS_CLIP);
}

fn rl_reward_fx(a: u32) -> i32 {
  ${channelBody}
}

fn rl_value_of_shared_obs(l: u32, tb: u32, hcur: u32) -> f32 {
  var xr: array<f32, ${S.nIn}>;
  for (var j = 0u; j < NIN; j++) { xr[j] = rl_norm(tb, j, obs_get(l, j)); }
  var total = theta[tb + OFF_BV];
  for (var i = 0u; i < HID; i++) {
    var acc = 0.0;
    for (var j = 0u; j < NIN; j++) { acc += theta[tb + OFF_W1 + j * HID + i] * xr[j]; }
    if (BIAS == 1u) { acc += theta[tb + OFF_B1 + i]; }
    var hn = rl_softsign(GAIN * acc);
    if (RECUR == 1u) {
      for (var m = 0u; m < HID; m++) { acc += theta[tb + OFF_WR + m * HID + i] * lane[hcur + m]; }
      let lk = theta[tb + OFF_LEAK + i];
      hn = lk * lane[hcur + i] + (1.0 - lk) * rl_softsign(GAIN * acc);
    }
    total += theta[tb + OFF_WV + i] * hn;
  }
  return total;
}

fn rl_pick_snapshot(seed: u32, epoch: u32) -> u32 {
  let valid = U.poolValid;
  if (valid <= 1u) { return 0u; }
  if (rl_unit(mix4(seed, epoch, 73u, 5u)) < U.latestBias) { return 0u; }
  return 1u + mix4(seed, epoch, 74u, 5u) % (valid - 1u);
}

fn rl_role_of(wi: u32, l: u32, slots: u32) -> u32 {
  if (l >= slots) { return 0u; }
  if (wi < EVAL_START) {
    if (U.leagueOn == 1u && U.poolValid > 0u && rl_unit(mix4(w_seed, U.leagueEpoch, 71u, 5u)) < U.leagueFraction && rl_unit(mix4(w_seed, U.leagueEpoch, l, 72u)) < U.slotFraction) { return ROLE_SNAPSHOT; }
    return ROLE_LIVE;
  }
  if (wi >= EVAL_B_START && wi < EVAL_C_START && U.poolValid > 0u && ((l + wi) & 1u) == 1u) { return ROLE_SNAPSHOT; }
  return ROLE_LIVE;
}

fn rl_base_of(wi: u32, role: u32) -> u32 {
  if (role != ROLE_SNAPSHOT) { return 0u; }
  if (wi < EVAL_START) { return (1u + rl_pick_snapshot(w_seed, U.leagueEpoch)) * PCOUNT; }
  return (1u + min(U.refSnap, U.poolValid - 1u)) * PCOUNT;
}

fn rl_world_selfplay(wi: u32) -> bool {
  return U.spOn == 1u && wi < EVAL_START && rl_unit(mix4(U.seedBase, wi, 0u, 77u)) < U.spFraction;
}

fn rl_init_mode(wi: u32) -> u32 {
  if (wi >= EVAL_B_START || rl_world_selfplay(wi)) { return 2u; }
  if (U.spOn == 1u && wi < EVAL_START) { return 0u; }
  return U.randomize;
}

fn rl_reset_lanes(wi: u32, t: u32) {
  if (t < LEARNERS) {
    let lb = (wi * LEARNERS + t) * LS;
    var life = 0.0;
    if (w_eval == 0u) { life = f32(mix4(w_seed, t, 0u, 11u) % u32(TRAIN_AGE)); }
    lane[lb + LS_LIFE] = life;
    lane[lb + LS_SCORE] = 0.0;
    lane[lb + LS_VALID] = 0.0;
    lane[lb + LS_BOOT] = 0.0;
    lane[lb + LS_ROLE] = 0.0;
    rl_zero_hidden(lb);
    for (var j = 0u; j < NIN; j++) { lane[lb + LS_XLAST + j] = 0.0; }
  }
}

var<workgroup> spCur: u32;

@compute @workgroup_size(64)
fn rl_init_world(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let wi = wg.x;
  if (t == 0u) {
    world[wi * WSTRIDE] = mix4(U.seedBase, wi, 0u, 9u);
    world[wi * WSTRIDE + 1u] = select(0u, 1u, wi >= EVAL_START);
    world[wi * WSTRIDE + 2u] = select(0u, 1u, rl_world_selfplay(wi));
    for (var i = 0u; i < 5u; i++) { world[wi * WSTRIDE + ROLE_STATS_WORD + i] = 0u; }
    for (var i = 0u; i < 16u; i++) { world[wi * WSTRIDE + STATS_OFF + i] = 0u; }
  }
  workgroupBarrier();
  rl_load_header(wi);
  workgroupBarrier();
  g_init(wi, w_seed, w_eval, rl_init_mode(wi), t);
  workgroupBarrier();
  rl_reset_lanes(wi, t);
}

@compute @workgroup_size(64)
fn rl_reinit_world(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let wi = wg.x;
  if (wi >= EVAL_START || U.spOn == 0u) { return; }
  let want = select(0u, 1u, rl_world_selfplay(wi));
  if (t == 0u) { spCur = world[wi * WSTRIDE + 2u]; }
  workgroupBarrier();
  if (workgroupUniformLoad(&spCur) == want) { return; }
  if (t == 0u) {
    world[wi * WSTRIDE] = mix4(U.seedBase, wi, U.tick0, 9u);
    world[wi * WSTRIDE + 2u] = want;
  }
  workgroupBarrier();
  rl_load_header(wi);
  workgroupBarrier();
  g_init(wi, w_seed, w_eval, rl_init_mode(wi), t);
  workgroupBarrier();
  rl_reset_lanes(wi, t);
}

@compute @workgroup_size(64)
fn rl_rollout(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let wi = wg.x;
  rl_load_header(wi);
  if (t < 20u) { atomicStore(&statAcc[t], 0); }
  if (t == 0u) { atomicStore(&deadCount, 0u); atomicStore(&truncCount, 0u); }
  workgroupBarrier();
  g_load(wi, t);
  let ln = t / QL;
  let sub = t % QL;
  let laneOk = ln < LEARNERS;
  let training = wi < EVAL_START;
  let kindB = wi >= EVAL_B_START && wi < EVAL_C_START;
  if (t < LEARNERS) {
    let lb = (wi * LEARNERS + t) * LS;
    lifeSh[t] = lane[lb + LS_LIFE];
    scoreSh[t] = lane[lb + LS_SCORE];
    validSh[t] = select(0u, 1u, lane[lb + LS_VALID] > 0.5);
    endedSh[t] = 0u;
  }
  workgroupBarrier();
  if ((U.flags & EVAL_RESET) != 0u && wi >= EVAL_START) {
    if (t == 0u) { for (var a = 0u; a < AGENTS; a++) { g_respawn(a, U.tick0); } }
    workgroupBarrier();
    if (t < LEARNERS) {
      lifeSh[t] = 0.0;
      scoreSh[t] = 0.0;
      validSh[t] = 0u;
      rl_zero_hidden((wi * LEARNERS + t) * LS);
    }
    workgroupBarrier();
  }
  if (t < LEARNERS) {
    let role = rl_role_of(wi, t, g_learner_slots());
    roleSh[t] = role;
    baseSh[t] = rl_base_of(wi, role);
    lane[(wi * LEARNERS + t) * LS + LS_ROLE] = f32(role);
    lane[(wi * LEARNERS + t) * LS + LS_BASE] = f32(baseSh[t] / PCOUNT);
  }
  workgroupBarrier();
  if (laneOk && roleSh[ln] != 0u) {
    let b = wi * LEARNERS + ln;
    let lb = b * LS;
    let tb = baseSh[ln];
    if (RECUR == 0u && validSh[ln] == 1u) {
      for (var i = sub; i < HID; i += QL) {
        var acc = 0.0;
        for (var j = 0u; j < NIN; j++) { acc += theta[tb + OFF_W1 + j * HID + i] * lane[lb + LS_XLAST + j]; }
        if (BIAS == 1u) { acc += theta[tb + OFF_B1 + i]; }
        lane[lb + LS_H0 + i] = rl_softsign(GAIN * acc);
      }
    }
    if (training) {
      for (var j = sub; j < NIN; j += QL) { rObs[b * OBS_LS + j] = lane[lb + LS_XLAST + j]; }
      if (RECUR == 1u) {
        let hb = hread(lb, U.tick0);
        for (var i = sub; i < HID; i += QL) { rObs[b * OBS_LS + OBS_HINIT + i] = lane[hb + i]; }
      }
    }
  }
  workgroupBarrier();
  for (var k = 0u; k < ROLL; k++) {
    let tick = U.tick0 + k;
    if (t < AGENTS) { deadBuf[t] = 0u; rewardBuf[t] = 0; }
    workgroupBarrier();
    g_observe(t, tick);
    workgroupBarrier();
    if (laneOk && roleSh[ln] != 0u) {
      let b = wi * LEARNERS + ln;
      let lb = b * LS;
      let tb = baseSh[ln];
      var xr: array<f32, ${S.nIn}>;
      for (var j = 0u; j < NIN; j++) { xr[j] = rl_norm(tb, j, obs_get(ln, j)); }
      var p2: array<f32, ${S.nOut}>;
      for (var kk = 0u; kk < NOUT; kk++) { p2[kk] = 0.0; }
      var pv = 0.0;
      let hold = hread(lb, tick);
      let hnew = hwrite(lb, tick);
      for (var i = sub; i < HID; i += QL) {
        var acc = 0.0;
        for (var j = 0u; j < NIN; j++) { acc += theta[tb + OFF_W1 + j * HID + i] * xr[j]; }
        if (BIAS == 1u) { acc += theta[tb + OFF_B1 + i]; }
        if (RECUR == 1u) {
          for (var m = 0u; m < HID; m++) { acc += theta[tb + OFF_WR + m * HID + i] * lane[hold + m]; }
        }
        let hp = lane[hold + i];
        var hn = rl_softsign(GAIN * acc);
        if (RECUR == 1u) {
          let lk = theta[tb + OFF_LEAK + i];
          hn = lk * hp + (1.0 - lk) * hn;
        }
        for (var kk = 0u; kk < NOUT; kk++) { p2[kk] += theta[tb + OFF_W2 + kk * HID + i] * hp; }
        lane[hnew + i] = hn;
        pv += theta[tb + OFF_WV + i] * hn;
      }
      let pb = lb + LS_PART + sub * (NOUT + 1u);
      for (var kk = 0u; kk < NOUT; kk++) { lane[pb + kk] = p2[kk]; }
      lane[pb + NOUT] = pv;
      if (training) {
        let ob = b * OBS_LS + (k + 1u) * NIN;
        for (var j = sub; j < NIN; j += QL) { rObs[ob + j] = xr[j]; }
      }
      if (k + 1u == ROLL) {
        for (var j = sub; j < NIN; j += QL) { lane[lb + LS_XLAST + j] = xr[j]; }
      }
    }
    storageBarrier();
    workgroupBarrier();
    if (laneOk && sub == 0u && roleSh[ln] != 0u) {
      let b = wi * LEARNERS + ln;
      let lb = b * LS;
      let tb = baseSh[ln];
      var pre2: array<f32, ${S.nOut}>;
      for (var kk = 0u; kk < NOUT; kk++) { pre2[kk] = 0.0; }
      var vsum = theta[tb + OFF_BV];
      for (var s = 0u; s < QL; s++) {
        let pb = lb + LS_PART + s * (NOUT + 1u);
        for (var kk = 0u; kk < NOUT; kk++) { pre2[kk] += lane[pb + kk]; }
        vsum += lane[pb + NOUT];
      }
      if (BIAS == 1u) { for (var kk = 0u; kk < NOUT; kk++) { pre2[kk] += theta[tb + OFF_B2 + kk]; } }
      var st = mix4(w_seed, tick, ln, 100u);
      var logp = 0.0;
      let rb = (b * ROLL + k) * REC;
      for (var kk = 0u; kk < NOUT; kk++) {
        let mu = rl_softsign(GAIN * pre2[kk]);
        let ls = theta[tb + OFF_LS + kk];
        var a = mu;
        var eps = 0.0;
        if (w_eval == 0u) {
          eps = rl_gauss(&st);
          a = mu + exp(ls) * eps;
        }
        logp += -0.5 * eps * eps - ls - HALF_LOG_2PI;
        outBuf[ln * NOUT + kk] = clamp(a, -1.0, 1.0);
        if (training) { rRec[rb + R_ACTION + kk] = a; }
      }
      if (training) {
        rRec[rb + R_LOGP] = logp;
        rRec[rb + R_VALUE] = vsum;
        rRec[rb + R_PREV] = f32(validSh[ln]);
      }
      validSh[ln] = 1u;
    }
    workgroupBarrier();
    g_step(t, tick);
    workgroupBarrier();
    if (t < LEARNERS) {
      let role = roleSh[t];
      var flag = 0u;
      if (role != 0u) {
        let b = wi * LEARNERS + t;
        let rewardFx = rl_reward_fx(t);
        let life = lifeSh[t] + 1.0;
        let score = scoreSh[t] + f32(rewardFx) / REWARD_SCALE;
        let counted = role == ROLE_LIVE || kindB;
        if (counted) {
          atomicAdd(&statAcc[STAT_REW_EVO], rewardFx);
          atomicAdd(&statAcc[STAT_TICKS_EVO], 1);
          atomicAdd(&statAcc[STAT_OPS_EVO], DENSE_EDGES);
        }
        if (kindB && U.poolValid > 0u) {
          let slot = select(18u, 16u, role == ROLE_LIVE);
          atomicAdd(&statAcc[slot], rewardFx);
          atomicAdd(&statAcc[slot + 1u], 1);
        }
        let cap = select(TRAIN_AGE, MAX_AGE, w_eval == 1u);
        if (deadBuf[t] == 1u) { flag = 1u; } else if (life >= cap) { flag = 2u; }
        if (training) {
          let rb = (b * ROLL + k) * REC;
          rRec[rb + R_REWARD] = f32(rewardFx) / REWARD_SCALE;
          rRec[rb + R_FLAG] = f32(flag);
          rRec[rb + R_TRUNC] = 0.0;
        }
        if (flag != 0u) {
          atomicAdd(&deadCount, 1u);
          if (counted) {
            atomicAdd(&statAcc[STAT_DEATHS], 1);
            atomicAdd(&statAcc[STAT_BIRTHS], 1);
            atomicAdd(&statAcc[STAT_LIFE_FIT], i32(floor(score * REWARD_SCALE)));
          }
          if (flag == 2u && training && role == ROLE_LIVE) { atomicAdd(&truncCount, 1u); }
        }
        lifeSh[t] = life;
        scoreSh[t] = score;
      }
      endedSh[t] = flag;
    }
    workgroupBarrier();
    if (t == 0u) {
      ndShared = atomicLoad(&deadCount);
      ntShared = atomicLoad(&truncCount);
      atomicStore(&deadCount, 0u);
      atomicStore(&truncCount, 0u);
    }
    workgroupBarrier();
    let nd = workgroupUniformLoad(&ndShared);
    let nt = workgroupUniformLoad(&ntShared);
    if (nt > 0u) {
      g_observe(t, tick + 1u);
      workgroupBarrier();
      if (t < LEARNERS && endedSh[t] == 2u && roleSh[t] == ROLE_LIVE) {
        rRec[((wi * LEARNERS + t) * ROLL + k) * REC + R_TRUNC] = rl_value_of_shared_obs(t, baseSh[t], hwrite((wi * LEARNERS + t) * LS, tick));
      }
      workgroupBarrier();
    }
    if (nd > 0u) {
      if (t == 0u) {
        for (var a = 0u; a < LEARNERS; a++) { if (endedSh[a] != 0u) { g_respawn(a, tick); } }
      }
      workgroupBarrier();
      if (t < LEARNERS && endedSh[t] != 0u) {
        lifeSh[t] = 0.0;
        scoreSh[t] = 0.0;
        validSh[t] = 0u;
        rl_zero_hidden((wi * LEARNERS + t) * LS);
      }
      workgroupBarrier();
    }
  }
  if (t < LEARNERS) {
    let lb = (wi * LEARNERS + t) * LS;
    lane[lb + LS_LIFE] = lifeSh[t];
    lane[lb + LS_SCORE] = scoreSh[t];
    lane[lb + LS_VALID] = f32(validSh[t]);
  }
  if (training) {
    g_observe(t, U.tick0 + ROLL);
    workgroupBarrier();
    if (laneOk && roleSh[ln] == ROLE_LIVE) {
      let lb = (wi * LEARNERS + ln) * LS;
      var pv = 0.0;
      var xb: array<f32, ${S.nIn}>;
      for (var j = 0u; j < NIN; j++) { xb[j] = rl_norm(0u, j, obs_get(ln, j)); }
      let hcur = hread(lb, U.tick0 + ROLL);
      for (var i = sub; i < HID; i += QL) {
        var acc = 0.0;
        for (var j = 0u; j < NIN; j++) { acc += theta[OFF_W1 + j * HID + i] * xb[j]; }
        if (BIAS == 1u) { acc += theta[OFF_B1 + i]; }
        var hn = rl_softsign(GAIN * acc);
        if (RECUR == 1u) {
          for (var m = 0u; m < HID; m++) { acc += theta[OFF_WR + m * HID + i] * lane[hcur + m]; }
          let lk = theta[OFF_LEAK + i];
          hn = lk * lane[hcur + i] + (1.0 - lk) * rl_softsign(GAIN * acc);
        }
        pv += theta[OFF_WV + i] * hn;
      }
      lane[lb + LS_PART + sub * (NOUT + 1u) + NOUT] = pv;
    }
    storageBarrier();
    workgroupBarrier();
    if (laneOk && sub == 0u && roleSh[ln] == ROLE_LIVE) {
      let lb = (wi * LEARNERS + ln) * LS;
      var vsum = theta[OFF_BV];
      for (var s = 0u; s < QL; s++) { vsum += lane[lb + LS_PART + s * (NOUT + 1u) + NOUT]; }
      lane[lb + LS_BOOT] = vsum;
    }
  }
  g_save(wi, t);
  if (t == 0u) { atomicStore(&statAcc[STAT_EDGES], i32(LEARNERS) * DENSE_EDGES); }
  workgroupBarrier();
  if (t == 0u) { rl_stat_flush(wi); }
}

@compute @workgroup_size(64)
fn rl_gae(@builtin(global_invocation_id) gid: vec3<u32>) {
  let b = gid.x;
  if (b >= TRAIN_LEARNERS) { return; }
  if (lane[b * LS + LS_ROLE] != 1.0) { return; }
  let scale = opt[CTL + C_SCALE];
  var carry = 0.0;
  for (var tt = ROLL; tt > 0u; tt--) {
    let t = tt - 1u;
    let rb = (b * ROLL + t) * REC;
    let r = clamp(rRec[rb + R_REWARD] * scale, -REWARD_CLIP, REWARD_CLIP);
    let flag = u32(rRec[rb + R_FLAG] + 0.5);
    let v = rRec[rb + R_VALUE];
    var next = 0.0;
    if (t == ROLL - 1u) { next = lane[b * LS + LS_BOOT]; } else { next = rRec[rb + REC + R_VALUE]; }
    var ends = false;
    if (flag == 1u) {
      next = 0.0;
      ends = true;
    } else if (flag == 2u) {
      next = rRec[rb + R_TRUNC];
      ends = true;
    }
    let delta = r + U.gamma * next - v;
    if (ends) { carry = delta; } else { carry = delta + U.gamma * U.lambda * carry; }
    rRec[rb + R_ADV] = carry;
    rRec[rb + R_RET] = clamp(carry + v, -VALUE_CLIP, VALUE_CLIP);
  }
}

var<workgroup> redA: array<f32, 256>;
var<workgroup> redB: array<f32, 256>;
var<workgroup> redC: array<f32, 256>;
var<workgroup> redD: array<f32, 256>;
var<workgroup> redE: array<f32, 256>;

@compute @workgroup_size(256)
fn rl_advstats(@builtin(local_invocation_index) tid: u32) {
  let n = TRAIN_LEARNERS * ROLL;
  var sa = 0.0;
  var sa2 = 0.0;
  var sr = 0.0;
  var sr2 = 0.0;
  var sc = 0.0;
  for (var s = tid; s < n; s += 256u) {
    if (lane[(s / ROLL) * LS + LS_ROLE] != 1.0) { continue; }
    let adv = rRec[s * REC + R_ADV];
    let ret = rRec[s * REC + R_RET];
    sa += adv;
    sa2 += adv * adv;
    sr += ret;
    sr2 += ret * ret;
    sc += 1.0;
  }
  redA[tid] = sa;
  redB[tid] = sa2;
  redC[tid] = sr;
  redD[tid] = sr2;
  redE[tid] = sc;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
    if (tid < stride) {
      redA[tid] += redA[tid + stride];
      redB[tid] += redB[tid + stride];
      redC[tid] += redC[tid + stride];
      redD[tid] += redD[tid + stride];
      redE[tid] += redE[tid + stride];
    }
    workgroupBarrier();
  }
  if (tid == 0u) {
    let count = max(redE[0], 1.0);
    let meanA = redA[0] / count;
    let varA = max(redB[0] / count - meanA * meanA, 0.0);
    let meanR = redC[0] / count;
    let varR = max(redD[0] / count - meanR * meanR, 0.0);
    opt[CTL + C_ADV_MEAN] = meanA;
    opt[CTL + C_ADV_INV] = 1.0 / sqrt(varA + 1e-8);
    opt[CTL + C_ADV_STD] = sqrt(varA);
    opt[CTL + C_RET_MEAN] = meanR;
    opt[CTL + C_RET_STD] = sqrt(varR);
    opt[CTL + C_ACTIVE] = redE[0];
    opt[CTL + C_ITER] = opt[CTL + C_ITER] + 1.0;
    opt[CTL + C_STEPS] = 0.0;
    for (var c = C_GRAD_NORM; c <= C_KL; c++) { opt[CTL + c] = 0.0; }
  }
}

var<workgroup> xs: array<f32, ${S.tile * S.obsStride}>;
var<workgroup> hs: array<f32, ${S.tile * S.hiddenStride}>;
var<workgroup> dzs: array<f32, ${S.tile * S.hiddenStride}>;
var<workgroup> mus: array<f32, ${S.tile * S.nOut}>;
var<workgroup> dpres: array<f32, ${S.tile * S.nOut}>;
var<workgroup> dlss: array<f32, ${S.tile * S.nOut}>;
var<workgroup> vmS: array<f32, ${S.tile}>;
var<workgroup> gwS: array<f32, ${S.tile}>;
var<workgroup> dvS: array<f32, ${S.tile}>;
var<workgroup> statRed: array<f32, ${S.tile * 8}>;
var<workgroup> stepSh: u32;
var<workgroup> skipSh: u32;

@compute @workgroup_size(256)
fn rl_grad(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let g = wg.x;
  if (tid == 0u) { stepSh = u32(opt[CTL + C_STEP]); }
  workgroupBarrier();
  let step = workgroupUniformLoad(&stepSh);
  let mb = step % U.minibatches;
  let salt = step / U.minibatches;
  let advMean = opt[CTL + C_ADV_MEAN];
  let advInv = opt[CTL + C_ADV_INV];
  var acc: array<f32, ${S.accumulators}>;
  for (var m = 0u; m < ACCN; m++) { acc[m] = 0.0; }
  var sPol = 0.0;
  var sVal = 0.0;
  var sEnt = 0.0;
  var sClip = 0.0;
  var sKl = 0.0;
  var sN = 0.0;
  let units = TRAIN_LEARNERS * NT;
  for (var u = g; u < units; u += GROUPS) {
    let b = u / NT;
    let tile = u % NT;
    if (tid == 0u) { skipSh = select(0u, 1u, mix4(salt, b, 7u, 9u) % U.minibatches != mb || lane[b * LS + LS_ROLE] != 1.0); }
    workgroupBarrier();
    if (workgroupUniformLoad(&skipSh) == 1u) { continue; }
    let j0 = tile * TL;
    let cnt = min(TL, ROLL + 1u - j0);
    let obase = b * OBS_LS + j0 * NIN;
    for (var idx = tid; idx < cnt * NIN; idx += 256u) {
      let e = idx / NIN;
      xs[e * XS + (idx - e * NIN)] = rObs[obase + idx];
    }
    for (var idx = tid; idx < cnt * HID; idx += 256u) {
      let e = idx / HID;
      dzs[e * HSS + (idx - e * HID)] = 0.0;
    }
    workgroupBarrier();
    for (var idx = tid; idx < cnt * HID; idx += 256u) {
      let e = idx / HID;
      let i = idx - e * HID;
      var pre = 0.0;
      for (var j = 0u; j < NIN; j++) { pre += theta[OFF_W1 + j * HID + i] * xs[e * XS + j]; }
      if (BIAS == 1u) { pre += theta[OFF_B1 + i]; }
      hs[e * HSS + i] = rl_softsign(GAIN * pre);
    }
    workgroupBarrier();
    if (tid < cnt) {
      let j = j0 + tid;
      vmS[tid] = select(0.0, 1.0, j < ROLL && rRec[(b * ROLL + j) * REC + R_PREV] > 0.5);
    }
    workgroupBarrier();
    for (var idx = tid; idx < cnt * NOUT; idx += 256u) {
      let e = idx / NOUT;
      let k = idx - e * NOUT;
      let j = j0 + e;
      var mu = 0.0;
      if (j < ROLL) {
        var pre = 0.0;
        if (vmS[e] > 0.5) {
          for (var i = 0u; i < HID; i++) { pre += theta[OFF_W2 + k * HID + i] * hs[e * HSS + i]; }
        }
        if (BIAS == 1u) { pre += theta[OFF_B2 + k]; }
        mu = rl_softsign(GAIN * pre);
      }
      mus[e * NOUT + k] = mu;
    }
    workgroupBarrier();
    if (tid < cnt) {
      let e = tid;
      let j = j0 + e;
      var gw = 0.0;
      var dv = 0.0;
      if (j < ROLL) {
        let rb = (b * ROLL + j) * REC;
        var logpNew = 0.0;
        var entropy = 0.0;
        for (var k = 0u; k < NOUT; k++) {
          let ls = theta[OFF_LS + k];
          let eps = (rRec[rb + R_ACTION + k] - mus[e * NOUT + k]) / exp(ls);
          logpNew += -0.5 * eps * eps - ls - HALF_LOG_2PI;
          entropy += ls + HALF_LOG_2PI + 0.5;
        }
        let logRatio = clamp(logpNew - rRec[rb + R_LOGP], -20.0, 20.0);
        let ratio = exp(logRatio);
        let adv = (rRec[rb + R_ADV] - advMean) * advInv;
        let unclipped = ratio * adv;
        let clipped = clamp(ratio, 1.0 - U.clip, 1.0 + U.clip) * adv;
        let pushes = unclipped <= clipped;
        gw = select(0.0, -adv * ratio, pushes);
        sPol += -min(unclipped, clipped);
        sClip += select(1.0, 0.0, pushes);
        sKl += ratio - 1.0 - logRatio;
        sEnt += entropy;
        sN += 1.0;
      }
      if (j >= 1u) {
        var v = theta[OFF_BV];
        for (var i = 0u; i < HID; i++) { v += theta[OFF_WV + i] * hs[e * HSS + i]; }
        let err = v - rRec[(b * ROLL + (j - 1u)) * REC + R_RET];
        sVal += 0.5 * err * err;
        dv = U.valueCoef * err;
      }
      gwS[e] = gw;
      dvS[e] = dv;
    }
    workgroupBarrier();
    for (var idx = tid; idx < cnt * NOUT; idx += 256u) {
      let e = idx / NOUT;
      let k = idx - e * NOUT;
      let j = j0 + e;
      var dpre = 0.0;
      var dls = 0.0;
      if (j < ROLL) {
        let rb = (b * ROLL + j) * REC;
        let mu = mus[e * NOUT + k];
        let ls = theta[OFF_LS + k];
        let sigma = exp(ls);
        let eps = (rRec[rb + R_ACTION + k] - mu) / sigma;
        let gw = gwS[e];
        dls = gw * (eps * eps - 1.0) - U.entropy;
        dpre = gw * (eps / sigma) * rl_slope(mu);
      }
      dpres[e * NOUT + k] = dpre;
      dlss[e * NOUT + k] = dls;
    }
    workgroupBarrier();
    for (var idx = tid; idx < cnt * HID; idx += 256u) {
      let e = idx / HID;
      let i = idx - e * HID;
      let j = j0 + e;
      var d = dvS[e] * theta[OFF_WV + i];
      if (j < ROLL && vmS[e] > 0.5) {
        for (var k = 0u; k < NOUT; k++) { d += dpres[e * NOUT + k] * theta[OFF_W2 + k * HID + i]; }
      }
      dzs[e * HSS + i] = d * rl_slope(hs[e * HSS + i]);
    }
    workgroupBarrier();
    for (var m = 0u; m < ACCN; m++) {
      let p = tid + 256u * m;
      if (p >= LEARN) { break; }
      var s = 0.0;
      if (p < OFF_W2) {
        let jj = p / HID;
        let ii = p - jj * HID;
        for (var e = 0u; e < cnt; e++) { s += dzs[e * HSS + ii] * xs[e * XS + jj]; }
      } else if (p < OFF_LS) {
        let q = p - OFF_W2;
        let kk = q / HID;
        let ii = q - kk * HID;
        for (var e = 0u; e < cnt; e++) { s += dpres[e * NOUT + kk] * hs[e * HSS + ii] * vmS[e]; }
      } else if (p < OFF_WV) {
        let kk = p - OFF_LS;
        for (var e = 0u; e < cnt; e++) { s += dlss[e * NOUT + kk]; }
      } else if (p < OFF_BV) {
        let ii = p - OFF_WV;
        for (var e = 0u; e < cnt; e++) { s += dvS[e] * hs[e * HSS + ii]; }
      } else if (p == OFF_BV) {
        for (var e = 0u; e < cnt; e++) { s += dvS[e]; }
      } else if (p < OFF_B2) {
        let ii = p - OFF_B1;
        for (var e = 0u; e < cnt; e++) { s += dzs[e * HSS + ii]; }
      } else {
        let kk = p - OFF_B2;
        for (var e = 0u; e < cnt; e++) { s += dpres[e * NOUT + kk]; }
      }
      acc[m] += s;
    }
    workgroupBarrier();
  }
  for (var m = 0u; m < ACCN; m++) {
    let p = tid + 256u * m;
    if (p < LEARN) { partials[g * PST + p] = acc[m]; }
  }
  if (tid < TL) {
    statRed[tid * 8u] = sPol;
    statRed[tid * 8u + 1u] = sVal;
    statRed[tid * 8u + 2u] = sEnt;
    statRed[tid * 8u + 3u] = sClip;
    statRed[tid * 8u + 4u] = sKl;
    statRed[tid * 8u + 5u] = sN;
  }
  workgroupBarrier();
  if (tid < 6u) {
    var total = 0.0;
    for (var e = 0u; e < TL; e++) { total += statRed[e * 8u + tid]; }
    partials[g * PST + PCOUNT + tid] = total;
  }
}

var<workgroup> sqX: array<f32, ${2 * S.obsStride}>;
var<workgroup> sqHp: array<f32, ${2 * S.hiddenStride}>;
var<workgroup> sqS: array<f32, ${2 * S.hiddenStride}>;
var<workgroup> sqDz: array<f32, ${2 * S.hiddenStride}>;
var<workgroup> sqDh: array<f32, ${2 * S.hiddenStride}>;
var<workgroup> sqMu: array<f32, ${S.rolloutTicks * S.nOut}>;
var<workgroup> sqDp: array<f32, ${S.rolloutTicks * S.nOut}>;
var<workgroup> sqGw: array<f32, ${S.rolloutTicks + 1}>;
var<workgroup> sqDv: array<f32, ${S.rolloutTicks + 1}>;
var<workgroup> sqValid: array<f32, ${S.rolloutTicks + 1}>;
var<workgroup> sqStat: array<f32, ${S.rolloutTicks * 8}>;

@compute @workgroup_size(256)
fn rl_grad_seq(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let g = wg.x;
  if (tid == 0u) { stepSh = u32(opt[CTL + C_STEP]); }
  workgroupBarrier();
  let step = workgroupUniformLoad(&stepSh);
  let mb = step % U.minibatches;
  let salt = step / U.minibatches;
  let advMean = opt[CTL + C_ADV_MEAN];
  let advInv = opt[CTL + C_ADV_INV];
  var acc: array<f32, ${S.accumulators}>;
  for (var m = 0u; m < ACCN; m++) { acc[m] = 0.0; }
  var sPol = 0.0;
  var sVal = 0.0;
  var sEnt = 0.0;
  var sClip = 0.0;
  var sKl = 0.0;
  var sN = 0.0;
  let scr = SCR_BASE + g * SCR_STRIDE;
  for (var b = g; b < TRAIN_LEARNERS; b += GROUPS) {
    if (tid == 0u) { skipSh = select(0u, 1u, mix4(salt, b, 7u, 9u) % U.minibatches != mb || lane[b * LS + LS_ROLE] != 1.0); }
    workgroupBarrier();
    if (workgroupUniformLoad(&skipSh) == 1u) { continue; }
    let ob = b * OBS_LS;
    if (tid < ROLL) { sqValid[tid] = select(0.0, 1.0, rRec[(b * ROLL + tid) * REC + R_PREV] > 0.5); }
    if (tid == ROLL) { sqValid[ROLL] = 1.0; }
    if (tid < HID) {
      let h0 = rObs[ob + OBS_HINIT + tid];
      partials[scr + tid] = h0;
      sqHp[tid] = h0 * select(0.0, 1.0, rRec[(b * ROLL) * REC + R_PREV] > 0.5);
    }
    for (var idx = tid; idx < ROLL * HID; idx += 256u) {
      let jm = idx / HID;
      let i = idx - jm * HID;
      let xo = ob + (jm + 1u) * NIN;
      var pre = 0.0;
      for (var c = 0u; c < NIN; c++) { pre += theta[OFF_W1 + c * HID + i] * rObs[xo + c]; }
      if (BIAS == 1u) { pre += theta[OFF_B1 + i]; }
      partials[scr + (TE + 1u + jm) * HID + i] = pre;
    }
    storageBarrier();
    workgroupBarrier();
    for (var j = 1u; j <= ROLL; j++) {
      let cur = (j & 1u) * HSS;
      let prv = ((j + 1u) & 1u) * HSS;
      if (tid < HID) {
        var pre = partials[scr + (TE + j) * HID + tid];
        for (var mm = 0u; mm < HID; mm++) { pre += theta[OFF_WR + mm * HID + tid] * sqHp[prv + mm]; }
        let s = rl_softsign(GAIN * pre);
        let lk = theta[OFF_LEAK + tid];
        let hn = lk * sqHp[prv + tid] + (1.0 - lk) * s;
        partials[scr + (TE + j) * HID + tid] = s;
        partials[scr + j * HID + tid] = hn;
        sqHp[cur + tid] = hn * sqValid[j];
      }
      workgroupBarrier();
    }
    storageBarrier();
    workgroupBarrier();
    for (var idx = tid; idx < ROLL * NOUT; idx += 256u) {
      let j = idx / NOUT;
      let k = idx - j * NOUT;
      var pre = 0.0;
      if (sqValid[j] > 0.5) {
        for (var i = 0u; i < HID; i++) { pre += theta[OFF_W2 + k * HID + i] * partials[scr + j * HID + i]; }
      }
      if (BIAS == 1u) { pre += theta[OFF_B2 + k]; }
      sqMu[idx] = rl_softsign(GAIN * pre);
    }
    workgroupBarrier();
    if (tid < ROLL) {
      let j = tid;
      let rb = (b * ROLL + j) * REC;
      var logpNew = 0.0;
      var entropy = 0.0;
      for (var k = 0u; k < NOUT; k++) {
        let ls = theta[OFF_LS + k];
        let eps = (rRec[rb + R_ACTION + k] - sqMu[j * NOUT + k]) / exp(ls);
        logpNew += -0.5 * eps * eps - ls - HALF_LOG_2PI;
        entropy += ls + HALF_LOG_2PI + 0.5;
      }
      let logRatio = clamp(logpNew - rRec[rb + R_LOGP], -20.0, 20.0);
      let ratio = exp(logRatio);
      let adv = (rRec[rb + R_ADV] - advMean) * advInv;
      let unclipped = ratio * adv;
      let clipped = clamp(ratio, 1.0 - U.clip, 1.0 + U.clip) * adv;
      let pushes = unclipped <= clipped;
      sqGw[j] = select(0.0, -adv * ratio, pushes);
      sPol += -min(unclipped, clipped);
      sClip += select(1.0, 0.0, pushes);
      sKl += ratio - 1.0 - logRatio;
      sEnt += entropy;
      sN += 1.0;
      var v = theta[OFF_BV];
      for (var i = 0u; i < HID; i++) { v += theta[OFF_WV + i] * partials[scr + (j + 1u) * HID + i]; }
      let err = v - rRec[rb + R_RET];
      sVal += 0.5 * err * err;
      sqDv[j] = U.valueCoef * err;
    }
    workgroupBarrier();
    for (var idx = tid; idx < ROLL * NOUT; idx += 256u) {
      let j = idx / NOUT;
      let k = idx - j * NOUT;
      let mu = sqMu[idx];
      let sigma = exp(theta[OFF_LS + k]);
      let eps = (rRec[(b * ROLL + j) * REC + R_ACTION + k] - mu) / sigma;
      sqDp[idx] = sqGw[j] * (eps / sigma) * rl_slope(mu);
    }
    workgroupBarrier();
    for (var jj = 0u; jj < ROLL; jj++) {
      let j = ROLL - jj;
      let cur = (j & 1u) * HSS;
      let nxt = ((j + 1u) & 1u) * HSS;
      let curx = (j & 1u) * XS;
      if (tid < HID) {
        var d = sqDv[j - 1u] * theta[OFF_WV + tid];
        if (j < ROLL && sqValid[j] > 0.5) {
          var e = 0.0;
          for (var k = 0u; k < NOUT; k++) { e += sqDp[j * NOUT + k] * theta[OFF_W2 + k * HID + tid]; }
          e += theta[OFF_LEAK + tid] * sqDh[nxt + tid];
          for (var mm = 0u; mm < HID; mm++) { e += theta[OFF_WR + tid * HID + mm] * sqDz[nxt + mm]; }
          d += e;
        }
        let sv = partials[scr + (TE + j) * HID + tid];
        sqDh[cur + tid] = d;
        sqS[cur + tid] = sv;
        sqDz[cur + tid] = d * (1.0 - theta[OFF_LEAK + tid]) * rl_slope(sv);
        sqHp[cur + tid] = sqValid[j - 1u] * partials[scr + (j - 1u) * HID + tid];
      }
      for (var c = tid; c < NIN; c += 256u) { sqX[curx + c] = rObs[ob + j * NIN + c]; }
      workgroupBarrier();
      for (var m = 0u; m < ACCN; m++) {
        let p = tid + 256u * m;
        if (p >= LEARN) { break; }
        if (p < OFF_W2) {
          let c = p / HID;
          let i = p - c * HID;
          acc[m] += sqDz[cur + i] * sqX[curx + c];
        } else if (p >= OFF_WR) {
          if (p < OFF_LEAK) {
            let q = p - OFF_WR;
            let mm = q / HID;
            let i = q - mm * HID;
            acc[m] += sqHp[cur + mm] * sqDz[cur + i];
          } else if (LEARN_LEAK == 1u) {
            let i = p - OFF_LEAK;
            acc[m] += sqDh[cur + i] * (sqHp[cur + i] - sqS[cur + i]);
          }
        } else if (BIAS == 1u && p >= OFF_B1 && p < OFF_B2) {
          acc[m] += sqDz[cur + p - OFF_B1];
        }
      }
    }
    for (var m = 0u; m < ACCN; m++) {
      let p = tid + 256u * m;
      if (p >= LEARN) { break; }
      var s = 0.0;
      if (p >= OFF_W2 && p < OFF_LS) {
        let q = p - OFF_W2;
        let kk = q / HID;
        let ii = q - kk * HID;
        for (var j = 0u; j < ROLL; j++) { s += sqDp[j * NOUT + kk] * partials[scr + j * HID + ii] * sqValid[j]; }
      } else if (p >= OFF_LS && p < OFF_WV) {
        let kk = p - OFF_LS;
        let sigma = exp(theta[p]);
        for (var j = 0u; j < ROLL; j++) {
          let eps = (rRec[(b * ROLL + j) * REC + R_ACTION + kk] - sqMu[j * NOUT + kk]) / sigma;
          s += sqGw[j] * (eps * eps - 1.0) - U.entropy;
        }
      } else if (p >= OFF_WV && p < OFF_BV) {
        let ii = p - OFF_WV;
        for (var j = 0u; j < ROLL; j++) { s += sqDv[j] * partials[scr + (j + 1u) * HID + ii]; }
      } else if (p == OFF_BV) {
        for (var j = 0u; j < ROLL; j++) { s += sqDv[j]; }
      } else if (BIAS == 1u && p >= OFF_B2 && p < OFF_B2 + NOUT) {
        let kk = p - OFF_B2;
        for (var j = 0u; j < ROLL; j++) { s += sqDp[j * NOUT + kk]; }
      }
      acc[m] += s;
    }
  }
  for (var m = 0u; m < ACCN; m++) {
    let p = tid + 256u * m;
    if (p < LEARN) { partials[g * PST + p] = acc[m]; }
  }
  if (tid < ROLL) {
    sqStat[tid * 8u] = sPol;
    sqStat[tid * 8u + 1u] = sVal;
    sqStat[tid * 8u + 2u] = sEnt;
    sqStat[tid * 8u + 3u] = sClip;
    sqStat[tid * 8u + 4u] = sKl;
    sqStat[tid * 8u + 5u] = sN;
  }
  workgroupBarrier();
  if (tid < 6u) {
    var total = 0.0;
    for (var e = 0u; e < ROLL; e++) { total += sqStat[e * 8u + tid]; }
    partials[g * PST + PCOUNT + tid] = total;
  }
}

${cfg.blockGrad ? blockKernelSource(S, F) : ''}

@compute @workgroup_size(256)
fn rl_reduce_grad(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.x;
  if (p >= PST) { return; }
  var total = 0.0;
  for (var g = 0u; g < GROUPS; g++) { total += partials[g * PST + p]; }
  if (p < PCOUNT) { total = total / max(opt[CTL + C_ACTIVE] / f32(U.minibatches), 1.0); }
  partials[GROUPS * PST + p] = total;
}

@compute @workgroup_size(256)
fn rl_adam(@builtin(local_invocation_index) tid: u32) {
  let gbase = GROUPS * PST;
  var normSq = 0.0;
  for (var p = tid; p < LEARN; p += 256u) {
    if (RECUR == 1u && LEARN_LEAK == 0u && p >= OFF_LEAK) { continue; }
    let gp = partials[gbase + p];
    normSq += gp * gp;
  }
  redA[tid] = normSq;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
    if (tid < stride) { redA[tid] += redA[tid + stride]; }
    workgroupBarrier();
  }
  let norm = sqrt(redA[0]);
  let scale = min(1.0, U.maxGrad / (norm + 1e-6));
  let stepIndex = opt[CTL + C_STEP] + 1.0;
  let c1 = 1.0 - pow(0.9, stepIndex);
  let c2 = 1.0 - pow(0.999, stepIndex);
  let lsLo = log(U.sigmaMin);
  let lsHi = log(U.sigmaMax);
  for (var p = tid; p < LEARN; p += 256u) {
    if (RECUR == 1u && LEARN_LEAK == 0u && p >= OFF_LEAK) { continue; }
    let gp = partials[gbase + p] * scale;
    let m = 0.9 * opt[p] + 0.1 * gp;
    let v = 0.999 * opt[PCOUNT + p] + 0.001 * gp * gp;
    opt[p] = m;
    opt[PCOUNT + p] = v;
    var value = theta[p] - U.lr * (m / c1) / (sqrt(v / c2) + 1e-8);
    if (p >= OFF_LS && p < OFF_WV) { value = clamp(value, lsLo, lsHi); }
    else if (RECUR == 1u && p >= OFF_LEAK) { value = clamp(value, 0.0, LEAK_MAX); }
    else if (p < OFF_LS || p > OFF_BV) { value = clamp(value, -U.weightMax, U.weightMax); }
    theta[p] = value;
  }
  if (tid == 0u) {
    let count = max(partials[gbase + PCOUNT + 5u], 1.0);
    opt[CTL + C_STEP] = opt[CTL + C_STEP] + 1.0;
    opt[CTL + C_STEPS] = opt[CTL + C_STEPS] + 1.0;
    opt[CTL + C_GRAD_NORM] = opt[CTL + C_GRAD_NORM] + norm;
    opt[CTL + C_POLICY] = opt[CTL + C_POLICY] + partials[gbase + PCOUNT] / count;
    opt[CTL + C_VALUE] = opt[CTL + C_VALUE] + partials[gbase + PCOUNT + 1u] / count;
    opt[CTL + C_ENTROPY] = opt[CTL + C_ENTROPY] + partials[gbase + PCOUNT + 2u] / (count * f32(NOUT));
    opt[CTL + C_CLIP] = opt[CTL + C_CLIP] + partials[gbase + PCOUNT + 3u] / count;
    opt[CTL + C_KL] = opt[CTL + C_KL] + partials[gbase + PCOUNT + 4u] / count;
  }
}

@compute @workgroup_size(64)
fn rl_rescale(@builtin(local_invocation_index) tid: u32) {
  let retStd = opt[CTL + C_RET_STD];
  if (U.adaptScale < 0.5 || retStd < 1e-6) { return; }
  let factor = clamp(pow(1.0 / retStd, U.adaptScale), 0.5, 2.0);
  for (var p = OFF_WV + tid; p <= OFF_BV; p += 64u) {
    theta[p] = theta[p] * factor;
    opt[p] = opt[p] * factor;
    opt[PCOUNT + p] = opt[PCOUNT + p] * factor * factor;
  }
  if (tid == 0u) { opt[CTL + C_SCALE] = opt[CTL + C_SCALE] * factor; }
}

@compute @workgroup_size(256)
fn rl_normstats(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  if (NORM == 0u) { return; }
  let f = tid % NIN;
  let sl = tid / NIN;
  if (sl >= NORM_SLICES) { return; }
  let n = TRAIN_LEARNERS * ROLL;
  var s1 = 0.0;
  var s2 = 0.0;
  for (var smp = wg.x * NORM_SLICES + sl; smp < n; smp += NORM_GROUPS * NORM_SLICES) {
    let b = smp / ROLL;
    if (lane[b * LS + LS_ROLE] != 1.0) { continue; }
    let k = smp - b * ROLL;
    let x = rObs[b * OBS_LS + (k + 1u) * NIN + f];
    s1 += x;
    s2 += x * x;
  }
  let o = NORM_PART + (wg.x * NORM_SLICES + sl) * 2u * NIN;
  partials[o + f] = s1;
  partials[o + NIN + f] = s2;
}

var<workgroup> nrmCount: f32;

@compute @workgroup_size(256)
fn rl_normapply(@builtin(local_invocation_index) tid: u32) {
  if (NORM == 0u) { return; }
  if (tid == 0u) { nrmCount = opt[CTL + C_NCOUNT]; }
  workgroupBarrier();
  let rn = workgroupUniformLoad(&nrmCount);
  let nb = max(opt[CTL + C_ACTIVE], 1.0);
  for (var f = tid; f < NIN; f += 256u) {
    var s1 = 0.0;
    var s2 = 0.0;
    for (var q = 0u; q < NORM_GROUPS * NORM_SLICES; q++) {
      let o = NORM_PART + q * 2u * NIN;
      s1 += partials[o + f];
      s2 += partials[o + NIN + f];
    }
    let sd0 = 1.0 / theta[OFF_NI + f];
    let mn = s1 / nb;
    let vn = max(s2 / nb - mn * mn, 0.0);
    let mb = theta[OFF_NM + f] + mn * sd0;
    let vb = vn * sd0 * sd0;
    var rm = mb;
    var rv = vb;
    if (rn > 0.0) {
      let rmo = opt[NORM_OPT + f];
      let rvo = opt[NORM_OPT + NIN + f];
      let tot = rn + nb;
      let delta = mb - rmo;
      rm = rmo + delta * nb / tot;
      rv = (rn * rvo + nb * vb + delta * delta * rn * nb / tot) / tot;
    }
    opt[NORM_OPT + f] = rm;
    opt[NORM_OPT + NIN + f] = rv;
    theta[OFF_NM + f] = rm;
    theta[OFF_NI + f] = 1.0 / max(sqrt(rv), NORM_FLOOR);
  }
  workgroupBarrier();
  if (tid == 0u) { opt[CTL + C_NCOUNT] = min(rn + nb, NORM_CAP); }
}

const STATS_ROWS: u32 = 68u;
var<workgroup> reduceRows: array<i32, ${64 * 68}>;

@compute @workgroup_size(64)
fn rl_reduce_stats(@builtin(local_invocation_index) t: u32) {
  var mine: array<i32, 68>;
  for (var w = t; w < WORLDS; w += 64u) {
    var group = 0u;
    if (w >= EVAL_C_START) { group = 3u; } else if (w >= EVAL_B_START) { group = 2u; } else if (w >= EVAL_START) { group = 1u; }
    let base = w * WSTRIDE + STATS_OFF;
    for (var i = 0u; i < 16u; i++) { mine[group * 16u + i] += bitcast<i32>(world[base + i]); }
    if (group == 2u) {
      for (var i = 0u; i < 4u; i++) { mine[STATS_ROLE_OFF + i] += bitcast<i32>(world[w * WSTRIDE + ROLE_STATS_WORD + i]); }
    }
    statsOut[STATS_WORLD_OFF + w * 2u] = bitcast<i32>(world[base + STAT_REW_EVO]);
    statsOut[STATS_WORLD_OFF + w * 2u + 1u] = bitcast<i32>(world[base + STAT_TICKS_EVO]);
    for (var i = 0u; i < 16u; i++) { if (i != STAT_EDGES) { world[base + i] = 0u; } }
    for (var i = 0u; i < 4u; i++) { world[w * WSTRIDE + ROLE_STATS_WORD + i] = 0u; }
  }
  for (var i = 0u; i < STATS_ROWS; i++) { reduceRows[t * STATS_ROWS + i] = mine[i]; }
  workgroupBarrier();
  for (var c = t; c < STATS_ROWS; c += 64u) {
    var sum = 0;
    for (var r = 0u; r < 64u; r++) { sum += reduceRows[r * STATS_ROWS + c]; }
    statsOut[c] = sum;
  }
}
`;
}
