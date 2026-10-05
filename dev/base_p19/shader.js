import { evoConfig } from './core.js';
export const SIM_BINDINGS = { edgePk: 0, edgeW: 1, brain: 2, world: 3, graveyard: 4, archive: 5, archiveOut: 6, statsOut: 7, scratch: 8, uniforms: 9 };
export const ENTRY_BINDINGS = {
  init_world: ['edgePk', 'edgeW', 'brain', 'world', 'graveyard', 'uniforms'],
  sim_step: ['edgePk', 'edgeW', 'brain', 'world', 'graveyard', 'archive', 'uniforms'],
  select_archive: ['graveyard', 'archive', 'archiveOut', 'uniforms'],
  reduce_stats: ['world', 'archive', 'statsOut', 'uniforms'],
  brain_test: ['edgePk', 'edgeW', 'brain', 'scratch', 'uniforms'],
  env_test: ['world', 'scratch', 'uniforms'],
  observe_test: ['world', 'scratch', 'uniforms']
};
export const UNIFORM_WORDS = 16;
export const BRAIN_STATE = { SKIP: 0, REWAVG: 1, OPS: 2, LASTACT: 3, SCORE: 4, LIFE: 5, EDGES: 6 };

export function shaderLayout(game, cfg) {
  const { nIn, nOut, nNodes } = game.dims;
  const learners = game.learners;
  const maxEdges = cfg.maxEdges;
  const genomeHeader = 24;
  const genomeStride = genomeHeader + 2 * maxEdges;
  const brainOffLastIn = 20;
  const brainOffState = brainOffLastIn + nIn;
  const brainOffAct = brainOffState + 8;
  const brainOffHist = brainOffAct + nNodes;
  const brainStride = brainOffHist + nOut;
  const worldStride = 24 + game.worldWords;
  const islandBlock = 4 + 64 * genomeStride;
  const islandMaxWorlds = Math.floor(cfg.evalStart / cfg.islands) + (cfg.evalStart % cfg.islands);
  const sortSize = nextPow2(64 + islandMaxWorlds * 4 + cfg.migrateCount);
  const statsPerIsland = 40;
  return {
    nIn, nOut, nNodes, learners, maxEdges, genomeHeader, genomeStride, brainOffLastIn, brainOffState, brainOffAct, brainOffHist, brainStride,
    worldStride, islandBlock, islandMaxWorlds, sortSize, statsPerIsland,
    worlds: cfg.worlds, islands: cfg.islands, migrateCount: cfg.migrateCount,
    brainCount: cfg.worlds * learners,
    workgroupBytes: (learners * (nNodes | 1) + learners * (nIn | 1) + 3 * game.agents + 16 + learners + 10) * 4 + game.workgroupBytes
  };
}

export function nextPow2(n) {
  let p = 1;
  while (p < n) p *= 2;
  return p;
}

const EDGE_UNROLL = 8;
const unroll = (n, f) => Array.from({ length: n }, (_, i) => f(i)).join('\n');

export function buildShader(game, cfg) {
  const L = shaderLayout(game, cfg);
  const evo = evoConfig(cfg.evo);
  const lit = (x) => (Number.isInteger(x) ? x + '.0' : String(x));
  return `
const NIN: u32 = ${L.nIn}u;
const NOUT: u32 = ${L.nOut}u;
const NODES: u32 = ${L.nNodes}u;
const ASTR: u32 = ${L.nNodes | 1}u;
const OSTR: u32 = ${L.nIn | 1}u;
const HID0: u32 = ${L.nIn + L.nOut}u;
const LEARNERS: u32 = ${game.learners}u;
const AGENTS: u32 = ${game.agents}u;
const MAXE: u32 = ${L.maxEdges}u;
const WORLDS: u32 = ${L.worlds}u;
const BR: u32 = ${L.brainCount}u;
const MAX_AGE: f32 = ${game.maxAge}.0;
const GH: u32 = ${L.genomeHeader}u;
const GS: u32 = ${L.genomeStride}u;
const IB: u32 = ${L.islandBlock}u;
const SORT_N: u32 = ${L.sortSize}u;
const MIGRATE_N: u32 = ${L.migrateCount}u;
const WSTRIDE: u32 = ${L.worldStride}u;
const STATS_OFF: u32 = 8u;
const GAME_OFF: u32 = 24u;
const BSTRIDE: u32 = ${L.brainStride}u;
const OFF_LASTIN: u32 = ${L.brainOffLastIn}u;
const OFF_STATE: u32 = ${L.brainOffState}u;
const OFF_ACT: u32 = ${L.brainOffAct}u;
const OFF_HIST: u32 = ${L.brainOffHist}u;
const NICHE_CAP: u32 = ${Math.round(evo.nicheCap)}u;
const NICHE_FRAC: f32 = ${lit(evo.nicheFrac)};
const S_SKIP: u32 = 0u;
const S_REWAVG: u32 = 1u;
const S_OPS: u32 = 2u;
const S_LASTACT: u32 = 3u;
const S_SCORE: u32 = 4u;
const S_LIFE: u32 = 5u;
const S_N: u32 = 6u;
const S_PARENTFIT: u32 = 7u;
const LIFE_WEIGHT: f32 = ${lit(evo.lifeWeight)};
const BIAS: u32 = ${evo.bias ? 1 : 0}u;
const PLAST: f32 = ${lit(evo.plast)};
const FRESH_FRAC: f32 = ${lit(evo.freshFrac)};
const ELITE_FRAC: f32 = ${lit(evo.eliteFrac)};
const ELITE_COUNT: u32 = ${Math.round(evo.eliteCount)}u;
const TOURN: u32 = ${Math.round(evo.tournament)}u;
const W_STEP: f32 = ${lit(evo.wStep)};
const PERTURB_P: f32 = ${lit(evo.perturbP)};
const CROSS_P: f32 = ${lit(evo.crossP)};
const CLONE_FRAC: f32 = ${lit(evo.cloneFrac)};
const MIN_EDGES: u32 = 30u;
const NEG_INF: f32 = -1e30;
const VALID_FIT: f32 = -1e29;
const REWARD_SCALE: f32 = 1024.0;
const RATE_EMA_ALPHA: f32 = 0.002;
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
  penalty: f32,
  mutation: f32,
  flags: u32,
  evalStart: u32,
  islandSize: u32,
  nIslands: u32,
  randomize: u32,
  seedBase: u32,
  decay: f32,
  migrate: u32,
  migrateCount: u32,
  difficulty: u32,
  pad1: u32,
  pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> edgePk: array<u32>;
@group(0) @binding(1) var<storage, read_write> edgeW: array<f32>;
@group(0) @binding(2) var<storage, read_write> brain: array<f32>;
@group(0) @binding(3) var<storage, read_write> world: array<u32>;
@group(0) @binding(4) var<storage, read_write> graveyard: array<u32>;
@group(0) @binding(5) var<storage, read> archive: array<u32>;
@group(0) @binding(6) var<storage, read_write> archiveOut: array<u32>;
@group(0) @binding(7) var<storage, read_write> statsOut: array<i32>;
@group(0) @binding(8) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(9) var<uniform> U: Uniforms;

var<private> PLO: array<f32, 20> = array<f32, 20>(0.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.1);
var<private> PHI: array<f32, 20> = array<f32, 20>(0.5, 1.0, 1.0, 1.0, 1.0, 0.01, 4.0, 0.2, 1.0, 2.0, 16.0, 0.2, 4.0, 0.2, 1.0, 0.9, 4.0, 1.0, 1.0, 4.0);
var<private> PDEF: array<f32, 20> = array<f32, 20>(0.05, 0.3, 0.0, 0.0, 0.0, 0.0005, 2.0, 0.02, 0.3, 0.1, 4.0, 0.03, 1.5, 0.02, 0.2, 0.3, 1.0, 0.0, 0.0, 1.0);

var<workgroup> act: array<f32, ${game.learners * (L.nNodes | 1)}>;
var<workgroup> obsBuf: array<atomic<u32>, ${game.learners * (L.nIn | 1)}>;
var<workgroup> actionBuf: array<u32, ${game.agents}>;
var<workgroup> rewardBuf: array<i32, ${game.agents}>;
var<workgroup> deadBuf: array<u32, ${game.agents}>;
var<workgroup> statAcc: array<atomic<i32>, 16>;
var<workgroup> deadCount: atomic<u32>;
var<workgroup> ndShared: u32;
var<workgroup> deadFit: array<f32, ${game.learners}>;
var<workgroup> gyAssign: array<u32, 4>;
var<workgroup> w_seed: u32;
var<workgroup> w_index: u32;
var<workgroup> w_eval: u32;
var<workgroup> w_rateEma: f32;
var<workgroup> sortKey: array<f32, ${L.sortSize}>;
var<workgroup> sortIdx: array<u32, ${L.sortSize}>;
var<workgroup> reduceRows: array<i32, 2048>;

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

fn world_difficulty() -> i32 { return select(i32(U.difficulty), 100, w_eval == 1u); }
fn action_of(a: u32) -> u32 { return actionBuf[a]; }
fn out_of(a: u32, k: u32) -> f32 { return act[a * ASTR + NIN + k]; }
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

struct Edge { pk: u32, w: f32 }

fn random_edge(st: ptr<function, u32>) -> Edge {
  let srcInput = rf(st) < 0.6;
  let sr = rf(st);
  let dstOutput = rf(st) < 0.5;
  let dr = rf(st);
  let wr = rn(st);
  var src = HID0 + u32(floor(sr * f32(NODES - HID0)));
  if (srcInput) { src = u32(floor(sr * f32(NIN))); }
  var dst = HID0 + u32(floor(dr * f32(NODES - HID0)));
  if (dstOutput) { dst = NIN + u32(floor(dr * f32(NOUT))); }
  return Edge(src | (dst << 8u), wr * 0.5);
}

fn brain_n(b: u32) -> u32 { return u32(brain[b * BSTRIDE + OFF_STATE + S_N]); }

fn reset_brain_state(b: u32, t: u32) {
  let bb = b * BSTRIDE;
  for (var i = 0u; i < NIN; i++) { brain[bb + OFF_LASTIN + i] = 0.0; }
  brain[bb + OFF_STATE + S_SKIP] = 1000.0;
  brain[bb + OFF_STATE + S_REWAVG] = 0.0;
  brain[bb + OFF_STATE + S_OPS] = 0.0;
  brain[bb + OFF_STATE + S_LASTACT] = 0.0;
  brain[bb + OFF_STATE + S_SCORE] = 0.0;
  brain[bb + OFF_STATE + S_LIFE] = 0.0;
  brain[bb + OFF_STATE + S_PARENTFIT] = 0.0;
  for (var k = 0u; k < NOUT; k++) { brain[bb + OFF_HIST + k] = 0.0; }
  for (var j = 0u; j < NODES; j++) { act[t * ASTR + j] = 0.0; }
}

fn random_genome_into(b: u32, st: ptr<function, u32>) {
  let bb = b * BSTRIDE;
  for (var i = 0u; i < 20u; i++) {
    let noise = rn(st);
    brain[bb + i] = clamp(PDEF[i] + noise * 0.25 * (PHI[i] - PLO[i]), PLO[i], PHI[i]);
  }
  for (var e = 0u; e < 60u; e++) {
    let edge = random_edge(st);
    edgePk[e * BR + b] = edge.pk;
    edgeW[e * BR + b] = clamp(edge.w, -brain[bb + 6u], brain[bb + 6u]);
  }
  brain[bb + OFF_STATE + S_N] = 60.0;
}

fn copy_archive_genome_into(b: u32, gbase: u32) {
  let bb = b * BSTRIDE;
  for (var i = 0u; i < 20u; i++) { brain[bb + i] = bf(archive[gbase + 2u + i]); }
  let n = archive[gbase + 1u];
  for (var e = 0u; e < n; e++) {
    edgePk[e * BR + b] = archive[gbase + GH + e];
    edgeW[e * BR + b] = bf(archive[gbase + GH + MAXE + e]);
  }
  brain[bb + OFF_STATE + S_N] = f32(n);
  brain[bb + OFF_STATE + S_PARENTFIT] = bf(archive[gbase]);
}

fn mutate_brain(b: u32, st: ptr<function, u32>, mutation: f32) {
  let bb = b * BSTRIDE;
  let rateNoise = rn(st);
  let p19 = clamp(brain[bb + 19u] * exp(0.2 * rateNoise), 0.1, 4.0);
  brain[bb + 19u] = p19;
  let m = mutation * p19;
  for (var i = 0u; i < 19u; i++) {
    let noise = rn(st);
    let applies = rf(st) < 0.5;
    if (applies) { brain[bb + i] = clamp(brain[bb + i] + noise * m * 0.05 * (PHI[i] - PLO[i]), PLO[i], PHI[i]); }
  }
  let wmax = brain[bb + 6u];
  var n = brain_n(b);
  var e = 0u;
  loop {
    if (e >= n) { break; }
    let removeRoll = rf(st);
    let noise = rn(st);
    let perturbs = rf(st) < PERTURB_P;
    if (removeRoll < m * 0.02 && n > MIN_EDGES) {
      n = n - 1u;
      edgePk[e * BR + b] = edgePk[n * BR + b];
      edgeW[e * BR + b] = edgeW[n * BR + b];
      continue;
    }
    edgeW[e * BR + b] = clamp(edgeW[e * BR + b] + select(0.0, noise * m * W_STEP, perturbs), -wmax, wmax);
    e = e + 1u;
  }
  let add0 = rf(st) < m * 0.3;
  let add1 = rf(st) < m * 0.1;
  let edge0 = random_edge(st);
  if (add0 && n < MAXE) {
    edgePk[n * BR + b] = edge0.pk;
    edgeW[n * BR + b] = clamp(edge0.w, -wmax, wmax);
    n = n + 1u;
  }
  let edge1 = random_edge(st);
  if (add1 && n < MAXE) {
    edgePk[n * BR + b] = edge1.pk;
    edgeW[n * BR + b] = clamp(edge1.w, -wmax, wmax);
    n = n + 1u;
  }
  brain[bb + OFF_STATE + S_N] = f32(n);
}

fn island_of(wi: u32) -> u32 {
  if (wi >= U.evalStart) { return (wi - U.evalStart) % U.nIslands; }
  return min(U.nIslands - 1u, wi / U.islandSize);
}

fn pick_parent(st: ptr<function, u32>, count: u32) -> u32 {
  let eliteRoll = rf(st);
  if (eliteRoll < ELITE_FRAC) { return u32(floor(rf(st) * f32(min(ELITE_COUNT, count)))); }
  var best = count;
  for (var k = 0u; k < TOURN; k++) { best = min(best, u32(floor(rf(st) * f32(count)))); }
  return best;
}

fn cross_from(b: u32, gbase: u32, st: ptr<function, u32>) {
  var n = brain_n(b);
  let m = archive[gbase + 1u];
  let wmax = brain[b * BSTRIDE + 6u];
  for (var e = 0u; e < m && n < MAXE; e++) {
    if (rf(st) < 0.5) {
      edgePk[n * BR + b] = archive[gbase + GH + e];
      edgeW[n * BR + b] = clamp(bf(archive[gbase + GH + MAXE + e]), -wmax, wmax);
      n = n + 1u;
    }
  }
  brain[b * BSTRIDE + OFF_STATE + S_N] = f32(n);
}

fn rebirth(t: u32, b: u32, wi: u32, tick: u32) {
  var st = mix4(w_seed, tick, t, 2u);
  let archBase = island_of(wi) * IB;
  let count = archive[archBase];
  reset_brain_state(b, t);
  if (w_eval == 1u) {
    if (count == 0u) { random_genome_into(b, &st); return; }
    let idx = min(t % 4u, count - 1u);
    copy_archive_genome_into(b, archBase + 4u + idx * GS);
    return;
  }
  let roll = rf(&st);
  if (count == 0u || roll < FRESH_FRAC) { random_genome_into(b, &st); return; }
  copy_archive_genome_into(b, archBase + 4u + pick_parent(&st, count) * GS);
  let crossRoll = rf(&st);
  if (CROSS_P > 0.0 && crossRoll < CROSS_P) { cross_from(b, archBase + 4u + pick_parent(&st, count) * GS, &st); }
  var cloneRoll = 1.0;
  if (CLONE_FRAC > 0.0) { cloneRoll = rf(&st); }
  if (cloneRoll >= CLONE_FRAC) { mutate_brain(b, &st, U.mutation); }
  atomicAdd(&statAcc[STAT_BIRTHS], 1);
}

fn think(t: u32, b: u32, tick: u32) -> u32 {
  let bb = b * BSTRIDE;
  let arow = t * ASTR;
  var diff = 0.0;
  let inScale = brain[bb + 16u];
  var i = 0u;
  for (; i + 8u <= NIN; i += 8u) {
${unroll(8, (u) => `    let l${u} = brain[bb + OFF_LASTIN + i + ${u}u];`)}
${unroll(8, (u) => `    let v${u} = obs_get(t, i + ${u}u);`)}
${unroll(8, (u) => `    act[arow + i + ${u}u] = v${u} * inScale;
    diff += abs(v${u} - l${u});`)}
  }
  for (; i < NIN; i++) {
    let v = obs_get(t, i);
    act[arow + i] = v * inScale;
    diff += abs(v - brain[bb + OFF_LASTIN + i]);
  }
  if (BIAS == 1u) { act[arow + NODES - 1u] = 1.0; }
  let skip = brain[bb + OFF_STATE + S_SKIP];
  if (skip < floor(brain[bb + 10u]) && diff < brain[bb + 9u]) {
    brain[bb + OFF_STATE + S_SKIP] = skip + 1.0;
    return u32(brain[bb + OFF_STATE + S_LASTACT]);
  }
  brain[bb + OFF_STATE + S_SKIP] = 0.0;
  for (var i = 0u; i < NIN; i++) { brain[bb + OFF_LASTIN + i] = obs_get(t, i); }
  var acc: array<f32, ${L.nNodes}>;
  let n = brain_n(b);
  var e = 0u;
  for (; e + ${EDGE_UNROLL}u <= n; e += ${EDGE_UNROLL}u) {
${unroll(EDGE_UNROLL, (u) => `    let pk${u} = edgePk[(e + ${u}u) * BR + b];
    let w${u} = edgeW[(e + ${u}u) * BR + b];`)}
${unroll(EDGE_UNROLL, (u) => `    acc[(pk${u} >> 8u) & 255u] += w${u} * act[arow + (pk${u} & 255u)];`)}
  }
  for (; e < n; e++) {
    let packed = edgePk[e * BR + b];
    acc[(packed >> 8u) & 255u] += edgeW[e * BR + b] * act[arow + (packed & 255u)];
  }
  let gain = brain[bb + 12u];
  let leak = brain[bb + 15u];
  for (var j = NIN; j < NODES - BIAS; j++) {
    let x = gain * acc[j];
    act[arow + j] = leak * act[arow + j] + (1.0 - leak) * (x / (1.0 + abs(x)));
  }
  brain[bb + OFF_STATE + S_OPS] += f32(n);
  var best = NIN;
  for (var j = NIN + 1u; j < NIN + NOUT; j++) { if (act[arow + j] > act[arow + best]) { best = j; } }
  var action = best - NIN;
  var st = mix4(w_seed, tick, t, 1u);
  let exploreRoll = rf(&st);
  let exploreAction = u32(floor(rf(&st) * f32(NOUT)));
  if (exploreRoll < brain[bb + 11u]) { action = exploreAction; }
  brain[bb + OFF_STATE + S_LASTACT] = f32(action);
  return action;
}

fn learn(t: u32, b: u32, rewardFx: i32) {
  let bb = b * BSTRIDE;
  let arow = t * ASTR;
  let reward = f32(rewardFx) / REWARD_SCALE;
  let rewAvg = brain[bb + OFF_STATE + S_REWAVG];
  let advantage = reward - rewAvg;
  brain[bb + OFF_STATE + S_REWAVG] = rewAvg + brain[bb + 13u] * (reward - rewAvg);
  brain[bb + OFF_STATE + S_SCORE] += reward;
  if (rewardFx == 0 || PLAST == 0.0) { return; }
  let eta = brain[bb] * PLAST;
  let A = brain[bb + 1u];
  let B = brain[bb + 2u];
  let C = brain[bb + 3u];
  let D = brain[bb + 4u];
  let decay = brain[bb + 5u];
  let wmax = brain[bb + 6u];
  let n = brain_n(b);
  var e = 0u;
  for (; e + ${EDGE_UNROLL}u <= n; e += ${EDGE_UNROLL}u) {
${unroll(EDGE_UNROLL, (u) => `    let pk${u} = edgePk[(e + ${u}u) * BR + b];
    let w${u} = edgeW[(e + ${u}u) * BR + b];`)}
${unroll(EDGE_UNROLL, (u) => `    let pre${u} = act[arow + (pk${u} & 255u)];
    let post${u} = act[arow + ((pk${u} >> 8u) & 255u)];
    edgeW[(e + ${u}u) * BR + b] = clamp(w${u} * (1.0 - decay) + eta * advantage * (A * pre${u} * post${u} + B * pre${u} + C * post${u} + D), -wmax, wmax);`)}
  }
  for (; e < n; e++) {
    let packed = edgePk[e * BR + b];
    let pre = act[arow + (packed & 255u)];
    let post = act[arow + ((packed >> 8u) & 255u)];
    let dw = eta * advantage * (A * pre * post + B * pre + C * post + D);
    edgeW[e * BR + b] = clamp(edgeW[e * BR + b] * (1.0 - decay) + dw, -wmax, wmax);
  }
  brain[bb + OFF_STATE + S_OPS] += f32(n);
}

fn structural(t: u32, b: u32, tick: u32) {
  let bb = b * BSTRIDE;
  let arow = t * ASTR;
  let pruneT = brain[bb + 7u];
  var n = brain_n(b);
  var e = 0u;
  loop {
    if (e >= n) { break; }
    if (abs(edgeW[e * BR + b]) < pruneT && n > MIN_EDGES) {
      n = n - 1u;
      edgePk[e * BR + b] = edgePk[n * BR + b];
      edgeW[e * BR + b] = edgeW[n * BR + b];
    } else {
      e = e + 1u;
    }
  }
  var st = mix4(w_seed, tick, t, 3u);
  let growRoll = rf(&st);
  var found = false;
  var src = 0u;
  var dst = 0u;
  for (var attempt = 0u; attempt < 4u; attempt++) {
    let rs = rf(&st);
    let rd = rf(&st);
    let s = u32(floor(rs * f32(NODES)));
    let d = NIN + u32(floor(rd * f32(NODES - NIN)));
    if (!found && abs(act[arow + s]) > 0.25 && abs(act[arow + d]) > 0.25) {
      found = true;
      src = s;
      dst = d;
    }
  }
  let weightRoll = rf(&st);
  if (growRoll < brain[bb + 8u] && found && n < MAXE) {
    edgePk[n * BR + b] = src | (dst << 8u);
    edgeW[n * BR + b] = clamp((weightRoll * 2.0 - 1.0) * brain[bb + 14u], -brain[bb + 6u], brain[bb + 6u]);
    n = n + 1u;
  }
  brain[bb + OFF_STATE + S_N] = f32(n);
  brain[bb + OFF_STATE + S_OPS] += f32(n);
}

fn load_brain_act(t: u32, b: u32) {
  for (var j = 0u; j < NODES; j++) { act[t * ASTR + j] = brain[b * BSTRIDE + OFF_ACT + j]; }
}

fn load_acts_coop(wi: u32, t: u32) {
  for (var i = t; i < LEARNERS * NODES; i += 64u) {
    let l = i / NODES;
    act[l * ASTR + (i - l * NODES)] = brain[(wi * LEARNERS + l) * BSTRIDE + OFF_ACT + (i - l * NODES)];
  }
}

fn save_acts_coop(wi: u32, t: u32) {
  for (var i = t; i < LEARNERS * NODES; i += 64u) {
    let l = i / NODES;
    brain[(wi * LEARNERS + l) * BSTRIDE + OFF_ACT + (i - l * NODES)] = act[l * ASTR + (i - l * NODES)];
  }
}

fn save_brain_act(t: u32, b: u32) {
  for (var j = 0u; j < NODES; j++) { brain[b * BSTRIDE + OFF_ACT + j] = act[t * ASTR + j]; }
}

fn copy_genome_to_graveyard(b: u32, slotBase: u32, fit: f32) {
  let bb = b * BSTRIDE;
  graveyard[slotBase] = fb(fit);
  let n = brain_n(b);
  graveyard[slotBase + 1u] = n;
  for (var i = 0u; i < 20u; i++) { graveyard[slotBase + 2u + i] = fb(brain[bb + i]); }
  var mask = 0u;
  let lifeSpan = max(brain[bb + OFF_STATE + S_LIFE], 1.0);
  for (var k = 0u; k < min(NOUT, 32u); k++) { if (brain[bb + OFF_HIST + k] >= NICHE_FRAC * lifeSpan) { mask = mask | (1u << k); } }
  graveyard[slotBase + 22u] = mask;
  for (var e = 0u; e < n; e++) {
    graveyard[slotBase + GH + e] = edgePk[e * BR + b];
    graveyard[slotBase + GH + MAXE + e] = fb(edgeW[e * BR + b]);
  }
}

fn stat_flush(wi: u32) {
  for (var i = 0u; i < 16u; i++) {
    let idx = wi * WSTRIDE + STATS_OFF + i;
    if (i == STAT_EDGES) {
      world[idx] = bitcast<u32>(atomicLoad(&statAcc[i]));
    } else {
      world[idx] = bitcast<u32>(bitcast<i32>(world[idx]) + atomicLoad(&statAcc[i]));
    }
  }
}

fn load_world_header(wi: u32) {
  w_seed = world[wi * WSTRIDE];
  w_eval = select(0u, 1u, wi >= U.evalStart);
  w_index = wi;
  w_rateEma = bf(world[wi * WSTRIDE + 2u]);
}

@compute @workgroup_size(64)
fn init_world(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let wi = wg.x;
  if (t == 0u) {
    world[wi * WSTRIDE] = mix4(U.seedBase, wi, 0u, 9u);
    world[wi * WSTRIDE + 1u] = select(0u, 1u, wi >= U.evalStart);
    world[wi * WSTRIDE + 2u] = 0u;
    for (var i = 0u; i < 16u; i++) { world[wi * WSTRIDE + STATS_OFF + i] = 0u; }
    for (var s = 0u; s < 4u; s++) { graveyard[(wi * 4u + s) * GS] = fb(NEG_INF); }
  }
  workgroupBarrier();
  load_world_header(wi);
  workgroupBarrier();
  g_init(wi, w_seed, w_eval, U.randomize, t);
  workgroupBarrier();
  if (t < LEARNERS) {
    let b = wi * LEARNERS + t;
    var st = mix4(w_seed, 0u, t, 6u);
    random_genome_into(b, &st);
    reset_brain_state(b, t);
    save_brain_act(t, b);
  }
}

@compute @workgroup_size(64)
fn sim_step(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let wi = wg.x;
  load_world_header(wi);
  if (t < 16u) { atomicStore(&statAcc[t], 0); }
  if (t == 0u) { atomicStore(&deadCount, 0u); }
  workgroupBarrier();
  g_load(wi, t);
  load_acts_coop(wi, t);
  workgroupBarrier();
  let evalRefresh = (U.flags & EVAL_RESET) != 0u && wi >= U.evalStart;
  if (evalRefresh) {
    if (t == 0u) { for (var a = 0u; a < AGENTS; a++) { g_respawn(a, U.tick0); } }
    workgroupBarrier();
    if (t < LEARNERS) { rebirth(t, wi * LEARNERS + t, wi, U.tick0); }
    workgroupBarrier();
  }
  for (var k = 0u; k < U.ticks; k++) {
    let tick = U.tick0 + k;
    if (t < AGENTS) { deadBuf[t] = 0u; rewardBuf[t] = 0; }
    workgroupBarrier();
    g_observe(t, tick);
    workgroupBarrier();
    if (t < LEARNERS) {
      let b = wi * LEARNERS + t;
      let chosen = think(t, b, tick);
      actionBuf[t] = chosen;
      if (NICHE_CAP > 0u) { brain[b * BSTRIDE + OFF_HIST + chosen] += 1.0; }
    }
    workgroupBarrier();
    g_step(t, tick);
    workgroupBarrier();
    if (t == 0u) {
      var total = 0;
      for (var a = 0u; a < LEARNERS; a++) { total += rewardBuf[a]; }
      w_rateEma += RATE_EMA_ALPHA * (f32(total) / REWARD_SCALE / f32(LEARNERS) - w_rateEma);
    }
    workgroupBarrier();
    if (t < LEARNERS) {
      let b = wi * LEARNERS + t;
      let bb = b * BSTRIDE;
      let opsBefore = brain[bb + OFF_STATE + S_OPS];
      learn(t, b, rewardBuf[t]);
      let life = brain[bb + OFF_STATE + S_LIFE] + 1.0;
      brain[bb + OFF_STATE + S_LIFE] = life;
      atomicAdd(&statAcc[STAT_REW_EVO], rewardBuf[t]);
      atomicAdd(&statAcc[STAT_TICKS_EVO], 1);
      let died = deadBuf[t] == 1u || life >= MAX_AGE;
      if (!died && (u32(life) % 32u) == 0u) { structural(t, b, tick); }
      atomicAdd(&statAcc[STAT_OPS_EVO], i32(brain[bb + OFF_STATE + S_OPS] - opsBefore));
      if (died) {
        let raw = brain[bb + OFF_STATE + S_SCORE] - w_rateEma * life - U.penalty * brain[bb + OFF_STATE + S_OPS] / 1000.0;
        var fit = LIFE_WEIGHT * raw + (1.0 - LIFE_WEIGHT) * brain[bb + OFF_STATE + S_PARENTFIT];
        if (!(fit == fit)) { fit = VALID_FIT; }
        deadFit[t] = fit;
        deadBuf[t] = 1u;
        atomicAdd(&deadCount, 1u);
        atomicAdd(&statAcc[STAT_DEATHS], 1);
        atomicAdd(&statAcc[STAT_LIFE_FIT], i32(floor(raw * REWARD_SCALE)));
      }
    }
    workgroupBarrier();
    if (t == 0u) {
      ndShared = atomicLoad(&deadCount);
      atomicStore(&deadCount, 0u);
    }
    workgroupBarrier();
    let nd = workgroupUniformLoad(&ndShared);
    if (nd > 0u) {
      if (t == 0u) {
        for (var s = 0u; s < 4u; s++) { gyAssign[s] = 0xffffffffu; }
        if (w_eval == 0u) {
          for (var a = 0u; a < LEARNERS; a++) {
            if (deadBuf[a] == 1u) {
              var minSlot = 0u;
              var minFit = bf(graveyard[(wi * 4u) * GS]);
              for (var s = 1u; s < 4u; s++) {
                let f = bf(graveyard[(wi * 4u + s) * GS]);
                if (f < minFit) { minFit = f; minSlot = s; }
              }
              if (deadFit[a] > minFit) {
                gyAssign[minSlot] = a;
                graveyard[(wi * 4u + minSlot) * GS] = fb(deadFit[a]);
              }
            }
          }
        }
      }
      workgroupBarrier();
      if (t < LEARNERS && deadBuf[t] == 1u) {
        for (var s = 0u; s < 4u; s++) {
          if (gyAssign[s] == t) { copy_genome_to_graveyard(wi * LEARNERS + t, (wi * 4u + s) * GS, deadFit[t]); }
        }
      }
      workgroupBarrier();
      if (t == 0u) {
        for (var a = 0u; a < LEARNERS; a++) { if (deadBuf[a] == 1u) { g_respawn(a, tick); } }
      }
      workgroupBarrier();
      if (t < LEARNERS && deadBuf[t] == 1u) { rebirth(t, wi * LEARNERS + t, wi, tick); }
      workgroupBarrier();
    }
  }
  save_acts_coop(wi, t);
  if (t < LEARNERS) { atomicAdd(&statAcc[STAT_EDGES], i32(brain_n(wi * LEARNERS + t))); }
  g_save(wi, t);
  workgroupBarrier();
  if (t == 0u) {
    world[wi * WSTRIDE + 2u] = fb(w_rateEma);
    stat_flush(wi);
  }
}

@compute @workgroup_size(256)
fn select_archive(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let island = wg.x;
  let firstWorld = island * U.islandSize;
  var lastWorld = firstWorld + U.islandSize;
  if (island == U.nIslands - 1u) { lastWorld = U.evalStart; }
  let worldCount = lastWorld - firstWorld;
  let archBase = island * IB;
  let count = archive[archBase];
  let prevBase = ((island + U.nIslands - 1u) % U.nIslands) * IB;
  let prevCount = archive[prevBase];
  let gyEnd = 64u + worldCount * 4u;
  for (var i = t; i < SORT_N; i += 256u) {
    var key = NEG_INF;
    if (i < 64u) {
      if (i < count) { key = bf(archive[archBase + 4u + i * GS]) - U.decay; }
    } else if (i < gyEnd) {
      let g = i - 64u;
      let w = firstWorld + g / 4u;
      if (w < U.evalStart) {
        let f = bf(graveyard[(w * 4u + (g % 4u)) * GS]);
        if (f > VALID_FIT) { key = f; }
      }
    } else if (i < gyEnd + MIGRATE_N && U.migrate == 1u) {
      let m = i - gyEnd;
      if (m < prevCount && m < U.migrateCount) { key = bf(archive[prevBase + 4u + m * GS]); }
    }
    sortKey[i] = key;
    sortIdx[i] = i;
  }
  workgroupBarrier();
  for (var k = 2u; k <= SORT_N; k = k << 1u) {
    for (var j = k >> 1u; j > 0u; j = j >> 1u) {
      for (var i = t; i < SORT_N; i += 256u) {
        let ixj = i ^ j;
        if (ixj > i) {
          let descending = (i & k) == 0u;
          let ki = sortKey[i];
          let kj = sortKey[ixj];
          let ii = sortIdx[i];
          let ij = sortIdx[ixj];
          let iBeforeJ = ki > kj || (ki == kj && ii < ij);
          if (descending != iBeforeJ) {
            sortKey[i] = kj;
            sortKey[ixj] = ki;
            sortIdx[i] = ij;
            sortIdx[ixj] = ii;
          }
        }
      }
      workgroupBarrier();
    }
  }
  if (NICHE_CAP > 0u) {
    if (t == 0u) {
      var cellMask: array<u32, 64>;
      var cellCount: array<u32, 64>;
      var cells = 0u;
      var picked = 0u;
      for (var r = 0u; r < SORT_N && picked < 64u; r++) {
        let key = sortKey[r];
        if (key <= VALID_FIT) { break; }
        let src = sortIdx[r];
        var mask = 0u;
        if (src < 64u) { mask = archive[archBase + 4u + src * GS + 22u]; }
        else if (src < gyEnd) { let g = src - 64u; mask = graveyard[(((firstWorld + g / 4u) * 4u) + (g % 4u)) * GS + 22u]; }
        else { mask = archive[prevBase + 4u + (src - gyEnd) * GS + 22u]; }
        var cell = cells;
        for (var c = 0u; c < cells; c++) { if (cellMask[c] == mask) { cell = c; } }
        if (cell == cells) { cellMask[cells] = mask; cellCount[cells] = 0u; cells = cells + 1u; }
        if (cellCount[cell] < NICHE_CAP) {
          cellCount[cell] = cellCount[cell] + 1u;
          sortKey[picked] = key;
          sortIdx[picked] = src;
          picked = picked + 1u;
        }
      }
      for (var r = picked; r < 64u; r++) { sortKey[r] = NEG_INF; }
    }
    workgroupBarrier();
  }
  let outBase = island * IB;
  for (var w = t; w < 64u * GS; w += 256u) {
    let rank = w / GS;
    let o = w % GS;
    let key = sortKey[rank];
    let src = sortIdx[rank];
    var value = 0u;
    if (key > VALID_FIT) {
      var base = 0u;
      if (src < 64u) {
        base = archBase + 4u + src * GS;
        value = archive[base + o];
      } else if (src < gyEnd) {
        let g = src - 64u;
        base = (((firstWorld + g / 4u) * 4u) + (g % 4u)) * GS;
        value = graveyard[base + o];
      } else {
        base = prevBase + 4u + (src - gyEnd) * GS;
        value = archive[base + o];
      }
    }
    if (o == 0u) { value = fb(key); }
    archiveOut[outBase + 4u + w] = value;
  }
  if (t == 0u) {
    var valid = 0u;
    var sum = 0.0;
    for (var r = 0u; r < 64u; r++) { if (sortKey[r] > VALID_FIT) { valid += 1u; sum += sortKey[r]; } }
    archiveOut[outBase] = valid;
    archiveOut[outBase + 1u] = fb(select(0.0, sortKey[0], valid > 0u));
    archiveOut[outBase + 2u] = fb(select(0.0, sum / f32(max(valid, 1u)), valid > 0u));
    archiveOut[outBase + 3u] = 0u;
  }
  workgroupBarrier();
  for (var g = t; g < worldCount * 4u; g += 256u) {
    graveyard[((firstWorld * 4u) + g) * GS] = fb(NEG_INF);
  }
}

fn reduce_world(w: u32, group: u32, mine: ptr<function, array<i32, 32>>) {
  let base = w * WSTRIDE + STATS_OFF;
  for (var i = 0u; i < 16u; i++) { (*mine)[group + i] += bitcast<i32>(world[base + i]); }
  statsOut[${cfg.islands * L.statsPerIsland}u + w * 2u] = bitcast<i32>(world[base + STAT_REW_EVO]);
  statsOut[${cfg.islands * L.statsPerIsland}u + w * 2u + 1u] = bitcast<i32>(world[base + STAT_TICKS_EVO]);
  for (var i = 0u; i < 16u; i++) { if (i != STAT_EDGES) { world[base + i] = 0u; } }
}

@compute @workgroup_size(64)
fn reduce_stats(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let island = wg.x;
  let firstWorld = island * U.islandSize;
  var lastWorld = firstWorld + U.islandSize;
  if (island == U.nIslands - 1u) { lastWorld = U.evalStart; }
  var mine: array<i32, 32>;
  for (var w = firstWorld + t; w < lastWorld; w += 64u) {
    reduce_world(w, 0u, &mine);
  }
  for (var w = U.evalStart + island + t * U.nIslands; w < WORLDS; w += 64u * U.nIslands) {
    reduce_world(w, 16u, &mine);
  }
  for (var i = 0u; i < 32u; i++) { reduceRows[t * 32u + i] = mine[i]; }
  workgroupBarrier();
  if (t < 32u) {
    var sum = 0;
    for (var r = 0u; r < 64u; r++) { sum += reduceRows[r * 32u + t]; }
    statsOut[island * ${L.statsPerIsland}u + t] = sum;
  }
  if (t == 32u) {
    statsOut[island * ${L.statsPerIsland}u + 32u] = bitcast<i32>(archive[island * IB]);
    statsOut[island * ${L.statsPerIsland}u + 33u] = bitcast<i32>(archive[island * IB + 1u]);
    statsOut[island * ${L.statsPerIsland}u + 34u] = bitcast<i32>(archive[island * IB + 2u]);
  }
}

@compute @workgroup_size(64)
fn brain_test(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let steps = U.ticks;
  let rewardBase = steps * LEARNERS * NIN;
  let actionBase = rewardBase + steps * LEARNERS;
  w_seed = 777u;
  workgroupBarrier();
  if (t < LEARNERS) { load_brain_act(t, t); }
  workgroupBarrier();
  for (var k = 0u; k < steps; k++) {
    if (t < LEARNERS) {
      for (var i = 0u; i < NIN; i++) { set_obs(t, i, scratch[(k * LEARNERS + t) * NIN + i]); }
      let action = think(t, t, k);
      scratch[actionBase + k * LEARNERS + t] = f32(action);
      let rewardFx = i32(scratch[rewardBase + k * LEARNERS + t]);
      learn(t, t, rewardFx);
      let life = brain[t * BSTRIDE + OFF_STATE + S_LIFE] + 1.0;
      brain[t * BSTRIDE + OFF_STATE + S_LIFE] = life;
      if ((u32(life) % 32u) == 0u) { structural(t, t, k); }
    }
    workgroupBarrier();
  }
  if (t < LEARNERS) { save_brain_act(t, t); }
}

@compute @workgroup_size(64)
fn observe_test(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let wi = wg.x;
  load_world_header(wi);
  workgroupBarrier();
  g_load(wi, t);
  workgroupBarrier();
  g_observe(t, U.tick0);
  workgroupBarrier();
  for (var i = t; i < LEARNERS * NIN; i += 64u) { scratch[i] = obs_get(i / NIN, i % NIN); }
}

@compute @workgroup_size(64)
fn env_test(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let wi = wg.x;
  load_world_header(wi);
  if (t < 16u) { atomicStore(&statAcc[t], 0); }
  workgroupBarrier();
  g_load(wi, t);
  workgroupBarrier();
  for (var k = 0u; k < U.ticks; k++) {
    let tick = U.tick0 + k;
    if (t < AGENTS) { deadBuf[t] = 0u; rewardBuf[t] = 0; }
    if (t < LEARNERS) {
      let rowBase = (k * LEARNERS + t) * (NOUT + 1u);
      actionBuf[t] = u32(scratch[rowBase]);
      for (var o = 0u; o < NOUT; o++) { act[t * ASTR + NIN + o] = scratch[rowBase + 1u + o]; }
    }
    workgroupBarrier();
    g_step(t, tick);
    workgroupBarrier();
    if (t == 0u) {
      for (var a = 0u; a < LEARNERS; a++) { if (deadBuf[a] == 1u) { g_respawn(a, tick); } }
    }
    workgroupBarrier();
  }
  g_save(wi, t);
  workgroupBarrier();
  if (t == 0u) { stat_flush(wi); }
}
`;
}
