import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { Brain, Rng, mix, evoConfig, genomeFromJSON, STAT, NSTATS, STRUCT_PERIOD, REWARD_SCALE } from '../src/core.js';

const SELF = fileURLToPath(import.meta.url);
const TOOL_DIR = path.dirname(SELF);
const EVAL_VERSION = 2;
const SUITES = ['bots', 'h2h', 'selfplay', 'botsRole', 'h2hSingle'];
const SINGLE_SUITES = ['bots', 'h2h', 'selfplay'];
const BUNDLE_SUITES = ['bots', 'botsRole', 'selfplay', 'h2hSingle'];
const BOT_SUITES = ['bots', 'botsRole'];
const DUEL_SUITES = ['h2h', 'h2hSingle'];
const ASSIGN_MODES = ['roundrobin', 'fraction'];
const HARVEST_TYPES = ['berry', 'tree', 'ore', 'spring', 'great'];
const GREAT_TOOL_TIER = 2;
const BOSS_WINDOW_TICKS = 60;
const POPULATION_FORMAT = 'npc-population-member/1';
const LABELS = ['idle', 'walk', 'interact', 'melee', 'range', 'mage', 'craftTool', 'craftWeap', 'buy'];
const CAUSES = ['mob', 'boss', 'animal', 'player', 'starve', 'age', 'other'];
const MOVE_THRESHOLD = 32;
const MIN_SPARSE_EDGES = 256;
const CHANNELS = [];
let WEIGHTS = [];

function parseArgs(argv) {
  const opts = { game: 'realm', suites: '', seeds: '2', baseSeed: 7, worlds: 4, periods: 1, top: 4, jobs: 0, gate: -1, opponent: '', assign: 'roundrobin', out: '', lesion: '', popWeights: '', json: false, write: true, files: [] };
  for (const arg of argv) {
    const m = /^--([a-zA-Z-]+)(?:=(.*))?$/.exec(arg);
    if (!m) { opts.files.push(arg); continue; }
    const key = m[1];
    if (key === 'json') opts.json = true;
    else if (key === 'no-write') opts.write = false;
    else if (key in opts && typeof opts[key] === 'number') opts[key] = Number(m[2]);
    else if (key in opts && typeof opts[key] === 'string') opts[key] = m[2];
    else throw new Error('unknown flag --' + key);
  }
  return opts;
}

function seedList(spec, baseSeed) {
  if (spec.includes(',')) return spec.split(',').map(Number);
  return Array.from({ length: Math.max(1, Number(spec)) }, (_, i) => baseSeed + i);
}

async function loadGameModule(spec) {
  const candidates = /[\\/.]/.test(spec) ? [path.resolve(TOOL_DIR, spec), path.resolve(process.cwd(), spec)] : [path.resolve(TOOL_DIR, '../src/games', spec + '.js')];
  const found = candidates.find((p) => fs.existsSync(p));
  if (!found) throw new Error('game module not found: ' + spec);
  const mod = await import(pathToFileURL(found).href);
  const game = Object.values(mod).find((v) => v && typeof v === 'object' && typeof v.createEnv === 'function');
  if (!game) throw new Error('no game definition exported by ' + found);
  return { game, info: mod.REALM_INFO && mod.REALM_INFO.F && mod.REALM_INFO.ACT ? mod.REALM_INFO : null };
}

function entryFromJson(json, game, file) {
  const dims = json.dims || game.dims;
  if (dims.nIn !== game.dims.nIn || dims.nOut !== game.dims.nOut) throw new Error(file + ': dims ' + JSON.stringify(dims) + ' do not match game ' + JSON.stringify(game.dims));
  const compat = /^npc-brain\/([4-9]|[1-9][0-9])$/.test(json.format) ? Object.assign({}, json, { format: 'npc-brain/3' }) : json;
  const params = json.params || [];
  const dense = (json.meta && json.meta.trainer === 'ppo') || (!params[0] && !params[7] && !params[8]);
  const bias = Boolean(json.meta && json.meta.bias);
  const declaredCap = json.meta && json.meta.maxEdges ? json.meta.maxEdges : 0;
  const maxEdges = dense ? json.edges.length : Math.max(MIN_SPARSE_EDGES, json.edges.length, declaredCap);
  return { genome: genomeFromJSON(compat, maxEdges), dims, dense, maxEdges, evo: evoConfig(bias ? { bias: 1 } : null), edges: json.edges.length, format: json.format };
}

function bundleMembers(parsed) {
  const members = Array.isArray(parsed) ? parsed : parsed && Array.isArray(parsed.policies) ? parsed.policies : null;
  return members && members.length > 0 && members.every((m) => m && typeof m === 'object' && m.genome) ? members : null;
}

function loadGenomes(file, game, top) {
  const parsed = JSON.parse(fs.readFileSync(file, 'utf8'));
  const list = (Array.isArray(parsed) ? parsed : [parsed]).slice(0, Math.max(1, top));
  return list.map((json) => entryFromJson(json, game, file));
}

function loadBundle(parsed, game, file) {
  const raw = bundleMembers(parsed);
  if (!raw) return null;
  const ordered = raw.map((m, i) => ({ m, i })).sort((a, b) => (Number.isFinite(a.m.index) && Number.isFinite(b.m.index) ? a.m.index - b.m.index : a.i - b.i)).map((x) => x.m);
  const seen = new Map();
  const members = ordered.map((m, k) => {
    const base = typeof m.role === 'string' && m.role ? m.role : 'p' + k;
    const count = (seen.get(base) || 0) + 1;
    seen.set(base, count);
    return { index: k, role: count > 1 ? base + '#' + count : base, weights: Array.isArray(m.weights) ? m.weights : null, channels: Array.isArray(m.channels) ? m.channels : null, fraction: Number.isFinite(m.fraction) ? m.fraction : null, format: m.format || POPULATION_FORMAT, entry: entryFromJson(m.genome, game, file) };
  });
  return { members, mix: !Array.isArray(parsed) && parsed.mix ? parsed.mix : null };
}

function assignRole(index, count, members, mode) {
  if (mode !== 'fraction') return index % members.length;
  const raw = members.map((m) => (m.fraction !== null && m.fraction > 0 ? m.fraction : 0));
  const sum = raw.reduce((a, v) => a + v, 0);
  const shares = sum > 0 ? raw.map((v) => v / sum) : members.map(() => 1 / members.length);
  const pos = (index + 0.5) / count;
  let cumulative = 0;
  for (let k = 0; k < shares.length; k++) {
    cumulative += shares[k];
    if (pos < cumulative && shares[k] > 0) return k;
  }
  return shares.map((v, k) => (v > 0 ? k : -1)).filter((k) => k >= 0).pop();
}

const RATION_TRADE_GOLD = 8;

function newSide(nOut) {
  return { ticks: 0, reward: 0, closed: 0, labelCounts: new Float64Array(Math.max(LABELS.length, 2 * nOut)), dist: 0, crafts: 0, buys: 0, moving: 0, safe: 0, ally: 0, allyLearner: 0, tool: 0, weap: 0, maxTool: 0, maxWeap: 0, attacks: 0, casts: 0, castStyleMix: new Float64Array(3), perk: 0, maxPerk: 0, coverTicks: 0, coverFirstHits: 0, tradeBought: 0, tradeSold: 0, mobArchetype: new Float64Array(5), bossPhaseHits: new Float64Array(4), saleTownUnits: new Float64Array(4), saleTownGold: new Float64Array(4), chan: new Float64Array(CHANNELS.length), causes: Object.fromEntries(CAUSES.map((c) => [c, 0])) };
}

function mergeSide(into, from) {
  for (const key of Object.keys(from)) {
    const value = from[key];
    if (typeof value === 'number') into[key] = key.startsWith('max') ? Math.max(into[key], value) : into[key] + value;
    else if (value instanceof Float64Array) for (let i = 0; i < value.length; i++) into[key][i] += value[i];
    else for (const k of Object.keys(value)) into[key][k] += value[k];
  }
}

function causeOf(killer, f, b, F, info) {
  if (killer >= info.BOSS0 && killer < info.ANIMAL0) return 'boss';
  if (killer >= info.MOB0 && killer < info.BOSS0) return 'mob';
  if (killer >= info.ANIMAL0) return 'animal';
  if (killer >= 0) return 'player';
  if (f[b + F.FOOD] === 0 || f[b + F.WATER] === 0) return 'starve';
  return 'other';
}

function actionLabel(env, e, ACT) {
  const a = env.act[e];
  if (a === ACT.ATTACK) return 3 + Math.min(2, env.atkStyle[e]);
  if (a === ACT.INTERACT && env.tgtNode[e] >= 0) return 2;
  if (a === ACT.CRAFT_TOOL) return 6;
  if (a === ACT.CRAFT_WEAP) return 7;
  if (a === ACT.BUY) return 8;
  return Math.hypot(env.ctlX[e], env.ctlY[e]) > MOVE_THRESHOLD ? 1 : 0;
}

function signedOutputLabel(outputs, offset, nOut) {
  let best = 0;
  for (let k = 1; k < nOut; k++) if (Math.abs(outputs[offset + k]) > Math.abs(outputs[offset + best])) best = k;
  return 2 * best + (outputs[offset + best] > 0 ? 1 : 0);
}

function newRoleSide(nOut, channels) {
  return Object.assign(newSide(nOut), { harvest: new Float64Array(HARVEST_TYPES.length), chan: new Float64Array(channels), chanPos: new Float64Array(channels), bossDmg: 0, pvpDmg: 0, pvpKills: 0, sales: 0, salesUnits: 0, salesGold: 0, crossAlly: 0, sameAlly: 0, bossKillsPart: 0, bossKillsMulti: 0 });
}

function harvestTypeIndex(info) {
  const byType = new Map([[info.nodeTypeOf(0), 0], [info.nodeTypeOf(60), 1], [info.nodeTypeOf(120), 2], [info.nodeTypeOf(168), 3]]);
  return (node) => (info.isGreat(node) ? 4 : byType.has(info.nodeTypeOf(node)) ? byType.get(info.nodeTypeOf(node)) : -1);
}

function runWorld(context, task) {
  const { game, info, bundle } = context;
  const nIn = game.dims.nIn;
  const nOut = game.dims.nOut;
  const bots = BOT_SUITES.includes(task.suite);
  const duel = DUEL_SUITES.includes(task.suite);
  const slots = bots ? game.learners : game.maxLearners;
  const worldSeed = mix(task.seed, task.world, 0, 9);
  const cfg = game.defaultCfg();
  if (!bots) cfg.learnerSlots = slots;
  const stats = new Int32Array(NSTATS);
  const env = game.createEnv(worldSeed, cfg, true, stats);
  env.difficulty = 100;
  if (context.gate >= 0) env.gateShift = context.gate;
  const swapped = duel && (task.world & 1) === 1;
  const sideOf = (slot) => (duel && ((slot + (swapped ? 1 : 0)) & 1) === 1 ? 'B' : 'A');
  const pools = { A: context.genomes, B: context.opponent };
  const roleOf = new Int16Array(slots).fill(-1);
  if (bundle) {
    for (let slot = 0; slot < slots; slot++) {
      if (task.suite === 'botsRole') roleOf[slot] = task.role;
      else if (task.suite === 'h2hSingle') roleOf[slot] = sideOf(slot) === 'A' ? assignRole(slot >> 1, slots >> 1, bundle.members, context.assign) : -1;
      else roleOf[slot] = assignRole(slot, slots, bundle.members, context.assign);
    }
  }
  const entryOf = (slot) => {
    if (roleOf[slot] >= 0) return bundle.members[roleOf[slot]].entry;
    const pool = pools[sideOf(slot)];
    return pool[(duel ? slot >> 1 : slot) % pool.length];
  };
  const brains = [];
  for (let slot = 0; slot < slots; slot++) { const entry = entryOf(slot); brains.push(new Brain(entry.genome, entry.maxEdges, entry.dims, entry.evo)); }
  const sides = { A: newSide(nOut), B: newSide(nOut) };
  const channelCount = (game.rewardChannels || []).length;
  const roleSides = bundle ? bundle.members.map(() => newRoleSide(nOut, channelCount)) : null;
  const popGlobal = { bossKills: 0, bossKillsMulti: 0 };
  const targetsOf = (slot) => (roleOf[slot] >= 0 ? [sides[sideOf(slot)], roleSides[roleOf[slot]]] : [sides[sideOf(slot)]]);
  const obs = new Float32Array(nIn);
  const actions = new Int32Array(slots);
  const outputs = new Float32Array(slots * nOut);
  const lifeTicks = new Int32Array(slots);
  const F = info && info.F;
  const fields = info ? info.FIELDS : 0;
  const f = env.f;
  const prev = info ? new Int32Array(slots * fields) : null;
  const hasAlly = info && typeof env.allyCount === 'function';
  const allyRadius2 = info ? info.ALLY_RADIUS * info.ALLY_RADIUS : 0;
  const harvestIndex = info ? harvestTypeIndex(info) : null;
  const lesionIdx = context.lesion && game.inputNames ? game.inputNames.indexOf(context.lesion) : -1;
  if (context.lesion && lesionIdx < 0) throw new Error('lesion input not found: ' + context.lesion + ' (have ' + (game.inputNames || []).length + ' names)');
  const popTracking = Boolean(roleSides && info);
  const hasChannels = channelCount > 0 && Boolean(env.rewardCh);
  const bossCount = info ? info.ANIMAL0 - info.BOSS0 : 0;
  const townBase = info && info.NODE_FIRST_TOWN !== undefined ? info.NODE_FIRST_TOWN : 200;
  const bossWasDead = new Uint8Array(bossCount);
  const bossLast = Array.from({ length: bossCount }, () => new Float64Array(bundle ? bundle.members.length : 0).fill(-Infinity));
  const closeLife = (slot, cause) => {
    for (const side of targetsOf(slot)) {
      side.closed++;
      side.causes[cause]++;
      if (info) { side.tool += f[slot * fields + F.TOOL]; side.weap += f[slot * fields + F.WEAP]; }
    }
  };
  const syncBosses = () => { for (let k = 0; k < bossCount; k++) bossWasDead[k] = f[(info.BOSS0 + k) * fields + F.DEAD] > 0 ? 1 : 0; };
  if (info) prev.set(f.subarray(0, slots * fields));
  if (popTracking) syncBosses();
  const ticks = task.periods * game.maxAge;
  for (let tick = 0; tick < ticks; tick++) {
    if (tick > 0 && tick % game.maxAge === 0) {
      for (let a = 0; a < game.agents; a++) env.respawn(a, tick);
      for (let slot = 0; slot < slots; slot++) brains[slot].load(entryOf(slot).genome);
      lifeTicks.fill(0);
      if (info) prev.set(f.subarray(0, slots * fields));
      if (popTracking) syncBosses();
    }
    for (let slot = 0; slot < slots; slot++) {
      env.observe(slot, obs, tick);
      if (lesionIdx >= 0) obs[lesionIdx] = 0;
      actions[slot] = brains[slot].step(obs, new Rng(mix(worldSeed, tick, slot, 1)));
      for (let k = 0; k < nOut; k++) outputs[slot * nOut + k] = brains[slot].act[nIn + k];
    }
    env.step(actions, outputs, tick);
    if (popTracking) trackPopulation(env, info, slots, tick, roleOf, roleSides, bossLast, bossWasDead, popGlobal, harvestIndex, prev, channelCount);
    const respawn = [];
    for (let slot = 0; slot < slots; slot++) {
      const targets = targetsOf(slot);
      for (const side of targets) {
        side.ticks++;
        side.reward += env.reward[slot] / REWARD_SCALE;
        if (hasChannels) for (let c = 0; c < channelCount; c++) side.chan[c] += env.rewardCh[slot * channelCount + c] / REWARD_SCALE;
        side.labelCounts[info ? actionLabel(env, slot, info.ACT) : signedOutputLabel(outputs, slot * nOut, nOut)]++;
      }
      if (info) {
        const b = slot * fields;
        const act = env.act[slot];
        const tgt = env.atkTgt[slot];
        const inCover = env.inCover(f[b + F.X], f[b + F.Y]);
        const soldUnits = (prev[b + F.WOOD] + prev[b + F.ORE]) - (f[b + F.WOOD] + f[b + F.ORE]);
        const sold = soldUnits > 0 && f[b + F.GOLD] > prev[b + F.GOLD];
        const townIdx = sold && env.tgtNode[slot] >= townBase && env.tgtNode[slot] < townBase + 4 ? env.tgtNode[slot] - townBase : -1;
        const perk = env.perkTier(slot);
        let near = false;
        let cross = false;
        let same = false;
        for (let other = 0; other < slots; other++) {
          if (other === slot || f[other * fields + F.DEAD] !== 0) continue;
          const dx = f[other * fields + F.X] - f[b + F.X];
          const dy = f[other * fields + F.Y] - f[b + F.Y];
          if (dx * dx + dy * dy < allyRadius2) {
            near = true;
            if (!popTracking) break;
            if (roleOf[other] < 0 || roleOf[slot] < 0) continue;
            if (roleOf[other] === roleOf[slot]) same = true; else cross = true;
            if (cross && same) break;
          }
        }
        for (const side of targets) {
          if (Math.hypot(env.ctlX[slot], env.ctlY[slot]) > MOVE_THRESHOLD) side.moving++;
          if (f[b + F.TOOL] > prev[b + F.TOOL]) side.crafts++;
          if (f[b + F.WEAP] > prev[b + F.WEAP]) side.crafts++;
          if (f[b + F.RAT] > prev[b + F.RAT] && f[b + F.GOLD] === prev[b + F.GOLD] - 10) side.buys++;
          const ratDelta = f[b + F.RAT] - prev[b + F.RAT];
          const goldDelta = f[b + F.GOLD] - prev[b + F.GOLD];
          if (goldDelta === -RATION_TRADE_GOLD && ratDelta > 0) side.tradeBought += ratDelta;
          if (goldDelta === RATION_TRADE_GOLD && ratDelta < 0) side.tradeSold -= ratDelta;
          side.dist += Math.hypot(f[b + F.X] - prev[b + F.X], f[b + F.Y] - prev[b + F.Y]);
          side.maxTool = Math.max(side.maxTool, f[b + F.TOOL]);
          side.maxWeap = Math.max(side.maxWeap, f[b + F.WEAP]);
          if (info.inSafeZone(env, f[b + F.X], f[b + F.Y])) side.safe++;
          if (hasAlly && env.allyCount(slot) > 0) side.ally++;
          if (near) side.allyLearner++;
          if (side.crossAlly !== undefined) { if (cross) side.crossAlly++; if (same) side.sameAlly++; }
          side.perk += perk;
          if (perk > side.maxPerk) side.maxPerk = perk;
          if (inCover) side.coverTicks++;
          if (act === info.ACT.ATTACK) {
            side.attacks++;
            if (env.atkCast[slot]) { side.casts++; side.castStyleMix[Math.min(2, env.atkStyle[slot])]++; }
            if (tgt >= info.MOB0 && tgt < info.BOSS0) side.mobArchetype[(tgt - info.MOB0) % 5]++;
            else if (info.isBoss(tgt)) side.bossPhaseHits[env.bossPhase(tgt)]++;
            if (inCover && tgt >= info.MOB0) side.coverFirstHits++;
          }
          if (townIdx >= 0) { side.saleTownUnits[townIdx] += soldUnits; side.saleTownGold[townIdx] += f[b + F.GOLD] - prev[b + F.GOLD]; }
        }
      }
      lifeTicks[slot]++;
      const died = env.dead[slot] !== 0;
      if (died || lifeTicks[slot] >= game.maxAge) {
        closeLife(slot, died ? (info ? causeOf(env.killer[slot], f, slot * fields, F, info) : 'other') : 'age');
        respawn.push(slot);
      }
    }
    for (let slot = 0; slot < slots; slot++) {
      const entry = entryOf(slot);
      if (entry.dense) continue;
      brains[slot].learn(env.reward[slot] / REWARD_SCALE);
      if (!respawn.includes(slot) && lifeTicks[slot] % STRUCT_PERIOD === 0) brains[slot].structural(new Rng(mix(worldSeed, tick, slot, 3)));
    }
    for (const slot of respawn) {
      env.respawn(slot, tick);
      brains[slot].load(entryOf(slot).genome);
      lifeTicks[slot] = 0;
    }
    if (info) prev.set(f.subarray(0, slots * fields));
  }
  return { sides, roles: roleSides, popGlobal, baseReward: stats[STAT.REW_BASE] / REWARD_SCALE, baseTicks: stats[STAT.TICKS_BASE], game: game.statNames.map((_, i) => stats[STAT.GAME0 + i]), slots };
}

function trackPopulation(env, info, slots, tick, roleOf, roleSides, bossLast, bossWasDead, popGlobal, harvestIndex, prev, channelCount) {
  const { F, ACT } = info;
  const f = env.f;
  const fields = info.FIELDS;
  const hasChannels = channelCount > 0 && env.rewardCh;
  for (let slot = 0; slot < slots; slot++) {
    const role = roleOf[slot];
    if (role < 0) continue;
    const rs = roleSides[role];
    const b = slot * fields;
    const act = env.act[slot];
    if (act === ACT.ATTACK) {
      const target = env.atkTgt[slot];
      if (info.isBoss(target)) { rs.bossDmg += env.atkDmg[slot]; bossLast[target - info.BOSS0][role] = tick; }
      else if (target < slots && target !== slot) rs.pvpDmg += env.atkDmg[slot];
    }
    if (act === ACT.INTERACT) {
      const node = env.tgtNode[slot];
      if (node >= 0) {
        if (info.isGreat(node)) {
          if (f[b + F.TOOL] >= GREAT_TOOL_TIER && env.gdone[node - info.GREAT0] === 1) rs.harvest[4]++;
        } else if (env.complete[slot]) {
          let taken = false;
          for (let j = 0; j < slot; j++) if (env.complete[j] && env.tgtNode[j] === node && env.act[j] === ACT.INTERACT) { taken = true; break; }
          const type = taken ? -1 : harvestIndex(node);
          if (type >= 0) rs.harvest[type]++;
        }
      }
    }
    const woodOre = f[b + F.WOOD] + f[b + F.ORE];
    const prevWoodOre = prev[b + F.WOOD] + prev[b + F.ORE];
    if (woodOre < prevWoodOre && f[b + F.GOLD] > prev[b + F.GOLD]) {
      rs.sales++;
      rs.salesUnits += prevWoodOre - woodOre;
      rs.salesGold += f[b + F.GOLD] - prev[b + F.GOLD];
    }
    if (hasChannels) for (let c = 0; c < channelCount; c++) {
      const value = env.rewardCh[slot * channelCount + c] / REWARD_SCALE;
      rs.chan[c] += value;
      if (value > 0) rs.chanPos[c] += value;
    }
    if (env.dead[slot] !== 0) {
      const killer = env.killer[slot];
      if (killer >= 0 && killer < slots && killer !== slot && roleOf[killer] >= 0) roleSides[roleOf[killer]].pvpKills++;
    }
  }
  for (let k = 0; k < bossLast.length; k++) {
    const dead = f[(info.BOSS0 + k) * fields + F.DEAD] > 0;
    if (dead && !bossWasDead[k]) {
      const involved = [];
      for (let role = 0; role < roleSides.length; role++) if (tick - bossLast[k][role] <= BOSS_WINDOW_TICKS) involved.push(role);
      if (involved.length > 0) {
        popGlobal.bossKills++;
        if (involved.length > 1) popGlobal.bossKillsMulti++;
        for (const role of involved) { roleSides[role].bossKillsPart++; if (involved.length > 1) roleSides[role].bossKillsMulti++; }
      }
      bossLast[k].fill(-Infinity);
    }
    bossWasDead[k] = dead ? 1 : 0;
  }
}

async function prepareContext(config) {
  const { game, info } = await loadGameModule(config.game);
  const parsed = JSON.parse(fs.readFileSync(config.genome, 'utf8'));
  const bundle = loadBundle(parsed, game, config.genome);
  const genomes = bundle ? null : loadGenomes(config.genome, game, config.top);
  const opponent = config.opponent ? loadGenomes(config.opponent, game, config.top) : null;
  CHANNELS.length = 0;
  CHANNELS.push(...(game.rewardChannels || []));
  return { game, info, genomes, opponent, bundle, assign: config.assign, lesion: config.lesion || '', gate: config.gate === undefined ? -1 : config.gate };
}

if (!isMainThread) {
  const contextPromise = prepareContext(workerData);
  parentPort.on('message', async (task) => {
    const started = performance.now();
    try {
      const result = runWorld(await contextPromise, task);
      parentPort.postMessage({ id: task.id, result, ms: performance.now() - started });
    } catch (error) {
      parentPort.postMessage({ id: task.id, error: error.stack || String(error) });
    }
  });
}

function entropyBits(counts) {
  const total = counts.reduce((s, v) => s + v, 0);
  if (total === 0) return null;
  return -counts.reduce((s, v) => (v > 0 ? s + (v / total) * Math.log2(v / total) : s), 0);
}

function maxShare(counts) {
  const total = counts.reduce((s, v) => s + v, 0);
  return total === 0 ? null : Math.max(...counts) / total;
}

function sideMetrics(side, hasInfo) {
  const lives = Math.max(1, side.closed);
  const per1k = (v) => (side.ticks > 0 ? (v / side.ticks) * 1000 : null);
  const mobTotal = side.mobArchetype ? side.mobArchetype.reduce((s, v) => s + v, 0) : 0;
  const bossTotal = side.bossPhaseHits ? side.bossPhaseHits.reduce((s, v) => s + v, 0) : 0;
  const townUnits = side.saleTownUnits ? side.saleTownUnits.reduce((s, v) => s + v, 0) : 0;
  const metrics = {
    rate: side.ticks > 0 ? side.reward / side.ticks : null,
    life: side.closed > 0 ? side.ticks / side.closed : null,
    actionEntropy: entropyBits(Array.from(side.labelCounts)),
    actionMaxShare: maxShare(Array.from(side.labelCounts))
  };
  for (let c = 0; c < CHANNELS.length; c++) metrics['ch_' + CHANNELS[c] + '1k'] = per1k(side.chan ? side.chan[c] : 0);
  WEIGHTS.forEach((w, i) => { metrics[WEIGHTS.length > 1 ? 'rateW' + (i + 1) : 'rateWeighted'] = per1k(w.reduce((s, ww, c) => s + ww * (side.chan ? side.chan[c] : 0), 0)); });
  if (!hasInfo) return metrics;
  Object.assign(metrics, {
    movingShare: side.ticks > 0 ? side.moving / side.ticks : null,
    dist1k: per1k(side.dist),
    crafts1k: per1k(side.crafts),
    buys1k: per1k(side.buys),
    toolTier: side.tool / lives,
    weaponTier: side.weap / lives,
    gearTierMax: Math.max(side.maxTool, side.maxWeap),
    safeShare: side.ticks > 0 ? side.safe / side.ticks : null,
    allyShare: side.ticks > 0 ? side.ally / side.ticks : null,
    allyLearnerShare: side.ticks > 0 ? side.allyLearner / side.ticks : null,
    perkTier: side.ticks > 0 ? side.perk / side.ticks : null,
    perkTierMax: side.maxPerk,
    attacks1k: per1k(side.attacks),
    casts1k: per1k(side.casts),
    castShare: side.attacks > 0 ? side.casts / side.attacks : null,
    coverShare: side.ticks > 0 ? side.coverTicks / side.ticks : null,
    coverFirstHits1k: per1k(side.coverFirstHits),
    tradeBought1k: per1k(side.tradeBought),
    tradeSold1k: per1k(side.tradeSold),
    mobArchetypeEntropy: entropyBits(Array.from(side.mobArchetype)),
    bossPhase3Share: bossTotal > 0 ? side.bossPhaseHits[3] / bossTotal : null,
    saleGoldPerUnit: townUnits > 0 ? side.saleTownGold.reduce((s, v) => s + v, 0) / townUnits : null
  });
  for (let k = 0; k < 5; k++) metrics['mobArch' + k] = mobTotal > 0 ? side.mobArchetype[k] / mobTotal : null;
  for (let k = 0; k < 4; k++) metrics['saleTown' + k] = townUnits > 0 ? side.saleTownUnits[k] / townUnits : null;
  for (const cause of CAUSES) metrics['death_' + cause] = side.causes[cause] / lives;
  return metrics;
}

function suiteSeedMetrics(suite, sideA, sideB, gameTotals, statNames, baseReward, baseTicks, hasInfo) {
  const metrics = sideMetrics(sideA, hasInfo);
  const closedAll = Math.max(1, sideA.closed + (DUEL_SUITES.includes(suite) ? sideB.closed : 0));
  statNames.forEach((name, i) => { metrics[name] = gameTotals[i] / closedAll; });
  if (BOT_SUITES.includes(suite)) {
    metrics.baseRate = baseTicks > 0 ? baseReward / baseTicks : null;
    metrics.ratio = metrics.baseRate ? metrics.rate / metrics.baseRate : null;
  }
  if (DUEL_SUITES.includes(suite)) {
    const opponent = sideMetrics(sideB, false);
    metrics.opponentRate = opponent.rate;
    metrics.opponentLife = opponent.life;
    metrics.ratio = opponent.rate ? metrics.rate / opponent.rate : null;
  }
  return metrics;
}

function meanSd(values) {
  const finite = values.filter((v) => typeof v === 'number' && Number.isFinite(v));
  if (finite.length === 0) return { mean: null, sd: null, n: 0 };
  const mean = finite.reduce((s, v) => s + v, 0) / finite.length;
  const sd = finite.length > 1 ? Math.sqrt(finite.reduce((s, v) => s + (v - mean) * (v - mean), 0) / (finite.length - 1)) : null;
  return { mean, sd, n: finite.length };
}

function summarizeSuite(perSeed) {
  const keys = new Set();
  for (const seed of perSeed) for (const key of Object.keys(seed.metrics)) keys.add(key);
  const summary = {};
  for (const key of keys) summary[key] = meanSd(perSeed.map((s) => s.metrics[key]));
  return summary;
}

function formatValue(stat, key) {
  if (!stat || stat.mean === null) return '-';
  const digits = /^(rate|baseRate|opponentRate|rateWeighted)$/.test(key) ? 5 : Math.abs(stat.mean) >= 100 ? 0 : Math.abs(stat.mean) >= 10 ? 1 : 3;
  const mean = stat.mean.toFixed(digits);
  return stat.sd === null ? mean : mean + ' +- ' + stat.sd.toFixed(digits);
}

function formatTable(result) {
  const suites = Object.keys(result.suites);
  const keys = [];
  for (const suite of suites) for (const key of Object.keys(result.suites[suite].summary)) if (!keys.includes(key)) keys.push(key);
  const rows = [['metric', ...suites]];
  for (const key of keys) rows.push([key, ...suites.map((suite) => formatValue(result.suites[suite].summary[key], key))]);
  const widths = rows[0].map((_, c) => Math.max(...rows.map((r) => r[c].length)));
  return rows.map((r) => r.map((cell, c) => (c === 0 ? cell.padEnd(widths[c]) : cell.padStart(widths[c]))).join('  ')).join('\n');
}

function distribution(counts) {
  const total = counts.reduce((s, v) => s + v, 0);
  return total > 0 ? counts.map((v) => v / total) : null;
}

function jsDivergenceBits(p, q) {
  let total = 0;
  for (let i = 0; i < p.length; i++) {
    const m = (p[i] + q[i]) / 2;
    if (p[i] > 0) total += 0.5 * p[i] * Math.log2(p[i] / m);
    if (q[i] > 0) total += 0.5 * q[i] * Math.log2(q[i] / m);
  }
  return total;
}

function meanPairwiseJs(countsByRole) {
  const dists = countsByRole.map(distribution).filter(Boolean);
  if (dists.length < 2) return null;
  let sum = 0;
  let pairs = 0;
  for (let i = 0; i < dists.length; i++) for (let j = i + 1; j < dists.length; j++) { sum += jsDivergenceBits(dists[i], dists[j]); pairs++; }
  return sum / pairs;
}

function roleLabelInformation(countsByRole) {
  const rows = countsByRole.filter((row) => row.some((v) => v > 0));
  const total = rows.reduce((s, row) => s + row.reduce((a, v) => a + v, 0), 0);
  if (rows.length < 2 || total === 0) return { mi: null, norm: null };
  const roleMass = rows.map((row) => row.reduce((a, v) => a + v, 0) / total);
  const labelMass = rows[0].map((_, l) => rows.reduce((s, row) => s + row[l], 0) / total);
  let mi = 0;
  rows.forEach((row, r) => row.forEach((v, l) => { if (v > 0) mi += (v / total) * Math.log2(v / total / (roleMass[r] * labelMass[l])); }));
  const roleEntropy = -roleMass.reduce((s, v) => s + v * Math.log2(v), 0);
  return { mi, norm: roleEntropy > 0 ? mi / roleEntropy : null };
}

function specialisation(roles) {
  const action = roles.map((r) => Array.from(r.labelCounts));
  const harvest = roles.map((r) => Array.from(r.harvest));
  const channel = roles.map((r) => Array.from(r.chanPos));
  const actionInfo = roleLabelInformation(action);
  const harvestInfo = roleLabelInformation(harvest);
  return { jsAction: meanPairwiseJs(action), jsHarvest: meanPairwiseJs(harvest), jsChannel: meanPairwiseJs(channel), miAction: actionInfo.mi, miActionNorm: actionInfo.norm, miHarvest: harvestInfo.mi, miHarvestNorm: harvestInfo.norm };
}

function roleMetrics(rs, ctx) {
  const metrics = sideMetrics(rs, ctx.hasInfo);
  if (!ctx.hasInfo) return metrics;
  const per1k = (v) => (rs.ticks > 0 ? (v / rs.ticks) * 1000 : null);
  const harvestTotal = rs.harvest.reduce((s, v) => s + v, 0);
  metrics.harvests1k = per1k(harvestTotal);
  HARVEST_TYPES.forEach((name, i) => { metrics['harvest_' + name] = harvestTotal > 0 ? rs.harvest[i] / harvestTotal : null; });
  metrics.bossDmgShare = ctx.bossTotal > 0 ? rs.bossDmg / ctx.bossTotal : null;
  metrics.bossDmg1k = per1k(rs.bossDmg);
  metrics.pvpKills1k = per1k(rs.pvpKills);
  metrics.pvpDmg1k = per1k(rs.pvpDmg);
  metrics.sales1k = per1k(rs.sales);
  metrics.salesUnits1k = per1k(rs.salesUnits);
  metrics.salesGold1k = per1k(rs.salesGold);
  metrics.crossAllyShare = rs.ticks > 0 ? rs.crossAlly / rs.ticks : null;
  metrics.sameAllyShare = rs.ticks > 0 ? rs.sameAlly / rs.ticks : null;
  metrics.bossKillsPart = rs.bossKillsPart;
  metrics.bossKillsMulti = rs.bossKillsMulti;
  ctx.channels.forEach((name, c) => { metrics['ch_' + name + '1k'] = per1k(rs.chan[c]); });
  if (ctx.weights) metrics.rateWeighted = per1k(ctx.weights.reduce((s, w, c) => s + w * (rs.chan[c] || 0), 0));
  if (ctx.baseRate) metrics.ratio = metrics.rate === null ? null : metrics.rate / ctx.baseRate;
  return metrics;
}

function sumSides(list, nOut, channels) {
  const total = newRoleSide(nOut, channels);
  for (const side of list) mergeSide(total, side);
  return total;
}

function populationSeed(bundle, keys, seed, tasks, results, game, hasInfo) {
  const channels = game.rewardChannels || [];
  const nOut = game.dims.nOut;
  const keyed = {};
  for (const key of keys) {
    const roles = bundle.members.map(() => newRoleSide(nOut, channels.length));
    const global = { bossKills: 0, bossKillsMulti: 0, baseReward: 0, baseTicks: 0 };
    tasks.forEach((task, i) => {
      if (task.key !== key || task.seed !== seed) return;
      results[i].roles.forEach((rs, k) => mergeSide(roles[k], rs));
      global.bossKills += results[i].popGlobal.bossKills;
      global.bossKillsMulti += results[i].popGlobal.bossKillsMulti;
      global.baseReward += results[i].baseReward;
      global.baseTicks += results[i].baseTicks;
    });
    keyed[key] = { roles, global };
  }
  const pools = {};
  for (const key of keys) if (!key.startsWith('botsRole:')) pools[key] = { roles: keyed[key].roles, globals: [keyed[key].global], bases: bundle.members.map(() => keyed[key].global) };
  const pureKeys = keys.filter((key) => key.startsWith('botsRole:'));
  if (pureKeys.length > 0) {
    const pureRoles = bundle.members.map((m, k) => (keyed['botsRole:' + m.role] ? keyed['botsRole:' + m.role].roles[k] : newRoleSide(nOut, channels.length)));
    pools.botsRole = { roles: pureRoles, globals: pureKeys.map((key) => keyed[key].global), bases: bundle.members.map((m) => (keyed['botsRole:' + m.role] ? keyed['botsRole:' + m.role].global : null)) };
  }
  const names = Object.keys(pools);
  if (names.length > 1) pools.pooled = { roles: bundle.members.map((m, k) => sumSides(names.map((n) => pools[n].roles[k]), nOut, channels.length)), globals: names.flatMap((n) => pools[n].globals), bases: null };
  const perRole = {};
  const team = {};
  const spec = {};
  for (const [name, pool] of Object.entries(pools)) {
    const bossTotal = pool.roles.reduce((s, r) => s + r.bossDmg, 0);
    perRole[name] = {};
    bundle.members.forEach((member, k) => {
      const base = pool.bases && pool.bases[k] && pool.bases[k].baseTicks > 0 && !DUEL_SUITES.includes(name) ? pool.bases[k].baseReward / pool.bases[k].baseTicks : null;
      perRole[name][member.role] = roleMetrics(pool.roles[k], { hasInfo, bossTotal, channels, weights: member.weights, baseRate: base });
    });
    const ticks = pool.roles.reduce((s, r) => s + r.ticks, 0);
    const kills = pool.globals.reduce((s, g) => s + g.bossKills, 0);
    const multi = pool.globals.reduce((s, g) => s + g.bossKillsMulti, 0);
    team[name] = { bossKills: kills, bossKillsMulti: multi, bossMultiShare: kills > 0 ? multi / kills : null, crossAllyShare: ticks > 0 ? pool.roles.reduce((s, r) => s + r.crossAlly, 0) / ticks : null, sameAllyShare: ticks > 0 ? pool.roles.reduce((s, r) => s + r.sameAlly, 0) / ticks : null, sales1k: ticks > 0 ? (pool.roles.reduce((s, r) => s + r.sales, 0) / ticks) * 1000 : null };
    spec[name] = specialisation(pool.roles);
  }
  return { seed, perRole, team, spec };
}

function flattenInto(value, pathParts, out) {
  if (value !== null && typeof value === 'object') for (const key of Object.keys(value)) flattenInto(value[key], pathParts.concat(key), out);
  else out.push([pathParts, value]);
}

function summarizePopulation(perSeed) {
  const groups = new Map();
  for (const entry of perSeed) {
    const flat = [];
    flattenInto({ perRole: entry.perRole, team: entry.team, specialisation: entry.spec }, [], flat);
    for (const [parts, value] of flat) {
      const id = JSON.stringify(parts);
      if (!groups.has(id)) groups.set(id, { parts, values: [] });
      groups.get(id).values.push(value);
    }
  }
  const summary = { perRole: {}, team: {}, specialisation: {} };
  for (const { parts, values } of groups.values()) {
    let node = summary;
    for (let i = 0; i < parts.length - 1; i++) node = node[parts[i]] || (node[parts[i]] = {});
    node[parts[parts.length - 1]] = meanSd(values);
  }
  return summary;
}

const ROLE_TABLE_KEYS = ['rate', 'ratio', 'rateWeighted', 'life', 'actionEntropy', 'toolTier', 'weaponTier', 'crafts1k', 'buys1k', 'sales1k', 'salesUnits1k', 'harvests1k', 'harvest_berry', 'harvest_tree', 'harvest_ore', 'harvest_spring', 'harvest_great', 'bossDmgShare', 'bossKillsPart', 'bossKillsMulti', 'pvpKills1k', 'pvpDmg1k', 'allyShare', 'crossAllyShare', 'sameAllyShare'];

function renderGrid(rows) {
  const widths = rows[0].map((_, c) => Math.max(...rows.map((r) => r[c].length)));
  return rows.map((r) => r.map((cell, c) => (c === 0 ? cell.padEnd(widths[c]) : cell.padStart(widths[c]))).join('  ')).join('\n');
}

function formatPopulation(population) {
  const lines = ['population roles: ' + population.roles.map((r) => r.index + ':' + r.role + (r.fraction !== null ? '(f=' + r.fraction + ')' : '') + (r.weights ? '(w=' + r.weights.join('/') + ')' : '')).join(' ') + ' assign=' + population.assign];
  for (const [pool, roles] of Object.entries(population.perRole)) {
    const names = Object.keys(roles);
    const keys = [];
    const channelKeys = [];
    for (const name of names) for (const key of Object.keys(roles[name])) if (key.startsWith('ch_') && !channelKeys.includes(key)) channelKeys.push(key);
    for (const key of ROLE_TABLE_KEYS.concat(channelKeys)) if (names.some((name) => roles[name][key] && roles[name][key].mean !== null)) keys.push(key);
    lines.push('', 'population ' + pool + ' per role');
    lines.push(renderGrid([['metric', ...names]].concat(keys.map((key) => [key, ...names.map((name) => formatValue(roles[name][key], key))]))));
  }
  const pools = Object.keys(population.team);
  const teamKeys = ['bossKills', 'bossKillsMulti', 'bossMultiShare', 'crossAllyShare', 'sameAllyShare', 'sales1k'];
  lines.push('', 'population team (trade / cooperation)');
  lines.push(renderGrid([['metric', ...pools]].concat(teamKeys.map((key) => [key, ...pools.map((pool) => formatValue(population.team[pool][key], key))]))));
  const specKeys = ['jsAction', 'jsHarvest', 'jsChannel', 'miAction', 'miActionNorm', 'miHarvest', 'miHarvestNorm'];
  lines.push('', 'population specialisation (bits; js = mean pairwise Jensen-Shannon between roles, mi = I(role;label), Norm = mi / H(role))');
  lines.push(renderGrid([['metric', ...pools]].concat(specKeys.map((key) => [key, ...pools.map((pool) => formatValue(population.specialisation[pool][key], key))]))));
  return lines.join('\n');
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  if (opts.files.length !== 1) throw new Error('usage: node tools/evalsuite.mjs [--game=realm] [--opponent=<genome file>] [--suites=bots,h2h,selfplay | bundle: bots,botsRole,selfplay,h2hSingle] [--assign=roundrobin|fraction] [--seeds=2|1,2,3] [--worlds=4] [--periods=1] [--top=4] [--jobs=N] [--out=file.json] [--json] [--no-write] <genome file or population bundle>');
  if (!ASSIGN_MODES.includes(opts.assign)) throw new Error('--assign must be one of ' + ASSIGN_MODES.join('|'));
  const genomeFile = path.resolve(opts.files[0]);
  const { game, info } = await loadGameModule(opts.game);
  const hasInfo = Boolean(info);
  const canSelfPlay = Boolean(game.maxLearners);
  const config = { game: opts.game, genome: genomeFile, opponent: opts.opponent ? path.resolve(opts.opponent) : '', top: opts.top, assign: opts.assign, lesion: opts.lesion, gate: opts.gate };
  const probe = await prepareContext(config);
  WEIGHTS = opts.popWeights ? opts.popWeights.split(';').map((v) => v.split(',').map(Number)) : [];
  if (WEIGHTS.length && !CHANNELS.length) throw new Error('--popWeights needs a game with rewardChannels');
  for (const w of WEIGHTS) if (w.length !== CHANNELS.length) throw new Error('--popWeights needs ' + CHANNELS.length + ' weights per vector, got ' + w.join(','));
  const bundle = probe.bundle;
  const defaults = bundle ? ['bots', 'botsRole'].concat(canSelfPlay ? ['selfplay'] : [], opts.opponent && canSelfPlay ? ['h2hSingle'] : []) : ['bots'].concat(opts.opponent && canSelfPlay ? ['h2h'] : [], canSelfPlay ? ['selfplay'] : []);
  const requested = opts.suites ? opts.suites.split(',') : defaults;
  const suites = requested.filter((suite) => {
    if (!SUITES.includes(suite)) throw new Error('unknown suite ' + suite);
    if (bundle && !BUNDLE_SUITES.includes(suite)) throw new Error('suite ' + suite + ' needs a single genome (population bundle: use ' + BUNDLE_SUITES.join(',') + ')');
    if (!bundle && !SINGLE_SUITES.includes(suite)) throw new Error('suite ' + suite + ' needs a population bundle');
    if (DUEL_SUITES.includes(suite) && !opts.opponent) throw new Error('suite ' + suite + ' needs --opponent=<genome file>');
    if (suite !== 'bots' && suite !== 'botsRole' && !canSelfPlay) { console.error('suite ' + suite + ' skipped: game ' + game.id + ' has no maxLearners'); return false; }
    return true;
  });
  const runs = suites.flatMap((suite) => (suite === 'botsRole' ? bundle.members.map((m) => ({ key: 'botsRole:' + m.role, suite, role: m.index })) : [{ key: suite, suite, role: -1 }]));
  const seeds = seedList(opts.seeds, opts.baseSeed);
  const started = performance.now();
  const tasks = [];
  for (const run of runs) for (const seed of seeds) for (let world = 0; world < opts.worlds; world++) tasks.push({ id: tasks.length, suite: run.suite, key: run.key, role: run.role, seed, world, periods: opts.periods });
  const jobs = Math.max(1, Math.min(tasks.length, opts.jobs || Math.max(1, Math.min(6, Math.floor(os.cpus().length / 2)))));
  const results = new Array(tasks.length);
  let cpuMs = 0;
  let next = 0;
  let finished = 0;
  await new Promise((resolve, reject) => {
    const workers = Array.from({ length: jobs }, () => new Worker(SELF, { workerData: config }));
    const feed = (worker) => { if (next < tasks.length) worker.postMessage(tasks[next++]); else worker.terminate(); };
    for (const worker of workers) {
      worker.on('error', reject);
      worker.on('message', (message) => {
        if (message.error) { reject(new Error(message.error)); return; }
        results[message.id] = message.result;
        cpuMs += message.ms;
        finished++;
        if (process.stderr.isTTY) process.stderr.write('\r' + finished + '/' + tasks.length + ' world runs');
        if (finished === tasks.length) { workers.forEach((w) => w.terminate()); resolve(); } else feed(worker);
      });
      feed(worker);
    }
  });
  if (process.stderr.isTTY) process.stderr.write('\n');
  const output = { tool: 'evalsuite', version: EVAL_VERSION, genome: genomeFile, opponent: config.opponent || null, game: game.id, lesion: opts.lesion || null, opts: { suites, seeds, worlds: opts.worlds, periods: opts.periods, top: opts.top, gate: opts.gate, popWeights: opts.popWeights || null }, suites: {}, when: new Date().toISOString() };
  if (bundle) output.opts.assign = opts.assign;
  for (const run of runs) {
    const perSeed = seeds.map((seed) => {
      const sideA = newSide(game.dims.nOut);
      const sideB = newSide(game.dims.nOut);
      const totals = new Float64Array(game.statNames.length);
      let baseReward = 0;
      let baseTicks = 0;
      tasks.forEach((task, i) => {
        if (task.key !== run.key || task.seed !== seed) return;
        const r = results[i];
        mergeSide(sideA, r.sides.A);
        mergeSide(sideB, r.sides.B);
        r.game.forEach((v, k) => { totals[k] += v; });
        baseReward += r.baseReward;
        baseTicks += r.baseTicks;
      });
      return { seed, metrics: suiteSeedMetrics(run.suite, sideA, sideB, totals, game.statNames, baseReward, baseTicks, hasInfo) };
    });
    output.suites[run.key] = { perSeed, summary: summarizeSuite(perSeed) };
  }
  if (bundle) {
    const keys = runs.map((run) => run.key);
    const perSeed = seeds.map((seed) => populationSeed(bundle, keys, seed, tasks, results, game, hasInfo));
    output.population = Object.assign({
      roles: bundle.members.map((m) => ({ index: m.index, role: m.role, weights: m.weights, channels: m.channels, fraction: m.fraction })),
      channels: game.rewardChannels || [],
      assign: opts.assign,
      mix: bundle.mix,
      pools: Object.keys(perSeed[0].perRole)
    }, summarizePopulation(perSeed), { perSeed });
  }
  output.timing = { wallSeconds: (performance.now() - started) / 1000, cpuSeconds: cpuMs / 1000, jobs, worldRuns: tasks.length, worldTicksPerSecond: (tasks.length * opts.periods * game.maxAge) / ((performance.now() - started) / 1000) };
  const outFile = opts.out || genomeFile + '.eval.json';
  if (opts.write || opts.out) fs.writeFileSync(outFile, JSON.stringify(output));
  if (opts.json) console.log(JSON.stringify(output));
  else {
    console.log('evalsuite ' + path.basename(genomeFile) + ' game=' + game.id + ' seeds=' + seeds.join(',') + ' worlds/seed=' + opts.worlds + ' periods=' + opts.periods + (config.opponent ? ' opponent=' + path.basename(config.opponent) : ''));
    if (WEIGHTS.length) console.log('weights  ' + WEIGHTS.map((w, i) => (WEIGHTS.length > 1 ? 'rateW' + (i + 1) : 'rateWeighted') + '=' + w.join(',')).join('   '));
    console.log(formatTable(output));
    if (output.population) console.log('\n' + formatPopulation(output.population));
    console.log('time ' + output.timing.wallSeconds.toFixed(1) + 's wall, ' + output.timing.cpuSeconds.toFixed(1) + 's cpu, ' + jobs + ' workers, ' + output.timing.worldRuns + ' world runs');
  }
}

if (isMainThread) main().catch((err) => { console.error(err.stack || String(err)); process.exit(1); });
