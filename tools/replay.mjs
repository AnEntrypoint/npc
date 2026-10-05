import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { Brain, Rng, mix, evoConfig, genomeFromJSON, randomGenome, STAT, STRUCT_PERIOD, REWARD_SCALE } from '../src/core.js';

const TOOL_DIR = path.dirname(fileURLToPath(import.meta.url));
const MAX_EDGES = 256;
const LABELS = ['idle', 'walk', 'interact', 'melee', 'range', 'mage', 'craftTool', 'craftWeap', 'buy'];
const CAUSES = ['mob', 'boss', 'animal', 'player', 'starve', 'age', 'other', 'alive'];
const MOVE_THRESHOLD = 32;

function parseArgs(argv) {
  const opts = { game: '../src/games/realm.js', ticks: 6000, worlds: 4, seed: 7, top: 4, json: false, selfplay: false, assign: 'roundrobin', files: [] };
  for (const arg of argv) {
    const m = /^--([a-z]+)(?:=(.*))?$/.exec(arg);
    if (!m) { opts.files.push(arg); continue; }
    if (m[1] === 'json') opts.json = true;
    else if (m[1] === 'selfplay') opts.selfplay = true;
    else if (m[1] === 'game') opts.game = m[2];
    else if (m[1] === 'assign') opts.assign = m[2];
    else if (m[1] in opts) opts[m[1]] = Number(m[2]);
    else throw new Error('unknown flag --' + m[1]);
  }
  return opts;
}

async function loadGame(spec) {
  const candidates = [path.resolve(TOOL_DIR, spec), path.resolve(process.cwd(), spec)];
  const found = candidates.find((p) => fs.existsSync(p));
  if (!found) throw new Error('game module not found: ' + spec);
  return import(pathToFileURL(found).href);
}

function loadGenomes(file, dims, top) {
  if (file === 'random') return Array.from({ length: top }, (_, i) => randomGenome(new Rng(mix(99, i, 0, 6)), MAX_EDGES, dims));
  const parsed = JSON.parse(fs.readFileSync(file, 'utf8'));
  const members = Array.isArray(parsed) ? parsed : parsed && Array.isArray(parsed.policies) ? parsed.policies : null;
  const bundle = members && members.length > 0 && members.every((m) => m && m.genome) ? members : null;
  const json = bundle ? bundle.map((m) => m.genome) : parsed;
  const list = Array.isArray(json) ? json : [json];
  const usable = list.filter((g) => g.dims && g.dims.nIn === dims.nIn && g.dims.nOut === dims.nOut && (g.dims.nNodes === dims.nNodes || (g.meta && g.meta.trainer === 'ppo')));
  if (usable.length === 0) return null;
  const picked = bundle ? usable : usable.slice(0, top);
  const genomes = picked.map((g) => Object.assign(genomeFromJSON(g, Math.max(MAX_EDGES, g.edges.length)), { dims: g.dims }));
  if (bundle && usable.length === bundle.length) genomes.fractions = bundle.map((m) => (Number.isFinite(m.fraction) && m.fraction > 0 ? m.fraction : 0));
  return genomes;
}

function roleSlot(slot, count, genomes, mode) {
  if (mode !== 'fraction' || !genomes.fractions) return slot % genomes.length;
  const sum = genomes.fractions.reduce((a, v) => a + v, 0);
  const shares = sum > 0 ? genomes.fractions.map((v) => v / sum) : genomes.fractions.map(() => 1 / genomes.length);
  let cumulative = 0;
  for (let k = 0; k < shares.length; k++) {
    cumulative += shares[k];
    if ((slot + 0.5) / count < cumulative && shares[k] > 0) return k;
  }
  return shares.map((v, k) => (v > 0 ? k : -1)).filter((k) => k >= 0).pop();
}

function causeOf(killer, f, b, F, info) {
  if (killer >= info.BOSS0 && killer < info.ANIMAL0) return 'boss';
  if (killer >= info.MOB0 && killer < info.BOSS0) return 'mob';
  if (killer >= info.ANIMAL0) return 'animal';
  if (killer >= 0) return 'player';
  if (f[b + F.FOOD] === 0 || f[b + F.WATER] === 0) return 'starve';
  return 'other';
}

function newLife() {
  return { t: 0, rew: 0, cnt: new Float64Array(LABELS.length), mv: 0, dist: 0, safe: 0, ally: 0, allyL: 0, wild: 0, crafts: 0, buys: 0, ch: [0, 0, 0, 0], harv: [0, 0, 0, 0, 0], arg: new Float64Array(12) };
}

function closeLife(life, cause, f, b, F, info) {
  const lv = [0, 1, 2, 3].map((k) => info.levelOf(f[b + F.XP0 + k]));
  return Object.assign(life, { cause, lv, gold: f[b + F.GOLD], tool: f[b + F.TOOL], weap: f[b + F.WEAP], rare: f[b + F.RARE] });
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

function runWorld(mod, genomes, worldSeed, ticks, selfplay, assign) {
  const { REALM, REALM_INFO: info } = mod;
  const { F, ACT, PLAYERS, FIELDS } = info;
  const LEARNERS = selfplay ? PLAYERS : info.LEARNERS;
  const dims = REALM.dims;
  const nOut = dims.nOut;
  const evo = evoConfig(null);
  const stats = new Int32Array(16);
  const cfg = REALM.defaultCfg();
  if (selfplay) cfg.learnerSlots = PLAYERS;
  const env = REALM.createEnv(worldSeed, cfg, true, stats);
  env.difficulty = 100;
  const brains = [];
  const gOf = (a) => genomes[roleSlot(a, LEARNERS, genomes, assign)];
  for (let a = 0; a < LEARNERS; a++) brains.push(new Brain(gOf(a), Math.max(MAX_EDGES, gOf(a).n), gOf(a).dims || dims, evo));
  const obs = new Float32Array(dims.nIn);
  const actions = new Int32Array(LEARNERS);
  const outputs = new Float32Array(LEARNERS * nOut);
  const lifeTicks = new Int32Array(LEARNERS);
  const prev = new Int32Array(PLAYERS * FIELDS);
  const cur = [];
  for (let e = 0; e < PLAYERS; e++) cur.push(newLife());
  const learnerLives = [];
  const botLives = [];
  const hasAlly = typeof env.allyCount === 'function';
  const f = env.f;
  const allyOf = (e) => {
    if (hasAlly) return env.allyCount(e);
    const r2 = info.ALLY_RADIUS * info.ALLY_RADIUS;
    let n = 0;
    for (let j = 0; j < PLAYERS; j++) {
      if (j === e || f[j * FIELDS + F.DEAD] !== 0) continue;
      const dx = f[j * FIELDS + F.X] - f[e * FIELDS + F.X], dy = f[j * FIELDS + F.Y] - f[e * FIELDS + F.Y];
      if (dx * dx + dy * dy < r2) n++;
    }
    return n;
  };
  prev.set(f.subarray(0, PLAYERS * FIELDS));
  for (let tick = 0; tick < ticks; tick++) {
    for (let a = 0; a < LEARNERS; a++) {
      env.observe(a, obs, tick);
      actions[a] = brains[a].step(obs, new Rng(mix(worldSeed, tick, a, 1)));
      for (let k = 0; k < nOut; k++) outputs[a * nOut + k] = brains[a].act[dims.nIn + k];
    }
    env.step(actions, outputs, tick);
    const respawn = [];
    for (let e = 0; e < PLAYERS; e++) {
      const b = e * FIELDS;
      const life = cur[e];
      const learner = e < LEARNERS;
      life.t++;
      life.rew += (learner ? env.reward[e] : env.rew[e]) / REWARD_SCALE;
      if (env.rewardCh) for (let c = 0; c < 4; c++) life.ch[c] += env.rewardCh[e * 4 + c] / REWARD_SCALE;
      life.cnt[actionLabel(env, e, ACT)]++;
      if (learner) life.arg[actions[e]]++;
      if (Math.hypot(env.ctlX[e], env.ctlY[e]) > MOVE_THRESHOLD) life.mv++;
      if (env.act[e] === ACT.INTERACT && env.tgtNode[e] >= 0) {
        const node = env.tgtNode[e];
        if (info.isGreat(node)) { if (env.gdone[node - info.GREAT0]) life.harv[4]++; }
        else if (env.complete[e] && f[b + F.CHAN] === 0) { const type = info.nodeTypeOf(node); if (type < 4) life.harv[type === 3 ? 3 : type]++; }
      }
      const reset = !learner && f[b + F.AGE] < prev[b + F.AGE];
      const ended = learner ? env.dead[e] !== 0 || life.t >= REALM.maxAge : reset;
      if (!reset) {
        const x = f[b + F.X], y = f[b + F.Y];
        if (info.FAR_ZONE) { env.nearestTown(e); if (env.town.d2 > info.FAR_ZONE * info.FAR_ZONE) life.wild++; }
        if (f[b + F.TOOL] > prev[b + F.TOOL]) life.crafts++;
        if (f[b + F.WEAP] > prev[b + F.WEAP]) life.crafts++;
        if (f[b + F.RAT] > prev[b + F.RAT] && f[b + F.GOLD] === prev[b + F.GOLD] - 10) life.buys++;
        life.dist += Math.hypot(x - prev[b + F.X], y - prev[b + F.Y]);
        if (info.inSafeZone(env, x, y)) life.safe++;
        if (allyOf(e) > 0) life.ally++;
        for (let j = 0; j < LEARNERS; j++) {
          if (j === e || f[j * FIELDS + F.DEAD] !== 0) continue;
          const dx = f[j * FIELDS + F.X] - x, dy = f[j * FIELDS + F.Y] - y;
          if (dx * dx + dy * dy < info.ALLY_RADIUS * info.ALLY_RADIUS) { life.allyL++; break; }
        }
      }
      if (!ended) continue;
      if (learner) {
        const cause = env.dead[e] !== 0 ? causeOf(env.killer[e], f, b, F, info) : 'age';
        learnerLives.push(closeLife(life, cause, f, b, F, info));
        respawn.push(e);
      } else {
        botLives.push(closeLife(life, causeOf(env.killer[e], prev, b, F, info), prev, b, F, info));
      }
      cur[e] = newLife();
    }
    for (let a = 0; a < LEARNERS; a++) {
      brains[a].learn(env.reward[a] / REWARD_SCALE);
      lifeTicks[a]++;
      if (!respawn.includes(a) && lifeTicks[a] % STRUCT_PERIOD === 0) brains[a].structural(new Rng(mix(worldSeed, tick, a, 3)));
    }
    for (const a of respawn) {
      env.respawn(a, tick);
      brains[a].load(gOf(a));
      lifeTicks[a] = 0;
    }
    prev.set(f.subarray(0, PLAYERS * FIELDS));
  }
  const tail = (lives, from, to) => {
    for (let e = from; e < to; e++) if (cur[e].t > 0) lives.push(closeLife(cur[e], 'alive', f, e * FIELDS, F, info));
  };
  tail(learnerLives, 0, LEARNERS);
  tail(botLives, LEARNERS, PLAYERS);
  const game = REALM.statNames.map((_, i) => stats[STAT.GAME0 + i]);
  return { learnerLives, botLives, game };
}

function summarize(lives, gameTotals, statNames, hasStats) {
  const n = lives.length;
  const ticks = lives.reduce((s, l) => s + l.t, 0);
  const cnt = new Float64Array(LABELS.length);
  const causes = Object.fromEntries(CAUSES.map((c) => [c, 0]));
  const chan = [0, 0, 0, 0];
  let rew = 0, dist = 0, safe = 0, ally = 0, allyL = 0, mv = 0, wild = 0, crafts = 0, buys = 0;
  const harv = [0, 0, 0, 0, 0];
  const arg = new Float64Array(12);
  const lv = [0, 0, 0, 0];
  let gold = 0, tool = 0, weap = 0, rare = 0;
  for (const l of lives) {
    rew += l.rew; dist += l.dist; safe += l.safe; ally += l.ally; allyL += l.allyL; mv += l.mv; wild += l.wild; crafts += l.crafts; buys += l.buys;
    for (let c = 0; c < 4; c++) chan[c] += l.ch[c];
    for (let k = 0; k < 5; k++) harv[k] += l.harv[k];
    for (let k = 0; k < 12; k++) arg[k] += l.arg[k];
    for (let k = 0; k < LABELS.length; k++) cnt[k] += l.cnt[k];
    causes[l.cause]++;
    for (let k = 0; k < 4; k++) lv[k] += l.lv[k];
    gold += l.gold; tool += l.tool; weap += l.weap; rare = Math.max(rare, l.rare);
  }
  const share = Array.from(cnt, (c) => c / ticks);
  const argTotal = arg.reduce((a, b) => a + b, 0);
  const argShare = Array.from(arg, (c) => (argTotal > 0 ? c / argTotal : 0));
  const argEntropy = argTotal > 0 ? -argShare.reduce((a, q) => (q > 0 ? a + q * Math.log2(q) : a), 0) : NaN;
  const entropy = -share.reduce((s, p) => (p > 0 ? s + p * Math.log2(p) : s), 0);
  const perLife = {};
  if (hasStats) statNames.forEach((name, i) => { perLife[name] = gameTotals[i] / n; });
  return {
    lives: n,
    ticks,
    meanLife: ticks / n,
    rewPerTick1000: (rew / ticks) * 1000,
    shares: Object.fromEntries(LABELS.map((name, k) => [name, share[k]])),
    movingShare: mv / ticks,
    wildShare: wild / ticks,
    channels1000: chan.map((v) => (v / ticks) * 1000),
    craftsPerLife: crafts / n,
    buysPerLife: buys / n,
    maxShare: Math.max(...share),
    entropy,
    distPer1000: (dist / ticks) * 1000,
    safeShare: safe / ticks,
    allyShare: ally / ticks,
    allyLearnerShare: allyL / ticks,
    argMax: argTotal > 0 ? Math.max(...argShare) : NaN,
    argEntropy,
    argShare,
    harvPerLife: harv.map((v) => v / n),
    causes: Object.fromEntries(CAUSES.map((c) => [c, causes[c] / n])),
    meanLevels: lv.map((v) => v / n),
    meanGold: gold / n,
    meanTool: tool / n,
    meanWeap: weap / n,
    maxRare: rare,
    perLife
  };
}

const pct = (v) => (v * 100).toFixed(0).padStart(3);
const num = (v, d = 1) => (Number.isFinite(v) ? v.toFixed(d) : '-');

function formatTable(rows) {
  const header = ['source', 'lives', 'life', 'r/t*1k', ...LABELS.map((l) => ({ craftTool: 'cTool', craftWeap: 'cWeap' }[l] || l.slice(0, 5))), 'mv%', 'max%', 'H(b)', 'argMax%', 'argH', 'dist/1k', 'safe%', 'ally%', 'allyL%', 'hBerry', 'hTree', 'hOre', 'hSpring', 'hGreat', ...CAUSES.map((c) => 'd:' + c), 'lv0', 'lv1', 'lv2', 'lv3', 'gold', 'tool', 'weap', 'rare', 'mobK', 'great', 'boss', 'pvpK', 'wild%', 'crafts', 'buys'];
  const table = [header];
  for (const { name, s } of rows) {
    const pl = s.perLife;
    table.push([
      name, String(s.lives), num(s.meanLife, 0), num(s.rewPerTick1000, 2),
      ...LABELS.map((l) => pct(s.shares[l])), pct(s.movingShare), pct(s.maxShare), num(s.entropy, 2), Number.isFinite(s.argMax) ? pct(s.argMax) : '-', num(s.argEntropy, 2), num(s.distPer1000, 0), pct(s.safeShare), pct(s.allyShare), pct(s.allyLearnerShare), ...s.harvPerLife.map((v) => num(v, 1)),
      ...CAUSES.map((c) => pct(s.causes[c])),
      ...s.meanLevels.map((v) => num(v, 1)), num(s.meanGold, 0), num(s.meanTool, 1), num(s.meanWeap, 1), String(s.maxRare),
      pl.mobKills === undefined ? '-' : num(pl.mobKills, 2), pl.greatHarvests === undefined ? '-' : num(pl.greatHarvests, 2), pl.bossKills === undefined ? '-' : num(pl.bossKills, 2), pl.pvpKills === undefined ? '-' : num(pl.pvpKills, 2), pct(s.wildShare), num(s.craftsPerLife, 1), num(s.buysPerLife, 1)
    ]);
  }
  const widths = header.map((_, c) => Math.max(...table.map((r) => r[c].length)));
  return table.map((r) => r.map((cell, c) => (c === 0 ? cell.padEnd(widths[c]) : cell.padStart(widths[c]))).join(' ')).join('\n');
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  if (opts.files.length === 0) throw new Error('usage: node tools/replay.mjs [--game=..] [--ticks=6000] [--worlds=4] [--seed=7] [--top=4] [--assign=roundrobin|fraction] [--json] file1 [file2 ...]');
  const mod = await loadGame(opts.game);
  const { REALM } = mod;
  const rows = [];
  const botPool = [];
  const skipped = [];
  for (const file of opts.files) {
    const genomes = loadGenomes(file, REALM.dims, opts.top);
    if (!genomes) { skipped.push(file); continue; }
    const lives = [];
    const gameTotals = new Float64Array(REALM.statNames.length);
    for (let w = 0; w < opts.worlds; w++) {
      const r = runWorld(mod, genomes, mix(opts.seed, w, 0, 9), opts.ticks, opts.selfplay, opts.assign);
      lives.push(...r.learnerLives);
      botPool.push(...r.botLives);
      r.game.forEach((v, i) => { gameTotals[i] += v; });
    }
    rows.push({ name: path.basename(file), s: summarize(lives, gameTotals, REALM.statNames, true), genomes: genomes.length });
  }
  rows.push({ name: 'bots', s: summarize(botPool, null, REALM.statNames, false) });
  if (skipped.length) console.error('skipped (dims mismatch): ' + skipped.join(', '));
  console.log(`game=${REALM.id} ticks=${opts.ticks} worlds=${opts.worlds} seed=${opts.seed} top=${opts.top}`);
  console.log(formatTable(rows));
  console.log('channels per 1000 ticks (survival progress combat cooperation): ' + rows.map((r) => r.name + ' ' + r.s.channels1000.map((v) => v.toFixed(2)).join('/')).join('  '));
  if (opts.json) console.log(JSON.stringify({ game: REALM.id, opts: { ticks: opts.ticks, worlds: opts.worlds, seed: opts.seed, top: opts.top }, skipped, rows }));
}

main().catch((err) => { console.error(err.stack || String(err)); process.exit(1); });
