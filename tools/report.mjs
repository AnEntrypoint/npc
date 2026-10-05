import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { lockStatus } from './gpulock.mjs';
import { nearestTick, evalRatio } from './abjudge.mjs';

const TOOL_DIR = path.dirname(fileURLToPath(import.meta.url));
const RUNS = path.resolve(TOOL_DIR, '..', 'runs');
const REPORT_FILE = path.join(RUNS, 'REPORT.md');
const QUEUE_STATE = path.join(RUNS, 'queue-state.json');
const IGNORED_LOGS = new Set(['queue.log']);

function skippedJobs() {
  const skipped = new Set();
  try {
    const state = JSON.parse(fs.readFileSync(path.join(RUNS, 'queue-state.json'), 'utf8'));
    for (const job of state.jobs || []) if (job.status === 'skipped') skipped.add(job.name);
  } catch {}
  return skipped;
}
const EVAL_VERSION = 2;
const SMOOTH_LINES = 3;
const PARTIAL_GRACE_MS = 5 * 60000;

const LOG_METRICS = [
  { key: 'ticks', label: 'ticks', dir: 0, digits: 0 },
  { key: 'evalReward', label: 'eval r/t x1e3', dir: 1, digits: 2, scale: 1000 },
  { key: 'evalBase', label: 'bots x1e3', dir: 0, digits: 2, scale: 1000 },
  { key: 'ratio', label: 'ratio', dir: 1, digits: 2 },
  { key: 'evalLife', label: 'eval life', dir: 1, digits: 0 },
  { key: 'h2hRatio', label: 'h2h ratio', dir: 1, digits: 2 },
  { key: 'selfPlayReward', label: 'self-play x1e3', dir: 1, digits: 2, scale: 1000 }
];
const SUITE_METRICS = [
  { key: 'suiteRatio', label: 'suite ratio', dir: 1, digits: 2, from: ['bots', 'ratio'] },
  { key: 'suiteLife', label: 'life', dir: 1, digits: 0, from: ['bots', 'life'] },
  { key: 'suiteH2h', label: 'suite h2h', dir: 1, digits: 2, from: ['h2h', 'ratio'] },
  { key: 'suiteSelfPlay', label: 'suite self-play x1e3', dir: 1, digits: 2, scale: 1000, from: ['selfplay', 'rate'] },
  { key: 'actionEntropy', label: 'act H', dir: 0, digits: 2, from: ['bots', 'actionEntropy'] },
  { key: 'actionMaxShare', label: 'max act%', dir: -1, digits: 2, from: ['bots', 'actionMaxShare'] },
  { key: 'dist1k', label: 'dist/1k', dir: 0, digits: 0, from: ['bots', 'dist1k'] },
  { key: 'crafts1k', label: 'craft/1k', dir: 0, digits: 2, from: ['bots', 'crafts1k'] },
  { key: 'buys1k', label: 'buy/1k', dir: 0, digits: 2, from: ['bots', 'buys1k'] },
  { key: 'gearTierMax', label: 'gear', dir: 1, digits: 0, from: ['bots', 'gearTierMax'] },
  { key: 'allyLearnerShare', label: 'ally%', dir: 0, digits: 2, from: ['bots', 'allyLearnerShare'] },
  { key: 'death_starve', label: 'starve/life', dir: -1, digits: 2, from: ['bots', 'death_starve'] },
  { key: 'bossKills', label: 'boss/life', dir: 0, digits: 2, from: ['bots', 'bossKills'] },
  { key: 'pvpKills', label: 'pvp/life', dir: 0, digits: 2, from: ['bots', 'pvpKills'] },
  { key: 'mobKills', label: 'mob/life', dir: 0, digits: 2, from: ['bots', 'mobKills'] }
];
const MAIN_COLUMNS = ['ticks', 'evalReward', 'evalBase', 'ratio', 'evalLife', 'h2hRatio', 'selfPlayReward', 'suiteRatio', 'suiteH2h', 'suiteSelfPlay'];
const BEHAVIOUR_COLUMNS = ['suiteLife', 'actionEntropy', 'actionMaxShare', 'dist1k', 'crafts1k', 'buys1k', 'gearTierMax', 'allyLearnerShare', 'death_starve', 'bossKills', 'pvpKills', 'mobKills'];
const POP_MIX_ROLE = 'MIX';
const POP_METRICS = [
  { key: 'popMean', label: 'pop mean x1e3', dir: 1, digits: 2, scale: 1000 },
  { key: 'popBest', label: 'pop best x1e3', dir: 1, digits: 2, scale: 1000 },
  { key: 'popRatio', label: 'pop ratio', dir: 1, digits: 2 },
  { key: 'popMix', label: 'mix x1e3', dir: 1, digits: 2, scale: 1000 },
  { key: 'popMixRatio', label: 'mix ratio', dir: 1, digits: 2 }
];
const POP_SUITE_METRICS = [
  { key: 'popJsAction', label: 'JS act', dir: 0, digits: 3, path: ['specialisation', 'bots', 'jsAction'] },
  { key: 'popJsHarvest', label: 'JS harv', dir: 0, digits: 3, path: ['specialisation', 'bots', 'jsHarvest'] },
  { key: 'popMiActionNorm', label: 'MI act/H', dir: 0, digits: 3, path: ['specialisation', 'bots', 'miActionNorm'] },
  { key: 'popJsChannel', label: 'JS chan', dir: 0, digits: 3, path: ['specialisation', 'bots', 'jsChannel'] },
  { key: 'popCrossAlly', label: 'cross ally%', dir: 0, digits: 2, path: ['team', 'bots', 'crossAllyShare'] },
  { key: 'popBossMulti', label: 'boss multi%', dir: 0, digits: 2, path: ['team', 'bots', 'bossMultiShare'] }
];
const POP_ROLE_COLUMNS = [
  { column: 'rate', tokens: ['rate', 'trainRate'], digits: 2, scale: 1000 },
  { column: 'eval', tokens: ['evalRate'], digits: 2, scale: 1000 },
  { column: 'life', tokens: ['life'], digits: 0, scale: 1 },
  { column: 'ratio', tokens: ['ratio'], digits: 2, scale: 1 },
  { column: 'mix', tokens: ['mixRate'], digits: 2, scale: 1000 },
  { column: 'h2h', tokens: ['h2h'], digits: 2, scale: 1 }
];
const ALL_METRICS = LOG_METRICS.concat(SUITE_METRICS, POP_METRICS, POP_SUITE_METRICS);

function parseArgs(argv) {
  const opts = { since: 72, limit: 60, baseline: path.join(RUNS, 'baseline.json'), match: '', all: false, setBaseline: null, eval: false, evalSeeds: 2, evalWorlds: 2, json: false, write: true, atTick: 0, files: [] };
  for (const arg of argv) {
    const m = /^--([a-zA-Z-]+)(?:=(.*))?$/.exec(arg);
    if (!m) { opts.files.push(arg); continue; }
    const key = m[1].replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    if (key === 'noWrite') opts.write = false;
    else if (key === 'setBaseline') opts.setBaseline = m[2] === undefined ? '' : m[2];
    else if (key in opts && typeof opts[key] === 'boolean') opts[key] = true;
    else if (key in opts && typeof opts[key] === 'number') opts[key] = Number(m[2]);
    else if (key in opts && typeof opts[key] === 'string') opts[key] = m[2];
    else throw new Error('unknown flag --' + m[1]);
  }
  return opts;
}

function parseKeyValues(body) {
  const out = {};
  for (const piece of body.split(',')) {
    const eq = piece.indexOf('=');
    if (eq > 0 && !(piece.slice(0, eq) in out)) out[piece.slice(0, eq)] = piece.slice(eq + 1);
  }
  return out;
}

function parsePopBody(body) {
  const roles = {};
  for (const entry of body.split(';')) {
    const colon = entry.indexOf(':');
    if (colon <= 0) continue;
    const head = entry.slice(0, colon).trim();
    const eq = head.indexOf('=');
    const role = eq < 0 ? head : head.slice(eq + 1) || head.slice(0, eq);
    roles[role] = parseKeyValues(entry.slice(colon + 1));
  }
  return roles;
}

function parseLine(line) {
  const groups = {};
  const rest = line.replace(/(\w+)\[([^\]]*)\]/g, (_, name, body) => { groups[name] = name === 'POP' ? parsePopBody(body) : parseKeyValues(body); return ''; });
  const fields = {};
  let label = null;
  for (const token of rest.split(/\s+/).filter(Boolean)) {
    const eq = token.indexOf('=');
    if (eq > 0) fields[token.slice(0, eq)] = token.slice(eq + 1);
    else if (/^\d+s$/.test(token)) fields.elapsed = parseInt(token, 10);
    else if (label === null && fields.tick === undefined && !/^(created|PASS|FAIL|tests|profile|bench|ERR|done)$/.test(token)) label = token;
  }
  return { label, fields, groups };
}

const num = (text) => { const v = Number(text); return Number.isFinite(v) ? v : null; };

function popFromEntries(entries) {
  const sums = {};
  for (const entry of entries) {
    for (const [role, tokens] of Object.entries(entry.groups.POP || {})) {
      for (const [token, text] of Object.entries(tokens)) {
        const value = num(text);
        if (value === null) continue;
        sums[role] = sums[role] || {};
        (sums[role][token] = sums[role][token] || []).push(value);
      }
    }
  }
  const roles = Object.keys(sums);
  if (roles.length === 0) return null;
  return Object.fromEntries(roles.map((role) => [role, Object.fromEntries(Object.entries(sums[role]).map(([token, values]) => [token, values.reduce((a, v) => a + v, 0) / values.length]))]));
}

function popAggregate(pop, kind) {
  const roles = Object.entries(pop).filter(([role]) => role !== POP_MIX_ROLE).map(([, tokens]) => tokens);
  const evalRates = roles.map((r) => r.evalRate).filter((v) => v !== undefined);
  if (kind === 'mean') return evalRates.length ? evalRates.reduce((a, v) => a + v, 0) / evalRates.length : null;
  if (kind === 'best') return evalRates.length ? Math.max(...evalRates) : null;
  const ratios = roles.map((r) => (r.ratio !== undefined ? r.ratio : r.evalRate !== undefined && r.evalBase ? r.evalRate / r.evalBase : undefined)).filter((v) => v !== undefined);
  return ratios.length ? ratios.reduce((a, v) => a + v, 0) / ratios.length : null;
}

function nearestToTick(entries, target) {
  const best = nearestTick(entries.map((entry) => ({ entry, tick: num(entry.fields.tick) })), target);
  return best === null ? null : best.entry;
}

function parseLog(file, atTick) {
  const name = path.basename(file, '.log');
  const stat = fs.statSync(file);
  const lines = fs.readFileSync(file, 'utf8').split(/\r?\n/).filter(Boolean);
  const byLabel = new Map();
  const meta = { game: null, seed: null, done: false, error: null };
  for (const line of lines) {
    if (/^done$/.test(line.trim()) || /^(tests|profile|bench) done/.test(line)) meta.done = true;
    if (line.startsWith('ERR ')) meta.error = line.slice(4, 120);
    if (line.startsWith('created')) {
      const gameMatch = /\bgame=(\w+)/.exec(line) || /"game":"(\w+)"/.exec(line);
      const seedMatch = /\bseed=(\d+)/.exec(line) || /"seed":(\d+)/.exec(line);
      if (gameMatch) meta.game = gameMatch[1];
      if (seedMatch) meta.seed = Number(seedMatch[1]);
      continue;
    }
    const parsed = parseLine(line);
    if (parsed.fields.tick === undefined || parsed.fields.EVAL === undefined) continue;
    const label = parsed.label === null ? null : parsed.label;
    if (!byLabel.has(label)) byLabel.set(label, []);
    byLabel.get(label).push(parsed);
  }
  const rows = [];
  for (const [label, entries] of byLabel) {
    const matched = atTick === null ? null : nearestToTick(entries, atTick);
    const recent = matched ? [matched] : entries.slice(-SMOOTH_LINES);
    const mean = (pick) => { const values = recent.map(pick).filter((v) => v !== null && v !== undefined); return values.length ? values.reduce((s, v) => s + v, 0) / values.length : null; };
    const last = matched || entries[entries.length - 1];
    const seedField = last.fields.seed !== undefined ? Number(last.fields.seed) : meta.seed;
    const evalBase = mean((e) => num(e.fields.evalBase));
    const evalReward = mean((e) => num(e.fields.EVAL));
    const pop = popFromEntries(recent);
    rows.push({
      name, label, game: meta.game, seed: seedField, mtime: stat.mtimeMs, done: meta.done, error: meta.error, lines: entries.length, matchedTick: matched ? num(matched.fields.tick) : null,
      values: {
        ticks: num(last.fields.tick),
        evalReward,
        evalBase,
        ratio: evalRatio(evalReward, evalBase),
        evalLife: mean((e) => num((e.fields.life || '').split('/').pop())),
        h2hRatio: mean((e) => (e.groups.H2H ? num(e.groups.H2H.ratio) : null)),
        selfPlayReward: mean((e) => (e.groups.SP ? num(e.groups.SP.rew) : null)),
        popMean: pop ? popAggregate(pop, 'mean') : null,
        popBest: pop ? popAggregate(pop, 'best') : null,
        popRatio: pop ? popAggregate(pop, 'ratio') : null,
        popMix: pop && pop[POP_MIX_ROLE] ? pop[POP_MIX_ROLE].evalRate ?? null : null,
        popMixRatio: pop && pop[POP_MIX_ROLE] ? pop[POP_MIX_ROLE].ratio ?? null : null
      },
      pop
    });
  }
  return rows;
}

function groupKey(row) {
  const seedInName = /-s(\d+)(?=-|$)/.exec(row.name);
  const base = seedInName ? row.name.replace(seedInName[0], '') : row.name;
  const suffix = row.label && row.label !== 'base' ? '-' + row.label : '';
  const seed = seedInName ? Number(seedInName[1]) : row.seed;
  return { key: base + suffix, seed };
}

function championPath(row) {
  const candidates = row.label === null ? [row.name] : [row.name + '-' + row.label, row.name];
  for (const candidate of candidates) {
    const file = path.join(RUNS, candidate);
    if (fs.existsSync(file) && fs.statSync(file).isFile()) return file;
  }
  return null;
}

function peekGame(file) {
  const fd = fs.openSync(file, 'r');
  const buffer = Buffer.alloc(240);
  const bytes = fs.readSync(fd, buffer, 0, 240, 0);
  fs.closeSync(fd);
  const m = /"game":"(\w+)"/.exec(buffer.toString('utf8', 0, bytes));
  return m ? m[1] : null;
}

function readEval(championFile, options, game) {
  const evalFile = championFile + '.eval.json';
  const fresh = () => fs.existsSync(evalFile) && fs.statSync(evalFile).mtimeMs >= fs.statSync(championFile).mtimeMs;
  if (!fresh() && options.eval) {
    const args = [path.join(TOOL_DIR, 'evalsuite.mjs'), '--game=' + (game || 'realm'), '--seeds=' + options.evalSeeds, '--worlds=' + options.evalWorlds, '--suites=' + (game === 'blob' ? 'bots' : 'bots,selfplay'), championFile];
    const result = spawnSync(process.execPath, args, { encoding: 'utf8', timeout: 3600000 });
    if (result.status !== 0) return { error: (result.stderr || '').trim().split('\n')[0] || 'evalsuite failed' };
  }
  if (!fresh()) return null;
  try { const parsed = JSON.parse(fs.readFileSync(evalFile, 'utf8')); return parsed.version === EVAL_VERSION ? parsed : null; } catch { return null; }
}

function suiteValues(evalJson) {
  const values = {};
  if (!evalJson || evalJson.error) return values;
  for (const metric of SUITE_METRICS) {
    const [suite, key] = metric.from;
    const stat = evalJson.suites[suite] && evalJson.suites[suite].summary[key];
    values[metric.key] = stat && stat.mean !== null ? stat.mean : null;
  }
  return values;
}

function popEvalValues(evalJson) {
  const values = {};
  if (!evalJson || evalJson.error || !evalJson.population) return values;
  for (const metric of POP_SUITE_METRICS) {
    const [section, pool, key] = metric.path;
    const entry = evalJson.population[section] && evalJson.population[section][pool] && evalJson.population[section][pool][key];
    values[metric.key] = entry && entry.mean !== null ? entry.mean : null;
  }
  return values;
}

function stat(values) {
  const finite = values.filter((v) => typeof v === 'number' && Number.isFinite(v));
  if (finite.length === 0) return null;
  const mean = finite.reduce((s, v) => s + v, 0) / finite.length;
  const sd = finite.length > 1 ? Math.sqrt(finite.reduce((s, v) => s + (v - mean) * (v - mean), 0) / (finite.length - 1)) : null;
  return { mean, sd, n: finite.length };
}

function formatStat(s, metric) {
  if (!s) return '-';
  const scale = metric.scale || 1;
  const mean = (s.mean * scale).toFixed(metric.digits);
  return s.sd === null ? mean : mean + ' ± ' + (s.sd * scale).toFixed(metric.digits);
}

function buildGroups(rows, options) {
  const groups = new Map();
  for (const row of rows) {
    const { key, seed } = groupKey(row);
    if (!groups.has(key)) groups.set(key, { key, members: new Map(), game: row.game, mtime: 0 });
    const group = groups.get(key);
    const id = seed === null ? row.name + ':' + row.label : String(seed);
    const existing = group.members.get(id);
    if (!existing || existing.mtime < row.mtime) group.members.set(id, Object.assign({ seed }, row));
    group.mtime = Math.max(group.mtime, row.mtime);
    group.game = group.game || row.game;
  }
  const out = [];
  for (const group of groups.values()) {
    const members = Array.from(group.members.values());
    const champion = members.map((m) => ({ member: m, file: championPath(m) }));
    for (const entry of champion) {
      if (entry.file) entry.member.game = entry.member.game || peekGame(entry.file);
      entry.evalJson = entry.file ? readEval(entry.file, options, entry.member.game) : null;
      entry.suite = Object.assign(suiteValues(entry.evalJson), popEvalValues(entry.evalJson));
    }
    const stats = {};
    for (const metric of LOG_METRICS) stats[metric.key] = stat(members.map((m) => m.values[metric.key]));
    for (const metric of SUITE_METRICS.concat(POP_SUITE_METRICS)) stats[metric.key] = stat(champion.map((c) => c.suite[metric.key]));
    for (const metric of POP_METRICS) stats[metric.key] = stat(members.map((m) => m.values[metric.key]));
    const popRoles = {};
    for (const member of members) {
      for (const [role, tokens] of Object.entries(member.pop || {})) {
        if (role === POP_MIX_ROLE) continue;
        for (const column of POP_ROLE_COLUMNS) {
          const token = column.tokens.find((t) => tokens[t] !== undefined);
          if (token) ((popRoles[role] = popRoles[role] || {})[column.column] = popRoles[role][column.column] || []).push(tokens[token]);
        }
      }
    }
    for (const role of Object.keys(popRoles)) for (const column of Object.keys(popRoles[role])) popRoles[role][column] = stat(popRoles[role][column]);
    const notes = [];
    if (members.length < 2) notes.push('n=1');
    if (members.some((m) => !m.done)) notes.push(members.every((m) => !m.done && Date.now() - m.mtime < PARTIAL_GRACE_MS) ? 'running' : 'partial');
    if (members.some((m) => m.error)) notes.push('ERR: ' + members.find((m) => m.error).error.slice(0, 50));
    const missingEval = champion.filter((c) => c.file && !c.evalJson).length;
    if (missingEval) notes.push(missingEval + ' without evalsuite');
    const evalErrors = champion.filter((c) => c.evalJson && c.evalJson.error);
    if (evalErrors.length) notes.push('evalsuite: ' + evalErrors[0].evalJson.error.slice(0, 60));
    out.push({ key: group.key, game: group.game || (members.find((m) => m.game) || {}).game, seeds: members.map((m) => m.seed).filter((s) => s !== null).sort((a, b) => a - b), n: members.length, mtime: group.mtime, stats, popRoles, notes, matchedTick: options.atTick ? stat(members.map((m) => m.matchedTick)) : undefined });
  }
  return out.sort((a, b) => b.mtime - a.mtime);
}

function compareToBaseline(group, baseline) {
  const reference = baseline && baseline.groups[group.key];
  if (!reference) return { flags: [], delta: null };
  const flags = [];
  for (const metric of ALL_METRICS) {
    const current = group.stats[metric.key];
    const before = reference[metric.key];
    if (!current || !before || metric.dir === 0) continue;
    const noise = Math.max(current.sd || 0, before.sd || 0);
    const tolerance = Math.max(2 * noise, 0.1 * Math.abs(before.mean));
    const change = metric.dir * (current.mean - before.mean);
    if (change < -tolerance) flags.push('REGRESSION ' + metric.label + ' ' + formatStat(before, metric) + ' -> ' + formatStat({ mean: current.mean, sd: null }, metric));
  }
  const ratio = group.stats.ratio;
  const ratioBefore = reference.ratio;
  return { flags, delta: ratio && ratioBefore && ratioBefore.mean ? (ratio.mean - ratioBefore.mean) / ratioBefore.mean : null };
}

function table(header, rows) {
  const lines = ['| ' + header.join(' | ') + ' |', '|' + header.map((_, c) => (c === 0 ? ' --- ' : ' ---: ')).join('|') + '|'];
  for (const row of rows) lines.push('| ' + row.join(' | ') + ' |');
  return lines.join('\n');
}

function queueSection() {
  if (!fs.existsSync(QUEUE_STATE)) return '';
  const state = JSON.parse(fs.readFileSync(QUEUE_STATE, 'utf8'));
  const lock = lockStatus();
  const counts = {};
  for (const job of state.jobs) counts[job.status] = (counts[job.status] || 0) + 1;
  const active = state.jobs.filter((j) => j.status === 'running' || j.status === 'waiting');
  const lines = ['## Queue', '', 'jobs: ' + Object.entries(counts).map(([k, v]) => v + ' ' + k).join(', ') + ' | gpu lock: ' + (lock.held ? lock.owner + (lock.job ? ' (' + lock.job + ')' : '') : 'free')];
  for (const job of active) lines.push('- ' + job.name + ' ' + job.status + (job.reason ? ': ' + job.reason : '') + (job.lastLine ? ' | ' + job.lastLine.slice(0, 120) : ''));
  for (const job of state.jobs.filter((j) => j.status === 'failed' || j.status === 'timeout')) lines.push('- ' + job.name + ' ' + job.status + ': ' + job.reason);
  return lines.join('\n') + '\n';
}

function saveBaseline(file, groups, names, existing) {
  const baseline = existing || { created: new Date().toISOString(), groups: {} };
  baseline.updated = new Date().toISOString();
  const wanted = names.length ? new Set(names) : null;
  for (const group of groups) {
    if (wanted && !wanted.has(group.key)) continue;
    baseline.groups[group.key] = {};
    for (const metric of ALL_METRICS) if (group.stats[metric.key]) baseline.groups[group.key][metric.key] = group.stats[metric.key];
  }
  fs.writeFileSync(file, JSON.stringify(baseline, null, 1));
  return baseline;
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  const cutoff = options.all ? 0 : Date.now() - options.since * 3600000;
  const matcher = options.match ? new RegExp(options.match) : null;
  const atTick = Number.isFinite(options.atTick) && options.atTick > 0 ? options.atTick : null;
  const rows = [];
  const skipped = skippedJobs();
  for (const file of fs.readdirSync(RUNS)) {
    if (!file.endsWith('.log') || IGNORED_LOGS.has(file) || skipped.has(file.slice(0, -4))) continue;
    const full = path.join(RUNS, file);
    if (fs.statSync(full).mtimeMs < cutoff || (matcher && !matcher.test(file))) continue;
    rows.push(...parseLog(full, atTick));
  }
  const groups = buildGroups(rows, options);
  const baseline = fs.existsSync(options.baseline) ? JSON.parse(fs.readFileSync(options.baseline, 'utf8')) : null;
  if (options.setBaseline !== null) {
    saveBaseline(options.baseline, groups, options.setBaseline ? options.setBaseline.split(',') : [], baseline);
    console.error('baseline written for ' + (options.setBaseline || groups.length + ' groups') + ' -> ' + options.baseline);
  }
  const metricsByKey = Object.fromEntries(ALL_METRICS.map((m) => [m.key, m]));
  const compared = groups.map((group) => Object.assign({ group }, compareToBaseline(group, options.setBaseline !== null ? null : baseline)));
  if (options.json) { console.log(JSON.stringify({ groups, regressions: compared.filter((c) => c.flags.length).map((c) => ({ group: c.group.key, flags: c.flags })) })); return; }
  const mainHeader = ['run', 'game', 'seeds'].concat(MAIN_COLUMNS.map((k) => metricsByKey[k].label), ['vs baseline', 'notes']);
  const mainRows = compared.slice(0, options.limit).map(({ group, flags, delta }) => [group.key, group.game || '-', group.seeds.length ? group.seeds.join(',') : String(group.n)].concat(MAIN_COLUMNS.map((k) => formatStat(group.stats[k], metricsByKey[k])), [delta === null ? '-' : (delta >= 0 ? '+' : '') + (delta * 100).toFixed(0) + '%', flags.length ? 'REGRESSION' : group.notes.join('; ') || '-']));
  const behaviourGroups = compared.slice(0, options.limit).filter(({ group }) => BEHAVIOUR_COLUMNS.some((k) => group.stats[k]));
  const behaviourRows = behaviourGroups.map(({ group }) => [group.key].concat(BEHAVIOUR_COLUMNS.map((k) => formatStat(group.stats[k], metricsByKey[k]))));
  const regressions = compared.filter((c) => c.flags.length);
  const popGroups = compared.slice(0, options.limit).filter(({ group }) => Object.keys(group.popRoles).length > 0);
  const popRoleNames = Array.from(new Set(popGroups.flatMap(({ group }) => Object.keys(group.popRoles)))).sort();
  const popColumns = POP_ROLE_COLUMNS.filter((c) => popGroups.some(({ group }) => Object.values(group.popRoles).some((r) => r[c.column])));
  const popSuiteMetrics = POP_SUITE_METRICS.filter((m) => popGroups.some(({ group }) => group.stats[m.key]));
  const popMetrics = POP_METRICS.filter((m) => popGroups.some(({ group }) => group.stats[m.key]));
  const popHeader = ['run', 'seeds'].concat(popMetrics.map((m) => m.label), popSuiteMetrics.map((m) => m.label), popRoleNames.flatMap((role) => popColumns.map((c) => role + '.' + c.column)));
  const popRows = popGroups.map(({ group }) => [group.key, group.seeds.length ? group.seeds.join(',') : String(group.n)].concat(
    popMetrics.map((m) => formatStat(group.stats[m.key], m)),
    popSuiteMetrics.map((m) => formatStat(group.stats[m.key], m)),
    popRoleNames.flatMap((role) => popColumns.map((c) => formatStat(group.popRoles[role] && group.popRoles[role][c.column], c)))
  ));
  const sections = [
    '# Training report',
    '',
    'generated ' + new Date().toISOString() + ' | logs modified in the last ' + (options.all ? 'all time' : options.since + 'h') + ' | mean ± sd across seeds (sample sd) | ' + (atTick === null ? 'eval columns are the mean of the last ' + SMOOTH_LINES + ' log windows' : 'eval columns are the single log window nearest tick ' + atTick + '; ticks = the tick matched per seed'),
    '',
    queueSection(),
    '## Runs',
    '',
    mainRows.length ? table(mainHeader, mainRows) : 'no logs found',
    '',
    '## Behaviour (evalsuite, vs bots fair start)',
    '',
    behaviourRows.length ? table(['run'].concat(BEHAVIOUR_COLUMNS.map((k) => metricsByKey[k].label)), behaviourRows) : 'no evalsuite results yet: run `node tools/evalsuite.mjs runs/<champion>` or `node tools/report.mjs --eval`',
    '',
    ...(popRows.length ? ['## Population (POP[..] log groups; per role: rate = train reward/tick x1e3, eval = pure-role benchmark x1e3, life, ratio vs bots, mix = rate inside the mixed team x1e3; MIX entry = mixed-team benchmark; JS/MI/cross ally/boss multi columns come from evalsuite bots)', '', table(popHeader, popRows), ''] : []),
    '## Regressions vs baseline',
    '',
    baseline ? (regressions.length ? regressions.map((c) => '- ' + c.group.key + ': ' + c.flags.join('; ')).join('\n') : 'none (baseline ' + baseline.updated + ', ' + Object.keys(baseline.groups).length + ' groups)') : 'no baseline: `node tools/report.mjs --set-baseline`'
  ];
  const text = sections.join('\n') + '\n';
  console.log(text);
  if (options.write) fs.writeFileSync(REPORT_FILE, text);
}

main();
