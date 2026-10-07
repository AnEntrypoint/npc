import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync, spawnSync } from 'node:child_process';

// Keeps `champion.json` (what play.html ships) pointed at the strongest genome.
// usage: node tools/publish.mjs [--candidates=a,b] [--seeds=7,8] [--worlds=4] [--max=8] [--dry] [--force]
// Guidance: >=2 seeds before adopting, and never run this while a GPU job is live --
// CPU eval batches and GPU training stall each other, so the guard refuses by default.

const ROOT = path.resolve(import.meta.dirname, '..');
const args = process.argv.slice(2);
const opt = (name, fallback) => {
  const hit = args.find((a) => a.startsWith('--' + name + '='));
  return hit ? hit.slice(name.length + 3) : fallback;
};
const has = (name) => args.includes('--' + name);

const CHAMPION = path.join(ROOT, opt('out', 'champion.json'));
const META = CHAMPION.replace(/\.json$/, '.meta.json');
const TMP = path.join(ROOT, 'runs', '.publishtmp');
const seeds = opt('seeds', '7,8,9,10');
const worlds = opt('worlds', '4');
const maxCandidates = Number(opt('max', '8'));

function lockHeld() {
  const file = path.join(ROOT, 'runs', '.gpu.lock');
  if (!fs.existsSync(file)) return null;
  try {
    const lock = JSON.parse(fs.readFileSync(file, 'utf8'));
    const alive = spawnSync('powershell', ['-NoProfile', '-NonInteractive', '-Command',
      `if (Get-Process -Id ${Number(lock.pid) || 0} -ErrorAction SilentlyContinue) { 'yes' } else { 'no' }`],
      { encoding: 'utf8', windowsHide: true }).stdout.trim();
    return alive === 'yes' ? lock : null;
  } catch { return null; }
}

function candidates() {
  const given = opt('candidates', '');
  if (given) return given.split(',').map((p) => path.resolve(ROOT, p.trim()));
  const dir = path.join(ROOT, 'runs');
  const all = fs.readdirSync(dir)
    .filter((f) => /-base$/.test(f) && !/-best$/.test(f))
    .map((f) => ({ file: path.join(dir, f), m: fs.statSync(path.join(dir, f)).mtimeMs }))
    .sort((a, b) => b.m - a.m);
  const pool = has('all') ? all : all.slice(0, maxCandidates);
  if (fs.existsSync(CHAMPION)) pool.unshift({ file: CHAMPION, m: fs.statSync(CHAMPION).mtimeMs });
  return [...new Set(pool.map((p) => p.file))];
}

function readable(file) {
  try {
    const g = JSON.parse(fs.readFileSync(file, 'utf8'));
    return g && (g.format === 'npc-brain/2' || g.format === 'npc-brain/3') ? g : null;
  } catch { return null; }
}

function score(file, tag) {
  const out = path.join(TMP, tag + '.json');
  const cmd = ['tools/evalsuite.mjs', '--game=realm', '--suites=bots', '--seeds=' + seeds,
    '--worlds=' + worlds, '--periods=1', '--jobs=1', '--out=' + out, file];
  const started = Date.now();
  const res = spawnSync(process.execPath, cmd, { cwd: ROOT, encoding: 'utf8', windowsHide: true });
  if (res.status !== 0) return { file, error: (res.stderr || res.stdout || '').split('\n').slice(0, 3).join(' ') };
  const j = JSON.parse(fs.readFileSync(out, 'utf8'));
  const m = j.suites.bots.summary;
  return {
    file,
    rate: m.rate && m.rate.mean,
    rateSd: m.rate && m.rate.sd,
    ratio: m.ratio && m.ratio.mean,
    life: m.life && m.life.mean,
    toolTier: m.toolTier && m.toolTier.mean,
    seconds: Math.round((Date.now() - started) / 1000)
  };
}

const lock = lockHeld();
if (lock && !has('force')) {
  console.error(`refusing: GPU lock held by ${lock.owner} job ${lock.job} (pid ${lock.pid}). CPU eval and GPU training stall each other -- pass --force to override.`);
  process.exit(2);
}

const list = candidates().filter((f) => fs.existsSync(f) && readable(f));
if (!list.length) { console.error('no readable npc-brain/2 or /3 candidates'); process.exit(2); }
fs.mkdirSync(TMP, { recursive: true });

console.log(`publish: ${list.length} candidate(s), seeds=${seeds} worlds=${worlds}`);
const results = [];
for (const file of list) {
  const r = score(file, path.basename(file).replace(/[^\w.-]/g, '_'));
  results.push(r);
  if (r.error) console.log(`  ${path.basename(file).padEnd(24)} FAILED ${r.error}`);
  else console.log(`  ${path.basename(file).padEnd(24)} rate ${r.rate.toFixed(5)} +- ${r.rateSd.toFixed(5)}  ratio ${r.ratio.toFixed(2)}  toolTier ${r.toolTier.toFixed(2)}  (${r.seconds}s)`);
}

const ok = results.filter((r) => !r.error && Number.isFinite(r.rate)).sort((a, b) => b.rate - a.rate);
if (!ok.length) { console.error('every candidate failed'); process.exit(1); }
const best = ok[0];
const current = fs.existsSync(CHAMPION) ? path.basename(readable(CHAMPION) ? CHAMPION : '') : '';
const previous = fs.existsSync(META) ? JSON.parse(fs.readFileSync(META, 'utf8')) : null;

console.log(`\nbest: ${path.basename(best.file)} rate ${best.rate.toFixed(5)} +- ${best.rateSd.toFixed(5)}`);
if (previous) console.log(`was:  ${path.basename(previous.source)} rate ${previous.rate.toFixed(5)} (${previous.when})`);
const improves = !previous || best.rate > previous.rate + (best.rateSd || 0);
console.log(improves ? 'verdict: REPLACE' : 'verdict: KEEP (inside one sd of the standing champion)');

if (has('dry') || !improves) process.exit(0);
fs.copyFileSync(best.file, CHAMPION);
fs.writeFileSync(META, JSON.stringify({
  source: path.relative(ROOT, best.file).split(path.sep).join('/'),
  format: readable(best.file).format,
  rate: best.rate, rateSd: best.rateSd, ratio: best.ratio, life: best.life, toolTier: best.toolTier,
  seeds, worlds, when: new Date().toISOString(),
  runnersUp: ok.slice(1, 5).map((r) => ({ source: path.relative(ROOT, r.file).split(path.sep).join('/'), rate: r.rate }))
}, null, 1));
console.log(`wrote ${path.relative(ROOT, CHAMPION)} and ${path.relative(ROOT, META)}`);
console.log('commit both: play.html serves ./champion.json');
