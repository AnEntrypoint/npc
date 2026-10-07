import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

// Tracks Kaggle engagement and keeps the CPU lane busy.
//
// Why: the local box can run GPU training or CPU eval but not both at speed --
// they stall each other -- so every eval question waits for a gap in the GPU
// queue. Kaggle is the second lane. `kaggle kernels list` exposes only
// lastRunTime (no durations, no quota), so usage has to be measured here:
// `push` stamps a start, `status` stamps the finish, and the difference is the
// wall clock we actually consumed.
//
// usage:
//   node tools/kaggle.mjs status              live status of every tracked kernel
//   node tools/kaggle.mjs push <slug> [note]  stamp a launch, then report
//   node tools/kaggle.mjs usage               runs, hours and core-hours, by week
//   node tools/kaggle.mjs watch [seconds]     poll until every tracked kernel is idle

const ROOT = path.resolve(import.meta.dirname, '..');
const LOG = path.join(ROOT, 'runs', 'kaggle-usage.jsonl');
const args = process.argv.slice(2);
const cmd = args[0] || 'status';

function kaggle(argv) {
  const r = spawnSync('kaggle', argv, { encoding: 'utf8', windowsHide: true, shell: true });
  return (r.stdout || '').trim();
}

function statusOf(slug) {
  const out = kaggle(['kernels', 'status', slug]);
  const m = /KernelWorkerStatus\.(\w+)/.exec(out);
  if (m) return m[1].toLowerCase();
  if (/Permission .* was denied|Cannot access/.test(out)) return 'unknown';
  return out ? out.slice(0, 40) : 'error';
}

function readLog() {
  if (!fs.existsSync(LOG)) return [];
  return fs.readFileSync(LOG, 'utf8').split('\n').filter(Boolean).map((l) => {
    try { return JSON.parse(l); } catch { return null; }
  }).filter(Boolean);
}

const PUSH = 'push';
const DONE = 'done';

function append(rec) {
  fs.mkdirSync(path.dirname(LOG), { recursive: true });
  fs.appendFileSync(LOG, JSON.stringify(rec) + '\n');
}

// A start is closed by the first later status that is not running/queued.
function states() {
  const rows = readLog();
  const open = new Map();
  const out = [];
  for (const r of rows) {
    if (r.event === PUSH) {
      const key = r.slug;
      if (open.has(key)) out.push({ ...open.get(key), seconds: null, end: null });
      open.set(key, { slug: r.slug, start: r.at, note: r.note || '', cores: r.cores || null });
    } else if (r.event === DONE && open.has(r.slug)) {
      const s = open.get(r.slug);
      out.push({ ...s, end: r.at, seconds: Math.round((new Date(r.at) - new Date(s.start)) / 1000) });
      open.delete(r.slug);
    }
  }
  for (const s of open.values()) out.push({ ...s, end: null, seconds: null });
  return out;
}

function isoWeek(d) {
  const t = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()));
  const day = t.getUTCDay() || 7;
  t.setUTCDate(t.getUTCDate() + 4 - day);
  const yearStart = new Date(Date.UTC(t.getUTCFullYear(), 0, 1));
  return `${t.getUTCFullYear()}-W${String(Math.ceil(((t - yearStart) / 86400000 + 1) / 7)).padStart(2, '0')}`;
}

function tracked() {
  const seen = new Set(readLog().map((r) => r.slug));
  return [...seen];
}

if (cmd === 'push') {
  append({ event: PUSH, slug: args[1], at: new Date().toISOString(), note: args.slice(2).join(' ') });
  console.log(`stamped ${PUSH} ${args[1]}`);
}

if (cmd === 'status' || cmd === 'push') {
  const live = new Map(tracked().map((s) => [s, statusOf(s)]));
  for (const [slug, st] of live) {
    const prev = (readLog().filter((r) => r.slug === slug).pop() || {}).event;
    if (prev === PUSH && st !== 'running' && st !== 'queued' && st !== 'unknown') {
      append({ event: DONE, slug, at: new Date().toISOString() });
    }
  }
  console.log('slug'.padEnd(34) + 'status');
  for (const [slug, st] of live) console.log(slug.padEnd(34) + st);
  const busy = [...live.values()].filter((s) => s === 'running' || s === 'queued').length;
  console.log(`\n${live.size} tracked, ${busy} busy, ${live.size - busy} idle`);
}

if (cmd === 'usage') {
  const rows = states().filter((r) => r.seconds !== null);
  if (!rows.length) { console.log('no completed runs recorded yet'); process.exit(0); }
  const byWeek = new Map();
  for (const r of rows) {
    const w = isoWeek(new Date(r.start));
    const b = byWeek.get(w) || { runs: 0, seconds: 0, cores: 0 };
    b.runs++;
    b.seconds += r.seconds;
    b.cores += r.seconds * (r.cores || 4);
    byWeek.set(w, b);
  }
  console.log('week'.padEnd(10) + 'runs'.padStart(5) + 'wall h'.padStart(9) + 'core-h'.padStart(9));
  for (const [w, b] of [...byWeek].sort()) {
    console.log(w.padEnd(10) + String(b.runs).padStart(5) +
      (b.seconds / 3600).toFixed(2).padStart(9) + (b.cores / 3600).toFixed(2).padStart(9));
  }
  const tot = rows.reduce((s, r) => s + r.seconds, 0);
  console.log(`\ntotal ${rows.length} runs, ${(tot / 3600).toFixed(2)} wall hours, ` +
    `${(rows.reduce((s, r) => s + r.seconds * (r.cores || 4), 0) / 3600).toFixed(2)} core-hours`);
  console.log('open runs: ' + states().filter((r) => r.seconds === null).map((r) => r.slug).join(', '));
}

if (cmd === 'watch') {
  const limit = Number(args[1] || 1800);
  const t0 = Date.now();
  for (;;) {
    const busy = tracked().map(statusOf).filter((s) => s === 'running' || s === 'queued');
    if (!busy.length) { console.log('all tracked kernels idle'); break; }
    if (Date.now() - t0 > limit * 1000) { console.log('watch timeout, still busy: ' + busy.join(',')); break; }
    await new Promise((r) => setTimeout(r, 30000));
  }
}
