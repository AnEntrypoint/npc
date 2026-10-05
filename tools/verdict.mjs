#!/usr/bin/env node
import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { nearestTick, evalRatio } from './abjudge.mjs';

const RUNS = new URL('../runs/', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1');
const MAX_TICK_DRIFT_FRACTION = 0.15;

function flags(argv) {
  const out = { _: [] };
  for (const a of argv) {
    if (!a.startsWith('--')) { out._.push(a); continue; }
    const i = a.indexOf('=');
    out[i < 0 ? a.slice(2) : a.slice(2, i)] = i < 0 ? true : a.slice(i + 1);
  }
  return out;
}

function windows(prefix, atTick) {
  const files = readdirSync(RUNS).filter((f) => f.startsWith(prefix + '-s') && f.endsWith('.log'));
  const runs = [];
  for (const f of files) {
    const lines = readFileSync(join(RUNS, f), 'utf8').split('\n').filter((l) => l.includes('tick='));
    if (!lines.length) continue;
    const tok = (l, k) => {
      const s = l.split(' ').find((x) => x.startsWith(k + '='));
      return s === undefined ? null : s.slice(k.length + 1);
    };
    const num = (v) => (v === null || v === '' ? NaN : Number(v));
    const best = nearestTick(lines.map((l) => ({ l, tick: num(tok(l, 'tick')) })), atTick);
    if (!best) continue;
    const l = best.l;
    const life = (tok(l, 'life') || '').split('/');
    runs.push({
      seed: f.slice(prefix.length + 2, -4),
      tick: best.tick,
      eval: num(tok(l, 'EVAL')),
      base: num(tok(l, 'evalBase')),
      ratio: num(tok(l, 'ratio')),
      life: num(life[1] ?? life[0]),
      h2h: num((l.match(/H2H\[[^\]]*ratio=([-\d.eE+]+)/) || [])[1]),
    });
  }
  return runs.sort((a, b) => a.seed.localeCompare(b.seed));
}

const stat = (xs) => {
  const v = xs.filter((x) => Number.isFinite(x));
  if (!v.length) return null;
  const mean = v.reduce((a, b) => a + b, 0) / v.length;
  const sd = v.length < 2 ? 0 : Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / (v.length - 1));
  return { mean, sd, n: v.length };
};
const pm = (s, k = 1e3, d = 2) => (s ? `${(s.mean * k).toFixed(d)} +- ${(s.sd * k).toFixed(d)}` : '-');
const row = (s, d = 0) => (s ? `${Math.round(s.mean)} +- ${Math.round(s.sd)}` : '-');

const usage = () => `usage: node tools/verdict.mjs <prefix> [--baseline=${baseline}] [--at-tick=${atTick}] [--adopt=${ADOPT}] [--floor=${FLOOR}] [--kill=${KILL}] [--row]`;
const f = flags(process.argv.slice(2));
const prefix = f._[0];
const baseline = f.baseline || 'popabL-single';
const atTick = Number(f['at-tick'] || 360000);
const ADOPT = Number(f.adopt || 2.5);
const FLOOR = Number(f.floor || 2.4);
const KILL = Number(f.kill || 2.45);
if (!prefix) { console.error(usage()); process.exit(1); }

const arm = windows(prefix, atTick);
const base = windows(baseline, atTick);
if (!arm.length) { console.error(`no runs/${prefix}-s*.log with a tick= line`); process.exit(1); }
if (!base.length) { console.error(`no runs/${baseline}-s*.log with a tick= line`); process.exit(1); }

const s = (runs, key) => stat(runs.map((r) => r[key]));
const a = { tick: s(arm, 'tick'), eval: s(arm, 'eval'), base: s(arm, 'base'), ratio: s(arm, 'ratio'), life: s(arm, 'life'), h2h: s(arm, 'h2h') };
const b = { ratio: s(base, 'ratio'), eval: s(base, 'eval'), base: s(base, 'base'), life: s(base, 'life'), tick: s(base, 'tick') };

const seeds = arm.map((r) => r.ratio).filter((r) => Number.isFinite(r));
const minSeed = seeds.length ? Math.min(...seeds) : NaN;
const ratio = { mean: evalRatio(a.eval?.mean, a.base?.mean), sd: s(arm, 'ratio')?.sd ?? NaN, n: arm.length };
const drift = arm.filter((r) => Math.abs(r.tick - atTick) > MAX_TICK_DRIFT_FRACTION * atTick).map((r) => `s${r.seed}@${r.tick}`);
let verdict = 'NEUTRAL';
if (!Number.isFinite(ratio.mean) || !Number.isFinite(a.base.mean)) verdict = 'NO DATA (missing EVAL or evalBase token)';
else if (ratio.mean >= ADOPT && minSeed >= FLOOR) verdict = 'ADOPT';
else if (ratio.mean <= KILL) verdict = 'KILL';
if (!seeds.length) verdict += ' (no ratio token in any seed window)';
if (arm.length < 2) verdict += ' (n=1, not adoptable)';
if (drift.length) verdict += ` (PARTIAL: ${drift.join(', ')} far from ${atTick})`;

console.log(`${prefix} vs ${baseline} at tick ${atTick}  (${arm.length} seeds: ${seeds.map((r) => r.toFixed(2)).join(', ')})`);
const bRatio = { mean: evalRatio(b.eval?.mean, b.base?.mean), sd: s(base, 'ratio')?.sd ?? NaN, n: base.length };
console.log(`  ratio   ${pm(ratio, 1, 2)}   baseline ${pm(bRatio, 1, 2)}   delta ${((ratio.mean / bRatio.mean - 1) * 100).toFixed(1)}%`);
console.log(`  eval    ${pm(a.eval)} vs ${pm(b.eval)}    bots ${pm(a.base)} vs ${pm(b.base)}`);
console.log(`  life    ${row(a.life)} vs ${row(b.life)}    h2h ${pm(a.h2h, 1, 2)}    matched tick ${row(a.tick)} vs ${row(b.tick)}`);
console.log(`  verdict ${verdict}   (adopt >= ${ADOPT} with no seed < ${FLOOR}; kill <= ${KILL})`);
if (f.row) console.log(`| ${prefix} | ${ratio.n} | ${row(a.tick)} | ${pm(a.eval)} | ${pm(a.base)} | ${pm(ratio, 1, 2)} | ${row(a.life)} | ${pm(a.h2h, 1, 2)} |`);
