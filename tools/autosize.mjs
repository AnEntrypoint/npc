import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const RUNS = path.join(ROOT, 'runs');
const WORKGROUP_LIMIT = 32768;

const ANCHOR = {
  realm: { hidden: 96, rolloutTicks: 31, bytes: 31476, perHidden: 40.9, perTick: 18.1 },
  blob: { hidden: 96, rolloutTicks: 31, bytes: 17796, perHidden: 189.5, perTick: 64 },
};

const BRACKET = [
  { group: 'size', tag: 'h64', params: { hidden: 64 } },
  { group: 'size', tag: 'h128', params: { hidden: 128 } },
  { group: 'size', tag: 'h192', params: { hidden: 192 } },
  { group: 'horizon', tag: 't16', params: { rolloutTicks: 16 } },
  { group: 'horizon', tag: 't64', params: { rolloutTicks: 64 } },
  { group: 'horizon', tag: 't128', params: { rolloutTicks: 128 } },
  { group: 'horizon', tag: 'g999', params: { gamma: 0.999 } },
  { group: 'horizon', tag: 'g99', params: { gamma: 0.99 } },
  { group: 'opt', tag: 'lr002', params: { lr: 0.002 } },
  { group: 'opt', tag: 'lr0005', params: { lr: 0.0005 } },
  { group: 'opt', tag: 'ep3', params: { epochs: 3 } },
  { group: 'opt', tag: 'clip03', params: { clip: 0.3 } },
];

export function predictBytesUpperBound(game, hidden, rolloutTicks) {
  const a = ANCHOR[game];
  if (!a) return null;
  return Math.round(a.bytes + (hidden - a.hidden) * a.perHidden + (rolloutTicks - a.rolloutTicks) * a.perTick);
}

function parseFlags(argv) {
  const options = {};
  const rest = [];
  for (const arg of argv) {
    if (!arg.startsWith('--')) { rest.push(arg); continue; }
    const eq = arg.indexOf('=');
    const key = (eq < 0 ? arg.slice(2) : arg.slice(2, eq)).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    options[key] = eq < 0 ? true : arg.slice(eq + 1);
  }
  return { options, rest };
}

function gen(argv) {
  const { options } = parseFlags(argv);
  const game = options.game;
  if (!game) throw new Error('usage: node tools/autosize.mjs gen --game=realm|blob [--worlds=512] [--ticks=120000] [--minutes=15] [--seeds=1] [--groups=size,horizon,opt] [--blockgrad=0] [--out=runs/jobs-autosize-<game>.json]');
  const worlds = Number(options.worlds || 512);
  const ticks = Number(options.ticks || 120000);
  const minutes = Number(options.minutes || 15);
  const seeds = String(options.seeds || '1').split(',').map((s) => Number(s.trim()));
  const groups = String(options.groups || 'size,horizon').split(',');
  const prefix = options.prefix || 'as' + game.replace(/[^a-z0-9]/gi, '');
  const jobs = [];
  const rows = [];
  const base = { game, worlds, ticks, logSeconds: 60, saveSeconds: 600 };
  jobs.push({ name: prefix + '-base', page: 'rllong', minutes, seeds, params: { ...base } });
  rows.push([prefix + '-base', 'defaults', '-']);
  for (const arm of BRACKET) {
    if (!groups.includes(arm.group)) continue;
    const hidden = arm.params.hidden ?? ANCHOR[game]?.hidden ?? 96;
    const rolloutTicks = arm.params.rolloutTicks ?? ANCHOR[game]?.rolloutTicks ?? 31;
    const bytes = predictBytesUpperBound(game, hidden, rolloutTicks);
    const note = bytes === null ? 'no anchor for this game' : bytes + ' B' + (bytes > WORKGROUP_LIMIT ? ' > 32768' : '');
    if (bytes !== null && bytes > WORKGROUP_LIMIT) {
      if (!options.blockgrad) { rows.push([prefix + '-' + arm.tag, 'SKIPPED', note + ' (use --blockgrad=0 to run it)']); continue; }
      rows.push([prefix + '-' + arm.tag, 'blockGrad 0', note + ' (compiles only without the block gradient kernel, ~37% slower)']);
      jobs.push({ name: prefix + '-' + arm.tag, page: 'rllong', minutes, seeds, params: { ...base, ...arm.params, blockGrad: 0 } });
      continue;
    }
    rows.push([prefix + '-' + arm.tag, JSON.stringify(arm.params), note]);
    jobs.push({ name: prefix + '-' + arm.tag, page: 'rllong', minutes, seeds, params: { ...base, ...arm.params } });
  }
  const out = path.resolve(options.out || path.join(RUNS, 'jobs-autosize-' + game + '.json'));
  fs.writeFileSync(out, JSON.stringify(jobs, null, 1) + '\n');
  const width = Math.max(...rows.map((r) => r[0].length));
  console.log(rows.map((r) => r[0].padEnd(width) + '  ' + r[1].padEnd(14) + '  ' + r[2]).join('\n'));
  console.log('\n' + jobs.length + ' jobs (' + jobs.reduce((n, j) => n + j.seeds.length, 0) + ' runs) -> ' + out);
  console.log('judge at equal ticks: node tools/report.mjs --at-tick=' + Math.round(ticks * 0.9) + ' --match=^' + prefix + '- --no-write');
}

function check(argv) {
  const { options } = parseFlags(argv);
  const game = options.game;
  const hidden = Number(options.hidden || ANCHOR[game]?.hidden || 96);
  const rolloutTicks = Number(options.rolloutTicks || ANCHOR[game]?.rolloutTicks || 31);
  const bytes = predictBytesUpperBound(game, hidden, rolloutTicks);
  if (bytes === null) { console.log('no measured anchor for "' + game + '": cannot predict (add one to ANCHOR in tools/autosize.mjs)'); return; }
  const ok = bytes <= WORKGROUP_LIMIT;
  console.log(game + ' hidden ' + hidden + ' rolloutTicks ' + rolloutTicks + ': ' + bytes + ' B of ' + WORKGROUP_LIMIT + ' -> ' + (ok ? 'compiles' : 'FAILS pipeline creation' + (options.blockgrad ? ' (blockGrad 0 moves the limit to the gradient kernel, which this does not predict)' : '; try --blockgrad=0')));
}

const [verb, ...argv] = process.argv.slice(2);
if (verb === 'gen') gen(argv);
else if (verb === 'check') check(argv);
else { console.error('usage: node tools/autosize.mjs gen --game=realm|blob [flags] | check --game=realm --hidden=128 --rolloutTicks=31'); process.exit(2); }
