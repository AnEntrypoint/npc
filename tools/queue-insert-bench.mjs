import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const ROOT = process.cwd();
const RUNS = join(ROOT, 'runs');
const stateFile = join(RUNS, 'queue-state.json');
const state = JSON.parse(readFileSync(stateFile, 'utf8'));
const specs = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const port = state.options.port || 8123;

const jobs = [];
for (const spec of specs) {
  const params = Object.assign({}, spec.params || {}, { name: spec.name }, spec.minutes ? { minutes: spec.minutes } : {});
  const query = Object.entries(params).map(([k, v]) => encodeURIComponent(k) + '=' + encodeURIComponent(String(v))).join('&');
  const labels = params.variants ? String(params.variants).split(';').filter(Boolean).map((v) => v.slice(0, v.indexOf(':'))) : ['base'];
  jobs.push({
    name: spec.name,
    base: spec.name,
    page: spec.page,
    seed: null,
    minutes: Number(params.minutes || 10),
    url: 'http://localhost:' + port + '/dev/' + spec.page + '.html?' + query,
    dependsOn: [],
    champions: labels.map((label) => spec.name + '-' + label),
    profileMode: spec.profileMode || 'shared',
    chromeFlags: spec.chromeFlags || [],
    status: 'pending',
    attempts: 0,
    reason: null,
    startedAt: null,
    endedAt: null,
    firstOutputMs: null,
    chromePid: null,
    profileDir: null,
  });
}

const known = new Set(state.jobs.map((j) => j.name));
const fresh = jobs.filter((j) => !known.has(j.name));
state.jobs = fresh.concat(state.jobs);
writeFileSync(stateFile, JSON.stringify(state));
console.log('inserted ' + fresh.length + ' bench jobs at the front; queue is now ' + state.jobs.length + ' jobs');
