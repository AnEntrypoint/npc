import fs from 'node:fs';
import net from 'node:net';
import http from 'node:http';
import path from 'node:path';
import { spawn, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { acquireLock, lockStatus } from './gpulock.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const RUNS = path.join(ROOT, 'runs');
const STATE_FILE = path.join(RUNS, 'queue-state.json');
const STOP_FILE = path.join(RUNS, 'queue.stop');
const RUNNER_LOG = path.join(RUNS, 'queue.log');
const OLD_DIR = path.join(RUNS, '.old');
const INBOX_DIR = path.join(RUNS, '.queue-inbox');
const SELF = fileURLToPath(import.meta.url);
const PROFILE_PREFIX = 'chrome-q-';
const SHARED_PROFILE = path.join(RUNS, 'chrome-shared');
const DONE_LINE = /^(done|tests done.*|profile done|bench done)$/;
const RETRIABLE = /device (was )?lost|GPU device lost|context lost|DXGI|requestDevice|no (webgpu )?adapter|out of memory|GPUValidationError|GPUInternalError|OperationError|external Instance|chrome exited|stalled/i;
const CHROME_FLAGS = ['--headless=new', '--enable-unsafe-webgpu', '--enable-webgpu-developer-features', '--ignore-gpu-blocklist', '--disable-gpu-watchdog', '--no-first-run', '--no-sandbox', '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows', '--disable-frame-rate-limit', '--disable-gpu-vsync', '--disable-extensions', '--disable-default-apps', '--disable-component-update', '--disable-sync', '--disable-domain-reliability', '--disable-client-side-phishing-detection', '--metrics-recording-only', '--mute-audio', '--disable-hang-monitor', '--disable-ipc-flooding-protection', '--disable-prompt-on-repost', '--disable-breakpad', '--disable-dev-shm-usage', '--disable-features=TranslateUI,OptimizationHints,MediaRouter,DialMediaRouteProvider,CalculateNativeWinOcclusion,AcceptCHFrame,AutofillServerCommunication,PasswordManagerOnboarding'];
const PAGE_PATTERN = /^[a-z0-9_-]+$/i;
const EXTERNAL_PAGE = /\/dev\/(rllong|longrun|ab|perf|harness|blobsp)\.html/;
const DEFAULTS = { port: 8123, graceSeconds: 240, stallSeconds: 600, pollSeconds: 5, exitGraceSeconds: 90, externalCheck: true, externalIdleSeconds: 300, retry: '', maxAttempts: 2, lockTimeoutHours: 12, chrome: '', profileMode: 'shared' };
const TERMINAL = new Set(['done', 'failed', 'timeout', 'skipped', 'stopped']);

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const now = () => Date.now();
const iso = (ms) => (ms ? new Date(ms).toISOString() : null);

function blockMs(ms) {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
}

function writeAtomic(file, text) {
  const temp = file + '.tmp' + process.pid;
  fs.writeFileSync(temp, text);
  for (let attempt = 0; attempt < 40; attempt++) {
    try { fs.renameSync(temp, file); return; } catch { blockMs(50); }
  }
  fs.writeFileSync(file, text);
  try { fs.unlinkSync(temp); } catch {}
}

function loadState() {
  for (let attempt = 0; attempt < 20; attempt++) {
    try { return JSON.parse(fs.readFileSync(STATE_FILE, 'utf8')); } catch (error) { if (error.code === 'ENOENT') return null; blockMs(50); }
  }
  return null;
}

function saveState(state) {
  state.updated = now();
  writeAtomic(STATE_FILE, JSON.stringify(state, null, 1));
}

function pidAlive(pid) {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try { process.kill(pid, 0); return true; } catch (error) { return error.code === 'EPERM'; }
}

function killTree(pid) {
  if (!pidAlive(pid)) return;
  if (process.platform === 'win32') spawnSync('taskkill', ['/PID', String(pid), '/T', '/F'], { stdio: 'ignore', windowsHide: true });
  else { spawnSync('pkill', ['-TERM', '-P', String(pid)], { stdio: 'ignore' }); try { process.kill(pid, 'SIGKILL'); } catch {} }
}

function profileOnCommand(command, profileDir) {
  if (!profileDir) return false;
  if (command.includes(profileDir)) return true;
  const base = path.basename(profileDir);
  return base !== path.basename(SHARED_PROFILE) && command.includes(base);
}

function killByProfile(profileDir) {
  if (!profileDir) return;
  for (const entry of processTable()) if (profileOnCommand(entry.command, profileDir)) killTree(entry.pid);
}

function profileAlive(profileDir) {
  if (!profileDir) return false;
  for (const entry of processTable()) if (profileOnCommand(entry.command, profileDir) && pidAlive(entry.pid)) return true;
  return false;
}

function sweepChrome(keepNames) {
  const keep = (keepNames || []).filter(Boolean);
  const kept = (name) => keep.some((k) => name.includes(k));
  const table = processTable();
  const livePids = new Set(table.map((e) => e.pid));
  let killed = 0;
  let orphans = 0;
  for (const entry of table) {
    const marker = /(chrome-q-[^\s"']+|chrome-shared)/.exec(entry.command);
    const parentless = /--type=/.test(entry.command) && !(entry.ppid && livePids.has(entry.ppid));
    if (parentless || (marker && !kept(marker[0]))) {
      if (pidAlive(entry.pid)) { killTree(entry.pid); parentless ? orphans++ : killed++; }
    }
  }
  let removed = 0;
  for (const dir of fs.readdirSync(RUNS)) {
    if (!dir.startsWith(PROFILE_PREFIX) || kept(dir)) continue;
    fs.rmSync(path.join(RUNS, dir), { recursive: true, force: true });
    removed++;
  }
  return { killed, orphans, removed };
}

function processTable() {
  if (process.platform === 'win32') {
    const script = "Get-CimInstance Win32_Process -Filter \"Name='chrome.exe'\" | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress";
    const result = spawnSync('powershell', ['-NoProfile', '-NonInteractive', '-Command', script], { encoding: 'utf8', windowsHide: true, timeout: 20000 });
    const text = (result.stdout || '').trim();
    if (!text) return [];
    const parsed = JSON.parse(text);
    return (Array.isArray(parsed) ? parsed : [parsed]).map((p) => ({ pid: p.ProcessId, ppid: p.ParentProcessId, command: p.CommandLine || '' }));
  }
  const result = spawnSync('ps', ['-eo', 'pid=,args='], { encoding: 'utf8' });
  return (result.stdout || '').split('\n').filter((l) => /chrome/i.test(l)).map((l) => { const m = /^\s*(\d+)\s+(.*)$/.exec(l); return { pid: Number(m[1]), command: m[2] }; });
}

function externalJobActive(command, idleSeconds) {
  const named = /[?&]name=([^&s"]+)/.exec(command);
  if (!named) return true;
  const lines = readLog(decodeURIComponent(named[1]));
  if (lines.length === 0) return true;
  if (lines.some((l) => DONE_LINE.test(l.trim()))) return false;
  const file = path.join(RUNS, decodeURIComponent(named[1]) + '.log');
  return Date.now() - fs.statSync(file).mtimeMs < idleSeconds * 1000;
}

function externalGpuJobs(state) {
  const ours = new Set(state.jobs.filter((j) => j.chromePid).map((j) => j.chromePid));
  return processTable().filter((p) => !/--type=/.test(p.command) && EXTERNAL_PAGE.test(p.command) && !ours.has(p.pid) && !p.command.includes(PROFILE_PREFIX) && !p.command.includes(SHARED_PROFILE) && externalJobActive(p.command, state.options.externalIdleSeconds));
}

function findChrome(preferred) {
  const candidates = [preferred, process.env.CHROME_PATH, 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe', 'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe', process.env.LOCALAPPDATA && path.join(process.env.LOCALAPPDATA, 'Google', 'Chrome', 'Application', 'chrome.exe'), '/usr/bin/google-chrome', '/usr/bin/chromium', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'];
  const found = candidates.find((c) => c && fs.existsSync(c));
  if (!found) throw new Error('chrome not found; set CHROME_PATH or pass --chrome=<path>');
  return found;
}

function portOpen(port) {
  return new Promise((resolve) => {
    const socket = net.connect({ port, host: '127.0.0.1' });
    const finish = (value) => { socket.destroy(); resolve(value); };
    socket.setTimeout(800, () => finish(false));
    socket.on('connect', () => finish(true));
    socket.on('error', () => finish(false));
  });
}

function httpStatus(port, urlPath) {
  return new Promise((resolve) => {
    const request = http.get({ host: '127.0.0.1', port, path: urlPath, timeout: 3000 }, (response) => { response.resume(); resolve(response.statusCode); });
    request.on('error', () => resolve(0));
    request.on('timeout', () => { request.destroy(); resolve(0); });
  });
}

async function ensureServer(state) {
  const port = state.options.port;
  if (await portOpen(port)) {
    state.server = { port, ownedByQueue: false, pid: null };
  } else {
    const child = spawn(process.execPath, [path.join(ROOT, 'serve.mjs'), String(port)], { cwd: ROOT, detached: true, stdio: 'ignore', windowsHide: true });
    child.unref();
    state.server = { port, ownedByQueue: true, pid: child.pid };
    for (let i = 0; i < 40 && !(await portOpen(port)); i++) await sleep(250);
    if (!(await portOpen(port))) throw new Error('serve.mjs did not start on port ' + port);
  }
  const status = await httpStatus(port, '/dev/rllong.html');
  if (status !== 200) throw new Error('server on port ' + port + ' does not serve /dev/rllong.html (status ' + status + ')');
  saveState(state);
}

function expandJobs(specs, options) {
  if (!Array.isArray(specs)) throw new Error('jobs file must be a JSON list');
  const jobs = [];
  for (const spec of specs) {
    if (!spec.name || !/^[A-Za-z0-9_.-]+$/.test(spec.name)) throw new Error('job needs a name of [A-Za-z0-9_.-]: ' + JSON.stringify(spec));
    if (!spec.page || !PAGE_PATTERN.test(spec.page) || !fs.existsSync(path.join(ROOT, 'dev', spec.page + '.html'))) throw new Error('job ' + spec.name + ': unknown page ' + spec.page);
    const seeds = Array.isArray(spec.seeds) && spec.seeds.length ? spec.seeds : [null];
    for (const seed of seeds) {
      const name = seed === null ? spec.name : spec.name + '-s' + seed;
      const params = Object.assign({}, spec.params || {}, { name }, seed === null ? {} : { seed }, spec.minutes ? { minutes: spec.minutes } : {});
      const query = Object.entries(params).map(([k, v]) => encodeURIComponent(k) + '=' + encodeURIComponent(String(v))).join('&');
      const labels = params.variants ? String(params.variants).split(';').filter(Boolean).map((v) => v.slice(0, v.indexOf(':'))) : ['base'];
      const champions = spec.page === 'longrun' ? [name] : labels.map((label) => name + '-' + label);
      jobs.push({ name, base: spec.name, page: spec.page, seed, minutes: Number(params.minutes || 10), url: 'http://localhost:' + options.port + '/dev/' + spec.page + '.html?' + query, dependsOn: spec.dependsOn || [], champions, profileMode: spec.profileMode || options.profileMode || 'shared', chromeFlags: spec.chromeFlags || [], status: 'pending', attempts: 0, reason: null, startedAt: null, endedAt: null, firstOutputMs: null, chromePid: null, profileDir: null });
    }
  }
  const names = new Set();
  for (const job of jobs) { if (names.has(job.name)) throw new Error('duplicate job name ' + job.name); names.add(job.name); }
  return jobs;
}

function readLog(name) {
  try { return fs.readFileSync(path.join(RUNS, name + '.log'), 'utf8').split(/\r?\n/).filter(Boolean); } catch { return []; }
}

function archiveLog(name) {
  const file = path.join(RUNS, name + '.log');
  if (!fs.existsSync(file)) return;
  fs.mkdirSync(OLD_DIR, { recursive: true });
  fs.renameSync(file, path.join(OLD_DIR, name + '.' + now() + '.log'));
}

async function removeProfile(dir) {
  if (!dir || !path.basename(dir).startsWith(PROFILE_PREFIX) || path.dirname(dir) !== RUNS) return;
  for (let attempt = 0; attempt < 10; attempt++) {
    try { fs.rmSync(dir, { recursive: true, force: true }); if (!fs.existsSync(dir)) return; } catch {}
    await sleep(1000);
  }
}

function dependencyVerdict(job, state) {
  for (const dep of job.dependsOn) {
    const matches = state.jobs.filter((j) => j.name === dep || j.base === dep);
    if (matches.length === 0) return { verdict: 'skip', reason: 'unknown dependency ' + dep };
    if (matches.some((j) => TERMINAL.has(j.status) && j.status !== 'done')) return { verdict: 'skip', reason: 'dependency ' + dep + ' did not finish' };
    if (matches.some((j) => j.status !== 'done')) return { verdict: 'wait' };
  }
  return { verdict: 'run' };
}

function pickNext(state) {
  let blocked = false;
  for (const job of state.jobs) {
    if (job.status !== 'pending') continue;
    const verdict = dependencyVerdict(job, state);
    if (verdict.verdict === 'skip') { job.status = 'skipped'; job.reason = verdict.reason; job.endedAt = now(); saveState(state); return pickNext(state); }
    if (verdict.verdict === 'run') return job;
    blocked = true;
  }
  if (blocked) for (const job of state.jobs) if (job.status === 'pending') { job.status = 'skipped'; job.reason = 'dependency cycle'; job.endedAt = now(); }
  return null;
}

function ingestInbox(state) {
  let count = 0;
  if (!fs.existsSync(INBOX_DIR)) return count;
  for (const file of fs.readdirSync(INBOX_DIR).filter((f) => f.endsWith('.json')).sort()) {
    const full = path.join(INBOX_DIR, file);
    try {
      const added = JSON.parse(fs.readFileSync(full, 'utf8'));
      const known = new Set(state.jobs.map((j) => j.name));
      for (const job of added) if (!known.has(job.name)) { state.jobs.push(job); count++; }
      fs.unlinkSync(full);
    } catch {}
  }
  if (count) saveState(state);
  return count;
}

function stopRequested() {
  return fs.existsSync(STOP_FILE);
}

async function waitForExternalGpu(state, job, handle) {
  if (!state.options.externalCheck) return;
  const started = now();
  for (;;) {
    const external = externalGpuJobs(state);
    if (external.length === 0) return;
    job.status = 'waiting';
    job.reason = 'external GPU job pid ' + external.map((p) => p.pid).join(',');
    saveState(state);
    ingestInbox(state);
    if (stopRequested()) return;
    if (now() - started > state.options.lockTimeoutHours * 3600000) throw new Error('external GPU job never finished');
    await sleep(5000);
    handle.heartbeat({ job: job.name });
  }
}

async function runAttempt(state, job, handle) {
  const options = state.options;
  const chromeExe = findChrome(options.chrome);
  archiveLog(job.name);
  const shared = job.profileMode !== 'fresh';
  job.profileDir = shared ? SHARED_PROFILE : path.join(RUNS, PROFILE_PREFIX + job.name + '-' + now());
  fs.mkdirSync(job.profileDir, { recursive: true });
  const flags = CHROME_FLAGS.concat(Array.isArray(job.chromeFlags) ? job.chromeFlags : []).concat(['--user-data-dir=' + job.profileDir, job.url]);
  const launchedAt = now();
  const chrome = spawn(chromeExe, flags, { stdio: 'ignore', windowsHide: true });
  let exitCode = null;
  let exitAt = null;
  chrome.on('exit', (code) => { exitCode = code === null ? -1 : code; exitAt = now(); });
  chrome.on('error', (error) => { exitCode = -2; job.reason = String(error); });
  job.chromePid = chrome.pid;
  job.status = 'running';
  job.startedAt = job.startedAt || now();
  job.reason = null;
  job.firstOutputMs = null;
  saveState(state);
  const deadline = now() + (job.minutes * 60 + options.graceSeconds) * 1000;
  let lastCount = 0;
  let lastGrowth = now();
  let outcome = null;
  try {
    while (!outcome) {
      await sleep(job.firstOutputMs === null ? 250 : options.pollSeconds * 1000);
      const lines = readLog(job.name);
      if (lines.length > lastCount) { lastCount = lines.length; lastGrowth = now(); if (job.firstOutputMs === null) job.firstOutputMs = now() - launchedAt; }
      const errLine = lines.find((l) => l.startsWith('ERR '));
      if (lines.some((l) => DONE_LINE.test(l.trim()))) outcome = { ok: true };
      else if (errLine) outcome = { ok: false, reason: errLine.slice(0, 300), retriable: RETRIABLE.test(errLine) };
      else if (!handle.heartbeat({ job: job.name })) outcome = { ok: false, reason: 'gpu lock taken over by another owner', retriable: false };
      else if (exitCode !== null && exitCode !== 0) outcome = { ok: false, reason: 'chrome exited with code ' + exitCode + ' before done', retriable: true };
      else if (exitCode === 0 && now() - exitAt > options.exitGraceSeconds * 1000 && !profileAlive(job.profileDir)) outcome = { ok: false, reason: 'chrome launcher exited 0 and no chrome for this profile is left', retriable: true };
      else if (stopRequested()) outcome = { ok: false, reason: 'stopped', stopped: true };
      else if (now() - lastGrowth > options.stallSeconds * 1000) outcome = { ok: false, reason: 'stalled: no log output for ' + options.stallSeconds + 's', retriable: true };
      else if (now() > deadline) outcome = { ok: false, reason: 'timeout after ' + Math.round((now() - job.startedAt) / 1000) + 's', timeout: true };
      job.lastLine = lines.length ? lines[lines.length - 1].slice(0, 200) : null;
    }
  } finally {
    killTree(chrome.pid);
    killByProfile(job.profileDir);
    for (let i = 0; i < 20 && exitCode === null; i++) await sleep(250);
    await removeProfile(job.profileDir);
    job.chromePid = null;
    job.profileDir = null;
  }
  return outcome;
}

async function runJob(state, job) {
  const options = state.options;
  job.status = 'waiting';
  job.reason = 'waiting for GPU lock';
  saveState(state);
  let handle = null;
  try {
    handle = await acquireLock('queue', { job: job.name, timeoutMs: options.lockTimeoutHours * 3600000, shouldAbort: stopRequested, onWait: (lock) => { job.reason = 'GPU lock held by ' + lock.owner + ' pid ' + lock.pid; saveState(state); } });
    const swept = sweepChrome(null);
    if (swept.killed || swept.orphans || swept.removed) console.log('swept ' + swept.killed + ' orphan chrome trees, ' + swept.orphans + ' parentless chrome children and ' + swept.removed + ' stale profiles before ' + job.name);
    await waitForExternalGpu(state, job, handle);
    if (stopRequested()) throw new Error('stopped');
    let outcome = null;
    while (job.attempts < options.maxAttempts) {
      job.attempts++;
      outcome = await runAttempt(state, job, handle);
      if (outcome.ok || outcome.stopped || outcome.timeout || !outcome.retriable) break;
      job.reason = 'attempt ' + job.attempts + ' failed (' + outcome.reason + '), retrying';
      saveState(state);
      await sleep(5000);
    }
    job.endedAt = now();
    if (outcome.ok) {
      const missing = job.champions.filter((c) => !fs.existsSync(path.join(RUNS, c)));
      job.status = 'done';
      job.reason = missing.length ? 'done, but champion files missing: ' + missing.join(',') : null;
    } else {
      job.status = outcome.stopped ? 'stopped' : outcome.timeout ? 'timeout' : 'failed';
      job.reason = outcome.reason;
    }
  } catch (error) {
    job.status = stopRequested() ? 'stopped' : 'failed';
    job.reason = error.message;
    job.endedAt = now();
  } finally {
    if (handle) handle.release();
    saveState(state);
  }
}

function cleanOrphans(state) {
  for (const job of state.jobs) {
    if (job.status !== 'running' && job.status !== 'waiting') continue;
    if (job.chromePid && pidAlive(job.chromePid)) {
      const entry = processTable().find((p) => p.pid === job.chromePid);
      if (entry && job.profileDir && profileOnCommand(entry.command, job.profileDir)) killTree(job.chromePid);
    }
    killByProfile(job.profileDir);
    if (job.profileDir && path.basename(job.profileDir).startsWith(PROFILE_PREFIX)) try { fs.rmSync(job.profileDir, { recursive: true, force: true }); } catch {}
    job.status = 'pending';
    job.chromePid = null;
    job.profileDir = null;
    job.reason = 'requeued after runner restart';
  }
}

async function runner() {
  const state = loadState();
  if (!state) throw new Error('no queue state');
  if (state.runnerPid && state.runnerPid !== process.pid && pidAlive(state.runnerPid)) throw new Error('another runner is alive: pid ' + state.runnerPid);
  state.runnerPid = process.pid;
  state.runnerStarted = now();
  cleanOrphans(state);
  saveState(state);
  const log = (text) => console.log(iso(now()) + ' ' + text);
  try {
    await ensureServer(state);
    log('server port ' + state.server.port + (state.server.ownedByQueue ? ' started by queue pid ' + state.server.pid : ' already running'));
    for (;;) {
      if (stopRequested()) { log('stop requested'); break; }
      ingestInbox(state);
      const job = pickNext(state);
      if (!job) { if (ingestInbox(state) > 0) continue; break; }
      log('job ' + job.name + ' starting');
      await runJob(state, job);
      log('job ' + job.name + ' ' + job.status + ' in ' + Math.round(((job.endedAt || now()) - job.startedAt) / 1000) + 's, first output +' + (job.firstOutputMs === null ? '?' : job.firstOutputMs) + 'ms' + (job.reason ? ' (' + job.reason + ')' : ''));
    }
  } finally {
    state.runnerPid = null;
    state.runnerEnded = now();
    saveState(state);
    try { fs.unlinkSync(STOP_FILE); } catch {}
  }
  log('queue finished: ' + state.jobs.map((j) => j.name + '=' + j.status).join(' '));
}

function parseFlags(argv) {
  const options = Object.assign({}, DEFAULTS);
  const rest = [];
  const flags = new Set();
  for (const arg of argv) {
    const m = /^--([a-zA-Z-]+)(?:=(.*))?$/.exec(arg);
    if (!m) { rest.push(arg); continue; }
    const camel = m[1].replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    if (camel === 'noExternalCheck') options.externalCheck = false;
    else if (camel in options) options[camel] = typeof options[camel] === 'number' ? Number(m[2]) : m[2];
    else flags.add(camel);
  }
  return { options, rest, flags };
}

function describe(state) {
  const lines = [];
  const lock = lockStatus();
  lines.push('runner: ' + (state.runnerPid && pidAlive(state.runnerPid) ? 'alive pid ' + state.runnerPid : 'not running') + ' | server: port ' + (state.server ? state.server.port + (state.server.ownedByQueue ? ' (started by queue, pid ' + state.server.pid + ')' : ' (external)') : '?') + ' | gpu lock: ' + (lock.held ? lock.owner + ' pid ' + lock.pid + ' job ' + lock.job : 'free'));
  const rows = [['job', 'status', 'try', 'min', 'elapsed', 'detail']];
  for (const job of state.jobs) {
    const elapsed = job.startedAt ? Math.round(((job.endedAt || now()) - job.startedAt) / 1000) + 's' : '-';
    const detail = job.reason || (job.status === 'running' ? job.lastLine || '' : '');
    rows.push([job.name, job.status, String(job.attempts), String(job.minutes), elapsed, String(detail).slice(0, 110)]);
  }
  const widths = rows[0].map((_, c) => Math.max(...rows.map((r) => r[c].length)));
  for (const r of rows) lines.push(r.map((cell, c) => cell.padEnd(widths[c])).join('  ').trimEnd());
  return lines.join('\n');
}

async function start(argv, resume) {
  const { options, rest, flags } = parseFlags(argv);
  const existing = loadState();
  if (existing && existing.runnerPid && pidAlive(existing.runnerPid)) throw new Error('a runner is already alive (pid ' + existing.runnerPid + '); use "queue status" or "queue stop"');
  let state;
  if (resume) {
    if (!existing) throw new Error('nothing to resume');
    state = existing;
    state.options = Object.assign({}, state.options, options);
    const requeue = flags.has('retryFailed') ? ['stopped', 'failed', 'timeout', 'skipped'] : ['stopped'];
    const named = new Set(options.retry ? options.retry.split(',') : []);
    for (const job of state.jobs) if (requeue.includes(job.status) || named.has(job.name)) Object.assign(job, { status: 'pending', attempts: 0, reason: null, startedAt: null, endedAt: null });
    state.options.retry = '';
  } else {
    const unfinished = existing ? existing.jobs.filter((j) => !TERMINAL.has(j.status) || j.status === 'stopped').length : 0;
    if (unfinished && !flags.has('replace')) throw new Error('the queue still has ' + unfinished + ' unfinished job(s); use "queue add <jobs.json>" to append, "queue resume" to continue, or --replace to discard them');
    if (rest.length !== 1) throw new Error('usage: node tools/queue.mjs start <jobs.json> [--foreground] [--port=8123] [--grace-seconds=240] [--stall-seconds=600] [--no-external-check]');
    const jobsFile = path.resolve(rest[0]);
    state = { version: 1, jobsFile, created: now(), options, jobs: expandJobs(JSON.parse(fs.readFileSync(jobsFile, 'utf8')), options), runnerPid: null };
  }
  fs.mkdirSync(RUNS, { recursive: true });
  try { fs.unlinkSync(STOP_FILE); } catch {}
  saveState(state);
  if (flags.has('foreground')) { await runner(); return; }
  const out = fs.openSync(RUNNER_LOG, 'a');
  const child = spawn(process.execPath, [SELF, 'run'], { cwd: ROOT, detached: true, stdio: ['ignore', out, out], windowsHide: true });
  child.unref();
  console.log('queue runner started pid ' + child.pid + ', ' + state.jobs.length + ' jobs, log runs/queue.log');
}

function add(argv) {
  const { options, rest } = parseFlags(argv);
  if (rest.length !== 1) throw new Error('usage: node tools/queue.mjs add <jobs.json>');
  const existing = loadState();
  const explicit = new Set(argv.filter((a) => a.startsWith('--')).map((a) => (/^--([a-zA-Z-]+)/.exec(a) || ['', ''])[1].replace(/-([a-z])/g, (_, c) => c.toUpperCase())));
  const merged = Object.assign({}, existing ? existing.options : {});
  for (const key of explicit) if (key in options) merged[key] = options[key];
  if (explicit.has('noExternalCheck')) merged.externalCheck = false;
  const jobs = expandJobs(JSON.parse(fs.readFileSync(path.resolve(rest[0]), 'utf8')), merged);
  const names = new Set(existing ? existing.jobs.map((j) => j.name) : []);
  const clash = jobs.find((j) => names.has(j.name));
  if (clash) throw new Error('job name already in the queue: ' + clash.name);
  if (existing && existing.runnerPid && pidAlive(existing.runnerPid)) {
    fs.mkdirSync(INBOX_DIR, { recursive: true });
    writeAtomic(path.join(INBOX_DIR, now() + '-' + process.pid + '.json'), JSON.stringify(jobs));
    console.log('added ' + jobs.length + ' jobs to the live runner (pid ' + existing.runnerPid + ')');
    return;
  }
  const state = existing || { version: 1, jobsFile: path.resolve(rest[0]), created: now(), options: merged, jobs: [], runnerPid: null };
  state.jobs.push(...jobs);
  saveState(state);
  console.log('added ' + jobs.length + ' jobs; no runner is alive, run "node tools/queue.mjs resume"');
}

async function stop() {
  const state = loadState();
  if (!state || !state.runnerPid || !pidAlive(state.runnerPid)) { console.log('no live runner'); return; }
  fs.writeFileSync(STOP_FILE, String(now()));
  for (let i = 0; i < 60 && pidAlive(state.runnerPid); i++) await sleep(1000);
  if (pidAlive(state.runnerPid)) {
    const fresh = loadState();
    for (const job of fresh.jobs) if (job.chromePid) killTree(job.chromePid);
    try { process.kill(state.runnerPid, 'SIGKILL'); } catch {}
    console.log('runner did not stop in 60s; killed it and its chrome');
  } else console.log('runner stopped');
  const swept = sweepChrome([]);
  if (swept.killed || swept.orphans) console.log('stop swept ' + swept.killed + ' leftover chrome trees and ' + swept.orphans + ' parentless chrome children');
  const final = loadState();
  for (const job of final.jobs) {
    if (job.profileDir && path.basename(job.profileDir).startsWith(PROFILE_PREFIX)) try { fs.rmSync(job.profileDir, { recursive: true, force: true }); } catch {}
    if (job.status === 'running' || job.status === 'waiting') { job.status = 'stopped'; job.chromePid = null; job.profileDir = null; }
  }
  final.runnerPid = null;
  saveState(final);
  try { fs.unlinkSync(STOP_FILE); } catch {}
}

async function bump(argv) {
  const { rest } = parseFlags(argv);
  if (!rest.length) throw new Error('usage: node tools/queue.mjs bump <name-prefix> [prefix...]');
  const match = (name) => rest.some((key) => name === key || name.startsWith(key));
  await stop();
  const state = loadState();
  if (!state) throw new Error('no queue state to bump');
  for (const job of state.jobs) if (job.status === 'stopped') Object.assign(job, { status: 'pending', attempts: 0, reason: null, startedAt: null, endedAt: null, lastLine: null });
  const moved = state.jobs.filter((job) => match(job.name) && job.status === 'pending');
  if (!moved.length) throw new Error('no pending job matches ' + rest.join(', ') + '; pending: ' + (state.jobs.filter((job) => job.status === 'pending').map((job) => job.name).join(', ') || '-'));
  moved.sort((a, b) => rest.findIndex((key) => a.name.startsWith(key)) - rest.findIndex((key) => b.name.startsWith(key)));
  const kept = state.jobs.filter((job) => !moved.includes(job));
  let at = kept.findIndex((job) => job.status === 'pending');
  if (at < 0) at = kept.length;
  state.jobs = [...kept.slice(0, at), ...moved, ...kept.slice(at)];
  saveState(state);
  console.log('bumped ' + moved.map((job) => job.name).join(', ') + ' ahead of the pending queue (' + at + ' finished job(s) before it)');
  await start([], true);
}

function sweep() {
  const state = loadState();
  const live = (state ? state.jobs : []).filter((j) => j.status === 'running' || j.status === 'waiting').map((j) => (j.profileDir ? path.basename(j.profileDir) : null));
  const result = sweepChrome(live);
  console.log('swept: killed ' + result.killed + ' orphan chrome trees, ' + result.orphans + ' parentless chrome children, removed ' + result.removed + ' stale profiles');
}

async function main() {
  const [verb, ...argv] = process.argv.slice(2);
  if (verb === 'sweep') sweep();
  else if (verb === 'start') await start(argv, false);
  else if (verb === 'resume') await start(argv, true);
  else if (verb === 'add') add(argv);
  else if (verb === 'run') await runner();
  else if (verb === 'stop') await stop();
  else if (verb === 'bump') await bump(argv);
  else if (verb === 'status') {
    const state = loadState();
    if (!state) { console.log('no queue state'); return; }
    console.log(argv.includes('--json') ? JSON.stringify(state) : describe(state));
  } else if (verb === 'list') {
    const { options, rest } = parseFlags(argv);
    if (rest.length !== 1) throw new Error('usage: node tools/queue.mjs list <jobs.json>');
    for (const job of expandJobs(JSON.parse(fs.readFileSync(rest[0], 'utf8')), options)) console.log(job.name + ' ' + job.minutes + 'min deps=' + (job.dependsOn.join(',') || '-') + ' ' + job.url);
  } else {
    console.error('usage: node tools/queue.mjs start <jobs.json> [flags] | add <jobs.json> | bump <name-prefix> | resume [--retry-failed | --retry=name,name] | status [--json] | stop | list <jobs.json>');
    process.exit(2);
  }
}

main().catch((error) => { console.error(error.stack || String(error)); process.exit(1); });
