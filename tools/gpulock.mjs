import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const RUNS_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', 'runs');
export const LOCK_FILE = path.join(RUNS_DIR, '.gpu.lock');
export const STALE_MS = 60000;
const TICKET_STALE_MS = 30000;
const WAIT_DIR = path.join(RUNS_DIR, '.gpu.wait');
export const HEARTBEAT_MS = 10000;

const sleepMs = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const blockMs = (ms) => Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);

export function pidAlive(pid) {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try { process.kill(pid, 0); return true; } catch (error) { return error.code === 'EPERM'; }
}

export function readLock() {
  for (let attempt = 0; attempt < 6; attempt++) {
    let text;
    try { text = fs.readFileSync(LOCK_FILE, 'utf8'); } catch (error) { if (error.code === 'ENOENT') return null; blockMs(20); continue; }
    try { return JSON.parse(text); } catch { blockMs(20); }
  }
  try {
    const age = Date.now() - fs.statSync(LOCK_FILE).mtimeMs;
    return { pid: 0, owner: 'unreadable', heartbeat: Date.now() - age };
  } catch { return null; }
}

export function isStale(lock, staleMs = STALE_MS) {
  return !lock || !pidAlive(lock.pid) || Date.now() - lock.heartbeat > staleMs;
}

export function lockStatus(staleMs = STALE_MS) {
  const lock = readLock();
  if (!lock) return { held: false };
  const stale = isStale(lock, staleMs);
  return Object.assign({ held: !stale, stale, ageSeconds: Math.round((Date.now() - lock.heartbeat) / 1000) }, lock);
}

function removeStale(expected) {
  const moved = LOCK_FILE + '.stale.' + process.pid;
  try { fs.renameSync(LOCK_FILE, moved); } catch { return false; }
  let taken = null;
  try { taken = JSON.parse(fs.readFileSync(moved, 'utf8')); } catch {}
  try { fs.unlinkSync(moved); } catch {}
  if (taken && expected && taken.pid !== expected.pid) tryCreate(taken);
  return true;
}

function tryCreate(record) {
  fs.mkdirSync(RUNS_DIR, { recursive: true });
  try { fs.writeFileSync(LOCK_FILE, JSON.stringify(record), { flag: 'wx' }); return true; } catch (error) { if (error.code === 'EEXIST') return false; throw error; }
}

export function ownsLock(pid = process.pid) {
  const lock = readLock();
  return Boolean(lock && lock.pid === pid);
}

export function releaseLock(pid = process.pid) {
  if (!ownsLock(pid)) return false;
  try { fs.unlinkSync(LOCK_FILE); return true; } catch { return false; }
}

function liveTickets() {
  let names = [];
  try { names = fs.readdirSync(WAIT_DIR); } catch { return []; }
  const live = [];
  for (const name of names) {
    const m = /^(d+)-(d+)$/.exec(name);
    if (!m) continue;
    const file = path.join(WAIT_DIR, name);
    let age = Infinity;
    try { age = Date.now() - fs.statSync(file).mtimeMs; } catch {}
    if (pidAlive(Number(m[2])) && age < TICKET_STALE_MS) live.push({ name, order: Number(m[1]) });
    else try { fs.unlinkSync(file); } catch {}
  }
  return live.sort((a, b) => a.order - b.order || (a.name < b.name ? -1 : 1));
}

export async function acquireLock(owner, options = {}) {
  const { timeoutMs = 6 * 3600 * 1000, staleMs = STALE_MS, pollMs = 2000, job = null, onWait = null, shouldAbort = null } = options;
  const started = Date.now();
  const record = { pid: process.pid, owner, job, since: Date.now(), heartbeat: Date.now() };
  fs.mkdirSync(WAIT_DIR, { recursive: true });
  const ticketName = Date.now() + '-' + process.pid;
  const ticketFile = path.join(WAIT_DIR, ticketName);
  fs.writeFileSync(ticketFile, owner);
  try {
    for (;;) {
      const lock = readLock();
      if (lock && lock.pid === process.pid) break;
      if (lock && isStale(lock, staleMs)) { removeStale(lock); continue; }
      const queue = liveTickets();
      if (!lock && (queue.length === 0 || queue[0].name === ticketName) && tryCreate(record)) break;
      if (shouldAbort && shouldAbort()) throw new Error('aborted while waiting for the GPU lock');
      if (Date.now() - started > timeoutMs) throw new Error('timed out after ' + Math.round((Date.now() - started) / 1000) + 's waiting for the GPU lock held by ' + (lock ? lock.owner + ' (pid ' + lock.pid + ')' : 'a waiter ahead in the queue'));
      if (onWait && lock) onWait(lock);
      try { fs.utimesSync(ticketFile, new Date(), new Date()); } catch {}
      await sleepMs(pollMs);
    }
  } finally {
    try { fs.unlinkSync(ticketFile); } catch {}
  }
  return {
    heartbeat(extra) {
      const lock = readLock();
      if (!lock) return tryCreate(Object.assign({}, record, extra || {}, { heartbeat: Date.now() }));
      if (lock.pid !== process.pid) return false;
      fs.writeFileSync(LOCK_FILE, JSON.stringify(Object.assign({}, lock, extra || {}, { heartbeat: Date.now() })));
      return true;
    },
    stillOwned: () => ownsLock(),
    release: () => releaseLock()
  };
}

async function runWithLock(owner, command, args) {
  const handle = await acquireLock(owner, { onWait: (lock) => process.stderr.write('waiting for GPU lock held by ' + lock.owner + '\n') });
  const timer = setInterval(() => handle.heartbeat(), HEARTBEAT_MS);
  const child = spawn(command, args, { stdio: 'inherit', shell: false });
  const code = await new Promise((resolve) => { child.on('exit', (c) => resolve(c === null ? 1 : c)); child.on('error', () => resolve(127)); });
  clearInterval(timer);
  handle.release();
  return code;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [verb, ...rest] = process.argv.slice(2);
  if (verb === 'status') {
    console.log(JSON.stringify(lockStatus()));
  } else if (verb === 'run' && rest.length >= 2) {
    process.exit(await runWithLock(rest[0], rest[1], rest.slice(2)));
  } else if (verb === 'clear') {
    const lock = readLock();
    if (lock && !isStale(lock)) { console.error('lock is live (' + lock.owner + ' pid ' + lock.pid + '), refusing'); process.exit(1); }
    if (lock) removeStale(lock);
    console.log('cleared');
  } else {
    console.error('usage: node tools/gpulock.mjs status | run <owner> <command> [args...] | clear');
    process.exit(2);
  }
}
