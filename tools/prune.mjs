import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { Brain, Rng, mix, evoConfig, genomeFromJSON, NSTATS } from '../src/core.js';
import { RealtimeNpc } from '../runtime/npc.js';
import * as P from '../src/prune.js';

const SELF = fileURLToPath(import.meta.url);
const TOOL_DIR = path.dirname(SELF);
const EVALSUITE = path.join(TOOL_DIR, 'evalsuite.mjs');
const METHODS = ['mag', 'act', 'fisher', 'structured', 'hybrid'];
const BUFFER_SLACK = 1.02;

function parseArgs(argv) {
  const opts = {
    genome: '', index: 0, game: 'realm', targets: '2048,1024,512,256,128', out: '', methods: 'fisher+rewire', scope: 'global', compareSuites: 'bots',
    jobs: 0, samples: 450000, chunk: 0, gap: 350, batch: 6144, window: 24, segment: 32, steps: 140, finalSteps: 420, ratio: 0.7, lr: 0.003,
    bias: 1, dagger: 0, daggerEvery: 4, daggerWorlds: 8, daggerShare: 0.6, cache: '', seed: 1, evalSeeds: 6, evalWorlds: 4, suites: 'bots,h2h,selfplay',
    skipEval: 0, rewireRounds: 8, rewireSteps: 300, rewireAlpha: 0.3, moveWeight: 2, thresholdWeight: 1, verbose: 1, plasticity: 0, apply: 0, summary: '', name: '', report: ''
  };
  for (const arg of argv) {
    const m = /^--([a-zA-Z0-9-]+)(?:=(.*))?$/.exec(arg);
    if (!m) throw new Error('unexpected argument ' + arg);
    const key = m[1];
    if (!(key in opts)) throw new Error('unknown flag --' + key);
    opts[key] = typeof opts[key] === 'number' ? Number(m[2] === undefined ? 1 : m[2]) : m[2];
  }
  return opts;
}

async function loadGameModule(spec) {
  const candidates = /[\\/.]/.test(spec) ? [path.resolve(TOOL_DIR, spec), path.resolve(process.cwd(), spec)] : [path.resolve(TOOL_DIR, '../src/games', spec + '.js')];
  const found = candidates.find((p) => fs.existsSync(p));
  if (!found) throw new Error('game module not found: ' + spec);
  const mod = await import(pathToFileURL(found).href);
  const game = Object.values(mod).find((v) => v && typeof v === 'object' && typeof v.createEnv === 'function');
  if (!game) throw new Error('no game definition exported by ' + found);
  return game;
}

function exponentialGap(rng, mean) {
  return Math.floor(-Math.log(Math.max(1e-9, rng.next())) * mean);
}

function collectWorld(context, task) {
  const { game, teacherGenome, teacherJson } = context;
  const { nIn, nOut } = game.dims;
  const worldSeed = mix(task.seed, task.index, 0, 9);
  const cfg = task.mode === 'rand' ? game.randomCfg(worldSeed) : game.defaultCfg();
  if (task.mode === 'selfplay') cfg.learnerSlots = game.maxLearners || game.learners;
  const slots = Math.min(cfg.learnerSlots || game.learners, game.maxLearners || game.learners);
  const stats = new Int32Array(NSTATS);
  const env = game.createEnv(worldSeed, cfg, true, stats);
  env.difficulty = task.difficulty;
  const evo = evoConfig(null);
  const studentGenome = task.student ? genomeFromJSON(task.student, task.student.edges.length) : null;
  const teachers = [];
  const students = [];
  const studentDriven = [];
  for (let s = 0; s < slots; s++) {
    teachers.push(new Brain(teacherGenome, teacherJson.edges.length, teacherJson.dims, evo));
    studentDriven.push(Boolean(studentGenome) && s / slots < task.studentShare);
    students.push(studentDriven[s] ? new Brain(studentGenome, task.student.edges.length, task.student.dims, evo) : null);
  }
  const teacherStateCount = teacherJson.dims.nNodes - nIn;
  const obs = new Float32Array(nIn);
  const actions = new Int32Array(slots);
  const outputs = new Float32Array(slots * nOut);
  const lifeTicks = new Int32Array(slots);
  const slotState = Array.from({ length: slots }, (_, s) => ({ rec: null, gap: 0, rng: new Rng(mix(task.seed, task.index, s, 11)) }));
  const chunks = [];
  const tally = { studentReward: 0, studentTicks: 0, teacherReward: 0, teacherTicks: 0, lives: 0 };
  const flush = (state) => {
    const rec = state.rec;
    state.rec = null;
    if (!rec || rec.n < 8) return;
    chunks.push({ obs: rec.obs.slice(0, rec.n * nIn), tout: rec.tout.slice(0, rec.n * nOut), flags: rec.flags.slice(0, rec.n), h0: rec.h0, length: rec.n });
  };
  const ticks = task.periods * game.maxAge;
  const respawnSlot = (slot, tick) => {
    env.respawn(slot, tick);
    teachers[slot].load(teacherGenome);
    if (students[slot]) students[slot].load(studentGenome);
    lifeTicks[slot] = 0;
    flush(slotState[slot]);
    slotState[slot].gap = 0;
  };
  for (let tick = 0; tick < ticks; tick++) {
    if (tick > 0 && tick % game.maxAge === 0) {
      for (let a = 0; a < game.agents; a++) env.respawn(a, tick);
      for (let s = 0; s < slots; s++) { teachers[s].load(teacherGenome); if (students[s]) students[s].load(studentGenome); lifeTicks[s] = 0; flush(slotState[s]); slotState[s].gap = 0; }
    }
    for (let s = 0; s < slots; s++) {
      env.observe(s, obs, tick);
      const state = slotState[s];
      const rng = new Rng(mix(worldSeed, tick, s, 1));
      if (!state.rec && state.gap <= 0) {
        state.rec = { obs: new Float32Array(task.chunk * nIn), tout: new Float32Array(task.chunk * nOut), flags: new Uint8Array(task.chunk), h0: Float32Array.from(teachers[s].act.subarray(nIn, nIn + teacherStateCount)), n: 0 };
        state.rec.flags[0] = 1 | (lifeTicks[s] === 0 ? 2 : 0);
      }
      actions[s] = teachers[s].step(obs, rng);
      const teacherOut = teachers[s].act.subarray(nIn, nIn + nOut);
      if (state.rec) {
        const rec = state.rec;
        rec.obs.set(obs, rec.n * nIn);
        rec.tout.set(teacherOut, rec.n * nOut);
        rec.n++;
      }
      if (students[s]) {
        actions[s] = students[s].step(obs, new Rng(mix(worldSeed, tick, s, 1)));
        for (let k = 0; k < nOut; k++) outputs[s * nOut + k] = students[s].act[nIn + k];
      } else for (let k = 0; k < nOut; k++) outputs[s * nOut + k] = teacherOut[k];
      if (state.rec && state.rec.n === task.chunk) { flush(state); state.gap = exponentialGap(state.rng, task.gap); }
      else if (!state.rec) state.gap--;
    }
    env.step(actions, outputs, tick);
    const respawn = [];
    for (let s = 0; s < slots; s++) {
      const reward = env.reward[s] / 1024;
      if (studentDriven[s]) { tally.studentReward += reward; tally.studentTicks++; } else { tally.teacherReward += reward; tally.teacherTicks++; }
      lifeTicks[s]++;
      if (env.dead[s] !== 0 || lifeTicks[s] >= game.maxAge) { tally.lives++; respawn.push(s); }
    }
    for (const s of respawn) respawnSlot(s, tick);
  }
  for (let s = 0; s < slots; s++) flush(slotState[s]);
  return { chunks, tally, slots };
}

class DataView_ {
  constructor(buffers, caps, dims) {
    this.obs = new Float32Array(buffers.obs);
    this.tout = new Float32Array(buffers.tout);
    this.flags = new Uint8Array(buffers.flags);
    this.chunkStart = new Int32Array(buffers.chunkStart);
    this.chunkLength = new Int32Array(buffers.chunkLength);
    this.h0 = new Float32Array(buffers.h0);
    this.teacherStateCount = dims.teacherStateCount;
    this.caps = caps;
  }
}

function makeBuffers(caps, dims) {
  const shared = (bytes) => new SharedArrayBuffer(bytes);
  return {
    obs: shared(caps.samples * dims.nIn * 4), tout: shared(caps.samples * dims.nOut * 4), flags: shared(caps.samples),
    chunkStart: shared(caps.chunks * 4), chunkLength: shared(caps.chunks * 4), h0: shared(caps.chunks * dims.teacherStateCount * 4)
  };
}

class Dataset {
  constructor(caps, dims) {
    this.caps = caps;
    this.dims = dims;
    this.buffers = makeBuffers(caps, dims);
    this.view = new DataView_(this.buffers, caps, dims);
    this.samples = 0;
    this.chunks = 0;
  }

  add(chunks) {
    const v = this.view;
    const { nIn, nOut, teacherStateCount } = this.dims;
    let added = 0;
    for (const chunk of chunks) {
      if (this.samples + chunk.length > this.caps.samples || this.chunks >= this.caps.chunks) break;
      v.obs.set(chunk.obs, this.samples * nIn);
      v.tout.set(chunk.tout, this.samples * nOut);
      v.flags.set(chunk.flags, this.samples);
      v.chunkStart[this.chunks] = this.samples;
      v.chunkLength[this.chunks] = chunk.length;
      v.h0.set(chunk.h0, this.chunks * teacherStateCount);
      this.samples += chunk.length;
      this.chunks++;
      added += chunk.length;
    }
    return added;
  }

  truncate(chunkCount) {
    if (chunkCount >= this.chunks) return;
    this.samples = this.view.chunkStart[chunkCount];
    this.chunks = chunkCount;
  }

  isValidation(chunk) {
    return (Math.imul(chunk + 1, 2654435761) >>> 0) % 8 === 0;
  }

  split() {
    const train = [];
    const validation = [];
    for (let c = 0; c < this.chunks; c++) (this.isValidation(c) ? validation : train).push(c);
    return { train: Int32Array.from(train), validation: Int32Array.from(validation) };
  }
}

function workerMain() {
  const state = { engine: null, context: null, data: null };
  const handlers = {
    async init(payload) {
      const game = await loadGameModule(payload.gameSpec);
      const teacherJson = payload.teacherJson;
      state.context = { game, teacherJson, teacherGenome: genomeFromJSON(teacherJson, teacherJson.edges.length) };
      state.data = new DataView_(payload.buffers, payload.caps, payload.dims);
      state.engine = new P.ChunkEngine(state.data);
      return true;
    },
    setNet(payload) {
      const net = payload.net;
      state.net = net;
      state.engine.configure(net, payload.spec, payload.segment);
      return { feedForward: state.engine.feedForward };
    },
    grad(payload) {
      const engine = state.engine;
      engine.setWeights(payload.w, payload.bias);
      const acc = P.newAccumulator(state.net, engine.spec, payload.wantStats);
      const items = payload.items;
      for (let i = 0; i < items.length; i += 3) engine.runItem(items[i], items[i + 1], items[i + 2], acc, payload.wantGrad);
      return acc;
    },
    collect(payload) {
      const result = collectWorld(state.context, payload);
      return result;
    }
  };
  parentPort.on('message', async ({ id, type, payload }) => {
    try {
      const result = await handlers[type](payload);
      const transfer = [];
      if (type === 'collect') for (const c of result.chunks) transfer.push(c.obs.buffer, c.tout.buffer, c.flags.buffer, c.h0.buffer);
      parentPort.postMessage({ id, result }, transfer);
    } catch (error) {
      parentPort.postMessage({ id, error: error.stack || String(error) });
    }
  });
}

if (!isMainThread) workerMain();

class Pool {
  constructor(size) {
    this.workers = Array.from({ length: size }, () => {
      const worker = new Worker(SELF, { workerData: {} });
      worker.pending = new Map();
      worker.nextId = 1;
      worker.on('message', (message) => {
        const entry = worker.pending.get(message.id);
        worker.pending.delete(message.id);
        if (message.error) entry.reject(new Error(message.error));
        else entry.resolve(message.result);
      });
      worker.on('error', (error) => { for (const entry of worker.pending.values()) entry.reject(error); });
      return worker;
    });
  }

  call(index, type, payload) {
    const worker = this.workers[index];
    return new Promise((resolve, reject) => {
      const id = worker.nextId++;
      worker.pending.set(id, { resolve, reject });
      worker.postMessage({ id, type, payload });
    });
  }

  broadcast(type, payload) {
    return Promise.all(this.workers.map((_, i) => this.call(i, type, payload)));
  }

  async map(type, payloads) {
    const results = new Array(payloads.length);
    let next = 0;
    const lane = async (index) => {
      while (next < payloads.length) {
        const mine = next++;
        results[mine] = await this.call(index, type, payloads[mine]);
      }
    };
    await Promise.all(this.workers.map((_, i) => lane(i)));
    return results;
  }

  async close() {
    await Promise.all(this.workers.map((w) => w.terminate()));
  }
}

function netPayload(net) {
  return { nIn: net.nIn, nOut: net.nOut, nNodes: net.nNodes, gain: net.gain, inScale: net.inScale, src: net.src, dst: net.dst, w: net.w, leak: net.leak, orig: net.orig };
}

class Distiller {
  constructor(pool, dataset, spec, opts) {
    this.pool = pool;
    this.dataset = dataset;
    this.spec = spec;
    this.opts = opts;
    this.rng = new Rng(opts.seed * 7919 + 13);
    this.refreshSplit();
  }

  refreshSplit() {
    this.split = this.dataset.split();
  }

  async setNet(net) {
    const info = await this.pool.broadcast('setNet', { net: netPayload(net), spec: this.spec, segment: this.opts.segment });
    this.feedForward = info[0].feedForward;
    this.net = net;
    this.validationItems = null;
  }

  tileItems(chunks, limit) {
    const v = this.dataset.view;
    const items = [];
    const take = limit ? chunks.slice(0, limit) : chunks;
    for (const c of take) {
      const length = v.chunkLength[c];
      if (!this.feedForward) { items.push(c, 0, length); continue; }
      for (let a = 0; a < length; a += this.opts.window) items.push(c, a, Math.min(this.opts.window, length - a));
    }
    return Int32Array.from(items);
  }

  sampleItems() {
    const v = this.dataset.view;
    const chunks = this.split.train;
    const items = [];
    if (this.feedForward) {
      const itemCount = Math.max(1, Math.round(this.opts.batch / this.opts.window));
      for (let i = 0; i < itemCount; i++) {
        const c = chunks[Math.floor(this.rng.next() * chunks.length)];
        const length = v.chunkLength[c];
        const span = Math.min(this.opts.window, length);
        items.push(c, Math.floor(this.rng.next() * (length - span + 1)), span);
      }
    } else {
      const itemCount = Math.max(1, Math.round(this.opts.batch / Math.max(1, this.opts.chunkLength)));
      for (let i = 0; i < itemCount; i++) {
        const c = chunks[Math.floor(this.rng.next() * chunks.length)];
        items.push(c, 0, v.chunkLength[c]);
      }
    }
    return Int32Array.from(items);
  }

  async runItems(net, items, wantGrad, wantStats) {
    const workers = this.pool.workers.length;
    const parts = Array.from({ length: workers }, () => []);
    for (let i = 0, k = 0; i < items.length; i += 3, k++) parts[k % workers].push(items[i], items[i + 1], items[i + 2]);
    const payloads = parts.filter((p) => p.length > 0).map((p) => ({ w: net.w, bias: net.bias, items: Int32Array.from(p), wantGrad, wantStats }));
    const results = await this.pool.map('grad', payloads);
    const total = results[0];
    for (let i = 1; i < results.length; i++) P.mergeAccumulator(total, results[i]);
    return total;
  }

  async evaluate(net) {
    if (this.net !== net) await this.setNet(net);
    if (!this.validationItems) this.validationItems = this.tileItems(this.split.validation, 240);
    const acc = await this.runItems(net, this.validationItems, false, false);
    const n = Math.max(1, acc.count);
    return {
      loss: acc.loss / n,
      rmse: Math.sqrt(acc.sqErr.reduce((s, v) => s + v, 0) / n / net.nOut),
      sideAgree: acc.sideAgree.length ? acc.sideAgree.reduce((s, v) => s + v, 0) / n / acc.sideAgree.length : null,
      thresholdRecall: acc.positiveTeacher.reduce((s, v) => s + v, 0) > 0 ? acc.positiveBoth.reduce((s, v) => s + v, 0) / acc.positiveTeacher.reduce((s, v) => s + v, 0) : null,
      moveCos: acc.moveCount ? acc.moveCos / acc.moveCount : null,
      moveMagErr: acc.moveCount ? acc.moveMagErr / n : null,
      samples: acc.count
    };
  }

  async nodeRms(net, itemCount) {
    if (this.net !== net) await this.setNet(net);
    const items = [];
    const v = this.dataset.view;
    for (let i = 0; i < itemCount; i++) {
      const c = this.split.train[Math.floor(this.rng.next() * this.split.train.length)];
      items.push(c, 0, this.feedForward ? Math.min(this.opts.window, v.chunkLength[c]) : v.chunkLength[c]);
    }
    const acc = await this.runItems(net, Int32Array.from(items), false, true);
    return P.nodeRmsFrom(acc, net);
  }

  async fisher(net, groups, itemsPerGroup) {
    if (this.net !== net) await this.setNet(net);
    const scores = new Float64Array(net.w.length);
    for (let g = 0; g < groups; g++) {
      const items = [];
      const v = this.dataset.view;
      for (let i = 0; i < itemsPerGroup; i++) {
        const c = this.split.train[Math.floor(this.rng.next() * this.split.train.length)];
        const length = v.chunkLength[c];
        items.push(c, 0, this.feedForward ? Math.min(this.opts.window, length) : length);
      }
      const acc = await this.runItems(net, Int32Array.from(items), true, false);
      const n = Math.max(1, acc.count);
      for (let e = 0; e < scores.length; e++) { const s = (net.w[e] * acc.grad[e]) / n; scores[e] += s * s / groups; }
    }
    return scores;
  }

  async rewireStep(net, teacherNet, alpha) {
    const { candidate, active } = P.buildCandidateNet(net, teacherNet);
    await this.setNet(candidate);
    const first = this.sampleItems();
    const second = this.sampleItems();
    const acc = await this.runItems(candidate, Int32Array.from([...first, ...second]), true, false);
    const rms = await this.nodeRms(net, 96);
    const growScores = Float64Array.from(acc.grad, Math.abs);
    const dropScores = Float64Array.from(candidate.w, (w, e) => Math.abs(w) * rms[candidate.src[e]]);
    let activeCount = 0;
    for (const flag of active) activeCount += flag;
    return P.rewireNet(candidate, active, growScores, dropScores, Math.round(alpha * activeCount));
  }

  async fit(net, steps, lrStart, lrEnd) {
    await this.setNet(net);
    const adamW = new P.Adam(net.w.length);
    const adamB = new P.Adam(net.nNodes - net.nIn);
    const wmax = net.params[6];
    let lossEma = null;
    for (let step = 0; step < steps; step++) {
      const phase = steps > 1 ? step / (steps - 1) : 1;
      const lr = lrEnd + 0.5 * (lrStart - lrEnd) * (1 + Math.cos(Math.PI * phase));
      const acc = await this.runItems(net, this.sampleItems(), true, false);
      if (!Number.isFinite(acc.loss)) throw new Error('non-finite loss during fit');
      const scale = 1 / Math.max(1, acc.count);
      let norm = 0;
      for (let e = 0; e < acc.grad.length; e++) norm += acc.grad[e] * acc.grad[e];
      for (let j = 0; j < acc.gradBias.length; j++) norm += acc.gradBias[j] * acc.gradBias[j];
      norm = Math.sqrt(norm) * scale;
      const clip = norm > 5 ? 5 / norm : 1;
      adamW.step(net.w, acc.grad, scale * clip, lr);
      for (let e = 0; e < net.w.length; e++) net.w[e] = Math.max(-wmax, Math.min(wmax, net.w[e]));
      if (this.opts.bias) adamB.step(net.bias.subarray(net.nIn), acc.gradBias, scale * clip, lr);
      const loss = acc.loss / Math.max(1, acc.count);
      lossEma = lossEma === null ? loss : 0.95 * lossEma + 0.05 * loss;
    }
    return lossEma;
  }
}

function rateOf(tally, kind) {
  const ticks = tally[kind + 'Ticks'];
  return ticks > 0 ? tally[kind + 'Reward'] / ticks : null;
}

async function collectData(pool, dataset, plan) {
  const modes = ['bots', 'selfplay', 'rand', 'bots', 'rand', 'selfplay'];
  const difficulties = [100, 100, 100, 60, 100, 100];
  const tasks = [];
  const perRun = plan.expectedPerRun;
  const runs = Math.max(1, Math.ceil(plan.samples / perRun));
  for (let i = 0; i < runs; i++) {
    const k = (plan.offset + i) % modes.length;
    tasks.push({ seed: plan.seed, index: plan.offset + i, mode: modes[k], difficulty: plan.lowDifficulty && i % 3 === 2 ? 40 : difficulties[k], periods: plan.periods || 1, chunk: plan.chunk, gap: plan.gap, student: plan.student || null, studentShare: plan.studentShare || 0 });
  }
  const results = await pool.map('collect', tasks);
  const tally = { studentReward: 0, studentTicks: 0, teacherReward: 0, teacherTicks: 0, lives: 0 };
  let added = 0;
  for (const r of results) {
    added += dataset.add(r.chunks);
    for (const key of Object.keys(tally)) tally[key] += r.tally[key];
  }
  return { added, tally, runs };
}

function log(opts, text) {
  if (opts.verbose) console.error(text);
}

function runEvalsuite(args) {
  return new Promise((resolve, reject) => {
    execFile(process.execPath, [EVALSUITE, ...args, '--json', '--no-write'], { maxBuffer: 1 << 28 }, (error, stdout) => {
      if (error) { reject(error); return; }
      resolve(JSON.parse(stdout));
    });
  });
}

function suiteStat(result, suite, key) {
  const s = result.suites[suite] && result.suites[suite].summary[key];
  return s ? { mean: s.mean, sd: s.sd } : null;
}

function benchBrain(genomeJson, obsSamples, repeats) {
  const genome = genomeFromJSON(genomeJson, genomeJson.edges.length);
  const brain = new Brain(genome, genomeJson.edges.length, genomeJson.dims, evoConfig(null));
  const rng = new Rng(5);
  const nIn = genomeJson.dims.nIn;
  const count = Math.min(4000, Math.floor(obsSamples.length / nIn));
  let best = Infinity;
  for (let r = 0; r < repeats; r++) {
    brain.load(genome);
    const start = performance.now();
    for (let i = 0; i < count; i++) brain.step(obsSamples.subarray(i * nIn, (i + 1) * nIn), rng);
    best = Math.min(best, (performance.now() - start) / count);
  }
  return best * 1000;
}

function verifyRuntime(genomeJson, obsSamples, ticks, plastic) {
  const nIn = genomeJson.dims.nIn;
  const nOut = genomeJson.dims.nOut;
  const cap = Math.max(genomeJson.edges.length + 64, (genomeJson.meta && genomeJson.meta.maxEdges) || 0);
  const genome = genomeFromJSON(genomeJson, cap);
  const brain = new Brain(genome, cap, genomeJson.dims, evoConfig(null));
  const npc = new RealtimeNpc(genomeJson, { seed: 3, agent: 1 });
  const net = P.netFromGenome(genomeJson);
  let state = new Float64Array(net.nNodes);
  let next = new Float64Array(net.nNodes);
  const rng = new Rng(9);
  let brainVsNpc = 0;
  let brainVsNet = 0;
  const count = Math.min(ticks, Math.floor(obsSamples.length / nIn));
  for (let t = 0; t < count; t++) {
    const inputs = obsSamples.subarray(t * nIn, (t + 1) * nIn);
    brain.step(inputs, rng);
    npc.decide(inputs, t);
    P.forwardStep(net, inputs, state, next);
    [state, next] = [next, state];
    for (let k = 0; k < nOut; k++) {
      brainVsNpc = Math.max(brainVsNpc, Math.abs(brain.act[nIn + k] - npc.outputs[k]));
      brainVsNet = Math.max(brainVsNet, Math.abs(brain.act[nIn + k] - state[nIn + k]));
    }
    const reward = plastic ? (t % 7 === 0 ? 0.6 : t % 3 === 0 ? -0.2 : 0) : 0;
    brain.learn(reward);
    if ((t + 1) % 32 === 0) brain.structural(new Rng(mix(3, t, 1, 3)));
    npc.reward(reward, t, false);
  }
  let weightDiff = 0;
  for (let e = 0; e < Math.min(brain.n, npc.n); e++) weightDiff = Math.max(weightDiff, Math.abs(brain.w[e] - npc.w[e]));
  return { ticks: count, brainVsRealtimeNpc: brainVsNpc, brainVsStudentMath: plastic ? null : brainVsNet, weightDiff, edgesAfter: { brain: brain.n, npc: npc.n } };
}

function parseMethod(text, defaultScope) {
  const flags = text.split('+');
  const [method, scope] = flags[0].split('/');
  return { text, method, scope: scope || defaultScope, dagger: flags.includes('dagger'), rewire: flags.includes('rewire'), label: text.replace(/\//g, '-').replace(/\+/g, '-') };
}

async function pruneStep(context, net, spec, nextEdges) {
  const { distiller } = context;
  const edgesNow = net.w.length;
  if (spec.method === 'structured' || spec.method === 'hybrid') {
    const hidden = P.hiddenCount(net);
    const rms = await distiller.nodeRms(net, 96);
    const fraction = spec.method === 'structured' ? nextEdges / edgesNow : Math.sqrt(nextEdges / edgesNow);
    const keepUnits = Math.max(1, Math.min(hidden - 1, Math.round(hidden * fraction)));
    let next = P.pruneUnitsByScore(net, keepUnits, P.unitScores(net, rms));
    if (spec.method === 'hybrid' && next.w.length > nextEdges) {
      const rmsNext = await distiller.nodeRms(next, 96);
      next = P.pruneEdgesByScore(next, nextEdges, P.edgeScores(next, 'act', rmsNext, null), 'global');
    }
    return next;
  }
  const rms = spec.method === 'act' ? await distiller.nodeRms(net, 96) : null;
  const fisher = spec.method === 'fisher' ? await distiller.fisher(net, 10, 24) : null;
  return P.pruneEdgesByScore(net, nextEdges, P.edgeScores(net, spec.method, rms, fisher), spec.scope);
}

async function daggerRound(context, net, seedOffset) {
  const { opts, pool, dataset, distiller } = context;
  const student = P.genomeFromNet(net, {});
  const plan = { samples: opts.daggerWorlds * context.perRun * 1.6, seed: opts.seed + 1000 + seedOffset, offset: 200 + seedOffset * 40, chunk: opts.chunkLength, gap: opts.gap, expectedPerRun: context.perRun * 1.6, student, studentShare: opts.daggerShare };
  const result = await collectData(pool, dataset, plan);
  distiller.refreshSplit();
  log(opts, '  dagger +' + result.added + ' samples, student reward/tick ' + (rateOf(result.tally, 'student') || 0).toFixed(5) + ' vs teacher ' + (rateOf(result.tally, 'teacher') || 0).toFixed(5));
  return { added: result.added, studentRate: rateOf(result.tally, 'student'), teacherRate: rateOf(result.tally, 'teacher') };
}

async function distill(context, spec) {
  const { opts, distiller, dataset } = context;
  dataset.truncate(context.baseChunks);
  distiller.refreshSplit();
  let net = P.cloneNet(context.net0);
  const snapshots = [];
  const history = [];
  let round = 0;
  const started = performance.now();
  for (const target of context.targets) {
    for (let guard = 0; net.w.length > target * 1.03 && guard < 80; guard++) {
      const nextEdges = Math.max(target, Math.floor(net.w.length * opts.ratio));
      net = await pruneStep(context, net, spec, nextEdges);
      const fitLoss = await distiller.fit(net, opts.steps, opts.lr, opts.lr * 0.15);
      round++;
      if (spec.dagger && round % opts.daggerEvery === 0) { await daggerRound(context, net, round); await distiller.fit(net, Math.round(opts.steps / 2), opts.lr * 0.5, opts.lr * 0.1); }
      const metrics = await distiller.evaluate(net);
      history.push({ edges: net.w.length, nodes: net.nNodes, fitLoss, valLoss: metrics.loss, sideAgree: metrics.sideAgree, moveCos: metrics.moveCos });
      log(opts, '  [' + spec.label + '] edges=' + net.w.length + ' nodes=' + net.nNodes + ' val=' + metrics.loss.toFixed(5) + ' agree=' + (metrics.sideAgree || 0).toFixed(3) + ' moveCos=' + (metrics.moveCos || 0).toFixed(3) + ' (' + ((performance.now() - started) / 1000).toFixed(0) + 's)');
    }
    if (spec.dagger) await daggerRound(context, net, 1000 + target);
    if (spec.rewire) {
      for (let r = 0; r < opts.rewireRounds; r++) {
        net = await distiller.rewireStep(net, context.net0, opts.rewireAlpha * (1 - r / opts.rewireRounds));
        await distiller.fit(net, opts.rewireSteps, opts.lr * 0.5, opts.lr * 0.1);
        const rewired = await distiller.evaluate(net);
        log(opts, '  [' + spec.label + '] rewire ' + (r + 1) + ' edges=' + net.w.length + ' val=' + rewired.loss.toFixed(5) + ' moveCos=' + (rewired.moveCos || 0).toFixed(3));
      }
    }
    await distiller.fit(net, opts.finalSteps, opts.lr * 0.6, opts.lr * 0.02);
    const metrics = await distiller.evaluate(net);
    log(opts, '* [' + spec.label + '] target=' + target + ' edges=' + net.w.length + ' nodes=' + net.nNodes + ' val=' + metrics.loss.toFixed(5) + ' rmse=' + metrics.rmse.toFixed(4) + ' agree=' + (metrics.sideAgree || 0).toFixed(3) + ' moveCos=' + (metrics.moveCos || 0).toFixed(3));
    snapshots.push({ target, net: P.cloneNet(net), metrics });
  }
  return { spec, snapshots, history, seconds: (performance.now() - started) / 1000 };
}

function evalArgs(context, suites, genomeFile, withOpponent) {
  const { opts, genomePath } = context;
  const args = ['--game=' + opts.game, '--seeds=' + opts.evalSeeds, '--worlds=' + opts.evalWorlds, '--suites=' + suites];
  if (withOpponent) args.push('--opponent=' + genomePath);
  args.push(genomeFile);
  return args;
}

function evalDigest(result) {
  const digest = {};
  for (const suite of Object.keys(result.suites)) {
    digest[suite] = {};
    for (const key of ['rate', 'ratio', 'life', 'baseRate', 'opponentRate', 'opponentLife']) {
      const stat = suiteStat(result, suite, key);
      if (stat) digest[suite][key] = stat;
    }
  }
  return digest;
}

function fixed(value, digits) {
  return value === null || value === undefined ? '-' : value.toFixed(digits);
}

function pm(stat, digits) {
  return stat ? fixed(stat.mean, digits) + (stat.sd === null || stat.sd === undefined ? '' : '+-' + fixed(stat.sd, digits)) : '-';
}

async function evaluateAll(context, results, teacherEval, isPrimary) {
  const { opts, outDir, base, obsSamples } = context;
  for (const result of results) {
    for (const snap of result.snapshots) {
      const json = P.genomeFromNet(snap.net, { teacher: path.basename(context.genomePath), method: result.spec.label, target: snap.target, edges: snap.net.w.length, maxEdges: Math.ceil(snap.net.w.length * 1.5) + 64 });
      const file = path.join(outDir, base + '-' + result.spec.label + '-s' + snap.target);
      fs.writeFileSync(file, JSON.stringify(json));
      snap.file = file;
      snap.json = json;
      snap.bench = Math.min(benchBrain(json, obsSamples, 5), benchBrain(json, obsSamples, 5));
      snap.verify = verifyRuntime(json, obsSamples, 600);
      if (!opts.skipEval) {
        const suites = isPrimary ? opts.suites : opts.compareSuites;
        const evaluation = await runEvalsuite(evalArgs(context, suites, file, suites.includes('h2h')));
        snap.eval = evalDigest(evaluation);
        fs.writeFileSync(file + '.eval.json', JSON.stringify(evaluation));
        const bots = snap.eval.bots;
        log(opts, 'eval [' + result.spec.label + ' s' + snap.target + '] edges=' + snap.net.w.length + ' bots rate ' + pm(bots && bots.rate, 5) + ' = ' + fixed(bots && teacherEval.bots.rate.mean ? bots.rate.mean / teacherEval.bots.rate.mean * 100 : null, 1) + '% of teacher, ' + snap.bench.toFixed(1) + ' us/tick');
      }
    }
  }
}

function writeReport(context, results, teacher) {
  const { opts, outDir, base } = context;
  const teacherBots = teacher.eval && teacher.eval.bots;
  const lines = [];
  lines.push('# Pruning report: ' + path.basename(context.genomePath));
  lines.push('');
  lines.push('teacher: ' + teacher.edges + ' edges, ' + teacher.nodes + ' nodes, ' + teacher.hidden + ' hidden, ' + teacher.bench.toFixed(1) + ' us per Brain.step' + (teacherBots ? ', bots rate ' + pm(teacherBots.rate, 5) + ' (ratio vs scripted ' + pm(teacherBots.ratio, 2) + ')' : ''));
  lines.push('');
  lines.push('options: ' + JSON.stringify({ targets: opts.targets, samples: opts.samples, steps: opts.steps, finalSteps: opts.finalSteps, ratio: opts.ratio, lr: opts.lr, bias: opts.bias, dagger: opts.dagger, evalSeeds: opts.evalSeeds, evalWorlds: opts.evalWorlds }));
  for (const result of results) {
    lines.push('');
    lines.push('## ' + result.spec.label + ' (' + result.seconds.toFixed(0) + ' s)');
    lines.push('');
    lines.push('| target | edges | %edges | nodes | hidden | us/tick | speedup | val rmse | move cos | thr agree | thr recall | bots rate | %teacher | bots ratio | h2h vs teacher | selfplay rate | life |');
    lines.push('| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |');
    for (const snap of result.snapshots) {
      const e = snap.eval || {};
      const bots = e.bots;
      const pct = bots && teacherBots ? bots.rate.mean / teacherBots.rate.mean * 100 : null;
      lines.push('| ' + [snap.target, snap.net.w.length, fixed(snap.net.w.length / teacher.edges * 100, 1), snap.net.nNodes, P.hiddenCount(snap.net), fixed(snap.bench, 1), fixed(teacher.bench / snap.bench, 1) + 'x', fixed(snap.metrics.rmse, 4), fixed(snap.metrics.moveCos, 3), fixed(snap.metrics.sideAgree, 3), fixed(snap.metrics.thresholdRecall, 3), pm(bots && bots.rate, 5), fixed(pct, 1), pm(bots && bots.ratio, 2), pm(e.h2h && e.h2h.ratio, 2), pm(e.selfplay && e.selfplay.rate, 5), pm(bots && bots.life, 0)].join(' | ') + ' |');
    }
  }
  if (teacherBots) {
    lines.push('');
    lines.push('## Recommended operating points (' + results[0].spec.label + ', bots reward/tick relative to the teacher, point estimates; seed noise is about 3%)');
    lines.push('');
    for (const level of [0.95, 0.9, 0.8]) {
      const fit = results[0].snapshots.filter((s) => s.eval && s.eval.bots && s.eval.bots.rate.mean >= level * teacherBots.rate.mean).sort((a, b) => a.net.w.length - b.net.w.length)[0];
      lines.push('- >= ' + level * 100 + '% of teacher reward: ' + (fit ? fit.net.w.length + ' edges (' + fixed(fit.net.w.length / teacher.edges * 100, 1) + '%), ' + fit.net.nNodes + ' nodes, ' + fixed(fit.bench, 1) + ' us/tick, ' + fixed(teacher.bench / fit.bench, 1) + 'x faster, file ' + path.basename(fit.file) : 'not reached'));
    }
  }
  const report = { tool: 'prune', when: new Date().toISOString(), teacher: Object.assign({}, teacher), options: opts, methods: results.map((r) => ({ label: r.spec.label, seconds: r.seconds, history: r.history, snapshots: r.snapshots.map((s) => ({ target: s.target, edges: s.net.w.length, nodes: s.net.nNodes, hidden: P.hiddenCount(s.net), file: s.file, fidelity: s.metrics, usPerTick: s.bench, verify: s.verify, eval: s.eval })) })) };
  const reportBase = opts.report || base;
  fs.writeFileSync(path.join(outDir, reportBase + '.prune.json'), JSON.stringify(report, null, 1));
  fs.writeFileSync(path.join(outDir, reportBase + '.prune.md'), lines.join('\n') + '\n');
  return lines.join('\n');
}

async function runAllMethods(context, methods) {
  const { opts, net0, teacherJson, obsSamples, genomePath } = context;
  const results = [];
  const teacher = { edges: net0.w.length, nodes: net0.nNodes, hidden: P.hiddenCount(net0), bench: Math.min(benchBrain(teacherJson, obsSamples, 3), benchBrain(teacherJson, obsSamples, 3)) };
  if (!opts.skipEval) {
    const evaluation = await runEvalsuite(['--game=' + opts.game, '--seeds=' + opts.evalSeeds, '--worlds=' + opts.evalWorlds, '--suites=bots,selfplay', genomePath]);
    teacher.eval = evalDigest(evaluation);
    log(opts, 'teacher eval bots rate ' + pm(teacher.eval.bots.rate, 5) + ' ratio ' + pm(teacher.eval.bots.ratio, 2));
  }
  context.teacherSummary = teacher;
  for (const text of methods) {
    log(opts, 'method ' + text);
    const result = await distill(context, parseMethod(text, opts.scope));
    results.push(result);
    await evaluateAll(context, [result], teacher.eval || null, results.length === 1);
    console.log(writeReport(context, results, teacher));
  }
  context.results = results;
}

const PARAM_INDEX = { eta: 0, A: 1, B: 2, C: 3, D: 4, decay: 5, pruneT: 7, growP: 8, growW: 14 };

function withParams(json, settings, label) {
  const copy = JSON.parse(JSON.stringify(json));
  for (const [name, value] of Object.entries(settings)) copy.params[PARAM_INDEX[name]] = value;
  copy.meta = Object.assign({}, copy.meta, { maxEdges: Math.ceil(json.edges.length * 1.6) + 96, variant: label });
  return copy;
}

function pairedDifference(variant, reference) {
  const a = variant.suites.bots.perSeed.map((s) => s.metrics.rate);
  const b = reference.suites.bots.perSeed.map((s) => s.metrics.rate);
  const diffs = a.map((v, i) => v - b[i]);
  const mean = diffs.reduce((s, v) => s + v, 0) / diffs.length;
  const sd = Math.sqrt(diffs.reduce((s, v) => s + (v - mean) * (v - mean), 0) / Math.max(1, diffs.length - 1));
  return { mean, se: sd / Math.sqrt(diffs.length), relative: mean / (b.reduce((s, v) => s + v, 0) / b.length) };
}

function sampleObservations(game, count) {
  const env = game.createEnv(77, game.defaultCfg(), true, new Int32Array(NSTATS));
  const nIn = game.dims.nIn;
  const out = new Float32Array(count * nIn);
  const obs = new Float32Array(nIn);
  const actions = new Int32Array(game.learners);
  const outputs = new Float32Array(game.learners * game.dims.nOut);
  for (let t = 0; t < count; t++) {
    env.observe(0, obs, t);
    out.set(obs, t * nIn);
    for (let k = 0; k < outputs.length; k++) outputs[k] = Math.sin(0.01 * t * (1 + (k % 5))) * 0.8;
    env.step(actions, outputs, t);
  }
  return out;
}

async function runPlasticity(opts) {
  const genomePath = path.resolve(opts.genome);
  const game = await loadGameModule(opts.game);
  const observations = sampleObservations(game, 700);
  const json = JSON.parse(fs.readFileSync(genomePath, 'utf8'));
  const outDir = path.resolve(opts.out || path.dirname(genomePath));
  const workDir = path.join(outDir, 'plasticity-' + path.basename(genomePath));
  fs.mkdirSync(workDir, { recursive: true });
  const groups = {
    plasticity: [
      ['off', {}],
      ['hebbA+ eta.001', { eta: 0.001, A: 0.3 }], ['hebbA+ eta.005', { eta: 0.005, A: 0.3 }], ['hebbA+ eta.02', { eta: 0.02, A: 0.3 }], ['hebbA+ eta.1', { eta: 0.1, A: 0.3 }], ['hebbA+ eta.5', { eta: 0.5, A: 0.3 }],
      ['hebbA- eta.005', { eta: 0.005, A: -0.3 }], ['hebbA- eta.1', { eta: 0.1, A: -0.3 }], ['hebbA- eta.5', { eta: 0.5, A: -0.3 }], ['pre B+ eta.005', { eta: 0.005, B: 0.3 }], ['post C+ eta.005', { eta: 0.005, C: 0.3 }], ['hebbA+ eta.005 decay.0005', { eta: 0.005, A: 0.3, decay: 0.0005 }]
    ],
    structure: [
      ['prune .002', { pruneT: 0.002 }], ['prune .01', { pruneT: 0.01 }], ['prune .03', { pruneT: 0.03 }], ['prune .06', { pruneT: 0.06 }],
      ['grow .02 w.02', { growP: 0.02, growW: 0.02, pruneT: 0.01 }], ['grow .1 w.02', { growP: 0.1, growW: 0.02, pruneT: 0.01 }], ['grow .3 w.02', { growP: 0.3, growW: 0.02, pruneT: 0.01 }],
      ['grow .1 w.2', { growP: 0.1, growW: 0.2, pruneT: 0.01 }], ['grow .3 w.2 prune.02', { growP: 0.3, growW: 0.2, pruneT: 0.02 }], ['grow .3 w.05 prune.005', { growP: 0.3, growW: 0.05, pruneT: 0.005 }]
    ]
  };
  const evaluateVariant = async (label, settings) => {
    const file = path.join(workDir, label.replace(/[^a-zA-Z0-9.+-]+/g, '_'));
    fs.writeFileSync(file, JSON.stringify(withParams(json, settings, label)));
    return runEvalsuite(['--game=' + opts.game, '--seeds=' + opts.evalSeeds, '--worlds=' + opts.evalWorlds, '--suites=bots', file]);
  };
  const reference = await evaluateVariant('off', {});
  const rows = [];
  const lines = ['# Plasticity and structure on ' + path.basename(genomePath) + ' (' + json.edges.length + ' edges)', '', '| group | variant | bots rate | vs off | paired se | relative | life | edges after 600 ticks | Brain vs RealtimeNpc | verdict |', '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |'];
  for (const [group, list] of Object.entries(groups)) {
    for (const [label, settings] of list) {
      const evaluation = label === 'off' ? reference : await evaluateVariant(label, settings);
      const diff = label === 'off' ? { mean: 0, se: 0, relative: 0 } : pairedDifference(evaluation, reference);
      const verdict = label === 'off' ? 'reference' : Math.abs(diff.mean) < 2 * diff.se ? 'neutral' : diff.mean > 0 ? 'helps' : 'hurts';
      const bots = evaluation.suites.bots.summary;
      const lockstep = verifyRuntime(withParams(json, settings, label), observations, 600, true);
      rows.push({ group, label, settings, rate: bots.rate, life: bots.life, diff, verdict, lockstep });
      lines.push('| ' + [group, label, pm(bots.rate, 5), fixed(diff.mean * 1000, 3) + 'e-3', fixed(diff.se * 1000, 3) + 'e-3', fixed(diff.relative * 100, 1) + '%', fixed(bots.life.mean, 0), lockstep.edgesAfter.brain, lockstep.brainVsRealtimeNpc.toExponential(1) + '/' + lockstep.weightDiff.toExponential(1), verdict].join(' | ') + ' |');
      log(opts, 'plasticity ' + group + ' ' + label + ' ' + verdict + ' ' + fixed(diff.relative * 100, 1) + '%');
    }
  }
  fs.writeFileSync(path.join(outDir, path.basename(genomePath) + '.plasticity.json'), JSON.stringify(rows, null, 1));
  const best = (group) => rows.filter((r) => r.group === group && r.verdict === 'helps').sort((a, b) => b.diff.relative - a.diff.relative)[0];
  const recommended = { maxEdges: Math.ceil(json.edges.length * 1.6) + 96, plasticity: best('plasticity') ? best('plasticity').settings : null, structure: best('structure') ? best('structure').settings : null, note: 'settings measured against the same genome with plasticity and structure off (bots suite, paired seeds); null = no setting helped beyond seed noise, keep the params at zero' };
  lines.push('', 'recommended: ' + JSON.stringify(recommended));
  fs.writeFileSync(path.join(outDir, path.basename(genomePath) + '.plasticity.md'), lines.join('\n') + '\n');
  if (opts.apply) {
    json.meta = Object.assign({}, json.meta, { maxEdges: recommended.maxEdges, recommended });
    fs.writeFileSync(genomePath, JSON.stringify(json));
  }
  fs.rmSync(workDir, { recursive: true, force: true });
  console.log(lines.join('\n'));
}

function summarizeReports(opts) {
  const files = opts.summary.split(',').map((f) => path.resolve(f));
  const lines = [];
  files.forEach((file) => {
    const report = JSON.parse(fs.readFileSync(file, 'utf8'));
    const teacher = report.teacher;
    const teacherBots = teacher.eval && teacher.eval.bots;
    lines.push('## ' + path.basename(report.options.genome) + ' / ' + path.basename(file).replace(/\.prune\.json$/, '') + ': teacher ' + teacher.edges + ' edges, ' + teacher.nodes + ' nodes, ' + teacher.hidden + ' hidden, ' + teacher.bench.toFixed(1) + ' us/step' + (teacherBots ? ', bots rate ' + pm(teacherBots.rate, 5) : ''), '');
    lines.push('| method | target | edges | %edges | nodes | us/tick | speedup | thr recall | move cos | bots rate | %teacher | bots ratio | h2h vs teacher | selfplay rate |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |');
    for (const method of report.methods) {
      for (const snap of method.snapshots) {
        const bots = snap.eval && snap.eval.bots;
        const pct = bots && teacherBots ? bots.rate.mean / teacherBots.rate.mean * 100 : null;
        lines.push('| ' + [method.label, snap.target, snap.edges, fixed(snap.edges / teacher.edges * 100, 1), snap.nodes, fixed(snap.usPerTick, 1), fixed(teacher.bench / snap.usPerTick, 1) + 'x', fixed(snap.fidelity.thresholdRecall, 3), fixed(snap.fidelity.moveCos, 3), pm(bots && bots.rate, 5), fixed(pct, 1), pm(bots && bots.ratio, 2), pm(snap.eval && snap.eval.h2h && snap.eval.h2h.ratio, 2), pm(snap.eval && snap.eval.selfplay && snap.eval.selfplay.rate, 5)].join(' | ') + ' |');
      }
    }
    lines.push('');
  });
  const out = path.resolve(opts.out || '.', 'PRUNE.md');
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, '# Pruning results (merged by tools/prune.mjs --summary)\n\n' + lines.join('\n') + '\n');
  console.log(out);
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  if (opts.summary) { summarizeReports(opts); return; }
  if (opts.plasticity) { await runPlasticity(opts); return; }
  if (!opts.genome) throw new Error('usage: node tools/prune.mjs --genome=<file> [--game=realm] [--targets=2048,1024,512,256] [--methods=act,mag,structured] [--out=<dir>] [--dagger=1] [--jobs=N]');
  const genomePath = path.resolve(opts.genome);
  const parsed = JSON.parse(fs.readFileSync(genomePath, 'utf8'));
  const teacherJson = Array.isArray(parsed) ? parsed[opts.index] : parsed;
  const game = await loadGameModule(opts.game);
  if (teacherJson.dims.nIn !== game.dims.nIn || teacherJson.dims.nOut !== game.dims.nOut) throw new Error('genome dims do not match game');
  const outDir = path.resolve(opts.out || path.dirname(genomePath));
  fs.mkdirSync(outDir, { recursive: true });
  const base = opts.name || path.basename(genomePath);
  const targets = opts.targets.split(',').map(Number).filter((n) => n > 0).sort((a, b) => b - a);
  const methods = opts.methods.split(',').filter(Boolean);
  for (const m of methods) if (!METHODS.includes(m.split('+')[0].split('/')[0])) throw new Error('unknown method ' + m);
  const net0 = P.compactNet(P.netFromGenome(teacherJson));
  const recurrent = P.groupCounts(net0)[2] > 0 || !P.isFeedForward(net0);
  const chunkLength = opts.chunk || (recurrent ? 192 : 64);
  opts.chunkLength = chunkLength;
  const jobs = opts.jobs || Math.max(2, Math.min(10, os.cpus().length - 4));
  const teacherStateCount = teacherJson.dims.nNodes - teacherJson.dims.nIn;
  const dims = { nIn: game.dims.nIn, nOut: game.dims.nOut, teacherStateCount };
  const sampleCap = Math.ceil((opts.samples * (opts.dagger ? 2.2 : 1.05) + 60000) * BUFFER_SLACK);
  const caps = { samples: sampleCap, chunks: Math.ceil(sampleCap / 8) };
  const dataset = new Dataset(caps, dims);
  const spec = P.lossSpecFor(game, { move: opts.moveWeight, threshold: opts.thresholdWeight });
  const pool = new Pool(jobs);
  const started = performance.now();
  await pool.broadcast('init', { gameSpec: opts.game, teacherJson, buffers: dataset.buffers, caps, dims });
  log(opts, 'teacher ' + path.basename(genomePath) + ' edges=' + net0.w.length + ' nodes=' + net0.nNodes + ' recurrent=' + recurrent + ' workers=' + jobs);
  let perRun = 20000;
  const cacheFile = opts.cache ? path.resolve(opts.cache) : '';
  if (cacheFile && fs.existsSync(cacheFile)) {
    const blob = JSON.parse(fs.readFileSync(cacheFile + '.json', 'utf8'));
    const bin = fs.readFileSync(cacheFile);
    const v = dataset.view;
    let offset = 0;
    const read =(typed, count, bytesPer) => { new Uint8Array(typed.buffer, typed.byteOffset, count * bytesPer).set(bin.subarray(offset, offset + count * bytesPer)); offset += count * bytesPer; };
    read(v.obs, blob.samples * dims.nIn, 4);
    read(v.tout, blob.samples * dims.nOut, 4);
    read(v.flags, blob.samples, 1);
    read(v.chunkStart, blob.chunks, 4);
    read(v.chunkLength, blob.chunks, 4);
    read(v.h0, blob.chunks * teacherStateCount, 4);
    dataset.samples = blob.samples;
    dataset.chunks = blob.chunks;
    log(opts, 'loaded cache ' + blob.samples + ' samples ' + blob.chunks + ' chunks');
  } else {
    const probe = await collectData(pool, dataset, { samples: 1, seed: opts.seed, offset: 0, chunk: chunkLength, gap: opts.gap, expectedPerRun: 1 });
    perRun = Math.max(1000, probe.added);
    const plan = { samples: opts.samples - probe.added, seed: opts.seed, offset: 1, chunk: chunkLength, gap: opts.gap, expectedPerRun: perRun * 1.6, lowDifficulty: true };
    const more = await collectData(pool, dataset, plan);
    log(opts, 'collected ' + dataset.samples + ' samples in ' + dataset.chunks + ' chunks (' + ((performance.now() - started) / 1000).toFixed(0) + 's), teacher tick reward ' + (rateOf(more.tally, 'teacher') || 0).toFixed(5));
    if (cacheFile) {
      const v = dataset.view;
      const pieces = [v.obs.subarray(0, dataset.samples * dims.nIn), v.tout.subarray(0, dataset.samples * dims.nOut), v.flags.subarray(0, dataset.samples), v.chunkStart.subarray(0, dataset.chunks), v.chunkLength.subarray(0, dataset.chunks), v.h0.subarray(0, dataset.chunks * teacherStateCount)];
      fs.writeFileSync(cacheFile, Buffer.concat(pieces.map((p) => Buffer.from(p.buffer, p.byteOffset, p.byteLength))));
      fs.writeFileSync(cacheFile + '.json', JSON.stringify({ samples: dataset.samples, chunks: dataset.chunks }));
    }
  }
  const baseChunks = dataset.chunks;
  const distiller = new Distiller(pool, dataset, spec, opts);
  const teacherMetrics = await distiller.evaluate(net0);
  log(opts, 'teacher self-consistency on validation: loss=' + teacherMetrics.loss.toExponential(2) + ' rmse=' + teacherMetrics.rmse.toExponential(2) + ' agree=' + teacherMetrics.sideAgree);
  const obsSamples = new Float32Array(dataset.view.obs.buffer, 0, Math.min(dataset.samples, 6000) * dims.nIn);
  const context = { perRun, opts, pool, dataset, distiller, spec, net0, teacherJson, targets, obsSamples, baseChunks, game, outDir, base, genomePath, recurrent };
  try {
    await runAllMethods(context, methods);
  } finally {
    await pool.close();
  }
}

if (isMainThread) main().catch((err) => { console.error(err.stack || String(err)); process.exit(1); });
