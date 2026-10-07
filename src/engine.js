import { evoConfig, difficultyAt, STAT, NSTATS, WORLD_HDR_WORDS, REWARD_SCALE, StatsAccumulator, DEFAULT_OPTS, evalWorldCount, penaltyAt, validateGame, mergeIntoArchive, makeGenome, randomGenome, Brain, Rng, mix, genomeToJSON, genomeFromJSON, NEG_INF_FIT, VALID_FIT, ARCHIVE_SIZE, ARCHIVE_DECAY_PER_TICK, STRUCT_PERIOD, NODE_BITS, NODE_MASK, PARAM_LO, PARAM_HI, CpuBackend } from './core.js';
import { buildShader, shaderLayout, UNIFORM_WORDS, BRAIN_STATE } from './shader.js';

const GPU_ENTRIES = ['init_world', 'sim_step', 'select_archive', 'reduce_stats', 'brain_test', 'env_test', 'observe_test'];
const WANTED_LIMITS = ['maxStorageBufferBindingSize', 'maxBufferSize', 'maxComputeWorkgroupStorageSize', 'maxStorageBuffersPerShaderStage', 'maxComputeInvocationsPerWorkgroup', 'maxComputeWorkgroupSizeX', 'maxComputeWorkgroupsPerDimension', 'maxBindGroups'];
const STEP_TARGET_MS = 40;
const STEP_MAX_MS = 200;
const STEP_SPLIT_MS = 120;
const FIXED_SHARE = 0.35;
const DEFAULT_SELECT_TICKS = 64;
const NOMINAL_PASS_TICKS = 16;
const FIXED_PROBE_PASSES = 96;
const SAMPLE_WINDOW = 7;
const MAX_AUTO_TICKS = 1024;
const STATS_FLUSH_PASSES = 64;
const SCRATCH_BYTES = 4 * 1024 * 1024;

let deviceContextPromise = null;

async function createDeviceContext() {
  if (typeof navigator === 'undefined' || !navigator.gpu) throw new Error('WebGPU is not available in this browser');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter found');
  const limits = {};
  for (const key in adapter.limits) if (typeof adapter.limits[key] === 'number') limits[key] = adapter.limits[key];
  const features = Array.from(adapter.features);
  const requiredFeatures = ['shader-f16', 'timestamp-query'].filter((f) => features.includes(f));
  let device;
  try {
    device = await adapter.requestDevice({ requiredFeatures, requiredLimits: limits });
  } catch (error) {
    const subset = {};
    for (const key of WANTED_LIMITS) if (key in limits) subset[key] = limits[key];
    device = await adapter.requestDevice({ requiredFeatures, requiredLimits: subset });
  }
  let adapterInfo = adapter.info || {};
  if (!adapterInfo.description && adapter.requestAdapterInfo) adapterInfo = await adapter.requestAdapterInfo();
  const context = { adapter, device, limits, features, granted: {}, adapterInfo: { vendor: adapterInfo.vendor || '', architecture: adapterInfo.architecture || '', device: adapterInfo.device || '', description: adapterInfo.description || '' }, lost: null };
  for (const key in device.limits) if (typeof device.limits[key] === 'number') context.granted[key] = device.limits[key];
  device.lost.then((info) => { context.lost = info; deviceContextPromise = null; });
  return context;
}

export function getDeviceContext() {
  if (!deviceContextPromise) deviceContextPromise = createDeviceContext().catch((error) => { deviceContextPromise = null; throw error; });
  return deviceContextPromise;
}

export function worldCapacity(ctx, game, opts) {
  const limit = Math.min(ctx.granted.maxStorageBufferBindingSize, ctx.granted.maxBufferSize);
  const learners = game.learners;
  const genomeStride = 24 + 2 * opts.maxEdges;
  const brainStride = 20 + game.dims.nIn + 8 + game.dims.nNodes + game.dims.nOut;
  const perWorld = [opts.maxEdges * learners * 4, learners * brainStride * 4, (WORLD_HDR_WORDS + game.worldWords) * 4, 4 * genomeStride * 4];
  return Math.floor(Math.min(...perWorld.map((bytes) => limit / bytes)));
}

function adaptedDifficulty(backend, o) {
  return o.difficultyAdaptive && backend.adaptiveLevel !== undefined ? Math.round(backend.adaptiveLevel) : difficultyAt(o, backend.tick);
}

function adaptDifficulty(backend, trainDelta) {
  const o = backend.opts;
  if (!o.difficultyAdaptive) return;
  if (backend.adaptiveLevel === undefined) backend.adaptiveLevel = o.difficultyStart;
  const deaths = trainDelta[STAT.DEATHS];
  if (deaths > 0 && trainDelta[STAT.TICKS_EVO] / deaths > o.adaptLife) backend.adaptiveLevel = Math.min(o.difficultyEnd, backend.adaptiveLevel + o.adaptStep);
}

const f32Bits = new Float32Array(1);
const u32Bits = new Uint32Array(f32Bits.buffer);
const bitsToF32 = (u) => { u32Bits[0] = u; return f32Bits[0]; };
const f32ToBits = (f) => { f32Bits[0] = f; return u32Bits[0]; };

export class GpuBackend {
  static async probe() {
    try {
      const ctx = await getDeviceContext();
      return { available: true, adapter: ctx.adapterInfo, limits: ctx.granted, adapterLimits: ctx.limits, features: ctx.features, shaderF16: ctx.features.includes('shader-f16'), timestampQuery: ctx.features.includes('timestamp-query') };
    } catch (error) {
      return { available: false, error: error.message, adapter: {}, limits: {}, features: [] };
    }
  }

  static async create(opts) {
    const ctx = opts.deviceCtx || (await getDeviceContext());
    const merged = Object.assign({}, DEFAULT_OPTS, opts);
    validateGame(merged.game);
    const warnings = [];
    const cap = worldCapacity(ctx, merged.game, merged);
    if (merged.worlds > cap) {
      warnings.push('worlds reduced from ' + merged.worlds + ' to ' + cap + ' by buffer limits');
      merged.worlds = cap;
    }
    for (let attempt = 0; attempt < 4; attempt++) {
      const backend = new GpuBackend(ctx, merged);
      backend.info.warnings = warnings.slice();
      try {
        await backend.init();
        return backend;
      } catch (error) {
        backend.destroyBuffers();
        if (!/out of memory|OOM/i.test(error.message) || merged.worlds <= 16) throw error;
        warnings.push('out of memory at ' + merged.worlds + ' worlds, halved');
        merged.worlds = Math.floor(merged.worlds / 2);
      }
    }
    throw new Error('could not allocate GPU memory');
  }

  constructor(ctx, opts) {
    this.ctx = ctx;
    this.device = ctx.device;
    this.opts = opts;
    this.game = opts.game;
    this.kind = 'gpu';
    this.evalCount = evalWorldCount(opts);
    this.evalStart = opts.worlds - this.evalCount;
    if (opts.islands > this.evalStart) throw new Error('islands must not exceed training worlds');
    this.islandSize = Math.floor(this.evalStart / opts.islands);
    this.layout = shaderLayout(this.game, { worlds: opts.worlds, islands: opts.islands, evalStart: this.evalStart, maxEdges: opts.maxEdges, migrateCount: opts.migrateCount });
    this.stats = new StatsAccumulator();
    this.tick = 0;
    this.passes = 0;
    this.passesSinceFlush = 0;
    this.archiveIndex = 0;
    this.autoTicks = 8;
    this.lastStepMs = 0;
    this.inflight = null;
    this.samples = [];
    this.fixedMs = 0;
    this.rttMs = 0;
    this.ticksSinceSelect = 0;
    this.ticksSinceMigrate = 0;
    this.lastPassEnd = 0;
    this.worldScores = new Float32Array(opts.worlds);
    this.archiveSummary = { best: 0, mean: 0, count: 0 };
    this.buffers = {};
    this.info = { kind: 'gpu', adapter: ctx.adapterInfo, limits: ctx.granted, features: ctx.features, worlds: opts.worlds, islands: opts.islands, maxEdges: opts.maxEdges, game: this.game.id, workgroupBytes: this.layout.workgroupBytes, warnings: [] };
  }

  makeBuffer(name, bytes, usage) {
    const size = Math.max(16, Math.ceil(bytes / 4) * 4);
    if (size > this.ctx.granted.maxBufferSize) throw new Error('buffer ' + name + ' of ' + size + ' bytes exceeds device maxBufferSize');
    if ((usage & GPUBufferUsage.STORAGE) && size > this.ctx.granted.maxStorageBufferBindingSize) throw new Error('buffer ' + name + ' of ' + size + ' bytes exceeds maxStorageBufferBindingSize');
    const buffer = this.device.createBuffer({ label: name, size, usage });
    this.buffers[name] = buffer;
    return buffer;
  }

  destroyBuffers() {
    for (const name in this.buffers) this.buffers[name].destroy();
    this.buffers = {};
  }

  async withErrorScopes(label, fn) {
    this.device.pushErrorScope('out-of-memory');
    this.device.pushErrorScope('validation');
    let value;
    let thrown = null;
    try {
      value = await fn();
    } catch (error) {
      thrown = error;
    }
    const validation = await this.device.popErrorScope();
    const memory = await this.device.popErrorScope();
    if (memory) throw new Error(label + ': out of memory: ' + memory.message);
    if (validation) throw new Error(label + ': ' + validation.message);
    if (thrown) throw thrown;
    return value;
  }

  async init() {
    const L = this.layout;
    const o = this.opts;
    const limits = this.ctx.granted;
    if (L.workgroupBytes > limits.maxComputeWorkgroupStorageSize) throw new Error('game needs ' + L.workgroupBytes + ' bytes of workgroup memory, device allows ' + limits.maxComputeWorkgroupStorageSize);
    if (limits.maxStorageBuffersPerShaderStage < 9) throw new Error('device allows only ' + limits.maxStorageBuffersPerShaderStage + ' storage buffers per stage, 9 required');
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    await this.withErrorScopes('allocation', async () => {
      this.makeBuffer('edgePk', o.maxEdges * L.brainCount * 4, S);
      this.makeBuffer('edgeW', o.maxEdges * L.brainCount * 4, S);
      this.makeBuffer('brain', L.brainCount * L.brainStride * 4, S);
      this.makeBuffer('world', o.worlds * L.worldStride * 4, S);
      this.makeBuffer('graveyard', o.worlds * 4 * L.genomeStride * 4, S);
      this.makeBuffer('archiveA', o.islands * L.islandBlock * 4, S);
      this.makeBuffer('archiveB', o.islands * L.islandBlock * 4, S);
      this.makeBuffer('statsOut', (o.islands * L.statsPerIsland + o.worlds * 2) * 4, S);
      this.makeBuffer('scratch', SCRATCH_BYTES, S);
      this.makeBuffer('uniforms', UNIFORM_WORDS * 4, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    });
    const built = buildShader(this.game, { worlds: o.worlds, islands: o.islands, evalStart: this.evalStart, maxEdges: o.maxEdges, migrateCount: o.migrateCount, evo: evoConfig(o) });
    const code = o.shaderTransform ? o.shaderTransform(built) : built;
    this.shaderCode = code;
    const module = this.device.createShaderModule({ label: 'train', code });
    const compilation = await module.getCompilationInfo();
    const errors = compilation.messages.filter((m) => m.type === 'error');
    if (errors.length > 0) {
      const lines = code.split('\n');
      throw new Error('WGSL compile failed:\n' + errors.slice(0, 8).map((m) => 'line ' + m.lineNum + ': ' + m.message + '\n    ' + (lines[m.lineNum - 1] || '')).join('\n'));
    }
    const storage = (binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } });
    const readonly = (binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } });
    const bindLayout = this.device.createBindGroupLayout({ entries: [storage(0), storage(1), storage(2), storage(3), storage(4), readonly(5), storage(6), storage(7), storage(8), { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }] });
    const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [bindLayout] });
    this.pipelines = {};
    await this.withErrorScopes('pipelines', async () => {
      await Promise.all((o.pipelineEntries || GPU_ENTRIES).map(async (entry) => {
        this.pipelines[entry] = await this.device.createComputePipelineAsync({ label: entry, layout: pipelineLayout, compute: { module, entryPoint: entry } });
      }));
    });
    const b = this.buffers;
    const makeGroup = (archive, archiveOut) => this.device.createBindGroup({ layout: bindLayout, entries: [b.edgePk, b.edgeW, b.brain, b.world, b.graveyard, archive, archiveOut, b.statsOut, b.scratch, b.uniforms].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    this.bindGroups = [makeGroup(b.archiveA, b.archiveB), makeGroup(b.archiveB, b.archiveA)];
    this.uniformData = new ArrayBuffer(UNIFORM_WORDS * 4);
    this.uniformU32 = new Uint32Array(this.uniformData);
    this.uniformF32 = new Float32Array(this.uniformData);
    this.writeUniforms({ ticks: 0 });
    const encoder = this.device.createCommandEncoder();
    this.dispatch(encoder, 'init_world', o.worlds);
    await this.submit(encoder);
  }

  writeUniforms(v) {
    const o = this.opts;
    const u = this.uniformU32;
    const f = this.uniformF32;
    u[0] = v.ticks || 0;
    u[1] = v.tick0 !== undefined ? v.tick0 : this.tick;
    f[2] = v.penalty !== undefined ? v.penalty : penaltyAt(o, this.tick);
    f[3] = o.mutation;
    u[4] = v.flags || 0;
    u[5] = this.evalStart;
    u[6] = this.islandSize;
    u[7] = o.islands;
    u[8] = o.randomize ? 1 : 0;
    u[9] = o.seed >>> 0;
    f[10] = o.archiveDecay * (v.decayTicks !== undefined ? v.decayTicks : v.ticks || 0);
    u[11] = v.migrate ? 1 : 0;
    u[12] = o.migrateCount;
    u[13] = v.difficulty !== undefined ? v.difficulty : adaptedDifficulty(this, o);
    u[14] = o.styleGate | 0;
    u[15] = o.styleGateEval | 0;
    this.device.queue.writeBuffer(this.buffers.uniforms, 0, this.uniformData);
  }

  dispatch(encoder, entry, groups, groupIndex) {
    const pass = encoder.beginComputePass({ label: entry });
    pass.setPipeline(this.pipelines[entry]);
    pass.setBindGroup(0, this.bindGroups[groupIndex === undefined ? this.archiveIndex : groupIndex]);
    pass.dispatchWorkgroups(groups);
    pass.end();
  }

  async submit(encoder) {
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
  }

  async readWords(buffer, byteOffset, byteLength) {
    const staging = this.device.createBuffer({ size: byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, byteOffset, staging, 0, byteLength);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const copy = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();
    return copy;
  }

  currentTicks() {
    return this.opts.ticksPerDispatch > 0 ? this.opts.ticksPerDispatch : this.autoTicks;
  }

  selectTicks() {
    return this.opts.selectTicks === undefined ? DEFAULT_SELECT_TICKS : this.opts.selectTicks;
  }

  perTickMs() {
    const usable = this.samples.filter((s) => s.ticks > 0).map((s) => Math.max(0, s.ms - this.fixedMs) / s.ticks).sort((a, b) => a - b);
    return usable.length ? usable[Math.floor(usable.length / 2)] : 0;
  }

  retuneTicks() {
    const p = this.perTickMs();
    if (!(p > 0) || this.samples.length < 3) return;
    const target = this.opts.stepTargetMs || STEP_TARGET_MS;
    const F = this.fixedMs;
    const cap = Math.max(1, Math.floor((STEP_MAX_MS - F) / p));
    const floorTicks = Math.min(cap, Math.max(1, Math.ceil(F / (FIXED_SHARE * p))));
    const wanted = Math.max(floorTicks, Math.floor((target - F) / p));
    const clamped = Math.max(1, Math.min(MAX_AUTO_TICKS, cap, wanted));
    const current = this.autoTicks;
    const bounded = Math.max(current * 0.7, Math.min(current * 1.5, clamped));
    this.autoTicks = Math.max(1, Math.min(MAX_AUTO_TICKS, Math.round(bounded)));
  }

  async probeFixedCost() {
    await this.settle();
    const empty = performance.now();
    this.device.queue.submit([]);
    await this.device.queue.onSubmittedWorkDone();
    const rtt = performance.now() - empty;
    this.writeUniforms({ ticks: 0, decayTicks: 0 });
    const encoder = this.device.createCommandEncoder();
    this.dispatch(encoder, 'sim_step', this.opts.worlds);
    const begin = performance.now();
    await this.submit(encoder);
    const kernel = Math.max(0, performance.now() - begin - rtt);
    this.rttMs = this.rttMs ? Math.min(this.rttMs, rtt) : rtt;
    this.fixedMs = this.fixedMs ? Math.min(this.fixedMs, kernel) : kernel;
    this.lastPassEnd = performance.now();
  }

  async settle() {
    const pending = this.inflight;
    if (!pending) return;
    this.inflight = null;
    const end = await pending.done;
    this.account(pending, end);
  }

  account(pass, end) {
    const ms = Math.max(0.01, end - Math.max(pass.submitted, this.lastPassEnd));
    this.lastPassEnd = end;
    this.lastStepMs = ms;
    this.samples.push({ ticks: pass.ticks, ms });
    if (this.samples.length > SAMPLE_WINDOW) this.samples.shift();
    if (this.opts.ticksPerDispatch <= 0) this.retuneTicks();
  }

  async step() {
    if (this.ctx.lost) throw new Error('GPU device lost: ' + this.ctx.lost.message);
    const o = this.opts;
    if (this.opts.ticksPerDispatch <= 0 && this.passes % FIXED_PROBE_PASSES === 2) await this.probeFixedCost();
    const ticks = this.currentTicks();
    const evalPeriod = this.game.maxAge * o.evalEvery;
    const evalReset = o.evalEvery > 0 && Math.floor((this.tick + ticks) / evalPeriod) > Math.floor(this.tick / evalPeriod);
    const sinceSelect = this.ticksSinceSelect + ticks;
    const doSelect = (this.passes + 1) % o.selectEvery === 0 && sinceSelect >= this.selectTicks();
    const sinceMigrate = this.ticksSinceMigrate + ticks;
    const migrate = doSelect && o.migrateEvery > 0 && sinceMigrate >= o.migrateEvery * NOMINAL_PASS_TICKS;
    const predicted = this.fixedMs + this.perTickMs() * ticks;
    const parts = predicted > STEP_SPLIT_MS && ticks > 1 ? Math.min(ticks, Math.ceil(predicted / STEP_SPLIT_MS)) : 1;
    const submitted = performance.now();
    let done = 0;
    for (let part = 0; part < parts; part++) {
      const partTicks = Math.floor(ticks / parts) + (part < ticks % parts ? 1 : 0);
      const last = part === parts - 1;
      this.writeUniforms({ ticks: partTicks, tick0: this.tick + done, flags: evalReset && part === 0 ? 1 : 0, migrate: last && migrate, decayTicks: last && doSelect ? sinceSelect : 0 });
      const encoder = this.device.createCommandEncoder();
      this.dispatch(encoder, 'sim_step', o.worlds);
      if (last && doSelect) this.dispatch(encoder, 'select_archive', o.islands);
      this.device.queue.submit([encoder.finish()]);
      done += partTicks;
    }
    if (doSelect) {
      this.archiveIndex ^= 1;
      this.ticksSinceSelect = 0;
      if (migrate) this.ticksSinceMigrate = 0;
      else this.ticksSinceMigrate = sinceMigrate;
    } else {
      this.ticksSinceSelect = sinceSelect;
      this.ticksSinceMigrate = sinceMigrate;
    }
    this.tick += ticks;
    this.passes++;
    this.passesSinceFlush++;
    const completion = { ticks, submitted, done: this.device.queue.onSubmittedWorkDone().then(() => performance.now()) };
    const previous = this.inflight;
    this.inflight = completion;
    if (previous) this.account(previous, await previous.done);
    if (this.passesSinceFlush >= STATS_FLUSH_PASSES) await this.flushStats();
    return { ticks, ms: this.lastStepMs };
  }

  async flushStats() {
    const o = this.opts;
    const L = this.layout;
    const encoder = this.device.createCommandEncoder();
    this.dispatch(encoder, 'reduce_stats', o.islands);
    await this.submit(encoder);
    const bytes = (o.islands * L.statsPerIsland + o.worlds * 2) * 4;
    const raw = await this.readWords(this.buffers.statsOut, 0, bytes);
    const rows = new Int32Array(raw);
    const rowsF = new Float32Array(raw);
    const train = new Float64Array(NSTATS);
    const evalDelta = new Float64Array(NSTATS);
    let bestFit = -Infinity;
    let fitSum = 0;
    let archiveCount = 0;
    for (let island = 0; island < o.islands; island++) {
      const base = island * L.statsPerIsland;
      for (let i = 0; i < NSTATS; i++) {
        train[i] += rows[base + i];
        evalDelta[i] += rows[base + NSTATS + i];
      }
      const count = rows[base + 2 * NSTATS] >>> 0;
      if (count > 0) {
        bestFit = Math.max(bestFit, rowsF[base + 2 * NSTATS + 1]);
        fitSum += rowsF[base + 2 * NSTATS + 2] * count;
        archiveCount += count;
      }
    }
    const worldBase = o.islands * L.statsPerIsland;
    for (let w = 0; w < o.worlds; w++) this.worldScores[w] = rows[worldBase + w * 2] / REWARD_SCALE / Math.max(1, rows[worldBase + w * 2 + 1]);
    adaptDifficulty(this, train);
    this.stats.add('train', train, this.evalStart * this.game.learners);
    this.stats.add('eval', evalDelta, this.evalCount * this.game.learners);
    this.archiveSummary = { best: archiveCount ? bestFit : 0, mean: archiveCount ? fitSum / archiveCount : 0, count: archiveCount };
    this.passesSinceFlush = 0;
  }

  setParams(partial) {
    Object.assign(this.opts, partial);
  }

  async readStats() {
    await this.flushStats();
    const train = this.stats.summary('train', this.game);
    const evalSummary = this.stats.summary('eval', this.game);
    this.stats.clearWindow();
    return { tick: this.tick, passes: this.passes, train, eval: evalSummary, archive: this.archiveSummary, worldScores: this.worldScores, stepMs: this.lastStepMs, ticksPerDispatch: this.currentTicks() };
  }

  async snapshot(worldIndex) {
    const L = this.layout;
    const all = new Uint32Array(await this.readWords(this.buffers.world, worldIndex * L.worldStride * 4, L.worldStride * 4));
    return this.game.snapshotFromWords(all.slice(WORLD_HDR_WORDS, WORLD_HDR_WORDS + this.game.worldWords), all[0]);
  }

  currentArchiveBuffer() {
    return this.archiveIndex === 0 ? this.buffers.archiveA : this.buffers.archiveB;
  }

  async readArchives() {
    const L = this.layout;
    const words = new Uint32Array(await this.readWords(this.currentArchiveBuffer(), 0, this.opts.islands * L.islandBlock * 4));
    const floats = new Float32Array(words.buffer);
    const archives = [];
    for (let island = 0; island < this.opts.islands; island++) {
      const base = island * L.islandBlock;
      const count = words[base];
      const genomes = [];
      for (let i = 0; i < count; i++) {
        const g0 = base + 4 + i * L.genomeStride;
        const genome = makeGenome(this.opts.maxEdges);
        genome.fit = floats[g0];
        genome.n = words[g0 + 1];
        genome.niche = words[g0 + 22];
        for (let p = 0; p < 20; p++) genome.p[p] = floats[g0 + 2 + p];
        for (let e = 0; e < genome.n; e++) {
          genome.pk[e] = words[g0 + L.genomeHeader + e];
          genome.w[e] = floats[g0 + L.genomeHeader + this.opts.maxEdges + e];
        }
        genomes.push(genome);
      }
      archives.push({ count, genomes });
    }
    return archives;
  }

  async writeArchives(archives) {
    const L = this.layout;
    const words = new Uint32Array(this.opts.islands * L.islandBlock);
    const floats = new Float32Array(words.buffer);
    for (let island = 0; island < this.opts.islands; island++) {
      const archive = archives[island % archives.length];
      const base = island * L.islandBlock;
      const count = Math.min(archive.genomes.length, ARCHIVE_SIZE);
      words[base] = count;
      floats[base + 1] = count ? archive.genomes[0].fit : 0;
      for (let i = 0; i < count; i++) {
        const g = archive.genomes[i];
        const g0 = base + 4 + i * L.genomeStride;
        floats[g0] = g.fit;
        words[g0 + 1] = g.n;
        words[g0 + 22] = g.niche || 0;
        for (let p = 0; p < 20; p++) floats[g0 + 2 + p] = g.p[p];
        for (let e = 0; e < g.n; e++) {
          words[g0 + L.genomeHeader + e] = g.pk[e];
          floats[g0 + L.genomeHeader + this.opts.maxEdges + e] = g.w[e];
        }
      }
    }
    this.device.queue.writeBuffer(this.currentArchiveBuffer(), 0, words);
    await this.device.queue.onSubmittedWorkDone();
  }

  async exportChampions(n) {
    const all = [];
    for (const archive of await this.readArchives()) for (const g of archive.genomes) all.push(g);
    all.sort((a, b) => b.fit - a.fit);
    return all.slice(0, n).map((g) => genomeToJSON(g, this.game));
  }

  async importGenomes(jsonArray) {
    const genomes = jsonArray.map((j) => genomeFromJSON(j, this.opts.maxEdges));
    const archives = await this.readArchives();
    for (const archive of archives) {
      const copies = genomes.map((g) => ({ fit: g.fit, genome: g }));
      mergeIntoArchive(archive, copies, []);
    }
    await this.writeArchives(archives);
  }

  async checkpoint() {
    const o = Object.assign({}, this.opts);
    delete o.game;
    delete o.deviceCtx;
    const archives = await this.readArchives();
    return { kind: 'gpu', game: this.game.id, tick: this.tick, passes: this.passes, opts: o, archives: archives.map((a) => a.genomes.map((g) => genomeToJSON(g, this.game))) };
  }

  async restore(obj) {
    this.tick = obj.tick;
    this.passes = obj.passes;
    const archives = obj.archives.map((list) => {
      const genomes = list.map((j) => genomeFromJSON(j, this.opts.maxEdges));
      return { count: genomes.length, genomes };
    });
    await this.writeArchives(archives);
  }

  async dispose() {
    await this.settle();
    this.destroyBuffers();
  }

  async tempBackend(overrides) {
    return GpuBackend.create(Object.assign({ game: this.game, maxEdges: this.opts.maxEdges, deviceCtx: this.ctx, randomize: true, seed: 4242, migrateCount: 4, difficultyStart: 100, difficultyEnd: 100, migrateEvery: 2, selectEvery: 1, selectTicks: 0, evalEvery: 4 }, evoConfig(this.opts), overrides));
  }

  uploadBrains(backend, genomes) {
    const L = backend.layout;
    const maxE = backend.opts.maxEdges;
    const BR = L.brainCount;
    const pk = new Uint32Array(maxE * BR);
    const w = new Float32Array(maxE * BR);
    const brain = new Float32Array(BR * L.brainStride);
    genomes.forEach((g, b) => {
      for (let e = 0; e < g.n; e++) {
        pk[e * BR + b] = g.pk[e];
        w[e * BR + b] = g.w[e];
      }
      const bb = b * L.brainStride;
      for (let p = 0; p < 20; p++) brain[bb + p] = g.p[p];
      brain[bb + L.brainOffState + BRAIN_STATE.SKIP] = 1000;
      brain[bb + L.brainOffState + BRAIN_STATE.EDGES] = g.n;
    });
    this.device.queue.writeBuffer(backend.buffers.edgePk, 0, pk);
    this.device.queue.writeBuffer(backend.buffers.edgeW, 0, w);
    this.device.queue.writeBuffer(backend.buffers.brain, 0, brain);
  }

  async testBrainParity() {
    const game = this.game;
    const dims = game.dims;
    const L0 = game.learners;
    const steps = 70;
    const tmp = await this.tempBackend({ worlds: 4, islands: 2, evalFraction: 0.5 });
    try {
      const L = tmp.layout;
      const maxE = tmp.opts.maxEdges;
      const genomes = [];
      for (let a = 0; a < L0; a++) genomes.push(randomGenome(new Rng(1000 + a), maxE, dims));
      this.uploadBrains(tmp, genomes);
      const inRng = new Rng(555);
      const inputs = new Float32Array(steps * L0 * dims.nIn);
      const rewardBase = steps * L0 * dims.nIn;
      const actionBase = rewardBase + steps * L0;
      const scratch = new Float32Array(actionBase + steps * L0);
      for (let i = 0; i < inputs.length; i++) scratch[i] = inRng.next() < 0.3 ? 0 : Math.fround(inRng.next());
      const rewards = new Int32Array(steps * L0);
      const rewardChoices = [0, 0, 0, 128, -256, 512, 1536, -1024];
      for (let i = 0; i < rewards.length; i++) {
        rewards[i] = rewardChoices[Math.floor(inRng.next() * rewardChoices.length)];
        scratch[rewardBase + i] = rewards[i];
      }
      this.device.queue.writeBuffer(tmp.buffers.scratch, 0, scratch);
      tmp.writeUniforms({ ticks: steps, tick0: 0 });
      const encoder = tmp.device.createCommandEncoder();
      tmp.dispatch(encoder, 'brain_test', 1);
      await tmp.submit(encoder);
      const gpuScratch = new Float32Array(await tmp.readWords(tmp.buffers.scratch, 0, scratch.byteLength));
      const gpuBrain = new Float32Array(await tmp.readWords(tmp.buffers.brain, 0, L.brainCount * L.brainStride * 4));
      const gpuW = new Float32Array(await tmp.readWords(tmp.buffers.edgeW, 0, maxE * L.brainCount * 4));
      const gpuPk = new Uint32Array(await tmp.readWords(tmp.buffers.edgePk, 0, maxE * L.brainCount * 4));
      let actionMismatch = 0;
      let nMismatch = 0;
      let maxWeightDiff = 0;
      let maxActDiff = 0;
      const jsInputs = new Float32Array(dims.nIn);
      for (let a = 0; a < L0; a++) {
        const brain = new Brain(genomes[a], maxE, dims, evoConfig(this.opts));
        let life = 0;
        for (let k = 0; k < steps; k++) {
          for (let i = 0; i < dims.nIn; i++) jsInputs[i] = scratch[(k * L0 + a) * dims.nIn + i];
          const action = brain.step(jsInputs, new Rng(mix(777, k, a, 1)));
          if (action !== gpuScratch[actionBase + k * L0 + a]) actionMismatch++;
          brain.learn(rewards[k * L0 + a] / REWARD_SCALE);
          life++;
          if (life % STRUCT_PERIOD === 0) brain.structural(new Rng(mix(777, k, a, 3)));
        }
        const bb = a * L.brainStride;
        const gpuN = gpuBrain[bb + L.brainOffState + BRAIN_STATE.EDGES];
        if (gpuN !== brain.n) nMismatch++;
        else {
          for (let e = 0; e < brain.n; e++) {
            if (gpuPk[e * L.brainCount + a] === brain.pk[e]) maxWeightDiff = Math.max(maxWeightDiff, Math.abs(gpuW[e * L.brainCount + a] - brain.w[e]));
          }
        }
        for (let j = 0; j < dims.nNodes; j++) maxActDiff = Math.max(maxActDiff, Math.abs(gpuBrain[bb + L.brainOffAct + j] - brain.act[j]));
      }
      const total = steps * L0;
      const detail = 'actions ' + (total - actionMismatch) + '/' + total + ' equal, edge counts differ in ' + nMismatch + '/' + L0 + ' brains, max weight diff ' + maxWeightDiff.toExponential(2) + ', max activation diff ' + maxActDiff.toExponential(2);
      if (actionMismatch > total * 0.02 || nMismatch > L0 * 0.15 || maxWeightDiff > 5e-3 || maxActDiff > 5e-3) throw new Error(detail);
      return detail;
    } finally {
      await tmp.dispose();
    }
  }

  policyOutputs(rng, count, nOut) {
    const out = new Float32Array(count * nOut);
    for (let i = 0; i < out.length; i++) out[i] = Math.fround(rng.next());
    return out;
  }

  async testInitParity() {
    const game = this.game;
    const tmp = await this.tempBackend({ worlds: 6, islands: 2, evalFraction: 0.34 });
    try {
      const L = tmp.layout;
      let checked = 0;
      for (let w = 0; w < tmp.opts.worlds; w++) {
        const words = new Uint32Array(await tmp.readWords(tmp.buffers.world, w * L.worldStride * 4, L.worldStride * 4));
        const seed = words[0];
        const isEval = words[1] === 1;
        const cfg = tmp.opts.randomize && !isEval ? game.randomCfg(seed, tmp.opts) : game.defaultCfg(tmp.opts);
        const env = game.createEnv(seed, cfg, isEval, new Int32Array(NSTATS));
        const expected = game.packEnv(env);
        for (let i = 0; i < expected.length; i++) {
          if (words[WORLD_HDR_WORDS + i] !== expected[i]) throw new Error('world ' + w + ' word ' + i + ': gpu ' + words[WORLD_HDR_WORDS + i] + ' js ' + expected[i]);
        }
        checked++;
      }
      return checked + ' worlds initialised identically to the JS env (' + game.worldWords + ' words each)';
    } finally {
      await tmp.dispose();
    }
  }

  async testEnvLockstep(dense) {
    const game = this.game;
    const dims = game.dims;
    const ticks = dense ? 120 : 160;
    const tmp = await this.tempBackend({ worlds: 4, islands: 2, evalFraction: 0.5 });
    try {
      const L = tmp.layout;
      const worldBytes = L.worldStride * 4;
      const header = new Uint32Array(await tmp.readWords(tmp.buffers.world, 0, worldBytes));
      const seed = header[0];
      const stats = new Int32Array(NSTATS);
      const env = game.createEnv(seed, game.randomCfg(seed, tmp.opts), false, stats);
      if (dense) {
        game.densify(env, new Rng(99));
        this.device.queue.writeBuffer(tmp.buffers.world, WORLD_HDR_WORDS * 4, game.packEnv(env));
        await this.device.queue.onSubmittedWorkDone();
      }
      const policyRng = new Rng(2024);
      const actions = new Int32Array(game.learners);
      for (let k = 0; k < ticks; k++) {
        const outputs = this.policyOutputs(policyRng, game.learners, dims.nOut);
        const scratch = new Float32Array(game.learners * (dims.nOut + 1));
        for (let a = 0; a < game.learners; a++) {
          let best = 0;
          for (let j = 1; j < dims.nOut; j++) if (outputs[a * dims.nOut + j] > outputs[a * dims.nOut + best]) best = j;
          actions[a] = best;
          scratch[a * (dims.nOut + 1)] = best;
          for (let j = 0; j < dims.nOut; j++) scratch[a * (dims.nOut + 1) + 1 + j] = outputs[a * dims.nOut + j];
        }
        this.device.queue.writeBuffer(tmp.buffers.scratch, 0, scratch);
        tmp.writeUniforms({ ticks: 1, tick0: k });
        const encoder = tmp.device.createCommandEncoder();
        tmp.dispatch(encoder, 'env_test', 1);
        await tmp.submit(encoder);
        env.step(actions, outputs, k);
        for (let a = 0; a < game.learners; a++) if (env.dead[a]) env.respawn(a, k);
        const gpuWords = new Uint32Array(await tmp.readWords(tmp.buffers.world, WORLD_HDR_WORDS * 4, game.worldWords * 4));
        const expected = game.packEnv(env);
        for (let i = 0; i < expected.length; i++) {
          if (gpuWords[i] !== expected[i]) {
            const label = game.wordLabel ? game.wordLabel(i) : 'word ' + i;
            const row = game.wordRow ? ' | ' + game.wordRow(i, gpuWords, expected) : '';
            throw new Error('diverged at tick ' + k + ' ' + label + ': gpu ' + (gpuWords[i] | 0) + ' js ' + (expected[i] | 0) + row);
          }
        }
      }
      return ticks + ' ticks identical word for word (' + game.worldWords + ' words per tick)' + (dense ? ', game events ' + game.statNames.map((name, i) => name + ' ' + stats[STAT.GAME0 + i]).join(', ') : '');
    } finally {
      await tmp.dispose();
    }
  }

  async testObserveParity() {
    const game = this.game;
    const dims = game.dims;
    const tmp = await this.tempBackend({ worlds: 4, islands: 2, evalFraction: 0.5 });
    try {
      const header = new Uint32Array(await tmp.readWords(tmp.buffers.world, 0, tmp.layout.worldStride * 4));
      const seed = header[0];
      const rng = new Rng(31337);
      const expected = new Float32Array(dims.nIn);
      let compared = 0;
      for (const dense of [false, true]) {
        const env = game.createEnv(seed, game.randomCfg(seed, tmp.opts), false, new Int32Array(NSTATS));
        if (dense && game.densify) game.densify(env, new Rng(5));
        const actions = new Int32Array(game.learners);
        for (let k = 0; k < 30; k++) env.step(actions, this.policyOutputs(rng, game.learners, dims.nOut), 1000 + k);
        this.device.queue.writeBuffer(tmp.buffers.world, WORLD_HDR_WORDS * 4, game.packEnv(env));
        for (const tick of [7, 2500, 3599, 3600 + 2450]) {
          tmp.writeUniforms({ ticks: 0, tick0: tick });
          const encoder = tmp.device.createCommandEncoder();
          tmp.dispatch(encoder, 'observe_test', 1);
          await tmp.submit(encoder);
          const gpu = new Float32Array(await tmp.readWords(tmp.buffers.scratch, 0, game.learners * dims.nIn * 4));
          for (let a = 0; a < game.learners; a++) {
            env.observe(a, expected, tick);
            for (let i = 0; i < dims.nIn; i++) {
              if (gpu[a * dims.nIn + i] !== expected[i]) throw new Error((dense ? 'dense' : 'natural') + ' tick ' + tick + ' learner ' + a + ' input ' + i + ' (' + (game.inputNames[i] || i) + '): gpu ' + gpu[a * dims.nIn + i] + ' js ' + expected[i]);
            }
            compared++;
          }
        }
      }
      return compared + ' learner observations (' + dims.nIn + ' inputs each, day and night ticks) identical between GPU and JS';
    } finally {
      await tmp.dispose();
    }
  }

  async testInvariants() {
    const game = this.game;
    const dims = game.dims;
    const tmp = await this.tempBackend({ worlds: 12, islands: 2, evalFraction: 0.34, ticksPerDispatch: 40 });
    try {
      for (let i = 0; i < 4; i++) await tmp.step();
      const L = tmp.layout;
      const maxE = tmp.opts.maxEdges;
      const brain = new Float32Array(await tmp.readWords(tmp.buffers.brain, 0, L.brainCount * L.brainStride * 4));
      const pk = new Uint32Array(await tmp.readWords(tmp.buffers.edgePk, 0, maxE * L.brainCount * 4));
      const w = new Float32Array(await tmp.readWords(tmp.buffers.edgeW, 0, maxE * L.brainCount * 4));
      for (let b = 0; b < L.brainCount; b++) {
        const bb = b * L.brainStride;
        const n = brain[bb + L.brainOffState + BRAIN_STATE.EDGES];
        if (!(n >= 30 && n <= maxE)) throw new Error('brain ' + b + ' edge count ' + n);
        for (let p = 0; p < 20; p++) {
          const v = brain[bb + p];
          if (!Number.isFinite(v) || v < PARAM_LO[p] - 1e-4 || v > PARAM_HI[p] + 1e-4) throw new Error('brain ' + b + ' param ' + p + ' = ' + v);
        }
        const wmax = brain[bb + 6];
        for (let e = 0; e < n; e++) {
          const packed = pk[e * L.brainCount + b];
          const src = packed & NODE_MASK;
          const dst = (packed >>> NODE_BITS) & NODE_MASK;
          const weight = w[e * L.brainCount + b];
          if (src >= dims.nNodes || dst < dims.nIn || dst >= dims.nNodes) throw new Error('brain ' + b + ' edge ' + e + ' endpoints ' + src + '->' + dst);
          if (!Number.isFinite(weight) || Math.abs(weight) > wmax + 1e-3) throw new Error('brain ' + b + ' edge ' + e + ' weight ' + weight + ' wmax ' + wmax);
        }
        for (let j = 0; j < dims.nNodes; j++) if (!Number.isFinite(brain[bb + L.brainOffAct + j])) throw new Error('brain ' + b + ' activation NaN');
      }
      for (let wi = 0; wi < tmp.opts.worlds; wi++) {
        const words = new Uint32Array(await tmp.readWords(tmp.buffers.world, (wi * L.worldStride + WORLD_HDR_WORDS) * 4, game.worldWords * 4));
        const problems = game.validateWords(words);
        if (problems.length) throw new Error('world ' + wi + ': ' + problems.slice(0, 3).join('; '));
      }
      const stats = await tmp.readStats();
      const archives = await tmp.readArchives();
      for (const archive of archives) {
        for (let i = 0; i < archive.count; i++) {
          if (!Number.isFinite(archive.genomes[i].fit)) throw new Error('archive fitness not finite');
          if (i > 0 && archive.genomes[i].fit > archive.genomes[i - 1].fit) throw new Error('archive not sorted');
        }
      }
      return L.brainCount + ' brains, ' + tmp.opts.worlds + ' worlds valid after ' + tmp.tick + ' ticks; archive holds ' + stats.archive.count + ' genomes, best fitness ' + stats.archive.best.toFixed(2);
    } finally {
      await tmp.dispose();
    }
  }

  async testStatisticalEquivalence() {
    const options = { worlds: 32, islands: 2, evalFraction: 0.125, randomize: false, ticksPerDispatch: 50, penaltyEnd: 0, penaltyStart: 0, difficultyStart: 100, difficultyEnd: 100, selectEvery: 1000000, evalEvery: 0, migrateEvery: 0 };
    const gpu = await this.tempBackend(options);
    const cpu = await CpuBackend.create(Object.assign({}, options, { game: this.game, maxEdges: this.opts.maxEdges, seed: 4242, migrateCount: 4 }));
    try {
      for (let i = 0; i < 6; i++) {
        await gpu.step();
        await cpu.step();
      }
      const g = await gpu.readStats();
      const c = await cpu.readStats();
      const mean = (arr, n) => arr.slice(0, n).reduce((s, v) => s + v, 0) / n;
      const std = (arr, n) => { const m = mean(arr, n); return Math.sqrt(arr.slice(0, n).reduce((s, v) => s + (v - m) * (v - m), 0) / (n - 1)); };
      const n = gpu.evalStart;
      const gm = mean(g.worldScores, n), cm = mean(c.worldScores, n);
      const se = Math.sqrt(std(g.worldScores, n) ** 2 / n + std(c.worldScores, n) ** 2 / n);
      const z = se > 0 ? Math.abs(gm - cm) / se : 0;
      const baseDiff = Math.abs(g.train.baseRate - c.train.baseRate) / Math.max(1e-9, Math.abs(c.train.baseRate));
      const detail = 'learner reward/tick gpu ' + gm.toFixed(5) + ' cpu ' + cm.toFixed(5) + ' (z ' + z.toFixed(2) + '), scripted baseline gpu ' + g.train.baseRate.toFixed(5) + ' cpu ' + c.train.baseRate.toFixed(5) + ' (rel diff ' + baseDiff.toFixed(3) + ')';
      if (z > 4 || baseDiff > 0.15) throw new Error(detail);
      return detail;
    } finally {
      await gpu.dispose();
    }
  }

  async runTests() {
    const results = [];
    const record = async (name, fn) => {
      const start = performance.now();
      try {
        const detail = await fn();
        results.push({ name, ok: true, detail, ms: performance.now() - start });
      } catch (error) {
        results.push({ name, ok: false, detail: error.message, ms: performance.now() - start });
      }
    };
    await record('workgroup memory fits device', async () => this.layout.workgroupBytes + ' of ' + this.ctx.granted.maxComputeWorkgroupStorageSize + ' bytes');
    await record('world init identical to JS env', () => this.testInitParity());
    await record('brain forward/learn/structural parity', () => this.testBrainParity());
    await record('env lockstep parity, natural start', () => this.testEnvLockstep(false));
    if (this.game.densify) await record('env lockstep parity, dense interactions', () => this.testEnvLockstep(true));
    await record('observation parity, GPU vs JS', () => this.testObserveParity());
    await record('invariants after training passes', () => this.testInvariants());
    await record('statistical equivalence with CPU', () => this.testStatisticalEquivalence());
    return results;
  }

  async timeKernel(entry, groups, repeats) {
    const times = [];
    for (let i = 0; i < repeats; i++) {
      const encoder = this.device.createCommandEncoder();
      this.dispatch(encoder, entry, groups);
      const start = performance.now();
      await this.submit(encoder);
      times.push(performance.now() - start);
    }
    times.sort((a, b) => a - b);
    return times[Math.floor(times.length / 2)];
  }

  async benchmark(progress) {
    const rows = [];
    const sizes = [64, 256, 1024, 2048, 4096];
    const cap = worldCapacity(this.ctx, this.game, this.opts);
    for (const worlds of sizes) {
      if (worlds > cap) {
        rows.push({ worlds, skipped: 'exceeds buffer limits (cap ' + cap + ')' });
        continue;
      }
      let tmp;
      try {
        tmp = await this.tempBackend({ worlds, islands: Math.max(2, Math.min(32, Math.floor(worlds / 16))), ticksPerDispatch: 16, evalFraction: 0.0625 });
        for (let i = 0; i < 2; i++) await tmp.step();
        const timings = [];
        for (let i = 0; i < 5; i++) timings.push((await tmp.step()).ms);
        timings.sort((a, b) => a - b);
        const ms = timings[2];
        const simMs = await tmp.timeKernel('sim_step', worlds, 5);
        const selectMs = await tmp.timeKernel('select_archive', tmp.opts.islands, 5);
        const reduceMs = await tmp.timeKernel('reduce_stats', tmp.opts.islands, 5);
        const worldTicks = (worlds * 16) / (simMs / 1000);
        rows.push({ worlds, ticksPerDispatch: 16, stepMs: ms, simMs, selectMs, reduceMs, worldTicksPerSec: worldTicks, agentTicksPerSec: worldTicks * this.game.agents, realtimeFactor: worldTicks / (this.game.tickHz || 30) });
        if (progress) progress(rows.slice());
      } catch (error) {
        rows.push({ worlds, skipped: error.message });
      } finally {
        if (tmp) await tmp.dispose();
      }
    }
    return { rows, device: this.ctx.adapterInfo };
  }
}
