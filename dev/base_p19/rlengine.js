import { getDeviceContext } from './engine.js';
import { DEFAULT_OPTS, StatsAccumulator, NSTATS, STAT, REWARD_SCALE, validateGame, evalWorldCount, difficultyAt, genomeFromJSON, Brain, Rng, evoConfig, mix } from './core.js';
import { RL_DEFAULTS, RL_CURRICULUM_DEFAULTS, RlCurriculum, rlLayout, rlRecordLayout, rlInitTheta, rlThetaToGenome, rlGenomeToTheta, rlHiddenActivations, rlPolicyStep, rlPolicyMean, rlValueOf, rlLogProb, rlGae, rlLossAndGrad, rlAdamStep, rlGradientCheck, rlNormMerge } from './rl.js';
import { buildRlShader, rlShaderLayout, rlFeatures, RL_ENTRIES, RL_UNIFORM_WORDS, RL_CTL, RL_LANE, RL_ROLE, RL_STATS } from './rlshader.js';

const RL_STATS_FLUSH_ITERATIONS = 8;
const RL_STORAGE_BINDINGS = 8;
const RL_TRACKED_KEYS = ['hidden', 'rolloutTicks', 'gamma', 'lambda', 'clip', 'lr', 'lrEnd', 'lrDecayIterations', 'entropy', 'valueCoef', 'epochs', 'minibatches', 'maxGrad', 'sigmaInit', 'sigmaMin', 'sigmaMax', 'weightMax', 'rewardScale', 'adaptScale', 'lifeCapScale', 'gradGroups', 'tileEntries', 'league', 'leagueFraction', 'leagueSlotFraction', 'poolSize', 'snapshotEvery', 'leaguePeriod', 'latestBias', 'leagueEvalFraction', 'referenceAge', 'curriculum', 'curriculumMetric', 'curriculumDifficultyStart', 'curriculumDifficultyStep', 'curriculumLifeUp', 'curriculumLifeDown', 'curriculumPatience', 'curriculumSmoothing', 'curriculumSelfPlay', 'curriculumSelfPlayStart', 'curriculumSelfPlayEnd', 'curriculumRatioLow', 'curriculumRatioHigh', 'bias', 'recurrent', 'obsNorm', 'recurrentInit', 'leakInit', 'leakTauMax', 'learnLeak', 'rewardClip', 'valueClip', 'obsNormFloor', 'obsNormCap', 'channelCaps'];

export function rlSlotCount(game) {
  return game.maxLearners || game.learners;
}

export function rlWorldCapacity(ctx, game, opts) {
  const limit = Math.min(ctx.granted.maxStorageBufferBindingSize, ctx.granted.maxBufferSize);
  const { nIn, nOut } = game.dims;
  const learners = rlSlotCount(game);
  const features = rlFeatures(opts);
  const layout = rlLayout(nIn, nOut, opts.hidden, features);
  const partFloats = Math.max(1, Math.floor(64 / learners)) * (nOut + 1);
  const obsPerLearner = (opts.rolloutTicks + 1) * nIn + (features.recurrent ? opts.hidden : 0);
  const perTrainWorld = [learners * obsPerLearner * 4, learners * opts.rolloutTicks * layout.recStride * 4];
  const perWorld = [(24 + game.worldWords) * 4, learners * (RL_LANE.HEADER + (features.recurrent ? 2 : 1) * opts.hidden + nIn + partFloats) * 4];
  const trainCap = Math.min(...perTrainWorld.map((bytes) => limit / bytes));
  const allCap = Math.min(...perWorld.map((bytes) => limit / bytes));
  const trainShare = 1 - opts.evalFraction - (opts.league ? 2 * opts.leagueEvalFraction : 0);
  return Math.floor(Math.min(allCap, trainCap / Math.max(0.05, trainShare)));
}

export class RlBackend {
  static async probe() {
    const ctx = await getDeviceContext();
    return { available: true, adapter: ctx.adapterInfo, limits: ctx.granted, features: ctx.features };
  }

  static async create(opts) {
    const ctx = opts.deviceCtx || (await getDeviceContext());
    const merged = Object.assign({}, DEFAULT_OPTS, RL_DEFAULTS, RL_CURRICULUM_DEFAULTS, { islands: 1 }, opts);
    merged.islands = 1;
    validateGame(merged.game);
    const dims = merged.game.dims;
    if (dims.nIn + dims.nOut + merged.hidden > 256) throw new Error('hidden size ' + merged.hidden + ' exceeds the 256 node genome limit for ' + merged.game.id);
    const warnings = [];
    const cap = rlWorldCapacity(ctx, merged.game, merged);
    if (merged.worlds > cap) {
      warnings.push('worlds reduced from ' + merged.worlds + ' to ' + cap + ' by buffer limits');
      merged.worlds = Math.max(4, cap);
    }
    for (let attempt = 0; attempt < 4; attempt++) {
      const backend = new RlBackend(ctx, merged);
      backend.info.warnings = warnings.slice();
      try {
        await backend.init();
        return backend;
      } catch (error) {
        backend.destroyBuffers();
        if (!/out of memory|OOM/i.test(error.message) || merged.worlds <= 8) throw error;
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
    this.slots = rlSlotCount(this.game);
    this.evalCount = evalWorldCount(opts);
    this.leagueEvalCount = opts.league ? Math.max(2, Math.round(opts.worlds * opts.leagueEvalFraction)) : 0;
    this.evalStart = opts.worlds - this.evalCount - 2 * this.leagueEvalCount;
    if (this.evalStart < 1) throw new Error('not enough worlds for the eval and league groups');
    this.evalBStart = this.evalStart + this.evalCount;
    this.evalCStart = this.evalBStart + this.leagueEvalCount;
    this.poolSize = opts.league ? Math.max(1, Math.round(opts.poolSize)) : 0;
    this.layout = rlShaderLayout(this.game, this.shaderConfig());
    this.params = this.layout.params;
    this.stats = new StatsAccumulator();
    this.leagueStats = { headToHead: new StatsAccumulator(), selfPlay: new StatsAccumulator() };
    this.roleWindow = new Float64Array(RL_STATS.ROLE_WORDS);
    this.pool = { valid: 0, events: 0, iterations: new Array(this.poolSize).fill(-1) };
    this.reference = { slot: 0, iteration: -1 };
    this.controller = opts.curriculum ? new RlCurriculum(opts, this.game.maxAge) : null;
    this.curriculum = { spOn: Boolean(this.controller && opts.curriculumSelfPlay), selfPlay: this.controller ? this.controller.selfPlay : 0 };
    this.tick = 0;
    this.passes = 0;
    this.iterationsSinceFlush = 0;
    this.lastStepMs = 0;
    this.worldScores = new Float32Array(opts.worlds);
    this.diagnostics = { gradNorm: 0, policyLoss: 0, valueLoss: 0, entropy: 0, clipFraction: 0, kl: 0, returnStd: 0, returnMean: 0, advStd: 0, rewardScale: opts.rewardScale, iteration: 0 };
    this.bestEval = -Infinity;
    this.bestTheta = null;
    this.lastWindowEval = 0;
    this.evalMark = { ticks: 0, reward: 0 };
    this.buffers = {};
    this.info = { kind: 'gpu', trainer: 'ppo', adapter: ctx.adapterInfo, limits: ctx.granted, features: ctx.features, worlds: opts.worlds, islands: 1, maxEdges: this.layout.denseEdges, game: this.game.id, hidden: opts.hidden, features: this.layout.features, workgroupBytes: this.layout.workgroupBytes, warnings: [] };
  }

  shaderConfig() {
    const o = this.opts;
    return { worlds: o.worlds, evalStart: this.evalStart, evalBStart: this.evalBStart, evalCStart: this.evalCStart, poolSize: this.poolSize, hidden: o.hidden, rolloutTicks: o.rolloutTicks, gradGroups: o.gradGroups, tileEntries: o.tileEntries, lifeCapScale: o.lifeCapScale, bias: o.bias, recurrent: o.recurrent, obsNorm: o.obsNorm, learnLeak: o.learnLeak, rewardClip: o.rewardClip, valueClip: o.valueClip, obsNormFloor: o.obsNormFloor, obsNormCap: o.obsNormCap, channelCaps: o.channelCaps };
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
    const limits = this.ctx.granted;
    if (L.workgroupBytes > limits.maxComputeWorkgroupStorageSize) throw new Error('ppo kernels need ' + L.workgroupBytes + ' bytes of workgroup memory, device allows ' + limits.maxComputeWorkgroupStorageSize);
    if (limits.maxStorageBuffersPerShaderStage < RL_STORAGE_BINDINGS) throw new Error('device allows only ' + limits.maxStorageBuffersPerShaderStage + ' storage buffers per stage, ' + RL_STORAGE_BINDINGS + ' required');
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    await this.withErrorScopes('allocation', async () => {
      this.makeBuffer('theta', L.thetaFloats * 4, S);
      this.makeBuffer('world', this.opts.worlds * L.worldStride * 4, S);
      this.makeBuffer('obs', L.obsFloats * 4, S);
      this.makeBuffer('rec', L.recFloats * 4, S);
      this.makeBuffer('lane', L.laneFloats * 4, S);
      this.makeBuffer('partials', L.partialFloats * 4, S);
      this.makeBuffer('opt', L.optFloats * 4, S);
      this.makeBuffer('statsOut', L.statsFloats * 4, S);
      this.makeBuffer('uniforms', RL_UNIFORM_WORDS * 4, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    });
    const code = buildRlShader(this.game, this.shaderConfig());
    this.shaderCode = this.opts.shaderTransform ? this.opts.shaderTransform(code) : code;
    const module = this.device.createShaderModule({ label: 'ppo', code: this.shaderCode });
    const compilation = await module.getCompilationInfo();
    const errors = compilation.messages.filter((m) => m.type === 'error');
    if (errors.length > 0) {
      const lines = code.split('\n');
      throw new Error('WGSL compile failed:\n' + errors.slice(0, 8).map((m) => 'line ' + m.lineNum + ': ' + m.message + '\n    ' + (lines[m.lineNum - 1] || '')).join('\n'));
    }
    const storage = (binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } });
    const bindLayout = this.device.createBindGroupLayout({ entries: [0, 1, 2, 3, 4, 5, 6, 7].map(storage).concat([{ binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }]) });
    const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [bindLayout] });
    this.pipelines = {};
    await this.withErrorScopes('pipelines', async () => {
      await Promise.all(RL_ENTRIES.map(async (entry) => {
        this.pipelines[entry] = await this.device.createComputePipelineAsync({ label: entry, layout: pipelineLayout, compute: { module, entryPoint: entry } });
      }));
    });
    const b = this.buffers;
    this.bindGroup = this.device.createBindGroup({ layout: bindLayout, entries: [b.theta, b.world, b.obs, b.rec, b.lane, b.partials, b.opt, b.statsOut, b.uniforms].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    this.uniformData = new ArrayBuffer(RL_UNIFORM_WORDS * 4);
    this.uniformU32 = new Uint32Array(this.uniformData);
    this.uniformF32 = new Float32Array(this.uniformData);
    this.writeTheta(rlInitTheta(this.params, this.opts.seed * 2654435761, this.opts));
    this.resetOptimizer();
    this.writeUniforms({});
    const encoder = this.device.createCommandEncoder();
    this.dispatch(encoder, 'rl_init_world', this.opts.worlds);
    await this.submit(encoder);
  }

  writeTheta(theta) {
    this.device.queue.writeBuffer(this.buffers.theta, 0, theta);
  }

  resetOptimizer(scale) {
    const opt = new Float32Array(this.layout.optFloats);
    opt[2 * this.params.count + RL_CTL.SCALE] = scale === undefined ? this.opts.rewardScale : scale;
    if (this.params.obsNorm) {
      const normBase = 2 * this.params.count + RL_CTL.SIZE;
      for (let j = 0; j < this.params.nIn; j++) opt[normBase + this.params.nIn + j] = 1;
    }
    this.device.queue.writeBuffer(this.buffers.opt, 0, opt);
  }

  learningRate() {
    const o = this.opts;
    const progress = Math.min(1, this.passes / Math.max(1, o.lrDecayIterations));
    return o.lr + (o.lrEnd - o.lr) * progress;
  }

  writeUniforms(v) {
    const o = this.opts;
    const u = this.uniformU32;
    const f = this.uniformF32;
    u.fill(0);
    u[0] = this.opts.rolloutTicks;
    u[1] = v.tick0 !== undefined ? v.tick0 : this.tick;
    u[2] = v.flags || 0;
    u[3] = this.evalStart;
    u[4] = o.seed >>> 0;
    u[5] = o.randomize ? 1 : 0;
    u[6] = v.difficulty !== undefined ? v.difficulty : this.controller ? Math.round(this.controller.difficulty) : difficultyAt(o, this.tick);
    u[7] = Math.max(1, Math.round(o.minibatches));
    f[8] = o.gamma;
    f[9] = o.lambda;
    f[10] = o.clip;
    f[11] = o.entropy;
    f[12] = o.valueCoef;
    f[13] = v.lr !== undefined ? v.lr : this.learningRate();
    f[14] = o.maxGrad;
    f[15] = o.sigmaMin;
    f[16] = o.sigmaMax;
    f[17] = o.weightMax;
    f[18] = o.adaptScale;
    f[19] = o.leagueFraction;
    f[20] = o.leagueSlotFraction;
    f[21] = o.latestBias;
    u[22] = this.pool.valid;
    u[23] = Math.floor(this.passes / Math.max(1, Math.round(o.leaguePeriod)));
    u[24] = this.reference.slot;
    u[25] = o.league ? 1 : 0;
    u[26] = this.curriculum.spOn ? 1 : 0;
    f[27] = this.curriculum.selfPlay;
    this.device.queue.writeBuffer(this.buffers.uniforms, 0, this.uniformData);
  }

  dispatch(encoder, entry, groups) {
    const pass = encoder.beginComputePass({ label: entry });
    pass.setPipeline(this.pipelines[entry]);
    pass.setBindGroup(0, this.bindGroup);
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

  async readFloats(name, floatOffset, floatCount) {
    return new Float32Array(await this.readWords(this.buffers[name], floatOffset * 4, floatCount * 4));
  }

  recordSnapshot(encoder) {
    const o = this.opts;
    if (!o.league || this.passes === 0 || this.passes % Math.max(1, Math.round(o.snapshotEvery)) !== 0) return;
    const pool = this.pool;
    const bytes = this.params.count * 4;
    const slotBytes = (slot) => (1 + slot) * bytes;
    pool.events++;
    if (pool.valid === 0) {
      pool.valid = 1;
    } else {
      let target = -1;
      if (pool.valid < this.poolSize) target = pool.valid++;
      else if (this.poolSize > 1 && mix(o.seed, pool.events, 0, 55) / 4294967296 < (this.poolSize - 1) / pool.events) target = 1 + (mix(o.seed, pool.events, 1, 55) % (this.poolSize - 1));
      if (target === this.reference.slot && this.reference.iteration >= 0) target = -1;
      if (target >= 0) {
        encoder.copyBufferToBuffer(this.buffers.theta, slotBytes(0), this.buffers.theta, slotBytes(target), bytes);
        pool.iterations[target] = pool.iterations[0];
      }
    }
    encoder.copyBufferToBuffer(this.buffers.theta, 0, this.buffers.theta, slotBytes(0), bytes);
    pool.iterations[0] = this.passes;
  }

  chooseReference() {
    const pool = this.pool;
    if (pool.valid === 0) return;
    const wanted = this.passes - this.opts.referenceAge;
    let best = pool.valid > 1 ? 1 : 0;
    for (let s = pool.valid > 1 ? 1 : 0; s < pool.valid; s++) if (Math.abs(pool.iterations[s] - wanted) < Math.abs(pool.iterations[best] - wanted)) best = s;
    this.reference = { slot: best, iteration: pool.iterations[best] };
  }

  encodeRollout(encoder) {
    this.dispatch(encoder, 'rl_rollout', this.opts.worlds);
  }

  encodeAdvantages(encoder) {
    this.dispatch(encoder, 'rl_gae', Math.max(1, Math.ceil(this.layout.trainLearners / 64)));
    this.dispatch(encoder, 'rl_advstats', 1);
  }

  encodeGradient(encoder) {
    this.dispatch(encoder, this.params.recurrent ? 'rl_grad_seq' : 'rl_grad', this.layout.gradGroups);
    this.dispatch(encoder, 'rl_reduce_grad', Math.ceil(this.params.partialStride / 256));
  }

  encodeUpdates(encoder) {
    const steps = Math.max(0, Math.round(this.opts.epochs)) * Math.max(1, Math.round(this.opts.minibatches));
    for (let s = 0; s < steps; s++) {
      this.encodeGradient(encoder);
      this.dispatch(encoder, 'rl_adam', 1);
    }
    if (steps > 0) this.dispatch(encoder, 'rl_rescale', 1);
    if (this.params.obsNorm) {
      this.dispatch(encoder, 'rl_normstats', 64);
      this.dispatch(encoder, 'rl_normapply', 1);
    }
  }

  async iterate(options) {
    if (this.ctx.lost) throw new Error('GPU device lost: ' + this.ctx.lost.message);
    const o = this.opts;
    const ticks = o.rolloutTicks;
    const evalPeriod = this.game.maxAge * o.evalEvery;
    const evalReset = o.evalEvery > 0 && Math.floor((this.tick + ticks) / evalPeriod) > Math.floor(this.tick / evalPeriod);
    const start = performance.now();
    const rolloutEncoder = this.device.createCommandEncoder();
    this.recordSnapshot(rolloutEncoder);
    if (evalReset || this.reference.iteration < 0) this.chooseReference();
    this.writeUniforms({ flags: evalReset ? 1 : 0 });
    this.encodeRollout(rolloutEncoder);
    this.encodeAdvantages(rolloutEncoder);
    this.device.queue.submit([rolloutEncoder.finish()]);
    if (!options || options.update !== false) {
      const updateEncoder = this.device.createCommandEncoder();
      this.encodeUpdates(updateEncoder);
      this.device.queue.submit([updateEncoder.finish()]);
    }
    await this.device.queue.onSubmittedWorkDone();
    this.lastStepMs = performance.now() - start;
    this.tick += ticks;
    this.passes++;
    this.iterationsSinceFlush++;
    return { ticks, ms: this.lastStepMs };
  }

  async step() {
    const result = await this.iterate();
    if (this.iterationsSinceFlush >= RL_STATS_FLUSH_ITERATIONS) await this.flushStats();
    return result;
  }

  async readControl() {
    const ctl = await this.readFloats('opt', 2 * this.params.count, RL_CTL.SIZE);
    const steps = Math.max(1, ctl[RL_CTL.STEPS]);
    this.diagnostics = {
      gradNorm: ctl[RL_CTL.GRAD_NORM] / steps,
      policyLoss: ctl[RL_CTL.POLICY] / steps,
      valueLoss: ctl[RL_CTL.VALUE] / steps,
      entropy: ctl[RL_CTL.ENTROPY] / steps,
      clipFraction: ctl[RL_CTL.CLIP] / steps,
      kl: ctl[RL_CTL.KL] / steps,
      returnStd: ctl[RL_CTL.RET_STD],
      returnMean: ctl[RL_CTL.RET_MEAN],
      advStd: ctl[RL_CTL.ADV_STD],
      rewardScale: ctl[RL_CTL.SCALE],
      iteration: ctl[RL_CTL.ITER],
      updates: ctl[RL_CTL.STEP]
    };
    return ctl;
  }

  async flushStats() {
    const o = this.opts;
    const encoder = this.device.createCommandEncoder();
    this.dispatch(encoder, 'rl_reduce_stats', 1);
    await this.submit(encoder);
    const rows = new Int32Array(await this.readWords(this.buffers.statsOut, 0, this.layout.statsFloats * 4));
    const group = (index) => Float64Array.from(rows.subarray(index * NSTATS, (index + 1) * NSTATS));
    for (let w = 0; w < o.worlds; w++) this.worldScores[w] = rows[RL_STATS.WORLD_OFFSET + w * 2] / REWARD_SCALE / Math.max(1, rows[RL_STATS.WORLD_OFFSET + w * 2 + 1]);
    this.stats.add('train', group(0), this.evalStart * this.slots);
    this.stats.add('eval', group(1), this.evalCount * this.slots);
    if (this.leagueEvalCount > 0) {
      this.leagueStats.headToHead.add('train', group(2), this.leagueEvalCount * this.slots);
      this.leagueStats.selfPlay.add('train', group(3), this.leagueEvalCount * this.slots);
      for (let i = 0; i < RL_STATS.ROLE_WORDS; i++) this.roleWindow[i] += rows[RL_STATS.ROLE_OFFSET + i];
    }
    await this.readControl();
    this.iterationsSinceFlush = 0;
  }

  setParams(partial) {
    Object.assign(this.opts, partial);
    if (this.controller) this.controller.configure(partial);
    this.curriculum.spOn = Boolean(this.controller && this.opts.curriculumSelfPlay);
  }

  async reinitWorlds() {
    this.writeUniforms({});
    const encoder = this.device.createCommandEncoder();
    this.dispatch(encoder, 'rl_reinit_world', this.evalStart);
    await this.submit(encoder);
  }

  async applyCurriculum(train, evalSummary) {
    const step = this.controller.update({ trainLife: train.meanLife, evalLife: evalSummary.meanLife, evalRate: evalSummary.rewRate, baseRate: evalSummary.baseRate });
    this.curriculum.selfPlay = step.selfPlay;
    this.curriculum.spOn = Boolean(this.opts.curriculumSelfPlay);
    if (step.selfPlayChanged && this.curriculum.spOn) await this.reinitWorlds();
    return Object.assign(this.controller.state(), { note: step.note });
  }

  async trackBestEval() {
    const total = this.stats.total.eval;
    const windowTicks = total[STAT.TICKS_EVO] - this.evalMark.ticks;
    if (windowTicks < this.game.maxAge * this.evalCount * this.slots) return;
    const rate = (total[STAT.REW_EVO] - this.evalMark.reward) / REWARD_SCALE / windowTicks;
    this.evalMark = { ticks: total[STAT.TICKS_EVO], reward: total[STAT.REW_EVO] };
    this.lastWindowEval = rate;
    if (rate > this.bestEval) {
      this.bestEval = rate;
      this.bestTheta = await this.readFloats('theta', 0, this.params.count);
    }
  }

  async readStats() {
    await this.flushStats();
    const train = this.stats.summary('train', this.game);
    const evalSummary = this.stats.summary('eval', this.game);
    this.stats.clearWindow();
    await this.trackBestEval();
    const curriculum = this.controller ? await this.applyCurriculum(train, evalSummary) : null;
    const d = this.diagnostics;
    const archive = { best: Number.isFinite(this.bestEval) ? this.bestEval : 0, mean: d.returnMean / Math.max(1e-6, d.rewardScale), count: this.passes };
    return { tick: this.tick, passes: this.passes, train, eval: evalSummary, archive, worldScores: this.worldScores, stepMs: this.lastStepMs, ticksPerDispatch: this.opts.rolloutTicks, ppo: Object.assign({ windowEval: this.lastWindowEval, bestEval: this.bestEval }, d), league: this.leagueSummary(), curriculum };
  }

  leagueSummary() {
    if (!this.opts.league) return null;
    const r = this.roleWindow;
    const latestRate = r[0] / REWARD_SCALE / Math.max(1, r[1]);
    const olderRate = r[2] / REWARD_SCALE / Math.max(1, r[3]);
    const pool = this.pool;
    const summary = {
      pool: { valid: pool.valid, iterations: pool.iterations.slice(0, pool.valid), events: pool.events },
      headToHead: { latestRate, olderRate, ratio: olderRate !== 0 ? latestRate / olderRate : NaN, diff: latestRate - olderRate, latestTicks: r[1], olderTicks: r[3], referenceIteration: this.reference.iteration, referenceAge: this.reference.iteration < 0 ? 0 : this.passes - this.reference.iteration },
      mixed: this.leagueStats.headToHead.summary('train', this.game),
      selfPlay: this.leagueStats.selfPlay.summary('train', this.game)
    };
    this.roleWindow.fill(0);
    this.leagueStats.headToHead.clearWindow();
    this.leagueStats.selfPlay.clearWindow();
    return summary;
  }

  async snapshot(worldIndex) {
    const L = this.layout;
    const all = new Uint32Array(await this.readWords(this.buffers.world, worldIndex * L.worldStride * 4, L.worldStride * 4));
    return this.game.snapshotFromWords(all.slice(24, 24 + this.game.worldWords), all[0]);
  }

  async exportChampions(n) {
    const current = await this.readFloats('theta', 0, this.params.count);
    const meta = { fitness: Number.isFinite(this.bestEval) ? this.bestEval : 0, iterations: this.passes, ticks: this.tick };
    const list = [rlThetaToGenome(current, this.params, this.game, this.opts, meta)];
    if (this.bestTheta && n > 1) list.push(rlThetaToGenome(this.bestTheta, this.params, this.game, this.opts, Object.assign({ best: true }, meta)));
    return list.slice(0, Math.max(1, n));
  }

  async importGenomes(jsonArray) {
    const theta = rlGenomeToTheta(jsonArray[0], this.params, this.opts, this.opts.seed);
    this.writeTheta(theta);
    this.resetOptimizer();
    this.bestTheta = null;
    this.bestEval = -Infinity;
    this.evalMark = { ticks: this.stats.total.eval[STAT.TICKS_EVO], reward: this.stats.total.eval[STAT.REW_EVO] };
    await this.device.queue.onSubmittedWorkDone();
  }

  async checkpoint() {
    const o = Object.assign({}, this.opts);
    delete o.game;
    delete o.deviceCtx;
    const count = this.params.count;
    const theta = await this.readFloats('theta', 0, this.layout.thetaFloats);
    const opt = await this.readFloats('opt', 0, this.layout.optFloats);
    const pool = { valid: this.pool.valid, events: this.pool.events, iterations: this.pool.iterations.slice(), floats: Array.from(theta.subarray(count)), reference: this.reference };
    return { kind: 'gpu', trainer: 'ppo', game: this.game.id, tick: this.tick, passes: this.passes, opts: o, theta: Array.from(theta.subarray(0, count)), pool, opt: Array.from(opt), bestEval: this.bestEval, bestTheta: this.bestTheta ? Array.from(this.bestTheta) : null, curriculum: this.controller ? this.controller.state() : null };
  }

  async restore(obj) {
    if (obj.trainer !== 'ppo' || obj.theta.length !== this.params.count) throw new Error('checkpoint does not match this PPO configuration');
    this.tick = obj.tick;
    this.passes = obj.passes;
    this.writeTheta(new Float32Array(obj.theta));
    if (obj.pool && obj.pool.floats.length === this.layout.thetaFloats - this.params.count) {
      this.device.queue.writeBuffer(this.buffers.theta, this.params.count * 4, new Float32Array(obj.pool.floats));
      this.pool = { valid: obj.pool.valid, events: obj.pool.events, iterations: obj.pool.iterations.slice() };
      this.reference = obj.pool.reference;
    }
    this.device.queue.writeBuffer(this.buffers.opt, 0, new Float32Array(obj.opt));
    if (this.controller && obj.curriculum) {
      this.controller.restore(obj.curriculum);
      this.curriculum.selfPlay = this.controller.selfPlay;
    }
    this.bestEval = obj.bestEval === null || obj.bestEval === undefined ? -Infinity : obj.bestEval;
    this.bestTheta = obj.bestTheta ? new Float32Array(obj.bestTheta) : null;
    await this.device.queue.onSubmittedWorkDone();
  }

  async dispose() {
    await this.device.queue.onSubmittedWorkDone();
    this.destroyBuffers();
  }

  async tempBackend(overrides) {
    const base = {};
    for (const key of RL_TRACKED_KEYS) base[key] = this.opts[key];
    return RlBackend.create(Object.assign(base, { game: this.game, deviceCtx: this.ctx, randomize: true, seed: 4242, difficultyStart: 100, difficultyEnd: 100, evalEvery: 4, evalFraction: 0.34, league: 0 }, overrides));
  }

  async readRollout() {
    const L = this.layout;
    const theta = await this.readFloats('theta', 0, L.thetaFloats);
    const obs = await this.readFloats('obs', 0, L.obsFloats);
    const rec = await this.readFloats('rec', 0, L.recFloats);
    const lane = await this.readFloats('lane', 0, L.laneFloats);
    const ctl = await this.readFloats('opt', 2 * this.params.count, RL_CTL.SIZE);
    return { theta, obs, rec, lane, ctl };
  }

  roleOf(data, b) {
    return data.lane[b * this.layout.laneStride + RL_LANE.ROLE];
  }

  thetaOf(data, b) {
    const base = Math.round(data.lane[b * this.layout.laneStride + RL_LANE.BASE]);
    return data.theta.subarray(base * this.params.count, (base + 1) * this.params.count);
  }

  obsBase(b) {
    return b * this.layout.obsLearnerStride;
  }

  obsRow(data, b, j) {
    const base = this.obsBase(b) + j * this.params.nIn;
    return data.obs.subarray(base, base + this.params.nIn);
  }

  hiddenInit(data, b) {
    const base = this.obsBase(b) + (this.opts.rolloutTicks + 1) * this.params.nIn;
    return data.obs.subarray(base, base + this.params.hidden);
  }

  replayLearner(theta, data, b, visit) {
    const P = this.params;
    const R = rlRecordLayout(P.nOut);
    const steps = this.opts.rolloutTicks;
    const h = new Float64Array(P.hidden);
    const hEff = new Float64Array(P.hidden);
    const hNew = new Float64Array(P.hidden);
    const mu = new Float64Array(P.nOut);
    if (P.recurrent) h.set(this.hiddenInit(data, b));
    else rlHiddenActivations(theta, P, this.obsRow(data, b, 0), h);
    for (let k = 0; k < steps; k++) {
      const base = (b * steps + k) * R.stride;
      const valid = data.rec[base + R.prevValid] > 0.5;
      for (let i = 0; i < P.hidden; i++) hEff[i] = valid ? h[i] : 0;
      rlPolicyMean(theta, P, hEff, mu);
      if (P.recurrent) rlPolicyStep(theta, P, this.obsRow(data, b, k + 1), hEff, hNew);
      else rlHiddenActivations(theta, P, this.obsRow(data, b, k + 1), hNew);
      visit(k, base, valid, mu, hNew);
      h.set(hNew);
    }
    return h;
  }

  async testDeterminism() {
    const a = await this.tempBackend({ worlds: 6, hidden: 16, rolloutTicks: 12, gradGroups: 8 });
    const b = await this.tempBackend({ worlds: 6, hidden: 16, rolloutTicks: 12, gradGroups: 8 });
    try {
      for (let i = 0; i < 3; i++) {
        await a.iterate();
        await b.iterate();
      }
      const ra = await a.readRollout();
      const rb = await b.readRollout();
      for (const key of ['theta', 'obs', 'rec', 'lane']) {
        for (let i = 0; i < ra[key].length; i++) if (!Object.is(ra[key][i], rb[key][i])) throw new Error(key + ' differs at ' + i + ': ' + ra[key][i] + ' vs ' + rb[key][i]);
      }
      return 'two backends with the same seed produce identical theta, observations, actions, values and lane state after 3 full iterations (' + ra.rec.length + ' record floats)';
    } finally {
      await a.dispose();
      await b.dispose();
    }
  }

  trajectoryOf(data, L, b) {
    const R = rlRecordLayout(L.nOut);
    const steps = this.opts.rolloutTicks;
    const reward = new Float64Array(steps);
    const flag = new Float64Array(steps);
    const value = new Float64Array(steps);
    const truncValue = new Float64Array(steps);
    for (let t = 0; t < steps; t++) {
      const base = (b * steps + t) * R.stride;
      reward[t] = data.rec[base + R.reward];
      flag[t] = data.rec[base + R.flag];
      value[t] = data.rec[base + R.value];
      truncValue[t] = data.rec[base + R.truncValue];
    }
    return { steps, reward, flag, value, truncValue, boot: data.lane[b * this.layout.laneStride + RL_LANE.BOOT] };
  }

  async testForwardParity() {
    const t = await this.tempBackend({ worlds: 6, hidden: 20, rolloutTicks: 40, gradGroups: 8 });
    try {
      for (let i = 0; i < 3; i++) await t.iterate();
      await t.iterate({ update: false });
      const data = await t.readRollout();
      const P = t.params;
      const R = rlRecordLayout(P.nOut);
      const steps = t.opts.rolloutTicks;
      let worstLogp = 0;
      let worstValue = 0;
      let checked = 0;
      let resets = 0;
      for (let b = 0; b < t.layout.trainLearners; b++) {
        if (t.roleOf(data, b) === RL_ROLE.INACTIVE) continue;
        const theta = t.thetaOf(data, b);
        t.replayLearner(theta, data, b, (k, base, valid, mu, hNew) => {
          if (!valid) resets++;
          const action = data.rec.subarray(base, base + P.nOut);
          worstLogp = Math.max(worstLogp, Math.abs(rlLogProb(theta, P, action, mu) - data.rec[base + R.logProb]));
          worstValue = Math.max(worstValue, Math.abs(rlValueOf(theta, P, hNew) - data.rec[base + R.value]));
          checked++;
        });
      }
      const detail = checked + ' samples (' + resets + ' fresh-life ticks, ' + (P.recurrent ? 'recurrent with carried state' : 'feed-forward') + (P.bias ? ', bias' : '') + (P.obsNorm ? ', normalised inputs' : '') + '): max log-prob error ' + worstLogp.toExponential(2) + ', max value error ' + worstValue.toExponential(2);
      if (!(worstLogp < 2e-3) || !(worstValue < 1e-3)) throw new Error(detail);
      return detail;
    } finally {
      await t.dispose();
    }
  }

  async testRolloutLockstep() {
    const game = this.game;
    const steps = 40;
    const iterations = 3;
    const t = await this.tempBackend({ worlds: 6, hidden: 12, rolloutTicks: steps, gradGroups: 4, lifeCapScale: 0.02, obsNorm: 0 });
    try {
      const P = t.params;
      const R = rlRecordLayout(P.nOut);
      const slots = t.slots;
      const trainAge = Math.max(1, Math.round(game.maxAge * 0.02));
      const checked = Math.min(t.evalStart, 4);
      const replays = [];
      for (let w = 0; w < checked; w++) {
        const header = new Uint32Array(await t.readWords(t.buffers.world, w * t.layout.worldStride * 4, 4));
        const seed = header[0];
        const env = game.createEnv(seed, game.randomCfg(seed), false, new Int32Array(NSTATS));
        env.difficulty = 100;
        const life = new Float64Array(slots);
        for (let a = 0; a < slots; a++) life[a] = mix(seed, a, 0, 11) % trainAge;
        replays.push({ env, life });
      }
      const actions = new Int32Array(slots);
      const outputs = new Float32Array(slots * P.nOut);
      const obs = new Float32Array(P.nIn);
      let ticks = 0;
      let deaths = 0;
      let truncations = 0;
      let inactiveSlots = 0;
      for (let it = 0; it < iterations; it++) {
        await t.iterate({ update: false });
        const data = await t.readRollout();
        for (let w = 0; w < checked; w++) {
          const { env, life } = replays[w];
          const learner = (a) => w * slots + a;
          for (let k = 0; k < steps; k++) {
            const tick = it * steps + k;
            outputs.fill(0);
            for (let a = 0; a < slots; a++) {
              if (t.roleOf(data, learner(a)) === RL_ROLE.INACTIVE) continue;
              const sample = (learner(a) * steps + k) * R.stride;
              env.observe(a, obs, tick);
              for (let c = 0; c < P.nIn; c++) {
                const recorded = t.obsRow(data, learner(a), k + 1)[c];
                if (recorded !== obs[c]) throw new Error('world ' + w + ' tick ' + tick + ' learner ' + a + ' input ' + c + ': gpu ' + recorded + ' js ' + obs[c]);
              }
              for (let c = 0; c < P.nOut; c++) outputs[a * P.nOut + c] = Math.max(-1, Math.min(1, data.rec[sample + R.action + c]));
            }
            env.step(actions, outputs, tick);
            const ended = [];
            for (let a = 0; a < slots; a++) {
              if (t.roleOf(data, learner(a)) === RL_ROLE.INACTIVE) {
                if (k === 0 && it === 0) inactiveSlots++;
                continue;
              }
              const sample = (learner(a) * steps + k) * R.stride;
              life[a]++;
              const expectedFlag = env.dead[a] ? 1 : life[a] >= trainAge ? 2 : 0;
              if (data.rec[sample + R.reward] * REWARD_SCALE !== env.reward[a]) throw new Error('world ' + w + ' tick ' + tick + ' learner ' + a + ' reward: gpu ' + data.rec[sample + R.reward] * REWARD_SCALE + ' js ' + env.reward[a]);
              if (data.rec[sample + R.flag] !== expectedFlag) throw new Error('world ' + w + ' tick ' + tick + ' learner ' + a + ' flag: gpu ' + data.rec[sample + R.flag] + ' js ' + expectedFlag);
              if (expectedFlag) {
                ended.push(a);
                life[a] = 0;
                if (expectedFlag === 1) deaths++;
                else truncations++;
              }
            }
            for (const a of ended) env.respawn(a, tick);
            ticks++;
          }
        }
      }
      for (let w = 0; w < checked; w++) {
        const words = new Uint32Array(await t.readWords(t.buffers.world, (w * t.layout.worldStride + 24) * 4, game.worldWords * 4));
        const expected = game.packEnv(replays[w].env);
        for (let i = 0; i < expected.length; i++) if (words[i] !== expected[i]) throw new Error('world ' + w + ' state word ' + i + ': gpu ' + (words[i] | 0) + ' js ' + (expected[i] | 0));
      }
      return checked + ' worlds x ' + steps * iterations + ' rollout ticks x ' + slots + ' learner slots (' + inactiveSlots + ' scripted slots) replayed in the JS env from the recorded actions: observations, rewards and end-of-life flags identical (' + deaths + ' deaths, ' + truncations + ' truncations), final world states identical word for word';
    } finally {
      await t.dispose();
    }
  }

  async testAdvantageParity() {
    const t = await this.tempBackend({ worlds: 6, hidden: 20, rolloutTicks: 40, gradGroups: 8, rewardScale: 3, rewardClip: 0.5, valueClip: 2 });
    try {
      for (let i = 0; i < 3; i++) await t.iterate();
      await t.iterate({ update: false });
      const data = await t.readRollout();
      const R = rlRecordLayout(t.params.nOut);
      const steps = t.opts.rolloutTicks;
      const hp = { gamma: t.opts.gamma, lambda: t.opts.lambda, rewardScale: data.ctl[RL_CTL.SCALE], rewardClip: t.opts.rewardClip, valueClip: t.opts.valueClip };
      let clippedRewards = 0;
      let clippedReturns = 0;
      let worst = 0;
      let terminals = 0;
      let truncations = 0;
      for (let b = 0; b < t.layout.trainLearners; b++) {
        if (t.roleOf(data, b) !== RL_ROLE.LIVE) continue;
        const trajectory = t.trajectoryOf(data, t.params, b);
        const expected = rlGae(trajectory, hp);
        for (let k = 0; k < steps; k++) {
          const base = (b * steps + k) * R.stride;
          worst = Math.max(worst, Math.abs(expected.advantage[k] - data.rec[base + R.advantage]), Math.abs(expected.returns[k] - data.rec[base + R.returns]));
          if (trajectory.flag[k] === 1) terminals++;
          if (trajectory.flag[k] === 2) truncations++;
          if (Math.abs(trajectory.reward[k] * hp.rewardScale) > hp.rewardClip) clippedRewards++;
          if (Math.abs(data.rec[base + R.returns]) >= hp.valueClip - 1e-6) clippedReturns++;
        }
      }
      const detail = 'GAE(' + hp.gamma + ',' + hp.lambda + ', reward clip ' + hp.rewardClip + ' hit ' + clippedRewards + ' times, value target clip ' + hp.valueClip + ' hit ' + clippedReturns + ' times) over ' + t.layout.trainLearners * steps + ' samples (' + terminals + ' deaths, ' + truncations + ' truncations): max error ' + worst.toExponential(2);
      if (!(worst < 1e-4)) throw new Error(detail);
      return detail;
    } finally {
      await t.dispose();
    }
  }

  async gradientBatch(t, data) {
    const steps = t.opts.rolloutTicks;
    const P = t.params;
    const learners = t.layout.trainLearners;
    const selected = (b) => t.roleOf(data, b) === RL_ROLE.LIVE;
    const obs = new Float64Array(learners * (steps + 1) * P.nIn);
    const hInit = new Float64Array(learners * P.hidden);
    for (let b = 0; b < learners; b++) {
      for (let j = 0; j <= steps; j++) obs.set(t.obsRow(data, b, j), (b * (steps + 1) + j) * P.nIn);
      if (P.recurrent) hInit.set(t.hiddenInit(data, b), b * P.hidden);
    }
    return { steps, learners, obs, rec: data.rec, hInit: P.recurrent ? hInit : null, normMean: data.ctl[RL_CTL.ADV_MEAN], normInvStd: data.ctl[RL_CTL.ADV_INV_STD], selected, divisor: data.ctl[RL_CTL.ACTIVE] / Math.max(1, Math.round(t.opts.minibatches)) };
  }

  async testUpdateParity() {
    const t = await this.tempBackend({ worlds: 6, hidden: 20, rolloutTicks: 40, gradGroups: 8, minibatches: 1, epochs: 1, maxGrad: 1e9, lifeCapScale: 0.02 });
    try {
      for (let i = 0; i < 3; i++) await t.iterate({ update: false });
      const before = await t.readRollout();
      const encoder = t.device.createCommandEncoder();
      t.encodeGradient(encoder);
      await t.submit(encoder);
      const P = t.params;
      const reduced = await t.readFloats('partials', t.layout.gradGroups * P.partialStride, P.partialStride);
      const hp = Object.assign({}, t.opts, { rewardScale: before.ctl[RL_CTL.SCALE] });
      const batch = await t.gradientBatch(t, before);
      const reference = rlLossAndGrad(before.theta, P, batch, hp, true);
      let worst = 0;
      let peak = 0;
      for (let p = 0; p < P.learn; p++) {
        peak = Math.max(peak, Math.abs(reference.grad[p]));
        worst = Math.max(worst, Math.abs(reference.grad[p] - reduced[p]));
      }
      const gradDetail = 'analytic gradient of ' + P.learn + ' parameters over ' + batch.learners * batch.steps + ' samples: max abs error ' + worst.toExponential(2) + ' against peak ' + peak.toExponential(2);
      if (!(worst < 2e-3 * Math.max(peak, 1e-6))) throw new Error(gradDetail);
      const adamEncoder = t.device.createCommandEncoder();
      t.dispatch(adamEncoder, 'rl_adam', 1);
      await t.submit(adamEncoder);
      const after = await t.readFloats('theta', 0, P.count);
      const m = new Float64Array(P.count);
      const v = new Float64Array(P.count);
      const expected = Float64Array.from(before.theta);
      rlAdamStep(expected, Float64Array.from(reduced.subarray(0, P.count)), m, v, 1, Object.assign({}, hp, { lr: t.learningRate(), maxGrad: t.opts.maxGrad }), P);
      let updateWorst = 0;
      for (let p = 0; p < P.learn; p++) updateWorst = Math.max(updateWorst, Math.abs(expected[p] - after[p]));
      const detail = gradDetail + '; Adam step max theta error ' + updateWorst.toExponential(2) + ' (lr ' + t.learningRate() + ')';
      if (!(updateWorst < Math.max(2e-6, t.learningRate() * 0.02))) throw new Error(detail);
      return detail;
    } finally {
      await t.dispose();
    }
  }

  async testExportReplay() {
    const t = await this.tempBackend({ worlds: 6, hidden: 20, rolloutTicks: 64, gradGroups: 8 });
    try {
      for (let i = 0; i < 3; i++) await t.iterate();
      await t.iterate({ update: false });
      const data = await t.readRollout();
      const [json] = await t.exportChampions(1);
      const P = t.params;
      const dims = json.dims;
      const genome = genomeFromJSON(json, json.edges.length);
      const brain = new Brain(genome, json.edges.length, dims, evoConfig(null));
      const rng = new Rng(1);
      const R = rlRecordLayout(P.nOut);
      const steps = t.opts.rolloutTicks;
      const hid0 = P.nIn + P.nOut;
      const norm = json.inputNorm;
      const inputs = new Float32Array(P.nIn);
      const mu = new Float64Array(P.nOut);
      let worst = 0;
      let compared = 0;
      const rawRow = (b, j) => {
        const row = t.obsRow(data, b, j);
        for (let c = 0; c < P.nIn; c++) inputs[c] = norm ? row[c] * norm.std[c] + norm.mean[c] : row[c];
        return inputs;
      };
      const learners = Math.min(t.layout.trainLearners, 40);
      for (let b = 0; b < learners; b++) {
        brain.load(genome);
        if (P.recurrent) {
          const h0 = t.hiddenInit(data, b);
          for (let i = 0; i < P.hidden; i++) brain.act[hid0 + i] = h0[i];
        } else if (data.rec[(b * steps) * R.stride + R.prevValid] > 0.5) brain.step(rawRow(b, 0), rng);
        for (let k = 0; k < steps; k++) {
          const base = (b * steps + k) * R.stride;
          if (data.rec[base + R.prevValid] <= 0.5) brain.load(genome);
          brain.step(rawRow(b, k + 1), rng);
          for (let c = 0; c < P.nOut; c++) mu[c] = brain.act[P.nIn + c];
          const action = data.rec.subarray(base, base + P.nOut);
          worst = Math.max(worst, Math.abs(rlLogProb(data.theta, P, action, mu) - data.rec[base + R.logProb]));
          compared++;
        }
      }
      const detail = compared + ' policy ticks replayed through core.js Brain with the exported ' + json.edges.length + '-edge ' + json.format + ' genome (dims ' + dims.nIn + '/' + dims.nOut + '/' + dims.nNodes + (P.recurrent ? ', hidden-to-hidden edges and per-node leak' : '') + (norm ? ', input normalisation folded' : '') + '): max log-prob error ' + worst.toExponential(2);
      if (!(worst < 5e-3)) throw new Error(detail);
      return detail;
    } finally {
      await t.dispose();
    }
  }

  async testLeague() {
    const native = /fn\s+g_learner_slots\s*\(/.test(this.game.wgsl);
    const inactive = native ? 0 : 4;
    const shim = native ? this.game : Object.assign({}, this.game, { wgsl: this.game.wgsl + '\nfn g_learner_slots() -> u32 { return ' + (this.slots - inactive) + 'u; }' });
    const t = await this.tempBackend({ game: shim, worlds: 12, evalFraction: 0.17, leagueEvalFraction: 0.1, hidden: 16, rolloutTicks: 16, gradGroups: 8, league: 1, leagueFraction: 1, leagueSlotFraction: 0.5, poolSize: 3, snapshotEvery: 2, leaguePeriod: 1, referenceAge: 2, minibatches: 1, epochs: 1, maxGrad: 1e9 });
    try {
      for (let i = 0; i < 9; i++) await t.iterate();
      await t.iterate({ update: false });
      const data = await t.readRollout();
      const P = t.params;
      const R = rlRecordLayout(P.nOut);
      const steps = t.opts.rolloutTicks;
      const slots = t.slots;
      const counts = { train: [0, 0, 0], evalA: [0, 0, 0], evalB: [0, 0, 0], evalC: [0, 0, 0] };
      for (let w = 0; w < t.opts.worlds; w++) {
        const kind = w < t.evalStart ? 'train' : w < t.evalBStart ? 'evalA' : w < t.evalCStart ? 'evalB' : 'evalC';
        let active = 0;
        while (active < slots && data.lane[(w * slots + active) * t.layout.laneStride + RL_LANE.ROLE] !== RL_ROLE.INACTIVE) active++;
        if (!native && active !== slots - inactive) throw new Error('world ' + w + ' has ' + active + ' active slots, expected ' + (slots - inactive));
        for (let l = 0; l < slots; l++) {
          const role = data.lane[(w * slots + l) * t.layout.laneStride + RL_LANE.ROLE];
          counts[kind][role]++;
          if ((role === RL_ROLE.INACTIVE) !== (l >= active)) throw new Error('world ' + w + ' slot ' + l + ' role ' + role + ' breaks the active-slot prefix of ' + active);
          if (kind === 'evalB' && l < active && role !== (((l + w) & 1) === 1 ? RL_ROLE.SNAPSHOT : RL_ROLE.LIVE)) throw new Error('head-to-head world ' + w + ' slot ' + l + ' has role ' + role);
          if ((kind === 'evalA' || kind === 'evalC') && role === RL_ROLE.SNAPSHOT) throw new Error(kind + ' world ' + w + ' contains a snapshot learner');
        }
      }
      if (counts.train[RL_ROLE.SNAPSHOT] === 0 || counts.train[RL_ROLE.LIVE] === 0) throw new Error('training worlds need both live and snapshot learners: ' + JSON.stringify(counts.train));
      if (counts.evalB[RL_ROLE.SNAPSHOT] === 0) throw new Error('head-to-head worlds have no older-snapshot learners');
      const pool = t.pool;
      if (pool.valid !== 3) throw new Error('pool holds ' + pool.valid + ' snapshots, expected 3');
      const slice = (s) => data.theta.subarray((1 + s) * P.count, (2 + s) * P.count);
      let identical = 0;
      for (let s = 0; s < pool.valid; s++) {
        let same = true;
        for (let p = 0; p < P.count && same; p++) if (slice(s)[p] !== data.theta[p]) same = false;
        if (same) identical++;
      }
      if (identical > 0) throw new Error(identical + ' pool snapshots equal the live theta');
      let worstLogp = 0;
      let snapshotSamples = 0;
      for (let b = 0; b < t.layout.trainLearners; b++) {
        const role = t.roleOf(data, b);
        if (role === RL_ROLE.INACTIVE) continue;
        const theta = t.thetaOf(data, b);
        t.replayLearner(theta, data, b, (k, base, valid, mu) => {
          worstLogp = Math.max(worstLogp, Math.abs(rlLogProb(theta, P, data.rec.subarray(base, base + P.nOut), mu) - data.rec[base + R.logProb]));
          if (role === RL_ROLE.SNAPSHOT) snapshotSamples++;
        });
      }
      if (!(worstLogp < 2e-3)) throw new Error('snapshot-driven samples do not follow the snapshot policy: log-prob error ' + worstLogp);
      const encoder = t.device.createCommandEncoder();
      t.encodeGradient(encoder);
      await t.submit(encoder);
      const reduced = await t.readFloats('partials', t.layout.gradGroups * P.partialStride, P.partialStride);
      const batch = await t.gradientBatch(t, data);
      const reference = rlLossAndGrad(data.theta, P, batch, Object.assign({}, t.opts), true);
      let worst = 0;
      let peak = 0;
      for (let p = 0; p < P.learn; p++) {
        peak = Math.max(peak, Math.abs(reference.grad[p]));
        worst = Math.max(worst, Math.abs(reference.grad[p] - reduced[p]));
      }
      const expectedActive = counts.train[RL_ROLE.LIVE] * steps;
      if (Math.abs(data.ctl[RL_CTL.ACTIVE] - expectedActive) > 0.5) throw new Error('advantage statistics cover ' + data.ctl[RL_CTL.ACTIVE] + ' samples, live learners hold ' + expectedActive);
      if (!(worst < 2e-3 * Math.max(peak, 1e-6))) throw new Error('masked gradient error ' + worst + ' against peak ' + peak);
      await t.flushStats();
      const roles = t.roleWindow;
      if (!(roles[1] > 0 && roles[3] > 0)) throw new Error('head-to-head ticks missing: latest ' + roles[1] + ', older ' + roles[3]);
      return 'roles per world kind (inactive/live/snapshot) ' + JSON.stringify(counts) + '; ' + snapshotSamples + ' snapshot-driven samples follow their snapshot (log-prob error ' + worstLogp.toExponential(2) + '); pool of 3 distinct snapshots; masked gradient error ' + worst.toExponential(2) + ' against peak ' + peak.toExponential(2) + '; head-to-head ticks latest ' + roles[1] + ' older ' + roles[3];
    } finally {
      await t.dispose();
    }
  }

  async testHiddenCarry() {
    if (!this.params.recurrent) return 'not applicable: this configuration is feed-forward (hidden state is recomputed from the stored last observation)';
    const t = await this.tempBackend({ worlds: 6, hidden: 16, rolloutTicks: 24, gradGroups: 8 });
    try {
      for (let i = 0; i < 2; i++) await t.iterate();
      await t.iterate({ update: false });
      const first = await t.readRollout();
      await t.iterate({ update: false });
      const second = await t.readRollout();
      const P = t.params;
      const R = rlRecordLayout(P.nOut);
      const steps = t.opts.rolloutTicks;
      let worst = 0;
      let carried = 0;
      let zeroed = 0;
      for (let b = 0; b < t.layout.trainLearners; b++) {
        if (t.roleOf(first, b) === RL_ROLE.INACTIVE) continue;
        const last = t.replayLearner(t.thetaOf(first, b), first, b, () => {});
        const h2 = t.hiddenInit(second, b);
        const valid = second.rec[b * steps * R.stride + R.prevValid] > 0.5;
        if (valid) carried++;
        else zeroed++;
        for (let i = 0; i < P.hidden; i++) worst = Math.max(worst, Math.abs(h2[i] - (valid ? last[i] : 0)));
      }
      const detail = carried + ' carried and ' + zeroed + ' reset initial states: max difference between the stored initial state of the next rollout and the JS replay of the previous one ' + worst.toExponential(2);
      if (!(worst < 1e-3) || carried === 0) throw new Error(detail);
      return detail;
    } finally {
      await t.dispose();
    }
  }

  async testNormalization() {
    if (!this.params.obsNorm) return 'not applicable: input normalisation is off in this configuration';
    const t = await this.tempBackend({ worlds: 6, hidden: 16, rolloutTicks: 24, gradGroups: 8 });
    try {
      const P = t.params;
      const steps = t.opts.rolloutTicks;
      const state = { count: 0, mean: new Float64Array(P.nIn), variance: new Float64Array(P.nIn).fill(1) };
      let worstMean = 0;
      let worstIstd = 0;
      let worstVar = 0;
      for (let round = 0; round < 3; round++) {
        await t.iterate({ update: false });
        const data = await t.readRollout();
        const prev = { mean: Float64Array.from(data.theta.subarray(P.normMean, P.normMean + P.nIn)), istd: Float64Array.from(data.theta.subarray(P.normIstd, P.normIstd + P.nIn)) };
        const s1 = new Float64Array(P.nIn);
        const s2 = new Float64Array(P.nIn);
        for (let b = 0; b < t.layout.trainLearners; b++) {
          if (t.roleOf(data, b) !== RL_ROLE.LIVE) continue;
          for (let k = 1; k <= steps; k++) {
            const row = t.obsRow(data, b, k);
            for (let f = 0; f < P.nIn; f++) {
              s1[f] += row[f];
              s2[f] += row[f] * row[f];
            }
          }
        }
        const nb = data.ctl[RL_CTL.ACTIVE];
        const merged = rlNormMerge(state, s1, s2, nb, prev, { cap: t.opts.obsNormCap, floor: t.opts.obsNormFloor });
        const encoder = t.device.createCommandEncoder();
        t.dispatch(encoder, 'rl_normstats', 64);
        t.dispatch(encoder, 'rl_normapply', 1);
        await t.submit(encoder);
        const theta = await t.readFloats('theta', 0, P.count);
        const opt = await t.readFloats('opt', 0, t.layout.optFloats);
        const normBase = 2 * P.count + RL_CTL.SIZE;
        for (let f = 0; f < P.nIn; f++) {
          worstMean = Math.max(worstMean, Math.abs(theta[P.normMean + f] - merged.mean[f]));
          worstIstd = Math.max(worstIstd, Math.abs(theta[P.normIstd + f] - merged.istd[f]) / merged.istd[f]);
          worstVar = Math.max(worstVar, Math.abs(opt[normBase + P.nIn + f] - merged.variance[f]));
        }
        state.count = merged.count;
        state.mean = merged.mean;
        state.variance = merged.variance;
        const count = opt[2 * P.count + RL_CTL.NCOUNT];
        if (Math.abs(count - merged.count) > 1) throw new Error('running count ' + count + ' expected ' + merged.count);
      }
      const detail = 'running mean and inverse std of ' + P.nIn + ' inputs after 3 merges of ' + steps * t.layout.trainLearners + '-sample batches: max mean error ' + worstMean.toExponential(2) + ', inverse std relative error ' + worstIstd.toExponential(2) + ', variance error ' + worstVar.toExponential(2);
      if (!(worstMean < 2e-3) || !(worstIstd < 2e-3)) throw new Error(detail);
      return detail;
    } finally {
      await t.dispose();
    }
  }

  async testRewardCaps() {
    const game = this.game;
    if (!game.rewardChannels || !/g_reward_channel/.test(game.wgsl)) return 'not applicable: this game exposes no reward channels';
    const cap = 0.002;
    const steps = 40;
    const t = await this.tempBackend({ worlds: 6, hidden: 12, rolloutTicks: steps, gradGroups: 4, lifeCapScale: 0.02, obsNorm: 0, channelCaps: game.rewardChannels.map(() => cap) });
    try {
      const P = t.params;
      const R = rlRecordLayout(P.nOut);
      const slots = t.slots;
      const limit = Math.round(cap * REWARD_SCALE);
      const trainAge = Math.max(1, Math.round(game.maxAge * 0.02));
      await t.iterate({ update: false });
      const data = await t.readRollout();
      let capped = 0;
      let compared = 0;
      for (let w = 0; w < Math.min(t.evalStart, 4); w++) {
        const header = new Uint32Array(await t.readWords(t.buffers.world, w * t.layout.worldStride * 4, 4));
        const seed = header[0];
        const env = game.createEnv(seed, game.randomCfg(seed), false, new Int32Array(NSTATS));
        env.difficulty = 100;
        const life = new Float64Array(slots);
        for (let a = 0; a < slots; a++) life[a] = mix(seed, a, 0, 11) % trainAge;
        const actions = new Int32Array(slots);
        const outputs = new Float32Array(slots * P.nOut);
        for (let k = 0; k < steps; k++) {
          outputs.fill(0);
          for (let a = 0; a < slots; a++) {
            if (t.roleOf(data, w * slots + a) === RL_ROLE.INACTIVE) continue;
            const sample = ((w * slots + a) * steps + k) * R.stride;
            for (let c = 0; c < P.nOut; c++) outputs[a * P.nOut + c] = Math.max(-1, Math.min(1, data.rec[sample + R.action + c]));
          }
          env.step(actions, outputs, k);
          const ended = [];
          for (let a = 0; a < slots; a++) {
            if (t.roleOf(data, w * slots + a) === RL_ROLE.INACTIVE) continue;
            let expected = 0;
            for (let c = 0; c < game.rewardChannels.length; c++) expected += Math.max(-limit, Math.min(limit, env.rewardCh[a * game.rewardChannels.length + c]));
            const sample = ((w * slots + a) * steps + k) * R.stride;
            const recorded = data.rec[sample + R.reward] * REWARD_SCALE;
            if (recorded !== expected) throw new Error('world ' + w + ' tick ' + k + ' learner ' + a + ': gpu capped reward ' + recorded + ' js ' + expected);
            if (expected !== env.reward[a]) capped++;
            compared++;
            life[a]++;
            if (env.dead[a] || life[a] >= trainAge) {
              ended.push(a);
              life[a] = 0;
            }
          }
          for (const a of ended) env.respawn(a, k);
        }
      }
      const detail = compared + ' learner ticks: reward equals the sum of per-channel clamps at +-' + cap + ' (' + game.rewardChannels.join('/') + '), ' + capped + ' of them differ from the uncapped total';
      if (capped === 0) throw new Error(detail + ' (caps never triggered, test is vacuous)');
      return detail;
    } finally {
      await t.dispose();
    }
  }

  async testCurriculum() {
    const walk = new RlCurriculum({ curriculum: 1, curriculumDifficultyStart: 30, curriculumDifficultyStep: 10, curriculumLifeUp: 0.2, curriculumLifeDown: 0.05, curriculumPatience: 2, curriculumSmoothing: 1, curriculumSelfPlay: 1, curriculumSelfPlayStart: 0.1, curriculumSelfPlayEnd: 0.9, curriculumRatioLow: 1, curriculumRatioHigh: 3 }, 1000);
    const feed = (life, ratio) => walk.update({ evalLife: life, trainLife: life, evalRate: ratio, baseRate: 1 });
    feed(300, 0.5);
    if (walk.difficulty !== 30) throw new Error('difficulty moved before the patience expired');
    feed(300, 0.5);
    if (walk.difficulty !== 40) throw new Error('difficulty did not rise after two good reports: ' + walk.difficulty);
    feed(20, 2);
    feed(20, 2);
    if (walk.difficulty !== 30) throw new Error('difficulty did not fall after two collapsed reports: ' + walk.difficulty);
    if (Math.abs(walk.selfPlay - 0.5) > 1e-9) throw new Error('self-play fraction ' + walk.selfPlay + ' does not follow the eval ratio');
    const t = await this.tempBackend({ worlds: 24, hidden: 12, rolloutTicks: 8, gradGroups: 4, curriculum: 1, curriculumSelfPlay: 1, curriculumSelfPlayStart: 0, curriculumSelfPlayEnd: 1, evalFraction: 0.17, leagueEvalFraction: 0.05 });
    try {
      const modes = async () => {
        const words = new Uint32Array(await t.readWords(t.buffers.world, 0, t.opts.worlds * t.layout.worldStride * 4));
        return Array.from({ length: t.evalStart }, (_, w) => words[w * t.layout.worldStride + 2]);
      };
      const slotsOf = async (w) => (await t.snapshot(w)).learnerSlots;
      const set = async (fraction) => {
        t.controller.selfPlay = fraction;
        t.curriculum.selfPlay = fraction;
        await t.reinitWorlds();
        return modes();
      };
      const none = await modes();
      if (none.some((m) => m !== 0)) throw new Error('worlds start in self-play mode with fraction 0');
      const all = await set(1);
      if (all.some((m) => m !== 1)) throw new Error('fraction 1 left bot worlds: ' + all.join(''));
      const half = await set(0.5);
      const halfCount = half.reduce((a, b) => a + b, 0);
      if (!(halfCount > 0 && halfCount < half.length)) throw new Error('fraction 0.5 gave ' + halfCount + ' of ' + half.length + ' self-play worlds');
      const slotList = [];
      for (let w = 0; w < t.evalStart; w++) slotList.push(await slotsOf(w));
      if (slotList[0] !== undefined) for (let w = 0; w < t.evalStart; w++) if ((half[w] === 1) !== (slotList[w] === t.slots)) throw new Error('world ' + w + ' mode ' + half[w] + ' but learner slots ' + slotList[w]);
      await t.iterate();
      const back = await set(0);
      if (back.some((m) => m !== 0)) throw new Error('fraction 0 left self-play worlds');
      await t.iterate();
      return 'controller: difficulty 30 -> 40 -> 30 on good then collapsed life, self-play fraction follows the eval ratio (0.5 at ratio 2 between 1 and 3); GPU worlds flip between bot and ' + t.slots + '-learner self-play as the fraction moves (0 -> 1 -> 0.5 gave ' + halfCount + ' of ' + half.length + ' -> 0), state re-initialised and training continues';
    } finally {
      await t.dispose();
    }
  }

  async testTrainingStability() {
    const t = await this.tempBackend({ worlds: 12, hidden: 24, rolloutTicks: 32, gradGroups: 16, epochs: 2, minibatches: 2, curriculum: 1, curriculumSelfPlay: 1 });
    try {
      for (let i = 0; i < 12; i++) await t.step();
      const s = await t.readStats();
      const theta = await t.readFloats('theta', 0, t.params.count);
      for (let i = 0; i < theta.length; i++) if (!Number.isFinite(theta[i])) throw new Error('theta[' + i + '] not finite');
      const d = s.ppo;
      if (![d.gradNorm, d.policyLoss, d.valueLoss, d.entropy, d.kl].every(Number.isFinite)) throw new Error('non finite diagnostics ' + JSON.stringify(d));
      return '12 iterations (' + t.tick + ' ticks) finite: grad norm ' + d.gradNorm.toFixed(3) + ', value loss ' + d.valueLoss.toFixed(4) + ', entropy ' + d.entropy.toFixed(3) + ', kl ' + d.kl.toExponential(2) + ', reward scale ' + d.rewardScale.toFixed(3);
    } finally {
      await t.dispose();
    }
  }

  async testCheckpointAndImport() {
    const t = await this.tempBackend({ worlds: 6, hidden: 16, rolloutTicks: 16, gradGroups: 8 });
    const u = await this.tempBackend({ worlds: 6, hidden: 16, rolloutTicks: 16, gradGroups: 8, seed: 99 });
    try {
      for (let i = 0; i < 3; i++) await t.step();
      const ckpt = JSON.parse(JSON.stringify(await t.checkpoint()));
      await u.restore(ckpt);
      const a = await t.readFloats('theta', 0, t.params.count);
      const b = await u.readFloats('theta', 0, u.params.count);
      for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) throw new Error('checkpoint theta differs at ' + i);
      const [genome] = await t.exportChampions(1);
      await u.importGenomes([JSON.parse(JSON.stringify(genome))]);
      const c = await u.readFloats('theta', 0, u.params.count);
      const tolerance = t.params.obsNorm ? 1e-5 : 0;
      for (let i = 0; i < a.length; i++) if (Math.abs(a[i] - c[i]) > tolerance * (1 + Math.abs(a[i]))) throw new Error('imported genome theta differs at ' + i + ': ' + a[i] + ' vs ' + c[i]);
      return 'checkpoint restore and genome import reproduce ' + a.length + ' parameters ' + (tolerance ? 'within ' + tolerance + ' (input normalisation is folded into layer 1 and unfolded again)' : 'exactly');
    } finally {
      await t.dispose();
      await u.dispose();
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
    await record('workgroup memory fits device', async () => this.layout.rolloutBytes + ' rollout / ' + this.layout.gradBytes + ' gradient of ' + this.ctx.granted.maxComputeWorkgroupStorageSize + ' bytes');
    await record('finite-difference gradient check (JS reference)', async () => {
      const combos = [{}, { bias: true }, { recurrent: true }, { bias: true, recurrent: true, obsNorm: true }];
      const runs = combos.flatMap((features) => [1, 2, 3].map((seed) => rlGradientCheck(seed, features)));
      const worst = Math.max(...runs.map((r) => r.worst));
      if (!(worst < 1e-5)) throw new Error('worst relative error ' + worst);
      return 'analytic vs numeric gradient over feed-forward, bias, recurrent (BPTT with leak) and full layouts x 3 seeds: worst relative error ' + worst.toExponential(2);
    });
    await record('rollout determinism', () => this.testDeterminism());
    await record('policy/value forward parity, GPU vs JS reference', () => this.testForwardParity());
    await record('rollout lockstep with the JS env', () => this.testRolloutLockstep());
    await record('GAE parity, GPU vs JS reference', () => this.testAdvantageParity());
    await record('gradient and Adam update parity, GPU vs JS reference', () => this.testUpdateParity());
    await record('exported genome replays through core.js Brain', () => this.testExportReplay());
    await record('league: roles, snapshot policies, masked gradients, head-to-head stats', () => this.testLeague());
    await record('recurrent hidden state carried across rollouts', () => this.testHiddenCarry());
    await record('input normalisation statistics, GPU vs JS reference', () => this.testNormalization());
    await record('reward hygiene: per-channel caps', () => this.testRewardCaps());
    await record('automatic curriculum: difficulty controller and self-play flips', () => this.testCurriculum());
    await record('training stability', () => this.testTrainingStability());
    await record('checkpoint and genome import round trip', () => this.testCheckpointAndImport());
    return results;
  }

  async timeEncoded(build, repeats) {
    const times = [];
    for (let i = 0; i < repeats; i++) {
      const encoder = this.device.createCommandEncoder();
      build(encoder);
      const start = performance.now();
      await this.submit(encoder);
      times.push(performance.now() - start);
    }
    times.sort((a, b) => a - b);
    return times[Math.floor(times.length / 2)];
  }

  async profileIteration() {
    if (!this.ctx.features.includes('timestamp-query')) return null;
    const o = this.opts;
    const steps = Math.max(0, Math.round(o.epochs)) * Math.max(1, Math.round(o.minibatches));
    const plan = [['rl_rollout', o.worlds], ['rl_gae', Math.max(1, Math.ceil(this.layout.trainLearners / 64))], ['rl_advstats', 1]];
    for (let s = 0; s < steps; s++) plan.push([this.params.recurrent ? 'rl_grad_seq' : 'rl_grad', this.layout.gradGroups], ['rl_reduce_grad', Math.ceil(this.params.partialStride / 256)], ['rl_adam', 1]);
    if (steps > 0) plan.push(['rl_rescale', 1]);
    if (this.params.obsNorm) plan.push(['rl_normstats', 64], ['rl_normapply', 1]);
    const querySet = this.device.createQuerySet({ type: 'timestamp', count: plan.length * 2 });
    const resolve = this.device.createBuffer({ size: plan.length * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const total = { name: 'total', ms: 0 };
    const sums = new Map();
    const rounds = 3;
    for (let round = 0; round < rounds; round++) {
      this.writeUniforms({});
      const encoder = this.device.createCommandEncoder();
      plan.forEach(([entry, groups], i) => {
        const pass = encoder.beginComputePass({ label: entry, timestampWrites: { querySet, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } });
        pass.setPipeline(this.pipelines[entry]);
        pass.setBindGroup(0, this.bindGroup);
        pass.dispatchWorkgroups(groups);
        pass.end();
      });
      encoder.resolveQuerySet(querySet, 0, plan.length * 2, resolve, 0);
      await this.submit(encoder);
      const stamps = new BigUint64Array(await this.readWords(resolve, 0, plan.length * 16));
      plan.forEach(([entry], i) => {
        const ms = Number(stamps[2 * i + 1] - stamps[2 * i]) / 1e6;
        sums.set(entry, (sums.get(entry) || 0) + ms / rounds);
        total.ms += ms / rounds;
      });
    }
    querySet.destroy();
    resolve.destroy();
    const rows = {};
    for (const [entry, ms] of sums) rows[entry] = ms;
    rows.total = total.ms;
    return rows;
  }

  async profileKernels() {
    const P = this.params;
    const perCall = async (entry, groups) => (await this.timeEncoded((e) => { for (let i = 0; i < 8; i++) this.dispatch(e, entry, groups); }, 3)) / 8;
    return {
      rollout: await perCall('rl_rollout', this.opts.worlds),
      gae: await perCall('rl_gae', Math.max(1, Math.ceil(this.layout.trainLearners / 64))),
      advstats: await perCall('rl_advstats', 1),
      grad: await perCall(this.params.recurrent ? 'rl_grad_seq' : 'rl_grad', this.layout.gradGroups),
      reduce: await perCall('rl_reduce_grad', Math.ceil(P.partialStride / 256)),
      adam: await perCall('rl_adam', 1)
    };
  }

  async benchmark(progress) {
    const rows = [];
    const cap = rlWorldCapacity(this.ctx, this.game, this.opts);
    for (const worlds of [128, 256, 512, 1024, 2048]) {
      if (worlds > cap) {
        rows.push({ worlds, skipped: 'exceeds buffer limits (cap ' + cap + ')' });
        continue;
      }
      let tmp;
      try {
        tmp = await this.tempBackend({ worlds, evalFraction: 0.0625, evalEvery: 0 });
        for (let i = 0; i < 2; i++) await tmp.iterate();
        const rolloutMs = await tmp.timeEncoded((e) => tmp.encodeRollout(e), 3);
        const adviceMs = await tmp.timeEncoded((e) => tmp.encodeAdvantages(e), 3);
        const gradientMs = await tmp.timeEncoded((e) => tmp.encodeGradient(e), 3);
        const iterationMs = (await tmp.iterate()).ms;
        const samples = tmp.layout.trainLearners * tmp.opts.rolloutTicks;
        const steps = tmp.opts.epochs * tmp.opts.minibatches;
        const kernels = await tmp.profileKernels();
        rows.push({ worlds, samples, kernels, rolloutMs, adviceMs, gradientMs, iterationMs, samplesPerSec: samples / (iterationMs / 1000), rolloutSamplesPerSec: samples / (rolloutMs / 1000), updatesPerMin: (steps * 60000) / iterationMs, iterationsPerMin: 60000 / iterationMs, worldTicksPerSec: (worlds * tmp.opts.rolloutTicks) / (iterationMs / 1000) });
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
