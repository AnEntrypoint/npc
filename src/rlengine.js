import { getDeviceContext } from './engine.js';
import { DEFAULT_OPTS, StatsAccumulator, NSTATS, STAT, REWARD_SCALE, validateGame, evalWorldCount, difficultyAt, genomeFromJSON, Brain, Rng, evoConfig, mix } from './core.js';
import { RL_DEFAULTS, RL_CURRICULUM_DEFAULTS, RL_POP_DEFAULTS, RL_POP_MIX_NAMES, RlCurriculum, rlLayout, rlRecordLayout, rlInitTheta, rlThetaToGenome, rlGenomeToTheta, rlHiddenActivations, rlPolicyStep, rlPolicyMean, rlValueOf, rlLogProb, rlGae, rlLossAndGrad, rlAdamStep, rlObgdStep, rlBoundedStep, rlOptimizerStep, rlGradientCheck, rlNormMerge, rlPopConfig, rlPopPolicyOf, rlWeightedReward } from './rl.js';
import { buildRlShader, rlShaderLayout, rlFeatures, rlLanePermute, rlRecPermute, RL_ENTRIES, RL_UNIFORM_WORDS, RL_CTL, RL_LANE, RL_ROLE, RL_STATS, RL_POP_GROUP, RL_POP_STAT_GROUPS, RL_POP_WEIGHT_SLOTS } from './rlshader.js';

const RL_STATS_FLUSH_ITERATIONS = 8;
const RL_ERROR_SCOPE_EVERY = 16;
const RL_STORAGE_BINDINGS = 8;
const RL_BOT_FREE = { curriculum: 1, curriculumSelfPlay: 1, curriculumSelfPlayStart: 1, curriculumSelfPlayEnd: 1, curriculumDifficultyStart: 100, curriculumLifeUp: 2, curriculumLifeDown: -1 };
const RL_TRACKED_KEYS = ['hidden', 'rolloutTicks', 'gamma', 'lambda', 'clip', 'lr', 'lrEnd', 'lrDecayIterations', 'entropy', 'entropyEnd', 'valueCoef', 'epochs', 'minibatches', 'maxGrad', 'sigmaInit', 'sigmaMin', 'sigmaMax', 'weightMax', 'rewardScale', 'adaptScale', 'lifeCapScale', 'gradGroups', 'tileEntries', 'league', 'leagueFraction', 'leagueSlotFraction', 'poolSize', 'snapshotEvery', 'leaguePeriod', 'latestBias', 'leagueEvalFraction', 'referenceAge', 'blockGrad', 'curriculum', 'curriculumMetric', 'curriculumDifficultyStart', 'curriculumDifficultyStep', 'curriculumLifeUp', 'curriculumLifeDown', 'curriculumPatience', 'curriculumSmoothing', 'curriculumSelfPlay', 'curriculumSelfPlayStart', 'curriculumSelfPlayEnd', 'curriculumRatioLow', 'curriculumRatioHigh', 'bias', 'recurrent', 'obsNorm', 'recurrentInit', 'leakInit', 'leakTauMax', 'learnLeak', 'rewardClip', 'valueClip', 'obsNormFloor', 'obsNormCap', 'channelCaps', 'popWeights', 'popWeightsEnd', 'hostPipeline', 'soa', 'fwdDirect', 'optimizer', 'obgdBudget'];

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
    const merged = Object.assign({}, DEFAULT_OPTS, RL_DEFAULTS, RL_CURRICULUM_DEFAULTS, RL_POP_DEFAULTS, { islands: 1 }, opts);
    merged.islands = 1;
    if (merged.botFree) Object.assign(merged, RL_BOT_FREE);
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
    this.pop = rlPopConfig(opts, this.game.rewardChannels);
    this.K = this.pop.k;
    this.kind = 'gpu';
    this.slots = rlSlotCount(this.game);
    this.evalCount = evalWorldCount(opts);
    this.leagueEvalCount = opts.league ? Math.max(2, Math.round(opts.worlds * opts.leagueEvalFraction)) : 0;
    this.evalStart = opts.worlds - this.evalCount - 2 * this.leagueEvalCount;
    if (this.evalStart < 1) throw new Error('not enough worlds for the eval and league groups');
    if (this.K > 1 && this.evalCount < this.K + 1) throw new Error('population of ' + this.K + ' policies needs at least ' + (this.K + 1) + ' benchmark eval worlds (pure worlds per role plus the mixed team), raise evalFraction or worlds');
    this.evalBStart = this.evalStart + this.evalCount;
    this.evalCStart = this.evalBStart + this.leagueEvalCount;
    this.poolSize = opts.league ? Math.max(1, Math.round(opts.poolSize)) : 0;
    this.layout = rlShaderLayout(this.game, this.shaderConfig());
    this.params = this.layout.params;
    this.stats = new StatsAccumulator();
    this.leagueStats = { headToHead: new StatsAccumulator(), selfPlay: new StatsAccumulator() };
    this.roleWindow = new Float64Array(RL_STATS.ROLE_WORDS);
    this.pools = Array.from({ length: this.K }, () => ({ valid: 0, events: 0, iterations: new Array(this.poolSize).fill(-1) }));
    this.pool = this.pools[0];
    this.references = Array.from({ length: this.K }, () => ({ slot: 0, iteration: -1 }));
    this.reference = this.references[0];
    this.popDiagnostics = Array.from({ length: this.K }, () => ({}));
    this.resetPopStats();
    this.controller = opts.curriculum ? new RlCurriculum(opts, this.game.maxAge) : null;
    this.curriculum = { spOn: Boolean(this.controller && opts.curriculumSelfPlay), selfPlay: this.controller ? this.controller.selfPlay : 0 };
    this.tick = 0;
    this.passes = 0;
    this.lastEntropy = this.entropyCoef();
    this.iterationsSinceFlush = 0;
    this.lastStepMs = 0;
    this.inflight = null;
    this.stagingBySize = new Map();
    this.worldScores = new Float32Array(opts.worlds);
    this.diagnostics = { gradNorm: 0, policyLoss: 0, valueLoss: 0, entropy: 0, clipFraction: 0, kl: 0, returnStd: 0, returnMean: 0, advStd: 0, rewardScale: opts.rewardScale, iteration: 0 };
    this.bestEval = -Infinity;
    this.bestTheta = null;
    this.lastWindowEval = 0;
    this.evalMark = { ticks: 0, reward: 0 };
    this.buffers = {};
    this.info = { kind: 'gpu', trainer: 'ppo', adapter: ctx.adapterInfo, limits: ctx.granted, features: ctx.features, worlds: opts.worlds, islands: 1, maxEdges: this.layout.denseEdges, game: this.game.id, hidden: opts.hidden, policies: this.K, roles: this.pop.names, features: this.layout.features, workgroupBytes: this.layout.workgroupBytes, warnings: [] };
  }

  shaderConfig() {
    const o = this.opts;
    return { policies: this.K, worlds: o.worlds, evalStart: this.evalStart, evalBStart: this.evalBStart, evalCStart: this.evalCStart, poolSize: this.poolSize, hidden: o.hidden, rolloutTicks: o.rolloutTicks, gradGroups: o.gradGroups, tileEntries: o.tileEntries, lifeCapScale: o.lifeCapScale, blockGrad: o.blockGrad, minibatches: o.minibatches, bias: o.bias, recurrent: o.recurrent, obsNorm: o.obsNorm, learnLeak: o.learnLeak, rewardClip: o.rewardClip, valueClip: o.valueClip, advClipMult: o.advClipMult, advClipBeta: o.advClipBeta, entropySign: o.entropySign, obsNormFloor: o.obsNormFloor, obsNormCap: o.obsNormCap, channelCaps: o.channelCaps, optimizer: o.optimizer, obgdBudget: o.obgdBudget };
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
      for (let p = 1; p < this.K; p++) this.makeBuffer('uniforms' + p, RL_UNIFORM_WORDS * 4, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      if (this.poolSize > 0) this.makeBuffer('snapshotScratch', L.params.count * 4, GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
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
      await Promise.all(RL_ENTRIES.filter((entry) => (this.layout.blockGrad || (entry !== 'rl_select' && entry !== 'rl_grad_blk')) && (this.K > 1 || entry !== 'rl_reduce_pop')).map(async (entry) => {
        this.pipelines[entry] = await this.device.createComputePipelineAsync({ label: entry, layout: pipelineLayout, compute: { module, entryPoint: entry } });
      }));
    });
    const b = this.buffers;
    this.bindGroups = Array.from({ length: this.K }, (_, p) => this.device.createBindGroup({ layout: bindLayout, entries: [b.theta, b.world, b.obs, b.rec, b.lane, b.partials, b.opt, b.statsOut, p === 0 ? b.uniforms : b['uniforms' + p]].map((buffer, binding) => ({ binding, resource: { buffer } })) }));
    this.bindGroup = this.bindGroups[0];
    this.uniformData = new ArrayBuffer(RL_UNIFORM_WORDS * 4);
    this.uniformU32 = new Uint32Array(this.uniformData);
    this.uniformF32 = new Float32Array(this.uniformData);
    for (let p = 0; p < this.K; p++) this.writeTheta(rlInitTheta(this.params, (this.opts.seed + p * 7919) * 2654435761, this.opts), p);
    this.resetOptimizer();
    this.writeUniforms({});
    const encoder = this.device.createCommandEncoder();
    this.dispatch(encoder, 'rl_init_world', this.opts.worlds);
    await this.submit(encoder);
  }

  writeTheta(theta, policy) {
    this.device.queue.writeBuffer(this.buffers.theta, (policy || 0) * this.params.count * 4, theta);
  }

  resetOptimizer(scale, policy) {
    const stride = this.layout.optStride;
    const opt = new Float32Array(stride);
    opt[2 * this.params.count + RL_CTL.SCALE] = scale === undefined ? this.opts.rewardScale : scale;
    if (this.params.obsNorm) {
      const normBase = 2 * this.params.count + RL_CTL.SIZE;
      for (let j = 0; j < this.params.nIn; j++) opt[normBase + this.params.nIn + j] = 1;
    }
    const policies = policy === undefined ? Array.from({ length: this.K }, (_, p) => p) : [policy];
    for (const p of policies) {
      opt.fill(0, this.layout.optWeights);
      this.pop.weights[p].forEach((w, c) => { opt[this.layout.optWeights + c] = w; });
      this.device.queue.writeBuffer(this.buffers.opt, p * stride * 4, opt);
    }
  }

  writePolicyWeights(policy) {
    const row = new Float32Array(RL_POP_WEIGHT_SLOTS);
    this.pop.weights[policy].forEach((w, c) => { row[c] = w; });
    this.device.queue.writeBuffer(this.buffers.opt, (policy * this.layout.optStride + this.layout.optWeights) * 4, row);
  }

  learningRate() {
    const o = this.opts;
    const progress = Math.min(1, this.passes / Math.max(1, o.lrDecayIterations));
    return o.lr + (o.lrEnd - o.lr) * progress;
  }

  entropyCoef() {
    const o = this.opts;
    const end = o.entropyEnd === undefined ? o.entropy : o.entropyEnd;
    if (end === o.entropy) return o.entropy;
    const progress = Math.min(1, this.passes / Math.max(1, o.lrDecayIterations));
    return o.entropy + (end - o.entropy) * progress;
  }

  scheduleProgress() {
    return Math.min(1, this.passes / Math.max(1, this.opts.lrDecayIterations));
  }

  applyChannelSchedule() {
    const targets = this.pop.weightsEnd;
    if (!targets) return null;
    const progress = this.scheduleProgress();
    for (let p = 0; p < this.K; p++) {
      const row = this.pop.weights[p];
      const start = this.pop.weightsStart[p];
      const end = targets[p];
      if (!end) continue;
      for (let c = 0; c < row.length; c++) row[c] = start[c] + (end[c] - start[c]) * progress;
      this.writePolicyWeights(p);
    }
    return progress;
  }

  writeUniforms(v) {
    const o = this.opts;
    this.channelProgress = this.applyChannelSchedule();
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
    f[11] = this.lastEntropy = this.entropyCoef();
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
    u[33] = this.K;
    u[34] = this.pop.mix;
    u[35] = this.pop.period > 0 ? Math.floor(this.passes / this.pop.period) : 0;
    f[36] = this.pop.pure;
    for (let p = 0; p < this.K; p++) {
      u[40 + p] = this.pools[p].valid;
      u[48 + p] = this.references[p].slot;
    }
    for (let p = 0; p < 8; p++) f[56 + p] = this.pop.cum[p];
    for (let p = 0; p < this.K; p++) {
      u[32] = p;
      this.device.queue.writeBuffer(p === 0 ? this.buffers.uniforms : this.buffers['uniforms' + p], 0, this.uniformData);
    }
  }

  dispatch(encoder, entry, groups, policy) {
    const pass = encoder.beginComputePass({ label: entry });
    pass.setPipeline(this.pipelines[entry]);
    pass.setBindGroup(0, this.bindGroups[policy || 0]);
    pass.dispatchWorkgroups(groups);
    pass.end();
  }

  async submit(encoder) {
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
  }

  async readWords(buffer, byteOffset, byteLength) {
    let staging = this.stagingBySize.get(byteLength);
    if (!staging) {
      staging = this.device.createBuffer({ size: byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      this.stagingBySize.set(byteLength, staging);
    }
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, byteOffset, staging, 0, byteLength);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const copy = staging.getMappedRange().slice(0);
    staging.unmap();
    return copy;
  }

  async readFloats(name, floatOffset, floatCount) {
    return new Float32Array(await this.readWords(this.buffers[name], floatOffset * 4, floatCount * 4));
  }

  recordSnapshot(encoder) {
    const o = this.opts;
    if (!o.league || this.passes === 0 || this.passes % Math.max(1, Math.round(o.snapshotEvery)) !== 0) return;
    for (let p = 0; p < this.K; p++) {
      const pool = this.pools[p];
      const reference = this.references[p];
      const slice = (slot) => this.K + p * this.poolSize + slot;
      const seed = o.seed + p * 7919;
      pool.events++;
      if (pool.valid === 0) {
        pool.valid = 1;
      } else {
        let target = -1;
        if (pool.valid < this.poolSize) target = pool.valid++;
        else if (this.poolSize > 1 && mix(seed, pool.events, 0, 55) / 4294967296 < (this.poolSize - 1) / pool.events) target = 1 + (mix(seed, pool.events, 1, 55) % (this.poolSize - 1));
        if (target === reference.slot && reference.iteration >= 0) target = -1;
        if (target >= 0) {
          this.copySlice(encoder, slice(0), slice(target));
          pool.iterations[target] = pool.iterations[0];
        }
      }
      this.copySlice(encoder, p, slice(0));
      pool.iterations[0] = this.passes;
    }
  }

  copySlice(encoder, from, to) {
    const bytes = this.params.count * 4;
    encoder.copyBufferToBuffer(this.buffers.theta, from * bytes, this.buffers.snapshotScratch, 0, bytes);
    encoder.copyBufferToBuffer(this.buffers.snapshotScratch, 0, this.buffers.theta, to * bytes, bytes);
  }

  chooseReference() {
    for (let p = 0; p < this.K; p++) {
      const pool = this.pools[p];
      if (pool.valid === 0) continue;
      const wanted = this.passes - this.opts.referenceAge;
      let best = pool.valid > 1 ? 1 : 0;
      for (let s = pool.valid > 1 ? 1 : 0; s < pool.valid; s++) if (Math.abs(pool.iterations[s] - wanted) < Math.abs(pool.iterations[best] - wanted)) best = s;
      this.references[p] = { slot: best, iteration: pool.iterations[best] };
    }
    this.reference = this.references[0];
  }

  encodeRollout(encoder) {
    this.dispatch(encoder, 'rl_rollout', this.opts.worlds);
  }

  encodeAdvantages(encoder) {
    this.dispatch(encoder, 'rl_gae', Math.max(1, Math.ceil(this.layout.trainLearners / 64)));
    for (let p = 0; p < this.K; p++) this.dispatch(encoder, 'rl_advstats', 1, p);
  }

  gradEntry() {
    if (this.layout.blockGrad) return 'rl_grad_blk';
    return this.params.recurrent ? 'rl_grad_seq' : 'rl_grad';
  }

  encodeGradient(encoder, policy) {
    const p = policy || 0;
    if (this.layout.blockGrad) this.dispatch(encoder, 'rl_select', 1, p);
    this.dispatch(encoder, this.gradEntry(), this.layout.gradGroups, p);
    this.dispatch(encoder, 'rl_reduce_grad', Math.ceil(this.params.partialStride / 256), p);
  }

  encodeUpdates(encoder) {
    const steps = Math.max(0, Math.round(this.opts.epochs)) * Math.max(1, Math.round(this.opts.minibatches));
    for (let s = 0; s < steps; s++) {
      for (let p = 0; p < this.K; p++) {
        this.encodeGradient(encoder, p);
        this.dispatch(encoder, 'rl_adam', 1, p);
      }
    }
    for (let p = 0; p < this.K; p++) {
      if (steps > 0) this.dispatch(encoder, 'rl_rescale', 1, p);
      if (this.params.obsNorm) {
        this.dispatch(encoder, 'rl_normstats', 64, p);
        this.dispatch(encoder, 'rl_normapply', 1, p);
      }
    }
  }

  async iterate(options) {
    if (this.ctx.lost) throw new Error('GPU device lost: ' + this.ctx.lost.message);
    const o = this.opts;
    const ticks = o.rolloutTicks;
    const evalPeriod = this.game.maxAge * o.evalEvery;
    const evalReset = o.evalEvery > 0 && Math.floor((this.tick + ticks) / evalPeriod) > Math.floor(this.tick / evalPeriod);
    const start = performance.now();
    const pipe = o.hostPipeline === 1;
    if (pipe && this.inflight) {
      await this.inflight;
      this.inflight = null;
    }
    const scoped = !pipe || this.passes % RL_ERROR_SCOPE_EVERY === 0;
    if (scoped) this.device.pushErrorScope('validation');
    try {
      const rolloutEncoder = this.device.createCommandEncoder();
      this.recordSnapshot(rolloutEncoder);
      if (evalReset || this.references.some((r) => r.iteration < 0)) this.chooseReference();
      this.writeUniforms({ flags: evalReset ? 1 : 0 });
      this.encodeRollout(rolloutEncoder);
      this.encodeAdvantages(rolloutEncoder);
      if (!pipe) {
        this.device.queue.submit([rolloutEncoder.finish()]);
        if (!options || options.update !== false) {
          const updateEncoder = this.device.createCommandEncoder();
          this.encodeUpdates(updateEncoder);
          this.device.queue.submit([updateEncoder.finish()]);
        }
        await this.device.queue.onSubmittedWorkDone();
      } else {
        if (!options || options.update !== false) this.encodeUpdates(rolloutEncoder);
        this.device.queue.submit([rolloutEncoder.finish()]);
        this.inflight = this.device.queue.onSubmittedWorkDone();
      }
    } finally {
      if (scoped) this.lastValidationError = await this.device.popErrorScope();
      else this.lastValidationError = null;
    }
    if (this.lastValidationError) throw new Error('GPU validation error during iteration: ' + this.lastValidationError.message);
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

  controlOf(ctl) {
    const steps = Math.max(1, ctl[RL_CTL.STEPS]);
    return {
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
      updates: ctl[RL_CTL.STEP],
      samples: ctl[RL_CTL.ACTIVE]
    };
  }

  async readControl() {
    const all = [];
    for (let p = 0; p < this.K; p++) all.push(await this.readFloats('opt', p * this.layout.optStride + 2 * this.params.count, RL_CTL.SIZE));
    this.popDiagnostics = all.map((ctl) => this.controlOf(ctl));
    if (this.K === 1) this.diagnostics = this.popDiagnostics[0];
    else {
      const mean = (key) => this.popDiagnostics.reduce((sum, d) => sum + d[key], 0) / this.K;
      this.diagnostics = Object.assign({}, this.popDiagnostics[0], { gradNorm: mean('gradNorm'), policyLoss: mean('policyLoss'), valueLoss: mean('valueLoss'), entropy: mean('entropy'), clipFraction: mean('clipFraction'), kl: mean('kl'), returnStd: mean('returnStd'), returnMean: mean('returnMean'), advStd: mean('advStd'), rewardScale: mean('rewardScale'), samples: this.popDiagnostics.reduce((sum, d) => sum + d.samples, 0) });
    }
    return all[0];
  }

  resetPopStats() {
    this.popAcc = { learner: Array.from({ length: RL_POP_STAT_GROUPS }, () => Array.from({ length: this.K }, () => new Float64Array(4))), sub: Array.from({ length: this.K + 1 }, () => new StatsAccumulator()) };
  }

  absorbPopStats(rows) {
    const K = this.K;
    const base = this.layout.popOffset;
    for (let g = 0; g < RL_POP_STAT_GROUPS; g++) {
      for (let p = 0; p < K; p++) for (let s = 0; s < 4; s++) this.popAcc.learner[g][p][s] += rows[base + (g * K + p) * 4 + s];
    }
    const subBase = base + RL_POP_STAT_GROUPS * K * 4;
    for (let sg = 0; sg <= K; sg++) this.popAcc.sub[sg].add('eval', Float64Array.from(rows.subarray(subBase + sg * 16, subBase + (sg + 1) * 16)), this.slots);
  }

  popSummary() {
    const K = this.K;
    const learner = this.popAcc.learner;
    const rate = (g, p) => {
      const a = learner[g][p];
      return { rate: a[0] / REWARD_SCALE / Math.max(1, a[1]), ticks: a[1], life: a[1] / Math.max(1, a[2]), lifeFit: a[2] > 0 ? a[3] / REWARD_SCALE / a[2] : 0 };
    };
    const roles = [];
    for (let p = 0; p < K; p++) {
      const latest = rate(RL_POP_GROUP.EVAL_B_LIVE, p);
      const older = rate(RL_POP_GROUP.EVAL_B_SNAPSHOT, p);
      roles.push({
        index: p,
        name: this.pop.names[p],
        weights: Array.from(this.pop.weights[p]),
        fraction: this.pop.fractions[p],
        train: rate(RL_POP_GROUP.TRAIN, p),
        evalPure: this.popAcc.sub[p].summary('eval', this.game),
        evalMixed: rate(RL_POP_GROUP.EVAL_MIXED, p),
        headToHead: { latestRate: latest.rate, olderRate: older.rate, ratio: older.rate !== 0 ? latest.rate / older.rate : NaN, latestTicks: latest.ticks, olderTicks: older.ticks },
        selfPlay: rate(RL_POP_GROUP.EVAL_C, p),
        diag: this.popDiagnostics[p],
        pool: { valid: this.pools[p].valid, iterations: this.pools[p].iterations.slice(0, this.pools[p].valid) }
      });
    }
    const mixed = this.popAcc.sub[K].summary('eval', this.game);
    const pureRates = roles.map((r) => r.evalPure.rewRate);
    const best = pureRates.reduce((bestIndex, v, i) => (v > pureRates[bestIndex] ? i : bestIndex), 0);
    const baseRate = roles.reduce((sum, r) => sum + r.evalPure.baseRate, 0) / K;
    const summary = { roles, mixed, evalMean: pureRates.reduce((a, b) => a + b, 0) / K, evalBest: pureRates[best], evalBestRole: roles[best].name, evalBase: baseRate, mix: RL_POP_MIX_NAMES[this.pop.mix] };
    for (const group of learner) for (const row of group) row.fill(0);
    for (const accumulator of this.popAcc.sub) accumulator.clearWindow();
    return summary;
  }

  async flushStats() {
    const o = this.opts;
    const encoder = this.device.createCommandEncoder();
    if (this.K > 1) this.dispatch(encoder, 'rl_reduce_pop', 1);
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
    if (this.K > 1) this.absorbPopStats(rows);
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
      this.bestTheta = await this.readFloats('theta', 0, this.K * this.params.count);
    }
  }

  async readStats() {
    await this.flushStats();
    const train = this.stats.summary('train', this.game);
    const evalSummary = this.stats.summary('eval', this.game);
    this.stats.clearWindow();
    await this.trackBestEval();
    const curriculum = this.controller ? await this.applyCurriculum(train, evalSummary) : null;
    const pop = this.K > 1 ? this.popSummary() : null;
    const d = this.diagnostics;
    const archive = { best: Number.isFinite(this.bestEval) ? this.bestEval : 0, mean: d.returnMean / Math.max(1e-6, d.rewardScale), count: this.passes };
    return { tick: this.tick, passes: this.passes, train, eval: evalSummary, archive, worldScores: this.worldScores, stepMs: this.lastStepMs, ticksPerDispatch: this.opts.rolloutTicks, ppo: Object.assign({ windowEval: this.lastWindowEval, bestEval: this.bestEval }, d), league: this.leagueSummary(), curriculum, pop };
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

  async exportPopulation(best) {
    const count = this.params.count;
    const source = best && this.bestTheta ? this.bestTheta : await this.readFloats('theta', 0, this.K * count);
    const base = { fitness: Number.isFinite(this.bestEval) ? this.bestEval : 0, iterations: this.passes, ticks: this.tick };
    const members = [];
    for (let p = 0; p < this.K; p++) {
      const weights = Array.from(this.pop.weights[p]);
      const meta = Object.assign({ role: this.pop.names[p], policy: p, policies: this.K, weights, channels: this.pop.channels }, best ? { best: true } : {}, base);
      members.push({ format: 'npc-population-member/1', index: p, role: this.pop.names[p], weights, channels: this.pop.channels.slice(), fraction: this.pop.fractions[p], genome: rlThetaToGenome(source.subarray(p * count, (p + 1) * count), this.params, this.game, this.opts, meta) });
    }
    return members;
  }

  async exportChampions(n) {
    if (this.K > 1) return this.exportPopulation(false);
    const current = await this.readFloats('theta', 0, this.params.count);
    const meta = { fitness: Number.isFinite(this.bestEval) ? this.bestEval : 0, iterations: this.passes, ticks: this.tick };
    const list = [rlThetaToGenome(current, this.params, this.game, this.opts, meta)];
    if (this.bestTheta && n > 1) list.push(rlThetaToGenome(this.bestTheta, this.params, this.game, this.opts, Object.assign({ best: true }, meta)));
    return list.slice(0, Math.max(1, n));
  }

  async importGenomes(jsonArray) {
    const source = Array.isArray(jsonArray) ? jsonArray : jsonArray.policies;
    const members = source.length > 0 && source[0].genome ? source : null;
    for (let p = 0; p < this.K; p++) {
      const entry = members ? members.find((m) => m.index === p) || members[p % members.length] : null;
      const genome = entry ? entry.genome : source[p % source.length];
      this.writeTheta(rlGenomeToTheta(genome, this.params, this.opts, this.opts.seed + p * 7919), p);
      if (entry && Array.isArray(entry.weights)) {
        entry.weights.forEach((w, c) => { if (c < this.pop.weights[p].length && Number.isFinite(w)) { this.pop.weights[p][c] = w; this.pop.weightsStart[p][c] = w; } });
        if (typeof entry.role === 'string' && entry.role) this.pop.names[p] = entry.role;
      }
    }
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
    const pop = this.K > 1 ? { k: this.K, names: this.pop.names.slice(), weights: this.pop.weights.map((row) => Array.from(row)), weightsStart: this.pop.weightsStart.map((row) => Array.from(row)), weightsEnd: this.pop.weightsEnd ? this.pop.weightsEnd.map((row) => (row ? Array.from(row) : null)) : null, fractions: Array.from(this.pop.fractions), mix: this.pop.mix, pure: this.pop.pure, period: this.pop.period, pools: this.pools.map((p) => ({ valid: p.valid, events: p.events, iterations: p.iterations.slice() })), references: this.references.map((r) => ({ slot: r.slot, iteration: r.iteration })) } : null;
    return { kind: 'gpu', trainer: 'ppo', game: this.game.id, tick: this.tick, passes: this.passes, opts: o, theta: Array.from(theta.subarray(0, count)), pool, opt: Array.from(opt), bestEval: this.bestEval, bestTheta: this.bestTheta ? Array.from(this.bestTheta) : null, curriculum: this.controller ? this.controller.state() : null, pop };
  }

  async restore(obj) {
    if (obj.trainer !== 'ppo' || obj.theta.length !== this.params.count) throw new Error('checkpoint does not match this PPO configuration');
    const savedSoa = obj.opts && obj.opts.soa ? 1 : 0;
    const mineSoa = this.opts.soa ? 1 : 0;
    if (savedSoa !== mineSoa) throw new Error('checkpoint was written with soa=' + savedSoa + ', this backend runs soa=' + mineSoa + ': lane/rec field layout differs, refuse to import');
    this.tick = obj.tick;
    this.passes = obj.passes;
    const savedK = obj.pop ? obj.pop.k : 1;
    if (savedK !== this.K) throw new Error('checkpoint holds ' + savedK + ' policies, this backend trains ' + this.K);
    this.writeTheta(new Float32Array(obj.theta));
    if (obj.pool && obj.pool.floats.length === this.layout.thetaFloats - this.params.count) {
      this.device.queue.writeBuffer(this.buffers.theta, this.params.count * 4, new Float32Array(obj.pool.floats));
      this.pools[0] = { valid: obj.pool.valid, events: obj.pool.events, iterations: obj.pool.iterations.slice() };
      this.pool = this.pools[0];
      this.references[0] = obj.pool.reference;
      this.reference = this.references[0];
    } else if (savedK > 1) throw new Error('checkpoint league pool does not match this configuration (poolSize changed?)');
    if (obj.pop) {
      this.pools = obj.pop.pools.map((p) => ({ valid: p.valid, events: p.events, iterations: p.iterations.slice() }));
      this.pool = this.pools[0];
      this.references = obj.pop.references.map((r) => ({ slot: r.slot, iteration: r.iteration }));
      this.reference = this.references[0];
      this.configurePopulation({ popRoles: obj.pop.names, popWeights: obj.pop.weightsStart || obj.pop.weights, popWeightsEnd: obj.pop.weightsEnd ? obj.pop.weightsEnd.map((row) => (row ? row.join(',') : '')) : '', popFractions: obj.pop.fractions, popMix: obj.pop.mix, popPure: obj.pop.pure, popPeriod: obj.pop.period }, false);
      for (let p = 0; p < this.K; p++) {
        const row = obj.pop.weights && obj.pop.weights[p];
        if (!row) continue;
        for (let c = 0; c < this.pop.weights[p].length; c++) if (Number.isFinite(row[c])) this.pop.weights[p][c] = row[c];
      }
    }
    const saved = new Float32Array(obj.opt);
    if (saved.length === this.layout.optFloats) this.device.queue.writeBuffer(this.buffers.opt, 0, saved);
    else if (saved.length === this.layout.optFloats - RL_POP_WEIGHT_SLOTS && this.K === 1) {
      const widened = new Float32Array(this.layout.optFloats);
      widened.set(saved);
      this.device.queue.writeBuffer(this.buffers.opt, 0, widened);
      this.writePolicyWeights(0);
    } else throw new Error('checkpoint optimiser state has ' + saved.length + ' floats, expected ' + this.layout.optFloats);
    if (this.controller && obj.curriculum) {
      this.controller.restore(obj.curriculum);
      this.curriculum.selfPlay = this.controller.selfPlay;
    }
    this.bestEval = obj.bestEval === null || obj.bestEval === undefined ? -Infinity : obj.bestEval;
    this.bestTheta = obj.bestTheta ? new Float32Array(obj.bestTheta) : null;
    await this.device.queue.onSubmittedWorkDone();
  }

  configurePopulation(partial, writeWeights) {
    const next = rlPopConfig(Object.assign({}, { policies: this.K, popMix: this.pop.mix, popPure: this.pop.pure, popPeriod: this.pop.period, popRoles: this.pop.names, popWeights: this.pop.weightsStart.map((row) => Array.from(row)), popWeightsEnd: this.pop.weightsEnd ? this.pop.weightsEnd.map((row) => (row ? Array.from(row) : '')) : '', popFractions: Array.from(this.pop.fractions) }, partial, { policies: this.K }), this.game.rewardChannels);
    this.pop = next;
    if (writeWeights !== false) for (let p = 0; p < this.K; p++) this.writePolicyWeights(p);
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
    const rawRec = await this.readFloats('rec', 0, L.recFloats);
    const rawLane = await this.readFloats('lane', 0, L.laneFloats);
    const rec = this.opts.soa ? rlRecPermute(L, rawRec, true) : rawRec;
    const lane = this.opts.soa ? rlLanePermute(L, rawLane, true) : rawLane;
    const ctls = [];
    for (let p = 0; p < this.K; p++) ctls.push(await this.readFloats('opt', p * L.optStride + 2 * this.params.count, RL_CTL.SIZE));
    return { theta, obs, rec, lane, ctl: ctls[0], ctls };
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
      const channelCount = (game.rewardChannels || []).length;
      const channelCapFx = game.rewardChannels ? game.rewardChannels.map((_n, c) => (Array.isArray(t.opts.channelCaps) && t.opts.channelCaps[c] !== undefined ? Math.round(t.opts.channelCaps[c] * REWARD_SCALE) : 1073741823)) : [];
      const popWeights = t.pop.weights[0];
      const weightedReward = channelCount > 0 && Array.from(popWeights).some((w) => w !== 1);
      const expectedRewardFx = (env, a) => rlWeightedReward(Array.from(env.rewardCh.subarray(a * channelCount, (a + 1) * channelCount)), popWeights, channelCapFx);
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
              const wantFx = channelCount > 0 ? expectedRewardFx(env, a) : env.reward[a];
              const gotFx = data.rec[sample + R.reward] * REWARD_SCALE;
              if (weightedReward ? Math.abs(gotFx - wantFx) > 1e-3 : gotFx !== wantFx) throw new Error('world ' + w + ' tick ' + tick + ' learner ' + a + ' reward: gpu ' + gotFx + ' js ' + wantFx);
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
    const variants = [{}, { advClipMult: 3, advClipBeta: 0.9 }];
    const parts = [];
    for (const variant of variants) {
    const t = await this.tempBackend({ worlds: 6, hidden: 20, rolloutTicks: 40, gradGroups: 8, rewardScale: 3, rewardClip: 0.5, valueClip: 2, ...variant });
    try {
      for (let i = 0; i < 3; i++) await t.iterate();
      await t.iterate({ update: false });
      const data = await t.readRollout();
      const R = rlRecordLayout(t.params.nOut);
      const steps = t.opts.rolloutTicks;
      const hp = { gamma: t.opts.gamma, lambda: t.opts.lambda, rewardScale: data.ctl[RL_CTL.SCALE], rewardClip: t.opts.rewardClip, valueClip: t.opts.valueClip, advClipMult: t.opts.advClipMult, advClipBeta: t.opts.advClipBeta };
      let clippedRewards = 0;
      let clippedReturns = 0;
      let clippedDeltas = 0;
      let worst = 0;
      let terminals = 0;
      let truncations = 0;
      for (let b = 0; b < t.layout.trainLearners; b++) {
        if (t.roleOf(data, b) !== RL_ROLE.LIVE) continue;
        const trajectory = t.trajectoryOf(data, t.params, b);
        const expected = rlGae(trajectory, hp);
        const plain = hp.advClipMult ? rlGae(trajectory, Object.assign({}, hp, { advClipMult: 0 })) : null;
        for (let k = 0; k < steps; k++) {
          const base = (b * steps + k) * R.stride;
          worst = Math.max(worst, Math.abs(expected.advantage[k] - data.rec[base + R.advantage]), Math.abs(expected.returns[k] - data.rec[base + R.returns]));
          if (plain && plain.advantage[k] !== expected.advantage[k]) clippedDeltas++;
          if (trajectory.flag[k] === 1) terminals++;
          if (trajectory.flag[k] === 2) truncations++;
          if (Math.abs(trajectory.reward[k] * hp.rewardScale) > hp.rewardClip) clippedRewards++;
          if (Math.abs(data.rec[base + R.returns]) >= hp.valueClip - 1e-6) clippedReturns++;
        }
      }
      const detail = 'GAE(' + hp.gamma + ',' + hp.lambda + ', reward clip ' + hp.rewardClip + ' hit ' + clippedRewards + ' times, value target clip ' + hp.valueClip + ' hit ' + clippedReturns + ' times' + (hp.advClipMult ? ', delta clip ' + hp.advClipMult + 'x RMS(beta ' + hp.advClipBeta + ') moved ' + clippedDeltas + ' samples' : '') + ') over ' + t.layout.trainLearners * steps + ' samples (' + terminals + ' deaths, ' + truncations + ' truncations): max error ' + worst.toExponential(2);
      if (!(worst < 1e-4)) throw new Error(detail);
      parts.push(detail);
    } finally {
      await t.dispose();
    }
    }
    return parts.join(' | ');
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
    const variants = [{}, { optimizer: 'obgd' }, { optimizer: 'obgd', obgdBudget: 16 }, { optimizer: 'bounded' }, { entropySign: 1 }];
    const parts = [];
    for (const variant of variants) parts.push(await this.updateParityOnce(variant));
    return parts.join('; ');
  }

  async updateParityOnce(variant) {
    const label = variant.entropySign ? 'entropy sign' : (variant.optimizer === 'obgd' ? 'obgd' + (variant.obgdBudget ? ' budget ' + variant.obgdBudget : '') : (variant.optimizer === 'bounded' ? 'bounded' : 'adam'));
    const t = await this.tempBackend(Object.assign({ worlds: 6, hidden: 20, rolloutTicks: 40, gradGroups: 8, minibatches: 1, epochs: 1, maxGrad: 1e9, lifeCapScale: 0.02 }, variant));
    try {
      for (let i = 0; i < 3; i++) await t.iterate({ update: false });
      const before = await t.readRollout();
      const encoder = t.device.createCommandEncoder();
      t.encodeGradient(encoder);
      await t.submit(encoder);
      const P = t.params;
      const reduced = await t.readFloats('partials', t.layout.gradGroups * P.partialStride, P.partialStride);
      const hp = Object.assign({}, t.opts, { rewardScale: before.ctl[RL_CTL.SCALE], entropy: t.lastEntropy });
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
      const stepHp = Object.assign({}, hp, { lr: t.learningRate(), maxGrad: t.opts.maxGrad });
      rlOptimizerStep(expected, Float64Array.from(reduced.subarray(0, P.count)), m, v, 1, stepHp, P);
      let updateWorst = 0;
      for (let p = 0; p < P.learn; p++) updateWorst = Math.max(updateWorst, Math.abs(expected[p] - after[p]));
      const detail = gradDetail + '; ' + label + ' step max theta error ' + updateWorst.toExponential(2) + ' (lr ' + t.learningRate() + ')';
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
      for (let s = 0; s < pool.valid; s++) {
        let magnitude = 0;
        for (let p = 0; p < P.count; p++) magnitude += Math.abs(slice(s)[p]);
        if (!(magnitude > 0) || !Number.isFinite(magnitude)) throw new Error('pool snapshot ' + s + ' is empty or not finite (sum |theta| ' + magnitude + ')');
      }
      if (data.ctl[RL_CTL.ITER] !== t.passes) throw new Error('only ' + data.ctl[RL_CTL.ITER] + ' of ' + t.passes + ' iterations executed their rollout: the command buffers of snapshot iterations were dropped');
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
      const reference = rlLossAndGrad(data.theta, P, batch, Object.assign({}, t.opts, { entropy: t.lastEntropy }), true);
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
      const capsFx = game.rewardChannels.map(() => limit);
      const weightedReward = Array.from(t.pop.weights[0]).some((w) => w !== 1);
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
            const channelFx = Array.from(env.rewardCh.subarray(a * game.rewardChannels.length, (a + 1) * game.rewardChannels.length));
            const expected = rlWeightedReward(channelFx, t.pop.weights[0], capsFx);
            const sample = ((w * slots + a) * steps + k) * R.stride;
            const recorded = data.rec[sample + R.reward] * REWARD_SCALE;
            if (weightedReward ? Math.abs(recorded - expected) > 1e-3 : recorded !== expected) throw new Error('world ' + w + ' tick ' + k + ' learner ' + a + ': gpu capped reward ' + recorded + ' js ' + expected);
            if (expected !== rlWeightedReward(channelFx, t.pop.weights[0], null)) capped++;
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

  polOf(data, b) {
    return data.lane[b * this.layout.laneStride + RL_LANE.POL];
  }

  async testPopulationRoles() {
    const outcomes = [];
    for (const mixMode of [0, 1, 2]) {
      const opts = { worlds: 40, hidden: 12, rolloutTicks: 8, gradGroups: 4, policies: 3, popMix: mixMode, popPure: 0.4, popPeriod: 1, evalFraction: 0.125, league: 1, leagueEvalFraction: 0.05, poolSize: 2, snapshotEvery: 1 };
      const a = await this.tempBackend(opts);
      const b = await this.tempBackend(opts);
      try {
        for (let i = 0; i < 3; i++) {
          await a.iterate({ update: false });
          await b.iterate({ update: false });
        }
        const epoch = a.passes - 1;
        const dataA = await a.readRollout();
        const dataB = await b.readRollout();
        const slots = a.slots;
        const bounds = { evalStart: a.evalStart, evalBStart: a.evalBStart };
        let checked = 0;
        const perWorld = [];
        for (let w = 0; w < a.opts.worlds; w++) {
          const header = new Uint32Array(await a.readWords(a.buffers.world, w * a.layout.worldStride * 4, 4));
          const effective = this.effectiveSlots(dataA, a, w);
          const counts = new Array(a.K).fill(0);
          for (let l = 0; l < effective; l++) {
            const index = w * slots + l;
            const got = a.polOf(dataA, index);
            if (got !== b.polOf(dataB, index)) throw new Error('mix ' + mixMode + ' world ' + w + ' slot ' + l + ': two backends with one seed assign different policies');
            const expected = rlPopPolicyOf(a.pop, header[0], w, l, effective, bounds, epoch);
            if (expected !== got) throw new Error('mix ' + mixMode + ' world ' + w + ' slot ' + l + ': gpu policy ' + got + ' js reference ' + expected);
            counts[got]++;
            checked++;
          }
          perWorld.push(counts);
        }
        const evalA = perWorld.slice(a.evalStart, a.evalBStart);
        evalA.forEach((counts, i) => {
          const sub = i % (a.K + 1);
          const used = counts.filter((c) => c > 0).length;
          if (sub < a.K && (used !== 1 || counts[sub] === 0)) throw new Error('pure benchmark world ' + (a.evalStart + i) + ' should hold only role ' + sub + ': ' + counts);
          if (sub === a.K && used !== a.K) throw new Error('mixed benchmark world ' + (a.evalStart + i) + ' should hold every role: ' + counts);
        });
        const train = perWorld.slice(0, a.evalStart);
        const roleTotals = new Array(a.K).fill(0);
        for (const counts of train) counts.forEach((c, p) => { roleTotals[p] += c; });
        if (roleTotals.some((c) => c === 0)) throw new Error('mix ' + mixMode + ' leaves a role without training slots: ' + roleTotals);
        if (mixMode === 0) {
          const spread = train.map((counts) => Math.max(...counts) - Math.min(...counts));
          if (Math.max(...spread) > 1) throw new Error('balanced mixing should give each role an equal share per world, spread ' + Math.max(...spread));
        }
        if (mixMode === 2) {
          const pure = train.filter((counts) => counts.filter((c) => c > 0).length === 1).length;
          if (pure === 0 || pure === train.length) throw new Error('archetype mixing produced ' + pure + ' pure worlds out of ' + train.length);
        }
        outcomes.push(['balanced', 'random', 'archetype'][mixMode] + ' ' + checked + ' slots, role totals ' + roleTotals.join('/'));
      } finally {
        await a.dispose();
        await b.dispose();
      }
    }
    return 'GPU role assignment equals the JS reference slot for slot and is identical across two backends with one seed (' + outcomes.join('; ') + '); pure-role benchmark worlds hold one role each, the mixed benchmark world holds all roles';
  }

  effectiveSlots(data, backend, w) {
    let active = 0;
    while (active < backend.slots && data.lane[(w * backend.slots + active) * backend.layout.laneStride + RL_LANE.ROLE] !== RL_ROLE.INACTIVE) active++;
    return active;
  }

  async testPopulationMasking() {
    const t = await this.tempBackend({ worlds: 24, hidden: 20, rolloutTicks: 24, gradGroups: 8, policies: 3, minibatches: 1, epochs: 1, maxGrad: 1e9, lifeCapScale: 0.02, league: 1, leagueFraction: 1, leagueSlotFraction: 0.4, leagueEvalFraction: 0.05, poolSize: 2, snapshotEvery: 1, leaguePeriod: 1, referenceAge: 1 });
    try {
      for (let i = 0; i < 4; i++) await t.iterate();
      await t.iterate({ update: false });
      const P = t.params;
      const data = await t.readRollout();
      const R = rlRecordLayout(P.nOut);
      const steps = t.opts.rolloutTicks;
      const stride = t.layout.partialStride;
      const reduce = async (policy) => {
        const encoder = t.device.createCommandEncoder();
        t.encodeGradient(encoder, policy);
        await t.submit(encoder);
        return t.readFloats('partials', t.layout.gradGroups * stride, stride);
      };
      const compare = (reduced, policy, label) => {
        const batch = t.gradientBatchOf(data, policy);
        const reference = rlLossAndGrad(data.theta.subarray(policy * P.count, (policy + 1) * P.count), P, batch, Object.assign({}, t.opts, { entropy: t.lastEntropy }), true);
        let worst = 0;
        let peak = 0;
        for (let p = 0; p < P.learn; p++) {
          peak = Math.max(peak, Math.abs(reference.grad[p]));
          worst = Math.max(worst, Math.abs(reference.grad[p] - reduced[p]));
        }
        if (!(peak > 0)) throw new Error(label + ': reference gradient is zero, test would be vacuous');
        if (!(worst < 2e-3 * Math.max(peak, 1e-6))) throw new Error(label + ': gradient error ' + worst + ' against peak ' + peak);
        return { worst, peak };
      };
      const gradients = [];
      const roleSamples = [];
      for (let p = 0; p < t.K; p++) {
        let live = 0;
        let snapshot = 0;
        for (let b = 0; b < t.layout.trainLearners; b++) {
          if (t.polOf(data, b) !== p) continue;
          if (t.roleOf(data, b) === RL_ROLE.LIVE) live++;
          else if (t.roleOf(data, b) === RL_ROLE.SNAPSHOT) snapshot++;
        }
        if (live === 0 || snapshot === 0) throw new Error('role ' + p + ' needs live and snapshot learners: ' + live + '/' + snapshot);
        if (Math.abs(data.ctls[p][RL_CTL.ACTIVE] - live * steps) > 0.5) throw new Error('role ' + p + ' advantage statistics cover ' + data.ctls[p][RL_CTL.ACTIVE] + ' samples, expected ' + live * steps);
        roleSamples.push(live + '/' + snapshot);
        const reduced = await reduce(p);
        gradients.push(reduced);
        compare(reduced, p, 'policy ' + p);
      }
      const before = Float32Array.from(gradients[0].subarray(0, P.learn));
      const recBuffer = Float32Array.from(data.rec);
      const obsBuffer = Float32Array.from(data.obs);
      const rng = new Rng(77);
      let corrupted = 0;
      for (let b = 0; b < t.layout.trainLearners; b++) {
        if (t.polOf(data, b) === 0) continue;
        corrupted++;
        for (let k = 0; k < steps; k++) {
          const base = (b * steps + k) * R.stride;
          recBuffer[base + R.advantage] = (rng.next() - 0.5) * 40;
          recBuffer[base + R.returns] = (rng.next() - 0.5) * 40;
          recBuffer[base + R.logProb] += (rng.next() - 0.5) * 3;
          for (let c = 0; c < P.nOut; c++) recBuffer[base + R.action + c] = (rng.next() - 0.5) * 4;
        }
        const ob = t.obsBase(b);
        for (let i = 0; i < t.layout.obsLearnerStride; i++) obsBuffer[ob + i] = (rng.next() - 0.5) * 6;
      }
      t.device.queue.writeBuffer(t.buffers.rec, 0, t.opts.soa ? rlRecPermute(t.layout, recBuffer, false) : recBuffer);
      t.device.queue.writeBuffer(t.buffers.obs, 0, obsBuffer);
      const after = await reduce(0);
      for (let p = 0; p < P.learn; p++) if (!Object.is(after[p], before[p])) throw new Error('policy 0 gradient changed at ' + p + ' after corrupting the samples of the other roles: ' + before[p] + ' -> ' + after[p]);
      const other = await reduce(1);
      let moved = 0;
      for (let p = 0; p < P.learn; p++) if (!Object.is(other[p], gradients[1][p])) moved++;
      if (moved === 0) throw new Error('corrupting the samples of other roles did not change the gradient of role 1: the corruption was not applied');
      const thetaBefore = await t.readFloats('theta', 0, t.layout.thetaFloats);
      const optBefore = await t.readFloats('opt', 0, t.layout.optFloats);
      const encoder = t.device.createCommandEncoder();
      t.dispatch(encoder, 'rl_adam', 1, 1);
      await t.submit(encoder);
      const thetaAfter = await t.readFloats('theta', 0, t.layout.thetaFloats);
      const optAfter = await t.readFloats('opt', 0, t.layout.optFloats);
      let touched = 0;
      for (let i = 0; i < thetaAfter.length; i++) {
        if (Object.is(thetaAfter[i], thetaBefore[i])) continue;
        touched++;
        if (i < P.count || i >= 2 * P.count) throw new Error('Adam of role 1 changed theta[' + i + '] outside slice 1');
      }
      for (let i = 0; i < optAfter.length; i++) if (!Object.is(optAfter[i], optBefore[i]) && (i < t.layout.optStride || i >= 2 * t.layout.optStride)) throw new Error('Adam of role 1 changed optimiser state ' + i + ' outside its own block');
      const block = t.layout.optStride;
      const m = Float64Array.from(optBefore.subarray(block, block + P.count));
      const v = Float64Array.from(optBefore.subarray(block + P.count, block + 2 * P.count));
      const stepIndex = optBefore[block + 2 * P.count + RL_CTL.STEP] + 1;
      const expected = Float64Array.from(thetaBefore.subarray(P.count, 2 * P.count));
      rlOptimizerStep(expected, Float64Array.from(other.subarray(0, P.count)), m, v, stepIndex, Object.assign({}, t.opts, { lr: t.learningRate() }), P);
      let worst = 0;
      for (let p = 0; p < P.learn; p++) worst = Math.max(worst, Math.abs(expected[p] - thetaAfter[P.count + p]));
      if (!(worst < Math.max(2e-6, t.learningRate() * 0.02))) throw new Error('role 1 Adam step differs from the JS reference by ' + worst);
      return t.K + ' roles with live/snapshot learners ' + roleSamples.join(' ') + ': each role gradient matches the JS reference over only its own live samples; rewriting the advantages, returns, actions and observations of ' + corrupted + ' learners of the other roles leaves the gradient of role 0 bit for bit identical; Adam of role 1 changes ' + touched + ' parameters, all inside its own theta slice and optimiser block (JS reference error ' + worst.toExponential(2) + ')';
    } finally {
      await t.dispose();
    }
  }

  gradientBatchOf(data, policy) {
    const steps = this.opts.rolloutTicks;
    const P = this.params;
    const learners = this.layout.trainLearners;
    const selected = (b) => this.roleOf(data, b) === RL_ROLE.LIVE && this.polOf(data, b) === policy;
    const obs = new Float64Array(learners * (steps + 1) * P.nIn);
    const hInit = new Float64Array(learners * P.hidden);
    for (let b = 0; b < learners; b++) {
      for (let j = 0; j <= steps; j++) obs.set(this.obsRow(data, b, j), (b * (steps + 1) + j) * P.nIn);
      if (P.recurrent) hInit.set(this.hiddenInit(data, b), b * P.hidden);
    }
    const ctl = data.ctls[policy];
    return { steps, learners, obs, rec: data.rec, hInit: P.recurrent ? hInit : null, normMean: ctl[RL_CTL.ADV_MEAN], normInvStd: ctl[RL_CTL.ADV_INV_STD], selected, divisor: ctl[RL_CTL.ACTIVE] / Math.max(1, Math.round(this.opts.minibatches)) };
  }

  async testPopulationEquivalence() {
    const common = { worlds: 20, hidden: 16, rolloutTicks: 16, gradGroups: 8, league: 1, leagueFraction: 1, leagueSlotFraction: 0.5, leagueEvalFraction: 0.05, poolSize: 3, snapshotEvery: 2, leaguePeriod: 1, referenceAge: 2, evalFraction: 0.2 };
    const single = await this.tempBackend(common);
    const duo = await this.tempBackend(Object.assign({}, common, { policies: 2, popFractions: '1/0' }));
    try {
      for (let i = 0; i < 6; i++) {
        await single.iterate();
        await duo.iterate();
      }
      const count = single.params.count;
      const a = await single.readRollout();
      const b = await duo.readRollout();
      let differing = 0;
      for (let i = 0; i < count; i++) if (!Object.is(a.theta[i], b.theta[i])) differing++;
      if (differing > 0) throw new Error(differing + ' live parameters of the single policy differ from policy 0 of a two-policy backend that gives all samples to policy 0');
      for (let s = 0; s < 3 * count; s++) if (!Object.is(a.theta[count + s], b.theta[2 * count + s])) throw new Error('snapshot pool of policy 0 differs at ' + s);
      for (const key of ['rec', 'obs']) for (let i = 0; i < a[key].length; i++) if (!Object.is(a[key][i], b[key][i])) throw new Error(key + ' differs at ' + i);
      const initial = rlInitTheta(duo.params, (duo.opts.seed + 7919) * 2654435761, duo.opts);
      for (let i = 0; i < count; i++) if (!Object.is(initial[i], b.theta[count + i])) throw new Error('policy 1 received updates without samples (parameter ' + i + ')');
      const optA = await single.readFloats('opt', 0, single.layout.optStride);
      const optB = await duo.readFloats('opt', 0, duo.layout.optStride);
      for (let i = 0; i < optA.length; i++) if (!Object.is(optA[i], optB[i])) throw new Error('optimiser state of policy 0 differs at ' + i);
      return '6 full iterations with league snapshots: a two-policy backend where policy 1 holds no slots reproduces the single-policy theta (' + count + ' parameters), snapshot pool, rollout records (' + a.rec.length + ' floats), observations and Adam state bit for bit, and leaves policy 1 untouched';
    } finally {
      await single.dispose();
      await duo.dispose();
    }
  }

  async testPopulationRewards() {
    const game = this.game;
    if (!game.rewardChannels || !/g_reward_channel/.test(game.wgsl)) return 'not applicable: this game exposes no reward channels';
    const channels = game.rewardChannels.length;
    const weights = [[1, 1, 1, 1], [0.5, 0.5, 2, 0.5], [0.5, 2, 0.5, 1], [2, 1, 0.3, 2]].map((row) => row.slice(0, channels));
    const ends = [[2, 1, 1, 1], [1, 1.5, 1, 1], [1, 1, 1.5, 0.5]].map((row) => row.slice(0, channels));
    const caps = game.rewardChannels.map((_, c) => (c === 1 ? 0.004 : 1000));
    const steps = 40;
    const t = await this.tempBackend({ worlds: 12, hidden: 12, rolloutTicks: steps, gradGroups: 4, lifeCapScale: 0.02, obsNorm: 0, policies: 3, popWeights: weights.slice(0, 3).map((row) => row.join(',')).join('/'), popWeightsEnd: ends.map((row) => row.join(',')).join('/'), lrDecayIterations: 5, channelCaps: caps });
    try {
      const P = t.params;
      const R = rlRecordLayout(P.nOut);
      const slots = t.slots;
      const trainAge = Math.max(1, Math.round(game.maxAge * 0.02));
      const limit = caps.map((c) => Math.round(c * REWARD_SCALE));
      const startWeights = t.pop.weightsStart.map((row) => Array.from(row));
      const endWeights = t.pop.weightsEnd.map((row) => Array.from(row));
      const compareWindow = async (worldCount) => {
        const data = await t.readRollout();
        let compared = 0;
        let worst = 0;
        let capped = 0;
        let generalistExact = 0;
        let differsFromStart = 0;
        const perRole = new Array(t.K).fill(0);
        for (let w = 0; w < Math.min(t.evalStart, worldCount); w++) {
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
            const pol = t.polOf(data, w * slots + a);
            const channelFx = Array.from(env.rewardCh.subarray(a * channels, (a + 1) * channels));
            const expected = rlWeightedReward(channelFx, t.pop.weights[pol], limit);
            const sample = ((w * slots + a) * steps + k) * R.stride;
            const recorded = data.rec[sample + R.reward] * REWARD_SCALE;
            const error = Math.abs(recorded - expected);
            worst = Math.max(worst, error);
            if (!(error <= 1e-3 * Math.max(1, Math.abs(expected)))) throw new Error('world ' + w + ' tick ' + k + ' learner ' + a + ' role ' + pol + ': gpu weighted reward ' + recorded + ' js ' + expected);
            if (channelFx[1] > limit[1]) capped++;
            if (pol === 0) {
              let clamped = 0;
              for (let c = 0; c < channels; c++) clamped += Math.max(-limit[c], Math.min(limit[c], channelFx[c]));
              if (recorded === clamped) generalistExact++;
            }
            perRole[pol]++;
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
        return { compared, worst, capped, generalistExact, perRole, differsFromStart };
      };
      await t.iterate({ update: false });
      for (let p = 0; p < t.K; p++) {
        for (let c = 0; c < channels; c++) if (Math.abs(t.pop.weights[p][c] - startWeights[p][c]) > 1e-12) throw new Error('channel weight ' + c + ' of role ' + p + ' is ' + t.pop.weights[p][c] + ' before the schedule moved, expected ' + startWeights[p][c]);
      }
      const first = await compareWindow(6);
      const startRows = weights.slice(0, t.K).map((row) => row.join(',')).join('/');
      const endRows = ends.map((row) => row.join(',')).join('/');
      const variant = async (w) => this.tempBackend({ worlds: 12, hidden: 12, rolloutTicks: steps, gradGroups: 4, lifeCapScale: 0.02, obsNorm: 0, policies: 3, popWeights: w, channelCaps: caps });
      const target = await variant(endRows);
      const origin = await variant(startRows);
      let matched = 0;
      let moved = 0;
      try {
        const total = Math.max(1, t.opts.lrDecayIterations);
        for (let i = 0; i < total; i++) await t.iterate({ update: false });
        for (let i = 0; i <= total; i++) {
          await target.iterate({ update: false });
          await origin.iterate({ update: false });
        }
        const a = await t.readRollout();
        const e = await target.readRollout();
        const s = await origin.readRollout();
        for (let b = 0; b < t.layout.trainLearners; b++) {
          if (t.roleOf(a, b) === RL_ROLE.INACTIVE) continue;
          for (let k = 0; k < steps; k++) {
            const at = (b * steps + k) * R.stride + R.reward;
            if (a.rec[at] !== e.rec[at]) throw new Error('learner ' + b + ' tick ' + k + ': after the anneal the GPU recorded ' + a.rec[at] + ', a backend pinned at the target weights records ' + e.rec[at]);
            if (a.rec[at] !== s.rec[at]) moved++;
            matched++;
          }
        }
        if (moved === 0) throw new Error('the anneal changes no recorded reward, test is vacuous');
      } finally {
        await target.dispose();
        await origin.dispose();
      }
      for (let p = 0; p < t.K; p++) {
        if (!Array.from(startWeights[p]).some((w, c) => Math.abs(w - endWeights[p][c]) > 1e-9)) throw new Error('role ' + p + ' anneals to its own start weights, test is vacuous');
        for (let c = 0; c < channels; c++) if (Math.abs(t.pop.weights[p][c] - endWeights[p][c]) > 1e-6) throw new Error('channel weight ' + c + ' of role ' + p + ' is ' + t.pop.weights[p][c] + ' after the anneal, expected the target ' + endWeights[p][c]);
      }
      if (first.perRole.some((c) => c === 0)) throw new Error('some role has no compared samples: ' + first.perRole);
      if (first.capped === 0) throw new Error('the progress cap never triggered, test is vacuous');
      return first.compared + ' learner ticks over ' + t.K + ' roles (' + first.perRole.join('/') + ' per role, weights ' + startRows + ', progress channel capped at 0.004 ' + first.capped + ' times): per-sample reward equals sum of w[c] * clamp(channel c) in the JS env replay, max error ' + first.worst.toExponential(2) + ' fixed-point units; the unit-weight role reproduces the capped channel total exactly on ' + first.generalistExact + ' ticks; annealing to ' + endRows + ' moves ' + moved + ' of ' + matched + ' recorded rewards away from the start weighting and every one of them matches a backend pinned at the target weights';
    } finally {
      await t.dispose();
    }
  }

  async testPopulationTraining() {
    const t = await this.tempBackend({ worlds: 20, hidden: 16, rolloutTicks: 16, gradGroups: 8, policies: 3, evalFraction: 0.25, league: 1, leagueFraction: 0.5, leagueEvalFraction: 0.1, poolSize: 2, snapshotEvery: 2, leaguePeriod: 1, referenceAge: 2, epochs: 2, minibatches: 2, obsNorm: 1 });
    try {
      for (let i = 0; i < 10; i++) await t.step();
      await t.flushStats();
      const weval = Float64Array.from(t.stats.window.eval);
      let rewardSum = 0;
      let tickSum = 0;
      for (const accumulator of t.popAcc.sub) {
        rewardSum += accumulator.window.eval[STAT.REW_EVO];
        tickSum += accumulator.window.eval[STAT.TICKS_EVO];
      }
      if (rewardSum !== weval[STAT.REW_EVO] || tickSum !== weval[STAT.TICKS_EVO]) throw new Error('benchmark subgroups (' + rewardSum + ' reward, ' + tickSum + ' ticks) do not add up to the eval group (' + weval[STAT.REW_EVO] + ', ' + weval[STAT.TICKS_EVO] + ')');
      let learnerTicks = 0;
      let learnerReward = 0;
      for (const g of [RL_POP_GROUP.EVAL_PURE, RL_POP_GROUP.EVAL_MIXED]) for (const row of t.popAcc.learner[g]) {
        learnerReward += row[0];
        learnerTicks += row[1];
      }
      if (learnerReward !== weval[STAT.REW_EVO] || learnerTicks !== weval[STAT.TICKS_EVO]) throw new Error('per-role learner accumulators (' + learnerReward + ', ' + learnerTicks + ') do not add up to the eval group');
      const s = await t.readStats();
      const pop = s.pop;
      if (!pop || pop.roles.length !== 3) throw new Error('no per-role summary');
      const theta = await t.readFloats('theta', 0, t.layout.thetaFloats);
      for (let i = 0; i < theta.length; i++) if (!Number.isFinite(theta[i])) throw new Error('theta[' + i + '] not finite');
      const checks = [];
      for (const role of pop.roles) {
        const d = role.diag;
        if (![d.gradNorm, d.policyLoss, d.valueLoss, d.entropy, d.kl, d.rewardScale].every(Number.isFinite)) throw new Error('role ' + role.name + ' has non finite diagnostics ' + JSON.stringify(d));
        if (!(d.samples > 0)) throw new Error('role ' + role.name + ' trained on no samples');
        checks.push(role.name + ' ' + d.samples + ' samples, scale ' + d.rewardScale.toFixed(2));
      }
      const thetaMoved = pop.roles.map((_, p) => {
        let moved = 0;
        const init = rlInitTheta(t.params, (t.opts.seed + p * 7919) * 2654435761, t.opts);
        for (let i = 0; i < t.params.learn; i++) if (init[i] !== theta[p * t.params.count + i]) moved++;
        return moved;
      });
      if (thetaMoved.some((m) => m === 0)) throw new Error('some role never left its initialisation: ' + thetaMoved);
      const distinct = pop.roles.map((_, p) => {
        let diff = 0;
        for (let i = 0; i < t.params.w1 + 64; i++) if (theta[p * t.params.count + i] !== theta[((p + 1) % 3) * t.params.count + i]) diff++;
        return diff;
      });
      if (distinct.some((d) => d === 0)) throw new Error('two roles hold identical weights');
      return '10 iterations of 3 roles x ' + t.opts.worlds + ' worlds with league, obsNorm, 2 epochs x 2 minibatches: finite, each role trained (' + checks.join('; ') + '); benchmark subgroups and per-role learner accumulators add up exactly to the eval group (' + weval[STAT.TICKS_EVO] + ' ticks)';
    } finally {
      await t.dispose();
    }
  }

  async testPopulationCheckpoint() {
    const common = { worlds: 20, hidden: 16, rolloutTicks: 16, gradGroups: 8, policies: 3, evalFraction: 0.25, league: 1, leagueFraction: 0.5, leagueEvalFraction: 0.1, poolSize: 3, snapshotEvery: 2, leaguePeriod: 1, referenceAge: 2, popWeights: '1,1,1,1/0.7,0.9,1.6,0.5/0.3,1.8,0.6,1.1', popWeightsEnd: '1,1,1,1/1,1,1,1/1,1,1,1', lrDecayIterations: 4, popRoles: 'alpha,beta,gamma', popFractions: '2/1/1', popMix: 1 };
    const t = await this.tempBackend(common);
    const u = await this.tempBackend(Object.assign({}, common, { seed: 99, popWeights: '', popRoles: '', popFractions: '', popMix: 0 }));
    try {
      for (let i = 0; i < 5; i++) await t.step();
      const ckpt = JSON.parse(JSON.stringify(await t.checkpoint()));
      await u.restore(ckpt);
      const a = await t.readFloats('theta', 0, t.layout.thetaFloats);
      const b = await u.readFloats('theta', 0, u.layout.thetaFloats);
      for (let i = 0; i < a.length; i++) if (!Object.is(a[i], b[i])) throw new Error('checkpoint theta differs at ' + i + ' (slice ' + Math.floor(i / t.params.count) + ')');
      const oa = await t.readFloats('opt', 0, t.layout.optFloats);
      const ob = await u.readFloats('opt', 0, u.layout.optFloats);
      for (let i = 0; i < oa.length; i++) if (!Object.is(oa[i], ob[i])) throw new Error('checkpoint optimiser state differs at ' + i);
      if (u.pop.names.join() !== 'alpha,beta,gamma' || u.pop.mix !== 1 || Math.abs(u.pop.fractions[0] - 0.5) > 1e-9) throw new Error('role names, mixing or fractions not restored: ' + u.pop.names + ' ' + u.pop.mix + ' ' + u.pop.fractions);
      for (let p = 0; p < t.K; p++) {
        if (u.pools[p].valid !== t.pools[p].valid || u.pools[p].events !== t.pools[p].events || u.pools[p].iterations.join() !== t.pools[p].iterations.join()) throw new Error('league pool of role ' + p + ' not restored');
        if (u.references[p].slot !== t.references[p].slot) throw new Error('reference snapshot of role ' + p + ' not restored');
        const w = Array.from(u.pop.weights[p]);
        if (w.join() !== Array.from(t.pop.weights[p]).join()) throw new Error('weights of role ' + p + ' not restored: ' + w);
      }
      let annealed = 0;
      if (t.pop.weightsEnd) {
        if (!u.pop.weightsEnd) throw new Error('the anneal targets were not restored');
        for (let p = 0; p < t.K; p++) for (let c = 0; c < t.pop.channels.length; c++) {
          if (Math.abs(u.pop.weightsEnd[p][c] - t.pop.weightsEnd[p][c]) > 1e-12) throw new Error('anneal target ' + c + ' of role ' + p + ' not restored: ' + u.pop.weightsEnd[p][c]);
          if (Math.abs(t.pop.weights[p][c] - t.pop.weightsStart[p][c]) > 1e-6) annealed++;
        }
        if (annealed === 0) throw new Error('the channel weights never moved during training, anneal test is vacuous');
      }
      await u.iterate();
      const resumed = await u.readFloats('theta', 0, u.layout.thetaFloats);
      for (let i = 0; i < resumed.length; i++) if (!Number.isFinite(resumed[i])) throw new Error('training after the restore produced a non finite parameter at ' + i);
      const bundle = JSON.parse(JSON.stringify(await t.exportPopulation(false)));
      if (!Array.isArray(bundle) || bundle.length !== 3 || bundle.some((m, p) => m.index !== p || m.format !== 'npc-population-member/1' || !m.genome || m.genome.dims.nIn !== t.params.nIn)) throw new Error('population bundle malformed');
      const v = await this.tempBackend(Object.assign({}, common, { seed: 7, popWeights: '', popRoles: '' }));
      try {
        await v.importGenomes(bundle);
        const imported = await v.readFloats('theta', 0, 3 * v.params.count);
        const original = await t.readFloats('theta', 0, 3 * t.params.count);
        let worst = 0;
        for (let p = 0; p < 3; p++) for (let i = 0; i < t.params.learn; i++) worst = Math.max(worst, Math.abs(imported[p * t.params.count + i] - original[p * t.params.count + i]));
        if (!(worst <= 1e-5)) throw new Error('bundle import differs by ' + worst);
        if (v.pop.names.join() !== 'alpha,beta,gamma') throw new Error('bundle import lost the role names');
        for (let p = 0; p < 3; p++) if (Array.from(v.pop.weights[p]).join() !== Array.from(t.pop.weights[p]).join()) throw new Error('bundle import lost the weights of role ' + p);
      } finally {
        await v.dispose();
      }
      return 'checkpoint after 5 iterations restores all ' + t.K + ' live policies, ' + t.K * t.poolSize + ' league snapshots, optimiser states (' + oa.length + ' floats), roles, weights, fractions, mixing and reference snapshots exactly and training resumes finite' + (t.pop.weightsEnd ? ', the anneal targets survive the round trip while ' + annealed + ' channel weights moved during training' : '') + '; the exported population bundle (' + bundle.length + ' members) imports back with its roles and weights (parameters within 1e-5)';
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
    await record('population: role assignment, GPU vs JS reference', () => this.testPopulationRoles());
    await record('population: per-role masked gradients and separate Adam updates', () => this.testPopulationMasking());
    await record('population: two policies with one idle equal the single-policy path', () => this.testPopulationEquivalence());
    await record('population: channel-weighted rewards, GPU vs JS env replay', () => this.testPopulationRewards());
    await record('population: training, per-role statistics and benchmark subgroups', () => this.testPopulationTraining());
    await record('population: checkpoint, restore and bundle import of all policies', () => this.testPopulationCheckpoint());
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
    for (let s = 0; s < steps; s++) plan.push(...(this.layout.blockGrad ? [['rl_select', 1]] : []), [this.gradEntry(), this.layout.gradGroups], ['rl_reduce_grad', Math.ceil(this.params.partialStride / 256)], ['rl_adam', 1]);
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
      grad: await perCall(this.gradEntry(), this.layout.gradGroups),
      reduce: await perCall('rl_reduce_grad', Math.ceil(P.partialStride / 256)),
      adam: await perCall('rl_adam', 1)
    };
  }

  async benchmark(progress) {
    const rows = [];
    const cap = rlWorldCapacity(this.ctx, this.game, this.opts);
    for (const worlds of [256, 512, 1024, 2048, 4096]) {
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
