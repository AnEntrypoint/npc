function appMain() {
  const $ = (id) => document.getElementById(id);
  const COLORS = { evo: '#4cc9f0', bot: '#f4a261', eval: '#80ed99', violet: '#b794f6', gold: '#ffd166', grid: '#252d3f', text: '#8592ab' };
  const TILE_COLORS = ['#16301f', '#4a5266', '#c2506b', '#d8b445'];
  const FACE_VECTORS = [[0, -1], [0, 1], [-1, 0], [1, 0]];
  const HISTORY_LIMIT = 4000;
  const CHECKPOINT_MS = 60000;
  const STATS_MS = 500;
  const DRAW_MS = 100;
  const DB_NAME = 'npc-brain-trainer';
  const DB_STORE = 'checkpoints';
  const COUNTER_KEYS = ['rewRate', 'baseRate', 'opsPerTick', 'edgesMean', 'births', 'deaths', 'lifeFit'];
  const TILE_PALETTE = ['#16301f', '#4a5266', '#c2506b', '#d8b445', '#4c8fd8', '#8f5cd8', '#5cd8a8', '#d85c5c', '#8a8a8a', '#a0522d', '#2e8b57', '#b8860b', '#5f9ea0', '#cd853f', '#708090', '#ffffff'];
  const HISTORY_KEYS = ['tick', 'evo', 'bot', 'evalEvo', 'evalBot', 'ops', 'edges', 'archBest', 'archMean'];
  const PPO_FIELDS = [['ppoHidden', 'hidden'], ['ppoRollout', 'rolloutTicks'], ['ppoLr', 'lr'], ['ppoLrEnd', 'lrEnd'], ['ppoEntropy', 'entropy'], ['ppoClip', 'clip'], ['ppoEpochs', 'epochs'], ['ppoMinibatches', 'minibatches'], ['ppoGamma', 'gamma'], ['ppoLambda', 'lambda'], ['ppoLeague', 'league'], ['ppoSnapshot', 'snapshotEvery'], ['ppoRecurrent', 'recurrent'], ['ppoBias', 'bias'], ['ppoObsNorm', 'obsNorm'], ['ppoLearnLeak', 'learnLeak'], ['ppoLeakTau', 'leakTauMax'], ['ppoRewardClip', 'rewardClip'], ['ppoValueClip', 'valueClip'], ['ppoCurriculum', 'curriculum'], ['ppoCurDiff', 'curriculumDifficultyStart'], ['ppoCurStep', 'curriculumDifficultyStep'], ['ppoCurUp', 'curriculumLifeUp'], ['ppoCurDown', 'curriculumLifeDown'], ['ppoCurSelf', 'curriculumSelfPlay'], ['ppoCurSpStart', 'curriculumSelfPlayStart'], ['ppoCurSpEnd', 'curriculumSelfPlayEnd'], ['ppoPolicies', 'policies'], ['ppoPopMix', 'popMix'], ['ppoPopPure', 'popPure'], ['ppoPopPeriod', 'popPeriod']];
  const PPO_LIVE_KEYS = ['lr', 'lrEnd', 'entropy', 'clip', 'epochs', 'minibatches'];
  const PPO_WORLDS = 512;
  const POP_COLORS = ['#4cc9f0', '#f4a261', '#80ed99', '#b794f6', '#ffd166', '#f07178', '#5cd8a8', '#cd853f'];

  const currentGame = () => GAMES[$('game').value];
  const gpuAvailable = () => typeof navigator !== 'undefined' && !!navigator.gpu && typeof GpuBackend !== 'undefined';
  const gpuUsable = () => gpuAvailable() && !!currentGame().wgsl;
  const wantsPpo = () => $('trainer').value === 'ppo';

  const app = {
    backend: null,
    running: false,
    loopActive: false,
    busy: false,
    chain: Promise.resolve(),
    stats: null,
    snapshot: null,
    world: 0,
    history: null,
    optsUsed: null,
    throughput: [],
    lastCheckpoint: 0,
    heat: { canvas: document.createElement('canvas'), cols: 1, min: 0, max: 1 },
    pendingDraw: false,
    watch: { active: false, starting: false },
  };

  const newHistory = () => Object.fromEntries(HISTORY_KEYS.map((k) => [k, []]));
  app.history = newHistory();

  const fmt = (v, digits = 3) => {
    if (v === undefined || v === null || Number.isNaN(v)) return '-';
    if (typeof v !== 'number') return String(v);
    const a = Math.abs(v);
    if (a >= 1e9) return (v / 1e9).toFixed(2) + 'G';
    if (a >= 1e6) return (v / 1e6).toFixed(2) + 'M';
    if (a >= 1e4) return (v / 1e3).toFixed(1) + 'k';
    if (Number.isInteger(v)) return String(v);
    return v.toFixed(digits);
  };

  const log = (text) => {
    $('output').textContent = typeof text === 'string' ? text : JSON.stringify(text, replacerForJson, 2);
  };

  const replacerForJson = (key, value) => {
    if (ArrayBuffer.isView(value)) return Array.from(value);
    if (value instanceof Set) return Array.from(value);
    if (typeof value === 'bigint') return Number(value);
    return value;
  };

  const showError = (message) => {
    const el = $('errorBanner');
    el.textContent = message;
    el.hidden = !message;
  };

  const errorText = (e) => (e && e.stack ? String(e.stack).split('\n').slice(0, 4).join(' | ') : String(e));

  const lock = (fn) => {
    const run = app.chain.then(fn, fn);
    app.chain = run.catch(() => {});
    return run;
  };

  const yieldUi = (() => {
    const channel = new MessageChannel();
    const waiting = [];
    channel.port1.onmessage = () => {
      const resolve = waiting.shift();
      if (resolve) resolve();
    };
    return () => new Promise((resolve) => {
      waiting.push(resolve);
      channel.port2.postMessage(0);
    });
  })();

  const num = (id, fallback) => {
    const v = parseFloat($(id).value);
    return Number.isFinite(v) ? v : fallback;
  };

  const readOptions = () => {
    const cpu = usingCpu();
    const worlds = Math.max(cpu ? 1 : 64, Math.min(4096, Math.round(num('worlds', 256))));
    const islands = Math.max(1, Math.min(worlds, Math.round(num('islands', 1))));
    $('worlds').value = worlds;
    $('islands').value = islands;
    const ppo = {};
    for (const [id, key] of PPO_FIELDS) ppo[key] = num(id, key in RL_DEFAULTS ? RL_DEFAULTS[key] : RL_POP_DEFAULTS[key]);
    const dims = currentGame().dims;
    ppo.hidden = Math.max(8, Math.min(256 - dims.nIn - dims.nOut, Math.round(ppo.hidden)));
    $('ppoHidden').value = ppo.hidden;
    ppo.rolloutTicks = Math.max(4, Math.min(256, Math.round(ppo.rolloutTicks)));
    for (const key of ['recurrent', 'bias', 'obsNorm', 'learnLeak', 'curriculum', 'curriculumSelfPlay']) ppo[key] = ppo[key] > 0 ? 1 : 0;
    for (const key of ['rewardClip', 'valueClip']) ppo[key] = ppo[key] > 0 ? ppo[key] : 1e9;
    ppo.policies = Math.max(1, Math.min(RL_POP_MAX, Math.round(ppo.policies)));
    $('ppoPolicies').value = ppo.policies;
    return {
      ...ppo,
      ...readPopEditor(),
      trainer: wantsPpo() ? 'ppo' : 'evolution',
      game: currentGame(),
      worlds,
      islands,
      maxEdges: Math.max(MIN_EDGES, Math.round(num('maxEdges', 256))),
      ticksPerDispatch: $('tpdAuto').checked ? 0 : Math.max(1, Math.round(num('tpd', 32))),
      mutation: num('mutation', 1),
      penaltyStart: num('penaltyStart', 0),
      penaltyEnd: num('penaltyEnd', 0.05),
      penaltyRampTicks: Math.max(1, num('penaltyRampTicks', 200000)),
      difficultyStart: Math.max(0, Math.min(100, num('difficultyStart', 100))),
      difficultyEnd: Math.max(0, Math.min(100, num('difficultyEnd', 100))),
      difficultyRampTicks: Math.max(1, num('difficultyRampTicks', 300000)),
      randomize: $('randomize').checked,
      evalFraction: Math.max(0, Math.min(1, num('evalFraction', 0.125))),
      seed: Math.round(num('seed', 1)),
      migrateEvery: Math.max(1, Math.round(num('migrateEvery', 8))),
      migrateCount: Math.max(0, Math.round(num('migrateCount', 4))),
      selectEvery: Math.max(1, Math.round(num('selectEvery', 1))),
      evalEvery: Math.max(0, Math.round(num('evalEvery', 1))),
    };
  };

  const gameChannels = () => currentGame().rewardChannels || [];

  const popEditorState = () => {
    const rows = Array.from(document.querySelectorAll('#popEditor tr[data-role]'));
    return rows.map((tr) => ({ name: tr.querySelector('.pop-name').value.trim(), fraction: parseFloat(tr.querySelector('.pop-fraction').value), weights: Array.from(tr.querySelectorAll('.pop-weight')).map((input) => parseFloat(input.value)) }));
  };

  const readPopEditor = () => {
    const k = Math.max(1, Math.min(RL_POP_MAX, Math.round(num('ppoPolicies', 1))));
    const rows = popEditorState();
    if (rows.length !== k) return { popRoles: '', popWeights: '', popFractions: '' };
    return { popRoles: rows.map((r) => r.name).join(','), popWeights: rows.map((r) => r.weights.map((w) => (Number.isFinite(w) ? w : 1)).join(',')).join('/'), popFractions: rows.map((r) => (Number.isFinite(r.fraction) && r.fraction >= 0 ? r.fraction : 1)).join('/') };
  };

  const renderPopEditor = (config) => {
    const k = Math.max(1, Math.min(RL_POP_MAX, Math.round(num('ppoPolicies', 1))));
    const previous = config ? config.names.map((name, p) => ({ name, fraction: config.fractions[p] * config.k, weights: Array.from(config.weights[p]) })) : popEditorState();
    const channels = gameChannels();
    const defaults = rlPopConfig({ policies: k }, channels);
    const head = '<tr><th>policy</th><th>role</th><th>share</th>' + channels.map((c) => '<th>' + escapeHtml(c) + '</th>').join('') + '</tr>';
    const body = Array.from({ length: k }, (_, p) => {
      const old = previous[p];
      const name = old && old.name ? old.name : defaults.names[p];
      const fraction = old && Number.isFinite(old.fraction) ? old.fraction : 1;
      const weights = channels.map((_, c) => (old && Number.isFinite(old.weights[c]) ? old.weights[c] : defaults.weights[p][c]));
      return '<tr data-role="' + p + '"><td>' + p + '</td><td><input class="pop-name" type="text" value="' + escapeHtml(name) + '"></td><td><input class="pop-fraction" type="number" min="0" step="0.1" value="' + fraction + '"></td>' + weights.map((w) => '<td><input class="pop-weight" type="number" min="0" step="0.1" value="' + w + '"></td>').join('') + '</tr>';
    }).join('');
    $('popEditor').innerHTML = k > 1 ? '<table>' + head + body + '</table>' : '';
    $('popEditor').hidden = k < 2;
  };

  const applyPopulation = () => {
    if (!app.backend || !app.backend.configurePopulation || app.backend.K < 2) return;
    if (popEditorState().length !== app.backend.K) {
      showError('The role table holds ' + popEditorState().length + ' policies but the running backend trains ' + app.backend.K + ': press Reset to change the number of policies.');
      return;
    }
    try {
      const o = readOptions();
      app.backend.configurePopulation({ popRoles: o.popRoles, popWeights: o.popWeights, popFractions: o.popFractions, popMix: o.popMix, popPure: o.popPure, popPeriod: o.popPeriod });
      log('Population roles, weights and mixing applied.');
    } catch (e) {
      showError(errorText(e));
    }
  };

  const liveParams = () => {
    const o = readOptions();
    const ppoLive = {};
    for (const key of PPO_LIVE_KEYS) ppoLive[key] = o[key];
    return {
      ...ppoLive,
      mutation: o.mutation,
      penaltyStart: o.penaltyStart,
      penaltyEnd: o.penaltyEnd,
      penaltyRampTicks: o.penaltyRampTicks,
      difficultyStart: o.difficultyStart,
      difficultyEnd: o.difficultyEnd,
      difficultyRampTicks: o.difficultyRampTicks,
      ticksPerDispatch: o.ticksPerDispatch,
      evalEvery: o.evalEvery,
    };
  };

  const usingCpu = () => $('forceCpu').checked || !gpuUsable();

  const createBackend = async (opts, forceKind) => {
    const wantGpu = forceKind ? forceKind === 'gpu' : !usingCpu();
    if (wantGpu && gpuUsable()) {
      try {
        return opts.trainer === 'ppo' ? await RlBackend.create(opts) : await GpuBackend.create(opts);
      } catch (e) {
        showError('GPU backend failed, falling back to CPU: ' + errorText(e));
        $('forceCpu').checked = true;
      }
    }
    if (opts.trainer === 'ppo') {
      showError('PPO needs the WebGPU backend, using Evolution on the CPU instead.');
      $('trainer').value = 'evolution';
    }
    return CpuBackend.create(opts);
  };

  const setBusy = (busy) => {
    app.busy = busy;
    updateButtons();
  };

  const updateButtons = () => {
    const has = !!app.backend;
    const idle = has && !app.busy;
    $('startBtn').disabled = !has || app.busy;
    $('startBtn').textContent = app.running ? 'Pause' : 'Start';
    $('resetBtn').disabled = app.busy;
    for (const id of ['benchBtn', 'testBtn', 'champBtn', 'importBtn', 'saveBtn', 'watchBtn']) $(id).disabled = !idle;
    $('popApplyBtn').disabled = !idle || !app.backend.configurePopulation || app.backend.K < 2;
    $('csvBtn').disabled = app.history.tick.length === 0;
    $('restoreBtn').disabled = app.busy;
    $('runDot').classList.toggle('run', app.running);
  };

  const buildBackend = async (opts, restoredCheckpoint) => {
    setBusy(true);
    showError('');
    try {
      if (app.backend) {
        await lock(() => app.backend.dispose());
        app.backend = null;
      }
      app.optsUsed = opts;
      app.backend = await createBackend(opts);
      if (restoredCheckpoint) await app.backend.restore(restoredCheckpoint);
      const info = app.backend.info || {};
      const kind = info.kind || (usingCpu() ? 'cpu' : 'gpu');
      $('kindText').textContent = kind === 'gpu' ? (info.trainer === 'ppo' ? 'WebGPU PPO' : 'WebGPU') : 'CPU fallback';
      $('kindPill').className = 'pill ' + kind;
      $('forceCpu').checked = kind === 'cpu';
      const worlds = info.worlds || opts.worlds;
      $('worldRange').max = worlds - 1;
      $('worldPick').max = worlds - 1;
      selectWorld(Math.min(app.world, worlds - 1), false);
      await refreshAll();
    } catch (e) {
      showError('Backend creation failed: ' + errorText(e));
      app.backend = null;
    } finally {
      setBusy(false);
    }
  };

  const pushHistory = (s) => {
    const h = app.history;
    h.tick.push(s.tick);
    h.evo.push(s.train.rewRate);
    h.bot.push(s.train.baseRate);
    h.evalEvo.push(s.eval.rewRate);
    h.evalBot.push(s.eval.baseRate);
    h.ops.push(s.train.opsPerTick);
    h.edges.push(s.train.edgesMean);
    h.archBest.push(s.archive.best);
    h.archMean.push(s.archive.mean);
    const len = h.tick.length;
    if (s.pop) {
      const series = s.pop.roles.map((r, i) => ['pe_' + i + '_' + r.name, r.evalPure.rewRate]).concat([['pe_mixed', s.pop.mixed.rewRate], ['pe_bots', s.pop.evalBase]]);
      for (const [key, value] of series) {
        if (!h[key]) h[key] = new Array(len - 1).fill(NaN);
        h[key].push(value);
      }
    }
    for (const [name, value] of Object.entries(s.train.game || {})) {
      const key = 'g_' + name;
      if (!h[key]) h[key] = new Array(len - 1).fill(NaN);
      h[key].push(value);
    }
    if (len > HISTORY_LIMIT) {
      for (const k of Object.keys(h)) h[k] = h[k].filter((_, i) => i % 2 === 0);
    }
  };

  const refreshStats = async () => {
    const s = await lock(() => app.backend.readStats());
    app.stats = s;
    pushHistory(s);
    return s;
  };

  const refreshSnapshot = async () => {
    if ($('headless').checked) return;
    app.snapshot = await lock(() => app.backend.snapshot(app.world));
  };

  const refreshAll = async () => {
    await refreshStats();
    await refreshSnapshot();
    renderAll();
  };

  const selectWorld = (index, redraw = true) => {
    const max = parseInt($('worldRange').max, 10) || 0;
    app.world = Math.max(0, Math.min(max, index | 0));
    $('worldRange').value = app.world;
    $('worldPick').value = app.world;
    $('worldTitle').textContent = '#' + app.world;
    if (redraw && app.backend && !app.running) {
      refreshSnapshot().then(renderAll).catch((e) => showError(errorText(e)));
    }
  };

  const noteThroughput = (ticks) => {
    const now = performance.now();
    app.throughput.push({ now, agentTicks: ticks * (app.backend.info.worlds || 1) * (app.optsUsed.game.maxLearners && app.backend.info.trainer === 'ppo' ? app.optsUsed.game.maxLearners : app.optsUsed.game.learners) });
    while (app.throughput.length > 1 && now - app.throughput[0].now > 3000) app.throughput.shift();
  };

  const throughputRate = () => {
    const t = app.throughput;
    if (t.length < 2) return 0;
    const span = (t[t.length - 1].now - t[0].now) / 1000;
    if (span <= 0) return 0;
    let sum = 0;
    for (let i = 1; i < t.length; i++) sum += t[i].agentTicks;
    return sum / span;
  };

  const runLoop = async () => {
    if (app.loopActive) return;
    app.loopActive = true;
    let lastStats = performance.now();
    let lastDraw = performance.now();
    app.lastCheckpoint = performance.now();
    try {
      while (app.running && app.backend) {
        const r = await lock(() => app.backend.step());
        if (r) noteThroughput(r.ticks);
        const now = performance.now();
        if (now - lastStats >= STATS_MS) {
          await refreshStats();
          lastStats = performance.now();
          renderStats();
        }
        if (now - lastDraw >= DRAW_MS) {
          await refreshSnapshot();
          lastDraw = performance.now();
          renderViews();
        }
        if (now - app.lastCheckpoint >= CHECKPOINT_MS) {
          await saveCheckpoint(true);
          app.lastCheckpoint = performance.now();
        }
        await yieldUi();
      }
    } catch (e) {
      app.running = false;
      showError('Training loop stopped: ' + errorText(e));
    } finally {
      app.loopActive = false;
      updateButtons();
    }
  };

  const setRunning = (on) => {
    if (on && !app.backend) return;
    app.running = on;
    updateButtons();
    updateHiddenBanner();
    if (on) runLoop();
    else refreshAll().catch((e) => showError(errorText(e)));
  };

  const updateHiddenBanner = () => {
    $('hiddenBanner').hidden = !(document.hidden && app.running);
  };

  const popChartDef = () => {
    const h = app.history;
    const roles = app.stats && app.stats.pop ? app.stats.pop.roles : null;
    $('chartPopCard').hidden = !roles;
    if (!roles) return [];
    const series = roles.map((r, i) => ({ label: r.name, color: POP_COLORS[i % POP_COLORS.length], data: h['pe_' + i + '_' + r.name] || [] }));
    series.push({ label: 'mixed team', color: '#ffffff', data: h.pe_mixed || [] });
    series.push({ label: 'scripted baseline', color: COLORS.bot, data: h.pe_bots || [], dash: true });
    return [{ id: 'chartPop', x: h.tick, series }];
  };

  const chartDefs = () => {
    const h = app.history;
    return [
      { id: 'chartReward', x: h.tick, series: [{ label: 'evolved', color: COLORS.evo, data: h.evo }, { label: 'scripted baseline', color: COLORS.bot, data: h.bot, dash: true }] },
      { id: 'chartEval', x: h.tick, series: [{ label: 'eval', color: COLORS.eval, data: h.evalEvo }, { label: 'scripted baseline', color: COLORS.bot, data: h.evalBot, dash: true }] },
      { id: 'chartOps', x: h.tick, series: [{ label: 'ops / agent tick', color: COLORS.violet, data: h.ops }] },
      { id: 'chartEdges', x: h.tick, series: [{ label: 'mean synapses', color: COLORS.gold, data: h.edges }] },
      { id: 'chartArchive', x: h.tick, series: [{ label: 'best', color: COLORS.eval, data: h.archBest }, { label: 'mean', color: COLORS.evo, data: h.archMean, dash: true }] },
    ].concat(popChartDef());
  };

  const sizeCanvas = (cv) => {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(50, Math.round(cv.clientWidth * dpr));
    const h = Math.max(50, Math.round(cv.clientHeight * dpr));
    if (cv.width !== w || cv.height !== h) {
      cv.width = w;
      cv.height = h;
    }
    return dpr;
  };

  const drawChart = (def) => {
    const cv = $(def.id);
    const dpr = sizeCanvas(cv);
    const ctx = cv.getContext('2d');
    const W = cv.width;
    const H = cv.height;
    ctx.clearRect(0, 0, W, H);
    const padL = 52 * dpr;
    const padR = 10 * dpr;
    const padT = 10 * dpr;
    const padB = 22 * dpr;
    const n = def.x.length;
    ctx.font = `${11 * dpr}px system-ui, sans-serif`;
    if (n < 2) {
      ctx.fillStyle = COLORS.text;
      ctx.fillText('waiting for data', padL, H / 2);
      return;
    }
    let lo = Infinity;
    let hi = -Infinity;
    for (const s of def.series) {
      for (const v of s.data) {
        if (Number.isFinite(v)) {
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
      }
    }
    if (!Number.isFinite(lo)) {
      lo = 0;
      hi = 1;
    }
    if (hi - lo < 1e-9) {
      hi = lo + 1;
    }
    const pad = (hi - lo) * 0.08;
    lo -= pad;
    hi += pad;
    const x0 = def.x[0];
    const x1 = def.x[n - 1] === x0 ? x0 + 1 : def.x[n - 1];
    const px = (x) => padL + ((x - x0) / (x1 - x0)) * (W - padL - padR);
    const py = (y) => padT + (1 - (y - lo) / (hi - lo)) * (H - padT - padB);
    ctx.strokeStyle = COLORS.grid;
    ctx.fillStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const y = lo + ((hi - lo) * i) / 4;
      const yy = Math.round(py(y)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(padL, yy);
      ctx.lineTo(W - padR, yy);
      ctx.stroke();
      ctx.fillText(fmt(y, 3), padL - 6 * dpr, yy + 4 * dpr);
    }
    ctx.textAlign = 'left';
    ctx.fillText(fmt(x0), padL, H - 6 * dpr);
    ctx.textAlign = 'right';
    ctx.fillText('tick ' + fmt(x1), W - padR, H - 6 * dpr);
    let legendX = padL + 6 * dpr;
    for (const s of def.series) {
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.8 * dpr;
      ctx.setLineDash(s.dash ? [6 * dpr, 4 * dpr] : []);
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < n; i++) {
        const v = s.data[i];
        if (!Number.isFinite(v)) continue;
        const X = px(def.x[i]);
        const Y = py(v);
        if (started) ctx.lineTo(X, Y);
        else ctx.moveTo(X, Y);
        started = true;
      }
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = s.color;
      ctx.textAlign = 'left';
      const last = s.data[n - 1];
      const label = `${s.label} ${fmt(last, 4)}`;
      ctx.fillText(label, legendX, padT + 10 * dpr);
      legendX += ctx.measureText(label).width + 14 * dpr;
    }
  };

  const drawCharts = () => {
    for (const def of chartDefs()) drawChart(def);
  };

  const drawGridSnapshot = (ctx, snap, size) => {
    const cols = snap.width;
    const rows = snap.height;
    const cell = size / Math.max(cols, rows);
    const tiles = snap.tile;
    for (let i = 0; i < tiles.length; i++) {
      const raw = tiles[i];
      const x = (i % cols) * cell;
      const y = Math.floor(i / cols) * cell;
      ctx.fillStyle = TILE_PALETTE[raw & 15];
      ctx.fillRect(x, y, cell, cell);
      if (((raw >>> 4) & 4095) > 0 && (raw & 15) >= 2) {
        ctx.fillStyle = 'rgba(11,14,20,0.55)';
        ctx.fillRect(x, y, cell, cell);
      }
      ctx.strokeStyle = 'rgba(255,255,255,0.04)';
      ctx.strokeRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
    }
    let evoAlive = 0;
    let otherAlive = 0;
    let hpSum = 0;
    let goldSum = 0;
    for (const a of snap.agents) {
      if (a.alive === false) continue;
      if (a.evo) evoAlive++;
      else otherAlive++;
      hpSum += a.hp;
      goldSum += a.gold || 0;
      const cx = a.x * cell + cell / 2;
      const cy = a.y * cell + cell / 2;
      ctx.fillStyle = a.evo ? COLORS.evo : COLORS.bot;
      ctx.beginPath();
      ctx.arc(cx, cy, cell * 0.3, 0, Math.PI * 2);
      ctx.fill();
      const f = FACE_VECTORS[(a.face | 0) & 3];
      ctx.strokeStyle = '#0b0e14';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + f[0] * cell * 0.4, cy + f[1] * cell * 0.4);
      ctx.stroke();
      const hpw = Math.max(0, Math.min(1, a.hp / 100)) * (cell - 4);
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.fillRect(a.x * cell + 2, a.y * cell + 1, cell - 4, 3);
      ctx.fillStyle = a.hp > 50 ? '#80ed99' : a.hp > 25 ? '#ffd166' : '#ff6b6b';
      ctx.fillRect(a.x * cell + 2, a.y * cell + 1, hpw, 3);
      if ((a.gold || 0) > 0) {
        ctx.fillStyle = COLORS.gold;
        ctx.fillRect(a.x * cell + cell - 6, a.y * cell + cell - 6, 4, 4);
      }
    }
    const alive = evoAlive + otherAlive || 1;
    $('worldInfo').textContent = `evolved ${evoAlive}, scripted ${otherAlive}, mean hp ${fmt(hpSum / alive, 1)}, gold ${goldSum}`;
  };

  const drawWorld = () => {
    const cv = $('worldCanvas');
    const ctx = cv.getContext('2d');
    const size = cv.width;
    ctx.clearRect(0, 0, size, size);
    const snap = app.watch.active ? app.watch.env.snapshot() : app.snapshot;
    const game = app.optsUsed && app.optsUsed.game;
    if (!snap) {
      ctx.fillStyle = COLORS.text;
      ctx.font = '14px system-ui';
      ctx.fillText($('headless').checked ? 'headless: drawing off' : 'no snapshot', 12, 24);
      return;
    }
    if (game && typeof game.render === 'function') {
      game.render(ctx, snap, size, size);
      const living = (snap.agents || []).filter((x) => x.alive !== false);
      const evolved = living.filter((x) => x.evo).length;
      $('worldInfo').textContent = 'evolved ' + evolved + ', scripted ' + (living.length - evolved);
    } else if (snap.kind === 'grid') {
      drawGridSnapshot(ctx, snap, size);
    } else {
      ctx.fillStyle = COLORS.text;
      ctx.font = '14px system-ui';
      ctx.fillText('snapshot kind "' + snap.kind + '" has no renderer', 12, 24);
    }
  };

  const heatColor = (t) => {
    const stops = [[18, 24, 48], [60, 60, 140], [180, 60, 140], [250, 170, 70], [255, 245, 200]];
    const p = Math.max(0, Math.min(1, t)) * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(p));
    const f = p - i;
    return stops[i].map((c, k) => Math.round(c + (stops[i + 1][k] - c) * f));
  };

  const drawHeat = () => {
    const cv = $('heatCanvas');
    const scores = app.stats && app.stats.worldScores;
    const ctx = cv.getContext('2d');
    ctx.imageSmoothingEnabled = false;
    if (!scores || !scores.length) {
      ctx.clearRect(0, 0, cv.width, cv.height);
      return;
    }
    const n = scores.length;
    const cols = Math.ceil(Math.sqrt(n));
    const rows = Math.ceil(n / cols);
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < n; i++) {
      const v = scores[i];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const span = hi - lo || 1;
    const off = app.heat.canvas;
    off.width = cols;
    off.height = rows;
    const octx = off.getContext('2d');
    const img = octx.createImageData(cols, rows);
    for (let i = 0; i < cols * rows; i++) {
      const c = i < n ? heatColor((scores[i] - lo) / span) : [11, 14, 20];
      img.data[i * 4] = c[0];
      img.data[i * 4 + 1] = c[1];
      img.data[i * 4 + 2] = c[2];
      img.data[i * 4 + 3] = 255;
    }
    octx.putImageData(img, 0, 0);
    const side = 512;
    if (cv.width !== side) {
      cv.width = side;
      cv.height = side;
    }
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = '#0b0e14';
    ctx.fillRect(0, 0, side, side);
    const cell = side / Math.max(cols, rows);
    ctx.drawImage(off, 0, 0, cols * cell, rows * cell);
    const sx = (app.world % cols) * cell;
    const sy = Math.floor(app.world / cols) * cell;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 3;
    ctx.strokeRect(sx + 1.5, sy + 1.5, cell - 3, cell - 3);
    app.heat.cols = cols;
    app.heat.cell = cell;
    $('heatRange').textContent = `${fmt(lo)} to ${fmt(hi)}`;
  };

  const statRows = (s) => {
    const rows = [];
    for (const key of COUNTER_KEYS) rows.push([key, fmt(s.train[key], 4), fmt(s.eval ? s.eval[key] : undefined, 4)]);
    for (const key of Object.keys(s.train.game || {})) rows.push([key, fmt(s.train.game[key], 4), fmt(s.eval && s.eval.game ? s.eval.game[key] : undefined, 4)]);
    return rows;
  };

  const renderStats = () => {
    const s = app.stats;
    if (!s) return;
    $('tickText').textContent = fmt(s.tick);
    $('passText').textContent = fmt(s.passes);
    $('rateText').textContent = fmt(throughputRate());
    const left = statRows(s);
    const ppoRows = s.ppo ? Object.entries(s.ppo).map(([key, value]) => `<tr><td>${key}</td><td>${fmt(value, 4)}</td></tr>`).join('') : '';
    const curriculumRows = s.curriculum ? [['curriculum difficulty', s.curriculum.difficulty], ['curriculum self-play fraction', s.curriculum.selfPlay], ['curriculum smoothed life', s.curriculum.life], ['curriculum steps', s.curriculum.events]].map(([k, v]) => `<tr><td>${k}</td><td>${fmt(v, 4)}</td></tr>`).join('') : '';
    const leagueRows = s.league ? [['h2h latest reward/tick', s.league.headToHead.latestRate], ['h2h older reward/tick', s.league.headToHead.olderRate], ['h2h ratio', s.league.headToHead.ratio], ['h2h reference age (iterations)', s.league.headToHead.referenceAge], ['self-play reward/tick', s.league.selfPlay.rewRate], ['self-play mean life', s.league.selfPlay.meanLife], ['pool snapshots', s.league.pool.valid]].map(([key, value]) => `<tr><td>${key}</td><td>${fmt(value, 4)}</td></tr>`).join('') : '';
    const popRows = s.pop ? '<table><thead><tr><th>role</th><th>train/tick</th><th>pure eval/tick</th><th>x bots</th><th>life</th><th>in mixed team</th><th>h2h</th><th>entropy</th><th>reward scale</th></tr></thead><tbody>' + s.pop.roles.map((r) => `<tr><td>${escapeHtml(r.name)} (${r.weights.map((w) => fmt(w, 2)).join('/')})</td><td>${fmt(r.train.rate, 4)}</td><td>${fmt(r.evalPure.rewRate, 4)}</td><td>${fmt(r.evalPure.baseRate ? r.evalPure.rewRate / r.evalPure.baseRate : NaN, 2)}</td><td>${fmt(r.evalPure.meanLife, 0)}</td><td>${fmt(r.evalMixed.rate, 4)}</td><td>${fmt(r.headToHead.ratio, 2)}</td><td>${fmt(r.diag.entropy, 3)}</td><td>${fmt(r.diag.rewardScale, 2)}</td></tr>`).join('') + `<tr><td>mixed team (${escapeHtml(s.pop.mix)})</td><td></td><td>${fmt(s.pop.mixed.rewRate, 4)}</td><td>${fmt(s.pop.mixed.baseRate ? s.pop.mixed.rewRate / s.pop.mixed.baseRate : NaN, 2)}</td><td>${fmt(s.pop.mixed.meanLife, 0)}</td><td></td><td></td><td></td><td></td></tr></tbody></table>` : '';
    const ppoTable = s.ppo ? `<table><thead><tr><th>ppo</th><th></th></tr></thead><tbody>${ppoRows}${leagueRows}${curriculumRows}</tbody></table>${popRows}` : '';
    const html = `<table><thead><tr><th>metric</th><th>train</th><th>eval</th></tr></thead><tbody>${left
      .map((r) => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td></tr>`)
      .join('')}</tbody></table><table><thead><tr><th>${s.ppo ? 'ppo return' : 'archive'}</th><th></th></tr></thead><tbody><tr><td>${s.ppo ? 'best eval reward/tick' : 'best'}</td><td>${fmt(s.archive.best, 4)}</td></tr><tr><td>${s.ppo ? 'mean discounted return' : 'mean'}</td><td>${fmt(s.archive.mean, 4)}</td></tr><tr><td>${s.ppo ? 'iterations' : 'count'}</td><td>${fmt(s.archive.count)}</td></tr></tbody></table>${ppoTable}`;
    $('statGrid').innerHTML = html;
  };

  const renderViews = () => {
    drawCharts();
    drawWorld();
    drawHeat();
  };

  const renderAll = () => {
    renderStats();
    renderViews();
  };

  const flatten = (value, prefix, out, depth) => {
    if (value === null || value === undefined) {
      out.push([prefix, String(value)]);
    } else if (value instanceof Set) {
      out.push([prefix, Array.from(value).join(', ')]);
    } else if (Array.isArray(value) && value.every((v) => typeof v !== 'object')) {
      out.push([prefix, value.join(', ')]);
    } else if (typeof value === 'object' && depth < 3) {
      for (const [k, v] of Object.entries(value)) flatten(v, prefix ? prefix + '.' + k : k, out, depth + 1);
    } else {
      out.push([prefix, typeof value === 'object' ? JSON.stringify(value) : String(value)]);
    }
  };

  const escapeHtml = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

  const renderDevice = (report, info) => {
    const rows = [];
    rows.push(['WebGPU in browser', navigator.gpu ? 'yes' : 'no']);
    rows.push(['Engine module', typeof GpuBackend !== 'undefined' ? 'present' : 'missing']);
    if (report) {
      const text = JSON.stringify(report, replacerForJson);
      rows.push(['shader-f16', /shader-f16/.test(text) ? 'present' : 'absent']);
      rows.push(['timestamp-query', /timestamp-query/.test(text) ? 'present' : 'absent']);
      flatten(report, '', rows, 0);
    } else if (navigator.gpu) {
      rows.push(['probe', 'no report']);
    }
    if (info) flatten(info, 'backend', rows, 0);
    const cell = (r) => `<span>${escapeHtml(r[0])}</span><span>${escapeHtml(r[1])}</span>`;
    const limitRows = rows.filter((r) => r[0].startsWith('limits.'));
    const otherRows = rows.filter((r) => !r[0].startsWith('limits.'));
    const limitBlock = limitRows.length ? `<details style="grid-column:1/-1"><summary>Limits (${limitRows.length})</summary><div class="kv">${limitRows.map(cell).join('')}</div></details>` : '';
    $('devicePanel').innerHTML = otherRows.map(cell).join('') + limitBlock;
  };

  const probeDevice = async () => {
    if (!gpuAvailable() || typeof GpuBackend.probe !== 'function') return null;
    try {
      return await GpuBackend.probe();
    } catch (e) {
      return { probeError: errorText(e) };
    }
  };

  const download = (name, text, type) => {
    const blob = new Blob([text], { type: type || 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  };

  const stamp = () => new Date().toISOString().replace(/[:.]/g, '-');

  const exportCsv = () => {
    const h = app.history;
    const keys = Object.keys(h);
    const lines = [keys.join(',')];
    for (let i = 0; i < h.tick.length; i++) lines.push(keys.map((k) => h[k][i]).join(','));
    download(`curves-${stamp()}.csv`, lines.join('\n'), 'text/csv');
  };

  const exportChampions = async () => {
    const n = Math.max(1, Math.round(num('champCount', 8)));
    const list = await lock(() => app.backend.exportChampions(n));
    download(`champions-${stamp()}.json`, JSON.stringify(list, replacerForJson));
    log(`Exported ${list.length} champion genome(s).`);
  };

  const importGenomes = async (file) => {
    const parsed = JSON.parse(await file.text());
    const list = Array.isArray(parsed) ? parsed : parsed.policies ? parsed.policies : parsed.genomes ? parsed.genomes : [parsed];
    await lock(() => app.backend.importGenomes(list));
    log(`Imported ${list.length} genome(s) into every island archive.`);
  };

  const openDb = () => new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(DB_STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });

  const dbOp = async (mode, fn) => {
    const db = await openDb();
    try {
      return await new Promise((resolve, reject) => {
        const tx = db.transaction(DB_STORE, mode);
        const req = fn(tx.objectStore(DB_STORE));
        tx.oncomplete = () => resolve(req.result);
        tx.onerror = () => reject(tx.error);
        tx.onabort = () => reject(tx.error);
      });
    } finally {
      db.close();
    }
  };

  const saveCheckpoint = async (quiet) => {
    if (!app.backend) return;
    try {
      const ckpt = await lock(() => app.backend.checkpoint());
      const { game: activeGame, ...plainOpts } = app.optsUsed;
      const record = { at: Date.now(), kind: app.backend.info.kind, gameId: activeGame.id, opts: plainOpts, history: app.history, tick: app.stats ? app.stats.tick : 0, ckpt };
      await dbOp('readwrite', (store) => store.put(record, 'latest'));
      $('ckptText').textContent = `Checkpoint saved ${new Date(record.at).toLocaleTimeString()} at tick ${fmt(record.tick)}.`;
      if (!quiet) log('Checkpoint saved to IndexedDB.');
    } catch (e) {
      $('ckptText').textContent = 'Checkpoint failed: ' + errorText(e);
    }
  };

  const restoreCheckpoint = async () => {
    setBusy(true);
    try {
      const record = await dbOp('readonly', (store) => store.get('latest'));
      if (!record) {
        log('No checkpoint stored.');
        return;
      }
      const wasRunning = app.running;
      app.running = false;
      while (app.loopActive) await yieldUi();
      if (!GAMES[record.gameId]) throw new Error('checkpoint game not registered: ' + record.gameId);
      $('game').value = record.gameId;
      renderGameInfo();
      const opts = { ...record.opts, game: GAMES[record.gameId] };
      $('forceCpu').checked = record.kind === 'cpu';
      $('trainer').value = opts.trainer === 'ppo' ? 'ppo' : 'evolution';
      for (const [id, key] of [...PPO_FIELDS, ['worlds', 'worlds'], ['islands', 'islands'], ['maxEdges', 'maxEdges'], ['mutation', 'mutation'], ['penaltyStart', 'penaltyStart'], ['penaltyEnd', 'penaltyEnd'], ['penaltyRampTicks', 'penaltyRampTicks'], ['difficultyStart', 'difficultyStart'], ['difficultyEnd', 'difficultyEnd'], ['difficultyRampTicks', 'difficultyRampTicks'], ['evalFraction', 'evalFraction'], ['migrateEvery', 'migrateEvery'], ['migrateCount', 'migrateCount'], ['selectEvery', 'selectEvery'], ['evalEvery', 'evalEvery'], ['seed', 'seed']]) {
        if (opts[key] !== undefined) $(id).value = (key === 'rewardClip' || key === 'valueClip') && opts[key] >= 1e8 ? 0 : opts[key];
      }
      $('randomize').checked = opts.randomize !== false;
      $('tpdAuto').checked = !opts.ticksPerDispatch;
      if (opts.ticksPerDispatch) $('tpd').value = opts.ticksPerDispatch;
      $('tpd').disabled = $('tpdAuto').checked;
      app.history = record.history || newHistory();
      app.throughput = [];
      $('ppoPolicies').value = opts.policies || 1;
      const popSource = record.ckpt && record.ckpt.pop ? { policies: record.ckpt.pop.k, popRoles: record.ckpt.pop.names, popWeights: record.ckpt.pop.weights, popFractions: record.ckpt.pop.fractions } : opts;
      renderPopEditor(rlPopConfig(popSource, gameChannels()));
      await buildBackend(Object.assign({}, opts, readPopEditor()), record.ckpt);
      log(`Restored checkpoint from ${new Date(record.at).toLocaleString()} (tick ${fmt(record.tick)}).`);
      if (wasRunning) setRunning(true);
    } catch (e) {
      showError('Restore failed: ' + errorText(e));
    } finally {
      setBusy(false);
    }
  };

  const tableText = (rows) => {
    if (!Array.isArray(rows) || !rows.length) return JSON.stringify(rows, replacerForJson, 2);
    if (typeof rows[0] !== 'object') return rows.join('\n');
    const keys = Object.keys(rows[0]);
    const cells = rows.map((r) => keys.map((k) => (typeof r[k] === 'number' ? fmt(r[k], 3) : typeof r[k] === 'object' ? JSON.stringify(r[k], replacerForJson) : String(r[k]))));
    const widths = keys.map((k, i) => Math.max(k.length, ...cells.map((c) => c[i].length)));
    const line = (arr) => arr.map((c, i) => c.padEnd(widths[i])).join('  ');
    return [line(keys), ...cells.map(line)].join('\n');
  };

  const cpuSelfTests = async () => {
    const results = [];
    const small = { game: currentGame(), worlds: 2, islands: 1, maxEdges: 64, ticksPerDispatch: 4, mutation: 0.1, penaltyStart: 0, penaltyEnd: 0.1, penaltyRampTicks: 1000, randomize: false, evalFraction: 0.5, seed: 7, migrateEvery: 2, migrateCount: 1, selectEvery: 2, evalEvery: 2 };
    try {
      const a = await CpuBackend.create(small);
      const b = await CpuBackend.create(small);
      for (let i = 0; i < 3; i++) {
        await a.step();
        await b.step();
      }
      const sa = await a.readStats();
      const sb = await b.readStats();
      results.push({ name: 'cpu determinism', ok: JSON.stringify(sa.train, replacerForJson) === JSON.stringify(sb.train, replacerForJson), detail: `tick ${sa.tick}` });
      const champs = await a.exportChampions(2);
      results.push({ name: 'champion export shape', ok: champs.every((g) => /^npc-brain\//.test(g.format) && g.params.length === NPARAMS), detail: `${champs.length} genomes` });
      const ckpt = await a.checkpoint();
      const c = await CpuBackend.create(small);
      await c.restore(ckpt);
      const sc = await c.readStats();
      results.push({ name: 'checkpoint round trip', ok: sc.tick === sa.tick, detail: `tick ${sc.tick} vs ${sa.tick}` });
      await a.importGenomes(champs);
      results.push({ name: 'genome import', ok: true, detail: 'accepted' });
      await a.dispose();
      await b.dispose();
      await c.dispose();
    } catch (e) {
      results.push({ name: 'cpu self tests', ok: false, detail: errorText(e) });
    }
    return results;
  };

  const runTests = async () => {
    const b = app.backend;
    let results;
    if (typeof b.runTests === 'function') results = await lock(() => b.runTests());
    else if (b.info.kind === 'gpu' && typeof GpuBackend.runTests === 'function') results = await lock(() => GpuBackend.runTests());
    else results = await cpuSelfTests();
    const pass = results.filter((r) => r.ok).length;
    log(`Tests: ${pass}/${results.length} passed\n` + results.map((r) => `${r.ok ? 'PASS' : 'FAIL'}  ${r.name}  ${r.detail === undefined ? '' : typeof r.detail === 'string' ? r.detail : JSON.stringify(r.detail)}`).join('\n'));
  };

  const cpuBenchmark = async () => {
    const rows = [];
    for (const worlds of [1, 4]) {
      const bench = await CpuBackend.create({ ...readOptions(), worlds, islands: 1 });
      const t0 = performance.now();
      let ticks = 0;
      for (let i = 0; i < 3; i++) ticks += (await bench.step()).ticks;
      const secs = (performance.now() - t0) / 1000;
      rows.push({ worlds, ticks, seconds: secs, agentTicksPerSec: (ticks * worlds * bench.game.learners) / secs });
      await bench.dispose();
    }
    return { rows };
  };

  const runBenchmark = async () => {
    const b = app.backend;
    let result;
    if (typeof b.benchmark === 'function') result = await lock(() => b.benchmark());
    else if (b.info.kind === 'gpu' && typeof GpuBackend.benchmark === 'function') result = await lock(() => GpuBackend.benchmark());
    else result = await cpuBenchmark();
    log('Benchmark\n' + tableText(result.rows || result));
  };

  const withPause = async (label, fn) => {
    if (!app.backend) return;
    const wasRunning = app.running;
    app.running = false;
    while (app.loopActive) await yieldUi();
    setBusy(true);
    log(label + '...');
    try {
      await fn();
    } catch (e) {
      log(label + ' failed: ' + errorText(e));
    } finally {
      setBusy(false);
      if (wasRunning) setRunning(true);
    }
  };

  const applyLive = () => {
    if (!app.backend) return;
    try {
      app.backend.setParams(liveParams());
    } catch (e) {
      showError(errorText(e));
    }
  };

  const doReset = async () => {
    const wasRunning = app.running;
    app.running = false;
    while (app.loopActive) await yieldUi();
    app.history = newHistory();
    app.throughput = [];
    app.stats = null;
    app.snapshot = null;
    await buildBackend(readOptions());
    if (wasRunning) setRunning(true);
  };

  const heatPick = (event) => {
    const cv = $('heatCanvas');
    const rect = cv.getBoundingClientRect();
    const scale = cv.width / rect.width;
    const cx = Math.floor(((event.clientX - rect.left) * scale) / app.heat.cell);
    const cy = Math.floor(((event.clientY - rect.top) * scale) / app.heat.cell);
    const n = app.stats && app.stats.worldScores ? app.stats.worldScores.length : 0;
    const idx = cy * app.heat.cols + cx;
    if (cx >= 0 && cx < app.heat.cols && idx >= 0 && idx < n) selectWorld(idx);
  };

  const editingText = (e) => e.target && /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName) && e.target.type !== 'checkbox' && e.target.type !== 'range';

  const fillGameSelect = () => {
    $('game').innerHTML = Object.values(GAMES).map((g) => `<option value="${escapeHtml(g.id)}">${escapeHtml(g.name || g.id)}${g.wgsl ? '' : ' (CPU only)'}</option>`).join('');
    renderGameInfo();
  };

  const renderGameInfo = () => {
    const g = currentGame();
    const rows = [
      ['id', g.id],
      ['nodes', `${g.dims.nIn} in, ${g.dims.nOut} out, ${g.dims.nNodes} total`],
      ['agents', `${g.agents} (${g.learners} learners)`],
      ['max age', g.maxAge],
      ['GPU kernel', g.wgsl ? 'available' : 'none, CPU only'],
      ['inputs', g.inputNames.join(', ')],
      ['actions', g.actionNames.join(', ')],
      ['game stats', g.statNames.join(', ')],
    ];
    $('gameInfo').innerHTML = rows.map((r) => `<span>${escapeHtml(r[0])}</span><span>${escapeHtml(r[1])}</span>`).join('');
  };

  const WATCH_SPEEDS = [0.25, 0.5, 1, 2, 4, 8, 16, 32, Infinity];
  const WATCH_BUDGET_MS = 12;
  const WATCH_MAX_CATCHUP_S = 0.25;

  const watchHz = (game) => game.tickHz || 10;

  const watchSpeed = () => WATCH_SPEEDS[parseInt($('watchSpeed').value, 10)];

  const renderWatchSpeed = () => {
    const v = watchSpeed();
    $('watchSpeedText').textContent = v === Infinity ? 'max' : v + 'x';
  };

  const watchTick = () => {
    const w = app.watch;
    const game = w.game;
    const nOut = game.dims.nOut;
    for (let a = 0; a < game.learners; a++) {
      w.env.observe(a, w.obs, w.tick);
      const brain = w.brains[a];
      w.actions[a] = brain.step(w.obs, new Rng(mix(w.seed, w.tick, a, 1)));
      for (let k = 0; k < nOut; k++) w.outputs[a * nOut + k] = brain.act[game.dims.nIn + k];
    }
    w.env.step(w.actions, w.outputs, w.tick);
    for (let a = 0; a < game.learners; a++) {
      const brain = w.brains[a];
      const rewardFx = w.env.reward[a];
      brain.learn(rewardFx / REWARD_SCALE);
      w.reward += rewardFx / REWARD_SCALE;
      w.lifeTicks[a]++;
      const died = w.env.dead[a] !== 0 || w.lifeTicks[a] >= game.maxAge;
      if (died) {
        w.deaths++;
        w.env.respawn(a, w.tick);
        brain.load(copyGenome(w.genomes[a % w.genomes.length], w.maxEdges));
        w.lifeTicks[a] = 0;
      } else if (w.lifeTicks[a] % STRUCT_PERIOD === 0) {
        brain.structural(new Rng(mix(w.seed, w.tick, a, 3)));
      }
    }
    w.tick++;
  };

  const watchFrame = (now) => {
    const w = app.watch;
    if (!w.active) return;
    const dt = Math.min(WATCH_MAX_CATCHUP_S, (now - w.last) / 1000);
    w.last = now;
    const speed = watchSpeed();
    const started = performance.now();
    try {
      if (speed === Infinity) {
        while (performance.now() - started < WATCH_BUDGET_MS) watchTick();
      } else {
        w.acc += dt * watchHz(w.game) * speed;
        while (w.acc >= 1 && performance.now() - started < WATCH_BUDGET_MS * 2) {
          watchTick();
          w.acc -= 1;
        }
        if (w.acc > watchHz(w.game)) w.acc = 0;
      }
    } catch (e) {
      stopWatch();
      showError('Watch stopped: ' + errorText(e));
      return;
    }
    drawWorld();
    const seconds = w.tick / watchHz(w.game);
    $('watchInfo').textContent = 'watch tick ' + fmt(w.tick) + ', ' + seconds.toFixed(1) + ' s game time, reward per learner-tick ' + fmt(w.reward / Math.max(1, w.tick * w.game.learners), 4) + ', deaths ' + w.deaths;
    requestAnimationFrame(watchFrame);
  };

  const stopWatch = () => {
    app.watch = { active: false, starting: false };
    $('watchBtn').textContent = 'Watch champions';
    $('watchInfo').textContent = '';
    if (app.backend) refreshSnapshot().then(renderAll).catch(() => {});
  };

  const startWatch = async () => {
    if (!app.backend || app.watch.starting) return;
    app.watch.starting = true;
    try {
      const game = app.optsUsed.game;
      const jsons = (await lock(() => app.backend.exportChampions(game.learners))).map((entry) => entry.genome || entry);
      const maxEdges = Math.max(app.optsUsed.maxEdges, ...jsons.map((j) => j.edges.length));
      const genomes = jsons.map((j) => genomeFromJSON(j, maxEdges));
      if (!genomes.length) {
        for (let i = 0; i < game.learners; i++) genomes.push(randomGenome(new Rng(mix(app.optsUsed.seed, 0, i, 77)), maxEdges, game.dims));
        log('No champions in the archive yet, watching random brains.');
      } else {
        log('Watching ' + genomes.length + ' champion genome(s) at ' + watchHz(game) + ' Hz.');
      }
      const seed = (app.optsUsed.seed + 0x5eed) >>> 0;
      app.watch = {
        active: true,
        starting: false,
        game,
        maxEdges,
        genomes,
        seed,
        env: game.createEnv(seed, game.defaultCfg(), true, new Int32Array(NSTATS)),
        brains: Array.from({ length: game.learners }, (_, a) => new Brain(copyGenome(genomes[a % genomes.length], maxEdges), maxEdges, game.dims)),
        lifeTicks: new Int32Array(game.learners),
        obs: new Float32Array(game.dims.nIn),
        actions: new Int32Array(game.learners),
        outputs: new Float32Array(game.learners * game.dims.nOut),
        tick: 0,
        acc: 0,
        reward: 0,
        deaths: 0,
        last: performance.now(),
      };
      $('watchBtn').textContent = 'Stop watching';
      requestAnimationFrame(watchFrame);
    } catch (e) {
      app.watch = { active: false, starting: false };
      showError('Watch failed: ' + errorText(e));
    }
  };

  const bind = () => {
    $('watchBtn').addEventListener('click', () => (app.watch.active ? stopWatch() : startWatch()));
    $('watchSpeed').addEventListener('input', renderWatchSpeed);
    renderWatchSpeed();
    $('ppoPolicies').addEventListener('change', () => {
      renderPopEditor();
      log('Policies changed: press Reset to rebuild for ' + Math.max(1, Math.min(RL_POP_MAX, Math.round(num('ppoPolicies', 1)))) + ' policies.');
    });
    $('game').addEventListener('change', () => {
      renderGameInfo();
      renderPopEditor();
      const cpu = usingCpu();
      $('forceCpu').checked = cpu;
      $('worlds').value = cpu ? 16 : 256;
      $('islands').value = cpu ? 2 : 4;
      log('Game changed: press Reset to rebuild for ' + currentGame().name + '.');
    });
    $('startBtn').addEventListener('click', () => setRunning(!app.running));
    $('resetBtn').addEventListener('click', () => doReset());
    $('benchBtn').addEventListener('click', () => withPause('Benchmark', runBenchmark));
    $('testBtn').addEventListener('click', () => withPause('Tests', runTests));
    $('csvBtn').addEventListener('click', exportCsv);
    $('champBtn').addEventListener('click', () => withPause('Export champions', exportChampions));
    $('importBtn').addEventListener('click', () => $('importFile').click());
    $('importFile').addEventListener('change', (e) => {
      const file = e.target.files[0];
      e.target.value = '';
      if (file) withPause('Import genomes', () => importGenomes(file));
    });
    $('saveBtn').addEventListener('click', () => withPause('Checkpoint', () => saveCheckpoint(false)));
    $('restoreBtn').addEventListener('click', () => restoreCheckpoint());
    for (const id of ['mutation', 'penaltyStart', 'penaltyEnd', 'penaltyRampTicks', 'difficultyStart', 'difficultyEnd', 'difficultyRampTicks', 'tpd', 'evalEvery']) $(id).addEventListener('change', applyLive);
    $('tpdAuto').addEventListener('change', () => {
      $('tpd').disabled = $('tpdAuto').checked;
      applyLive();
    });
    for (const [id] of PPO_FIELDS) $(id).addEventListener('change', applyLive);
    $('ppoPolicies').addEventListener('change', renderPopEditor);
    $('popApplyBtn').addEventListener('click', applyPopulation);
    $('trainer').addEventListener('change', () => {
      $('worlds').value = wantsPpo() ? PPO_WORLDS : usingCpu() ? 16 : 256;
      log('Trainer changed: press Reset to rebuild with ' + (wantsPpo() ? 'PPO' : 'Evolution') + '.');
    });
    $('forceCpu').addEventListener('change', () => {
      $('worlds').value = usingCpu() ? 16 : 256;
      $('islands').value = usingCpu() ? 2 : 4;
      log('Backend changed: press Reset to rebuild with the new backend.');
    });
    $('headless').addEventListener('change', () => {
      if ($('headless').checked) app.snapshot = null;
      if (app.backend && !app.running) refreshAll().catch((e) => showError(errorText(e)));
    });
    $('worldRange').addEventListener('input', (e) => selectWorld(parseInt(e.target.value, 10)));
    $('worldPick').addEventListener('change', (e) => selectWorld(parseInt(e.target.value, 10) || 0));
    $('prevWorld').addEventListener('click', () => selectWorld(app.world - 1));
    $('nextWorld').addEventListener('click', () => selectWorld(app.world + 1));
    $('heatCanvas').addEventListener('click', heatPick);
    $('heatCanvas').addEventListener('keydown', (e) => {
      if (e.key === 'ArrowRight') selectWorld(app.world + 1);
      if (e.key === 'ArrowLeft') selectWorld(app.world - 1);
      if (e.key === 'ArrowDown') selectWorld(app.world + app.heat.cols);
      if (e.key === 'ArrowUp') selectWorld(app.world - app.heat.cols);
    });
    $('worldCanvas').addEventListener('keydown', (e) => {
      if (e.key === 'ArrowRight') selectWorld(app.world + 1);
      if (e.key === 'ArrowLeft') selectWorld(app.world - 1);
    });
    document.addEventListener('keydown', (e) => {
      if (editingText(e) || e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === ' ' && e.target.tagName !== 'BUTTON' && e.target.tagName !== 'SUMMARY') {
        e.preventDefault();
        if (!$('startBtn').disabled) setRunning(!app.running);
      } else if (e.key === '[') selectWorld(app.world - 1);
      else if (e.key === ']') selectWorld(app.world + 1);
      else if (e.key === 'h' || e.key === 'H') {
        $('headless').checked = !$('headless').checked;
        $('headless').dispatchEvent(new Event('change'));
      }
    });
    document.addEventListener('visibilitychange', updateHiddenBanner);
    window.addEventListener('resize', () => renderViews());
    if (typeof ResizeObserver !== 'undefined') {
      const ro = new ResizeObserver(() => renderViews());
      for (const id of ['chartReward', 'chartEval', 'chartOps', 'chartEdges', 'chartArchive', 'chartPop']) ro.observe($(id));
    }
  };

  const start = async () => {
    bind();
    fillGameSelect();
    renderPopEditor();
    if (!gpuUsable()) {
      $('forceCpu').checked = true;
      $('worlds').value = 16;
      $('islands').value = 2;
    } else {
      $('forceCpu').checked = false;
    }
    const report = await probeDevice();
    renderDevice(report, null);
    if (report && report.probeError) showError('WebGPU probe failed: ' + report.probeError);
    await buildBackend(readOptions());
    renderDevice(report, app.backend ? app.backend.info : null);
    updateHiddenBanner();
    window.trainApp = app;
  };

  window.addEventListener('error', (e) => showError('Uncaught: ' + e.message));
  window.addEventListener('unhandledrejection', (e) => showError('Unhandled: ' + errorText(e.reason)));
  start();
}

appMain();
