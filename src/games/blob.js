import { STAT, mix } from '../core.js';
import { BLOB_WGSL } from './blob.wgsl.js';

const BLOB_WORLD = 8192;
const BLOB_PELLETS = 256;
const BLOB_AGENTS = 48;
const BLOB_LEARNERS = 24;
const BLOB_START_MASS = 160;
const BLOB_PELLET_MASS = 8;
const BLOB_VIEW = 2048;
const BLOB_VIEW2 = BLOB_VIEW * BLOB_VIEW;
const BLOB_RADIUS_K = 6;
const BLOB_SPEED_NUM = 540;
const BLOB_BOOST_MIN = 300;
const BLOB_FLEE2 = 1600 * 1600;
const BLOB_CHASE2 = 1800 * 1800;
const BLOB_FAR = 1 << 30;
const BLOB_WORDS = 904;
const BLOB_STAT = { PELLETS: STAT.GAME0, KILLS: STAT.GAME0 + 1, BOOST_TICKS: STAT.GAME0 + 2, MASS_GAINED: STAT.GAME0 + 3, EATEN: STAT.GAME0 + 4 };
const BLOB_DIR8 = [[128, 0], [90, 90], [0, 128], [-90, 90], [-128, 0], [-90, -90], [0, -128], [90, -90]];

export function blobIsqrt(x) {
  return Math.floor(Math.sqrt(x));
}

export function blobSector(dx, dy) {
  const ax = Math.abs(dx);
  const ay = Math.abs(dy);
  if (ay * 12 <= ax * 5) return dx >= 0 ? 0 : 4;
  if (ax * 12 <= ay * 5) return dy >= 0 ? 2 : 6;
  if (dx > 0) return dy > 0 ? 1 : 7;
  return dy > 0 ? 3 : 5;
}

const blobRadius = (mass) => blobIsqrt(mass) * BLOB_RADIUS_K;
const blobNear = (d2) => (d2 >= BLOB_VIEW2 ? 0 : (BLOB_VIEW - blobIsqrt(d2)) * (1 / BLOB_VIEW));

export class BlobEnv {
  constructor(seed, cfg, isEval, stats) {
    this.seed = seed >>> 0;
    this.cfg = cfg;
    this.isEval = isEval;
    this.stats = stats;
    this.px = new Int32Array(BLOB_PELLETS);
    this.py = new Int32Array(BLOB_PELLETS);
    this.ptimer = new Int32Array(BLOB_PELLETS);
    this.x = new Int32Array(BLOB_AGENTS);
    this.y = new Int32Array(BLOB_AGENTS);
    this.vx = new Int32Array(BLOB_AGENTS);
    this.vy = new Int32Array(BLOB_AGENTS);
    this.mass = new Int32Array(BLOB_AGENTS);
    this.mass0 = new Int32Array(BLOB_AGENTS);
    this.age = new Int32Array(BLOB_AGENTS);
    this.radius = new Int32Array(BLOB_AGENTS);
    this.ctlX = new Int32Array(BLOB_AGENTS);
    this.ctlY = new Int32Array(BLOB_AGENTS);
    this.boosting = new Int32Array(BLOB_AGENTS);
    this.boostCost = new Int32Array(BLOB_AGENTS);
    this.gain = new Int32Array(BLOB_AGENTS);
    this.eatenBy = new Int32Array(BLOB_AGENTS);
    this.secFood = new Int32Array(8);
    this.secThreat = new Int32Array(8);
    this.secPrey = new Int32Array(8);
    this.near = { food: BLOB_FAR, fx: 0, fy: 0, threat: BLOB_FAR, tx: 0, ty: 0, prey: BLOB_FAR, qx: 0, qy: 0 };
    this.reward = new Int32Array(BLOB_LEARNERS);
    this.dead = new Uint8Array(BLOB_LEARNERS);
    for (let p = 0; p < BLOB_PELLETS; p++) this.placePellet(p, 0);
    for (let a = 0; a < BLOB_AGENTS; a++) this.placeAgent(a, 0);
  }

  placePellet(p, tick) {
    this.px[p] = mix(this.seed, tick, p, 21) % BLOB_WORLD;
    this.py[p] = mix(this.seed, tick, p, 22) % BLOB_WORLD;
    this.ptimer[p] = 0;
  }

  placeAgent(a, tick) {
    this.x[a] = mix(this.seed, tick, a, 4) % BLOB_WORLD;
    this.y[a] = mix(this.seed, tick, a, 5) % BLOB_WORLD;
    this.vx[a] = 0;
    this.vy[a] = 0;
    this.mass[a] = BLOB_START_MASS;
    this.age[a] = 0;
    this.radius[a] = blobRadius(BLOB_START_MASS);
  }

  respawn(a, tick) {
    this.placeAgent(a, tick);
  }

  scan(a) {
    const sf = this.secFood, st = this.secThreat, sp = this.secPrey;
    sf.fill(BLOB_FAR);
    st.fill(BLOB_FAR);
    sp.fill(BLOB_FAR);
    const near = this.near;
    near.food = near.threat = near.prey = BLOB_FAR;
    const ax = this.x[a], ay = this.y[a], ar = this.radius[a];
    for (let p = 0; p < BLOB_PELLETS; p++) {
      if (this.ptimer[p] !== 0) continue;
      const dx = this.px[p] - ax, dy = this.py[p] - ay;
      const d2 = dx * dx + dy * dy;
      if (d2 >= BLOB_VIEW2) continue;
      const s = blobSector(dx, dy);
      if (d2 < sf[s]) sf[s] = d2;
      if (d2 < near.food) { near.food = d2; near.fx = dx; near.fy = dy; }
    }
    for (let j = 0; j < BLOB_AGENTS; j++) {
      if (j === a) continue;
      const dx = this.x[j] - ax, dy = this.y[j] - ay;
      const d2 = dx * dx + dy * dy;
      if (d2 >= BLOB_VIEW2) continue;
      const s = blobSector(dx, dy);
      const jr = this.radius[j];
      if (jr * 100 > ar * 115) {
        if (d2 < st[s]) st[s] = d2;
        if (d2 < near.threat) { near.threat = d2; near.tx = dx; near.ty = dy; }
      } else if (ar * 100 > jr * 115) {
        if (d2 < sp[s]) sp[s] = d2;
        if (d2 < near.prey) { near.prey = d2; near.qx = dx; near.qy = dy; }
      }
    }
  }

  observe(a, out, tick) {
    this.scan(a);
    for (let s = 0; s < 8; s++) {
      out[s] = blobNear(this.secFood[s]);
      out[8 + s] = blobNear(this.secThreat[s]);
      out[16 + s] = blobNear(this.secPrey[s]);
    }
    out[24] = blobIsqrt(this.mass[a]) * (1 / 128);
    out[25] = this.mass[a] > BLOB_BOOST_MIN ? 1 : 0;
    out[26] = this.x[a] * (1 / BLOB_WORLD);
    out[27] = (BLOB_WORLD - this.x[a]) * (1 / BLOB_WORLD);
    out[28] = this.y[a] * (1 / BLOB_WORLD);
    out[29] = (BLOB_WORLD - this.y[a]) * (1 / BLOB_WORLD);
  }

  normalized(dx, dy, a) {
    const len = blobIsqrt(dx * dx + dy * dy);
    this.ctlX[a] = len === 0 ? 0 : Math.trunc((dx * 128) / len);
    this.ctlY[a] = len === 0 ? 0 : Math.trunc((dy * 128) / len);
  }

  botControl(a, tick) {
    this.scan(a);
    const near = this.near;
    this.boosting[a] = 0;
    if (near.threat < BLOB_FLEE2) {
      this.normalized(-near.tx, -near.ty, a);
      this.boosting[a] = this.mass[a] > BLOB_BOOST_MIN ? 1 : 0;
    } else if (this.cfg.aggro === 1 && near.prey < BLOB_CHASE2) this.normalized(near.qx, near.qy, a);
    else if (near.food < BLOB_FAR) this.normalized(near.fx, near.fy, a);
    else {
      const dir = BLOB_DIR8[mix(this.seed, tick >>> 5, a, 22) & 7];
      this.ctlX[a] = dir[0];
      this.ctlY[a] = dir[1];
    }
  }

  teacherOutputs(a, tick, out) {
    this.botControl(a, tick);
    out.fill(0);
    out[0] = Math.max(this.ctlX[a], 0) / 128;
    out[1] = Math.max(-this.ctlX[a], 0) / 128;
    out[2] = Math.max(this.ctlY[a], 0) / 128;
    out[3] = Math.max(-this.ctlY[a], 0) / 128;
    out[4] = this.boosting[a] ? 1 : 0;
  }

  learnerControl(a, outputs) {
    const base = a * 5;
    const ox = Math.fround(outputs[base] - outputs[base + 1]);
    const oy = Math.fround(outputs[base + 2] - outputs[base + 3]);
    this.ctlX[a] = Math.max(-128, Math.min(128, Math.floor(ox * 128)));
    this.ctlY[a] = Math.max(-128, Math.min(128, Math.floor(oy * 128)));
    this.boosting[a] = outputs[base + 4] > 0.5 && this.mass[a] > BLOB_BOOST_MIN ? 1 : 0;
  }

  move(a) {
    const boost = this.boosting[a];
    let speed = Math.trunc(BLOB_SPEED_NUM / (blobIsqrt(this.mass[a]) + 8)) * (boost ? 2 : 1);
    const ox = this.ctlX[a], oy = this.ctlY[a];
    const denom = Math.max(blobIsqrt(ox * ox + oy * oy), 128);
    const tx = Math.trunc((ox * speed) / denom);
    const ty = Math.trunc((oy * speed) / denom);
    this.vx[a] += (tx - this.vx[a]) >> 2;
    this.vy[a] += (ty - this.vy[a]) >> 2;
    let nx = this.x[a] + this.vx[a];
    let ny = this.y[a] + this.vy[a];
    if (nx < 0) { nx = 0; this.vx[a] = 0; }
    if (nx > BLOB_WORLD - 1) { nx = BLOB_WORLD - 1; this.vx[a] = 0; }
    if (ny < 0) { ny = 0; this.vy[a] = 0; }
    if (ny > BLOB_WORLD - 1) { ny = BLOB_WORLD - 1; this.vy[a] = 0; }
    this.x[a] = nx;
    this.y[a] = ny;
    this.boostCost[a] = boost ? this.mass[a] >> 6 : 0;
    this.mass[a] -= this.boostCost[a];
    this.radius[a] = blobRadius(this.mass[a]);
    if (boost) this.stats[BLOB_STAT.BOOST_TICKS]++;
  }

  pelletPhase(tick) {
    for (let p = 0; p < BLOB_PELLETS; p++) {
      if (this.ptimer[p] > 0) {
        this.ptimer[p]--;
        if (this.ptimer[p] === 0) this.placePellet(p, tick);
        continue;
      }
      for (let a = 0; a < BLOB_AGENTS; a++) {
        const dx = this.px[p] - this.x[a], dy = this.py[p] - this.y[a];
        const r = this.radius[a];
        if (dx * dx + dy * dy < r * r) {
          this.gain[a] += BLOB_PELLET_MASS;
          this.ptimer[p] = this.cfg.pelletRespawn;
          this.stats[BLOB_STAT.PELLETS]++;
          break;
        }
      }
    }
  }

  victimPhase() {
    for (let j = 0; j < BLOB_AGENTS; j++) {
      this.eatenBy[j] = -1;
      let bestRadius = -1;
      for (let i = 0; i < BLOB_AGENTS; i++) {
        if (i === j) continue;
        const ri = this.radius[i], rj = this.radius[j];
        if (ri * 100 <= rj * 115) continue;
        const reach = ri - (rj >> 2);
        const dx = this.x[j] - this.x[i], dy = this.y[j] - this.y[i];
        if (reach > 0 && dx * dx + dy * dy < reach * reach && ri > bestRadius) {
          bestRadius = ri;
          this.eatenBy[j] = i;
        }
      }
      if (this.eatenBy[j] >= 0) {
        this.gain[this.eatenBy[j]] += (this.mass[j] * 3) >> 2;
        this.stats[BLOB_STAT.KILLS]++;
        if (j < BLOB_LEARNERS) this.stats[BLOB_STAT.EATEN]++;
      }
    }
  }

  applyPhase() {
    for (let a = 0; a < BLOB_AGENTS; a++) {
      const gained = this.gain[a];
      this.stats[BLOB_STAT.MASS_GAINED] += gained;
      this.mass[a] += gained;
      this.mass[a] -= this.mass[a] >> this.cfg.decayShift;
      this.age[a]++;
      let rewardFx = (blobIsqrt(this.mass[a] * 64) - blobIsqrt(this.mass0[a] * 64)) * 16;
      if (this.eatenBy[a] >= 0) rewardFx -= 1024;
      if (a < BLOB_LEARNERS) {
        this.reward[a] = rewardFx;
        this.dead[a] = this.eatenBy[a] >= 0 ? 1 : 0;
      } else {
        this.stats[STAT.REW_BASE] += rewardFx;
        this.stats[STAT.TICKS_BASE]++;
      }
    }
    for (let a = 0; a < BLOB_AGENTS; a++) this.radius[a] = blobRadius(this.mass[a]);
  }

  step(actions, outputs, tick) {
    for (let a = 0; a < BLOB_AGENTS; a++) {
      this.mass0[a] = this.mass[a];
      if (a < BLOB_LEARNERS) this.learnerControl(a, outputs);
      else this.botControl(a, tick);
    }
    for (let a = 0; a < BLOB_AGENTS; a++) this.move(a);
    this.gain.fill(0);
    this.pelletPhase(tick);
    this.victimPhase();
    this.applyPhase();
    for (let a = BLOB_LEARNERS; a < BLOB_AGENTS; a++) if (this.eatenBy[a] >= 0) this.respawn(a, tick);
  }

  pack() {
    const words = new Uint32Array(BLOB_WORDS);
    for (let p = 0; p < BLOB_PELLETS; p++) {
      words[p] = (this.px[p] | (this.py[p] << 16)) >>> 0;
      words[256 + p] = this.ptimer[p];
    }
    for (let a = 0; a < BLOB_AGENTS; a++) {
      const base = 512 + a * 8;
      words[base] = this.x[a] >>> 0;
      words[base + 1] = this.y[a] >>> 0;
      words[base + 2] = this.vx[a] >>> 0;
      words[base + 3] = this.vy[a] >>> 0;
      words[base + 4] = this.mass[a] >>> 0;
      words[base + 5] = this.age[a] >>> 0;
    }
    words[896] = this.cfg.pelletRespawn;
    words[897] = this.cfg.decayShift;
    words[898] = this.cfg.aggro;
    return words;
  }

  snapshot() {
    const pellets = [];
    for (let p = 0; p < BLOB_PELLETS; p++) if (this.ptimer[p] === 0) pellets.push({ x: this.px[p], y: this.py[p] });
    const agents = [];
    for (let a = 0; a < BLOB_AGENTS; a++) agents.push({ x: this.x[a], y: this.y[a], r: this.radius[a], mass: this.mass[a], evo: a < BLOB_LEARNERS, alive: true, boosting: this.boosting[a] === 1 });
    return { kind: 'blobs', width: BLOB_WORLD, height: BLOB_WORLD, pellets, agents };
  }
}

export function blobSnapshotFromWords(words) {
  const pellets = [];
  for (let p = 0; p < BLOB_PELLETS; p++) if (words[256 + p] === 0) pellets.push({ x: words[p] & 0xffff, y: words[p] >>> 16 });
  const agents = [];
  for (let a = 0; a < BLOB_AGENTS; a++) {
    const base = 512 + a * 8;
    const mass = words[base + 4] | 0;
    agents.push({ x: words[base] | 0, y: words[base + 1] | 0, r: blobRadius(mass), mass, evo: a < BLOB_LEARNERS, alive: true, boosting: false });
  }
  return { kind: 'blobs', width: BLOB_WORLD, height: BLOB_WORLD, pellets, agents };
}

export function blobValidate(words) {
  const problems = [];
  for (let a = 0; a < BLOB_AGENTS; a++) {
    const base = 512 + a * 8;
    const x = words[base] | 0, y = words[base + 1] | 0, mass = words[base + 4] | 0;
    if (x < 0 || x >= BLOB_WORLD || y < 0 || y >= BLOB_WORLD) problems.push('agent ' + a + ' out of bounds');
    if (mass <= 0) problems.push('agent ' + a + ' mass ' + mass);
  }
  return problems;
}

export function blobDensify(env, rng) {
  for (let a = 0; a < BLOB_AGENTS; a++) {
    env.x[a] = 3800 + Math.floor(rng.next() * 600);
    env.y[a] = 3800 + Math.floor(rng.next() * 600);
    env.mass[a] = 100 + Math.floor(rng.next() * rng.next() * 3000);
    env.vx[a] = 0;
    env.vy[a] = 0;
    env.radius[a] = blobRadius(env.mass[a]);
  }
  for (let p = 0; p < 64; p++) {
    env.px[p] = 3800 + Math.floor(rng.next() * 600);
    env.py[p] = 3800 + Math.floor(rng.next() * 600);
    env.ptimer[p] = 0;
  }
}

export function blobRender(ctx, snap, w, h) {
  const scale = Math.min(w, h) / snap.width;
  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = '#3fb950';
  for (const p of snap.pellets) ctx.fillRect(p.x * scale - 1, p.y * scale - 1, 2, 2);
  const ordered = snap.agents.slice().sort((a, b) => a.r - b.r);
  for (const a of ordered) {
    ctx.beginPath();
    ctx.arc(a.x * scale, a.y * scale, Math.max(2, a.r * scale), 0, Math.PI * 2);
    ctx.fillStyle = a.evo ? 'rgba(88,166,255,0.75)' : 'rgba(248,81,73,0.65)';
    ctx.fill();
    ctx.strokeStyle = a.boosting ? '#f0e130' : 'rgba(255,255,255,0.35)';
    ctx.stroke();
  }
}

export const BLOB = {
  id: 'blob',
  name: 'Blob arena (.io)',
  tickHz: 30,
  dims: { nIn: 30, nOut: 5, nNodes: 64 },
  agents: BLOB_AGENTS,
  learners: BLOB_LEARNERS,
  maxAge: 2700,
  worldWords: BLOB_WORDS,
  workgroupBytes: 4760,
  inputNames: ['foodE', 'foodSE', 'foodS', 'foodSW', 'foodW', 'foodNW', 'foodN', 'foodNE', 'threatE', 'threatSE', 'threatS', 'threatSW', 'threatW', 'threatNW', 'threatN', 'threatNE', 'preyE', 'preySE', 'preyS', 'preySW', 'preyW', 'preyNW', 'preyN', 'preyNE', 'mass', 'canBoost', 'wallLeft', 'wallRight', 'wallTop', 'wallBottom'],
  actionNames: ['+x', '-x', '+y', '-y', 'boost'],
  statNames: ['pellets', 'kills', 'boostTicks', 'massGained', 'eaten'],
  defaultCfg: () => ({ pelletRespawn: 60, decayShift: 10, aggro: 1 }),
  randomCfg: (seed) => ({ pelletRespawn: 30 + (mix(seed, 0, 0, 8) % 91), decayShift: 9 + (mix(seed, 1, 0, 8) % 3), aggro: mix(seed, 2, 0, 8) % 4 === 0 ? 1 : 0 }),
  createEnv: (seed, cfg, isEval, stats) => new BlobEnv(seed, cfg, isEval, stats),
  teacher: (env, a, tick, out) => env.teacherOutputs(a, tick, out),
  packEnv: (env) => env.pack(),
  snapshotFromWords: blobSnapshotFromWords,
  validateWords: blobValidate,
  densify: blobDensify,
  render: blobRender,
  wgsl: BLOB_WGSL
};
