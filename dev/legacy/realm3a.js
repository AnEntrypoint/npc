import { STAT, mix } from '../../src/core.js';
import { REALM_WGSL } from './realm3a.wgsl.js';

const R_WORLD = 8192;
const R_GRID = 32;
const R_NODES = 256;
const R_ENT = 64;
const R_LEARNERS = 16;
const R_PLAYERS = 32;
const R_MOB0 = 32;
const R_ANIMAL0 = 56;
const R_FIELDS = 32;
const R_BOSS0 = 52;
const R_GREAT0 = 204;
const R_GREAT_END = 236;
const R_GREATS = 32;
const R_NODE_TIMER_OFF = 256;
const R_ENT_OFF = 512;
const R_CFG_OFF = 2560;
const R_SAT_OFF = 2564;
const R_GPROG_OFF = 2568;
const R_WORDS = 2600;
const GREAT_PROGRESS_NEED = 60;
const GREAT_PROGRESS_CAP = 90;
const GREAT_TIMER = 1500;
const BOSS_HP = 500;
const BOSS_LEVEL = 10;
const BOSS_RESPAWN = 1200;
const BOSS_DAMAGE_BASE = 44;
const BOSS_COOLDOWN = 30;
const BOSS_WINDOW = 60;
const BOSS_GOLD = 60;
const BOSS_XP = 200;
const GREAT_GOLD = 15;
const GREAT_TOGETHER = 2;
const GUARD_PCT = 10;
const GUARD_CAP = 5;
const RARE_PER_GREAT = 3;
const RARE_CAP = 8;
const RARE_TIER3 = 2;
const GREAT_XP = 40;
const ALLY_RADIUS = 800;
const RAID_RADIUS = 2600;
const MARKET_FLOOR = 30;
const MARKET_CAP = 150;
const MARKET_DECAY = 6;
const ALIVE_REWARD = 1;
const GREAT_REWARD = 1024;
const BOSS_RARE = 2;
const BOSS_REWARD = 2048;
const XP_CAP = 1296;
const ARMOR_PCT = 10;
const AURA_CAP = 3;
const SHARE_XP = 3;
const SPRING_REGROW = 700;
const BERRY_REGROW = 700;
const MOB_DAMAGE_STEP = 4;
const MILESTONE = { GREAT: 0, BOSS: 1, MOB: 2 };
const MILESTONE_REWARD = [1024, 2048, 256];
const NEED_LEVEL = 60;
const NEED_REWARD = 4;
const R_FAR = 1 << 30;
const R_DAY_TICKS = 3600;
const R_NIGHT_FROM = 2400;
const R_MAX_AGE = 6000;
const R_TOWN_CELLS = [[8, 8], [24, 8], [8, 24], [24, 24]];
const R_DIR8 = [[128, 0], [90, 90], [0, 128], [-90, 90], [-128, 0], [-90, -90], [0, -128], [90, -90]];

const T_GRASS = 0, T_FOREST = 1, T_ROCK = 2, T_WATER = 3, T_SAND = 4;
const N_BERRY = 0, N_TREE = 1, N_ORE = 2, N_SPRING = 3, N_TOWN = 4, N_NONE = 7;
const NODE_FIRST_TOWN = 200;
const CLASSES = 9;
const CLASS_BOSS = 9;
const NEAR_SLOTS = 10;

const F = { X: 0, Y: 1, VX: 2, VY: 3, HP: 4, FOOD: 5, WATER: 6, CD: 7, CHAN: 8, STYLE: 9, XP0: 10, XP1: 11, XP2: 12, XP3: 13, WOOD: 14, ORE: 15, GOLD: 16, RAT: 17, TOOL: 18, WEAP: 19, AGE: 20, DEAD: 21, HX: 22, HY: 23, CHAN_NODE: 24, LAST_HIT: 25, LEVEL: 26, SPRINT: 27, RARE: 28, BOSS_ID: 29, BOSS_TICK: 30, ACH: 31 };
const ACT = { NONE: 0, INTERACT: 1, ATTACK: 2, CRAFT_TOOL: 3, CRAFT_WEAP: 4, BUY: 5 };
const R_STAT = { MOB_KILLS: STAT.GAME0, GREAT_HARVESTS: STAT.GAME0 + 1, BOSS_KILLS: STAT.GAME0 + 2, CRAFT_UPS: STAT.GAME0 + 3, HARVESTS: STAT.GAME0 + 4, ALLY_TICKS: STAT.GAME0 + 5, PLAYER_DEATHS: STAT.GAME0 + 6 };
const STYLE_RANGE = [200, 800, 500];
const STYLE_COOLDOWN = [10, 18, 24];
const MOB_STYLE_RANGE = [200, 800, 500];
const REACH_NODE = 320;
const REACH_TOWN = 500;
const SAFE_RADIUS = 600;
const BOSS_FLEE = 1300;
const STEER_AHEAD = 192;
const STEER_RAY = 20;
const SHAPE_NEED = 50;
const SHAPE_CLAMP = 8;

export function realmIsqrt(x) {
  return Math.floor(Math.sqrt(x));
}

export function realmSector(dx, dy) {
  const ax = Math.abs(dx);
  const ay = Math.abs(dy);
  if (ay * 12 <= ax * 5) return dx >= 0 ? 0 : 4;
  if (ax * 12 <= ay * 5) return dy >= 0 ? 2 : 6;
  if (dx > 0) return dy > 0 ? 1 : 7;
  return dy > 0 ? 3 : 5;
}

export function realmTerrain(seed, cx, cy) {
  for (const town of R_TOWN_CELLS) if (Math.abs(cx - town[0]) <= 3 && Math.abs(cy - town[1]) <= 3) return T_GRASS;
  const n3 = mix(seed, cx >> 3, cy >> 3, 30) & 255;
  const n1 = mix(seed, cx >> 1, cy >> 1, 31) & 255;
  const n0 = mix(seed, cx, cy, 32) & 255;
  const h = Math.trunc((n3 * 4 + n1 * 2 + n0) / 7);
  if (h < 60) return T_WATER;
  if (h < 80) return T_SAND;
  if (h < 170) return T_GRASS;
  if (h < 215) return T_FOREST;
  return T_ROCK;
}

const passableType = (t) => t !== T_WATER && t !== T_ROCK;
const nodeTypeOf = (j) => (j < 60 ? N_BERRY : j < 120 ? N_TREE : j < 168 ? N_ORE : j < 200 ? N_SPRING : j < 204 ? N_TOWN : j < 220 ? N_TREE : j < R_GREAT_END ? N_ORE : N_NONE);
const isGreat = (j) => j >= R_GREAT0 && j < R_GREAT_END;
const isBoss = (e) => e >= R_BOSS0 && e < R_ANIMAL0;
const damageScale = (d) => 30 + Math.trunc((70 * d) / 100);
const levelOf = (xp) => 1 + Math.min(9, realmIsqrt(xp >> 4));
const styleBeats = (a, b) => (a === 0 && b === 1) || (a === 1 && b === 2) || (a === 2 && b === 0);
const isNight = (tick) => tick % R_DAY_TICKS >= R_NIGHT_FROM;
const bossResist = (e) => (e - R_BOSS0) % 3;

export class RealmEnv {
  constructor(seed, cfg, isEval, stats) {
    this.seed = seed >>> 0;
    this.cfg = cfg;
    this.isEval = isEval;
    this.stats = stats;
    this.difficulty = 100;
    this.gprog = new Int32Array(R_GREATS);
    this.gdone = new Int32Array(R_GREATS);
    this.gcount = new Int32Array(R_GREATS);
    this.sat = new Int32Array(4);
    this.sunit = new Int32Array(R_PLAYERS);
    this.allies = new Int32Array(R_PLAYERS);
    this.part = new Int32Array(R_ENT);
    this.nodeX = new Int32Array(R_NODES);
    this.nodeY = new Int32Array(R_NODES);
    this.nodeTimer = new Int32Array(R_NODES);
    this.f = new Int32Array(R_ENT * R_FIELDS);
    this.ctlX = new Int32Array(R_ENT);
    this.ctlY = new Int32Array(R_ENT);
    this.act = new Int32Array(R_ENT);
    this.tgtNode = new Int32Array(R_ENT);
    this.atkTgt = new Int32Array(R_ENT);
    this.atkDmg = new Int32Array(R_ENT);
    this.atkStyle = new Int32Array(R_ENT);
    this.complete = new Int32Array(R_ENT);
    this.incoming = new Int32Array(R_ENT);
    this.killer = new Int32Array(R_ENT);
    this.victimGold = new Int32Array(R_ENT);
    this.victimLevel = new Int32Array(R_ENT);
    this.dies = new Int32Array(R_ENT);
    this.rew = new Int32Array(R_ENT);
    this.nd2 = new Int32Array(NEAR_SLOTS);
    this.ndx = new Int32Array(NEAR_SLOTS);
    this.ndy = new Int32Array(NEAR_SLOTS);
    this.nidx = new Int32Array(NEAR_SLOTS);
    this.sec = new Int32Array(CLASSES * 8);
    this.town = { d2: R_FAR, dx: 0, dy: 0, idx: -1 };
    this.reward = new Int32Array(R_LEARNERS);
    this.dead = new Uint8Array(R_LEARNERS);
    for (let j = 0; j < R_NODES; j++) this.placeNode(j);
    for (let e = 0; e < R_ENT; e++) this.spawn(e, 0);
  }

  terrainAt(x, y) {
    return realmTerrain(this.seed, x >> 8, y >> 8);
  }

  passableAt(x, y) {
    return passableType(this.terrainAt(x, y));
  }

  nearRock(cx, cy) {
    return realmTerrain(this.seed, (cx + 1) & 31, cy) === T_ROCK || realmTerrain(this.seed, (cx + 31) & 31, cy) === T_ROCK || realmTerrain(this.seed, cx, (cy + 1) & 31) === T_ROCK || realmTerrain(this.seed, cx, (cy + 31) & 31) === T_ROCK;
  }

  placeNode(j) {
    const type = nodeTypeOf(j);
    this.nodeTimer[j] = 0;
    if (type === N_NONE) { this.nodeX[j] = 0; this.nodeY[j] = 0; return; }
    if (type === N_TOWN) {
      const town = R_TOWN_CELLS[j - NODE_FIRST_TOWN];
      this.nodeX[j] = town[0] * 256 + 128;
      this.nodeY[j] = town[1] * 256 + 128;
      return;
    }
    let x = 0, y = 0;
    for (let k = 0; k < 8; k++) {
      x = mix(this.seed, j, k, 33) % R_WORLD;
      y = mix(this.seed, j, k, 34) % R_WORLD;
      const t = this.terrainAt(x, y);
      if (!passableType(t)) continue;
      const preferred = type === N_BERRY ? t === T_GRASS : type === N_TREE ? t === T_FOREST : type === N_ORE ? this.nearRock(x >> 8, y >> 8) : t === T_SAND;
      if (preferred || k >= 6) break;
    }
    if (!this.passableAt(x, y)) { x = 2048 + (j & 7) * 64; y = 2048 + (j >> 3) * 16; }
    this.nodeX[j] = x;
    this.nodeY[j] = y;
  }

  nearestTownAt(ex, ey) {
    const t = this.town;
    t.d2 = R_FAR; t.idx = -1;
    for (let j = NODE_FIRST_TOWN; j < NODE_FIRST_TOWN + 4; j++) {
      const dx = this.nodeX[j] - ex, dy = this.nodeY[j] - ey;
      const d2 = dx * dx + dy * dy;
      if (d2 < t.d2) { t.d2 = d2; t.dx = dx; t.dy = dy; t.idx = j; }
    }
  }

  nearestTown(e) {
    this.nearestTownAt(this.f[e * R_FIELDS + F.X], this.f[e * R_FIELDS + F.Y]);
  }

  homeNearTown(e) {
    this.nearestTownAt(this.f[e * R_FIELDS + F.HX], this.f[e * R_FIELDS + F.HY]);
    return this.town.d2 < RAID_RADIUS * RAID_RADIUS;
  }

  allyCount(e) {
    const f = this.f;
    const ex = f[e * R_FIELDS + F.X], ey = f[e * R_FIELDS + F.Y];
    let count = 0;
    for (let j = 0; j < R_PLAYERS; j++) {
      if (j === e || !this.alive(j)) continue;
      const dx = f[j * R_FIELDS + F.X] - ex, dy = f[j * R_FIELDS + F.Y] - ey;
      if (dx * dx + dy * dy < ALLY_RADIUS * ALLY_RADIUS) count++;
    }
    return count;
  }

  maxHp(e) {
    const b = e * R_FIELDS;
    if (e >= R_PLAYERS) return 30 + 15 * this.f[b + F.LEVEL];
    return 90 + 10 * Math.max(levelOf(this.f[b + F.XP0]), levelOf(this.f[b + F.XP1]), levelOf(this.f[b + F.XP2]));
  }

  power(e) {
    const b = e * R_FIELDS;
    return levelOf(this.f[b + F.XP0]) + levelOf(this.f[b + F.XP1]) + levelOf(this.f[b + F.XP2]) + 2 * this.f[b + F.WEAP];
  }

  spawn(e, tick) {
    const b = e * R_FIELDS;
    const f = this.f;
    f.fill(0, b, b + R_FIELDS);
    f[b + F.CHAN_NODE] = -1;
    if (e < R_PLAYERS) {
      const townIdx = mix(this.seed, tick, e, 40) & 3;
      const town = R_TOWN_CELLS[townIdx];
      f[b + F.X] = town[0] * 256 + 128 + (mix(this.seed, tick, e, 41) % 601) - 300;
      f[b + F.Y] = town[1] * 256 + 128 + (mix(this.seed, tick, e, 42) % 601) - 300;
      f[b + F.HP] = 100;
      f[b + F.FOOD] = 80;
      f[b + F.WATER] = 80;
      f[b + F.LAST_HIT] = 1000;
      return;
    }
    let x = 0, y = 0;
    let placed = false;
    if (e < R_BOSS0 && isNight(tick) && mix(this.seed, tick, e, 45) % 100 < this.difficulty) {
      const town = R_TOWN_CELLS[mix(this.seed, tick, e, 46) & 3];
      const dir = R_DIR8[mix(this.seed, tick, e, 47) & 7];
      const reach = 9 + (mix(this.seed, tick, e, 48) % 5);
      x = town[0] * 256 + 128 + dir[0] * reach;
      y = town[1] * 256 + 128 + dir[1] * reach;
      placed = this.passableAt(x, y);
    }
    if (!placed) {
      const minTown = e >= R_BOSS0 ? 2400 : 1200;
      for (let k = 0; k < 8; k++) {
        x = mix(this.seed, tick, e * 8 + k, 43) % R_WORLD;
        y = mix(this.seed, tick, e * 8 + k, 44) % R_WORLD;
        if (!this.passableAt(x, y)) continue;
        this.nearestTownAt(x, y);
        if (e >= R_ANIMAL0 || this.town.d2 > minTown * minTown) break;
      }
      if (!this.passableAt(x, y)) { x = 4096; y = 4096 + e; }
    }
    f[b + F.X] = x;
    f[b + F.Y] = y;
    f[b + F.HX] = x;
    f[b + F.HY] = y;
    this.nearestTownAt(x, y);
    if (isBoss(e)) {
      f[b + F.LEVEL] = BOSS_LEVEL;
      f[b + F.STYLE] = 1 + (e & 1);
      f[b + F.HP] = BOSS_HP;
    } else {
      if (e < R_ANIMAL0) {
        f[b + F.LEVEL] = 1 + Math.min(7, Math.floor(realmIsqrt(this.town.d2) / 600));
        f[b + F.STYLE] = e % 3;
      } else f[b + F.LEVEL] = 1;
      f[b + F.HP] = e < R_ANIMAL0 ? 30 + 15 * f[b + F.LEVEL] : 20;
    }
    f[b + F.FOOD] = 100;
    f[b + F.WATER] = 100;
  }

  respawn(e, tick) {
    this.spawn(e, tick);
  }

  alive(e) {
    return this.f[e * R_FIELDS + F.DEAD] === 0;
  }

  viewRadius(tick) {
    return isNight(tick) ? 1400 : 2048;
  }

  scan(e, tick, withSectors) {
    const f = this.f;
    const b = e * R_FIELDS;
    this.nd2.fill(R_FAR);
    if (withSectors) this.sec.fill(R_FAR);
    const view = this.viewRadius(tick);
    const view2 = view * view;
    const ex = f[b + F.X], ey = f[b + F.Y];
    const asBot = !withSectors;
    for (let j = 0; j < R_NODES; j++) {
      const type = nodeTypeOf(j);
      if (type === N_NONE || this.nodeTimer[j] !== 0) continue;
      if (asBot && isGreat(j)) continue;
      const dx = this.nodeX[j] - ex, dy = this.nodeY[j] - ey;
      const d2 = dx * dx + dy * dy;
      if (d2 >= view2) continue;
      this.consider(type, dx, dy, d2, j, withSectors);
    }
    const myPower = this.power(e);
    for (let j = 0; j < R_ENT; j++) {
      if (j === e || !this.alive(j)) continue;
      const jb = j * R_FIELDS;
      const dx = f[jb + F.X] - ex, dy = f[jb + F.Y] - ey;
      const d2 = dx * dx + dy * dy;
      if (d2 >= view2) continue;
      const cls = j < R_PLAYERS ? (this.power(j) >= myPower ? 8 : 7) : isBoss(j) && asBot ? CLASS_BOSS : j < R_ANIMAL0 ? 5 : 6;
      this.consider(cls, dx, dy, d2, j, withSectors);
    }
  }

  consider(cls, dx, dy, d2, idx, withSectors) {
    if (withSectors) {
      const s = realmSector(dx, dy);
      if (d2 < this.sec[cls * 8 + s]) this.sec[cls * 8 + s] = d2;
    }
    if (d2 < this.nd2[cls]) { this.nd2[cls] = d2; this.ndx[cls] = dx; this.ndy[cls] = dy; this.nidx[cls] = idx; }
  }

  observe(a, out, tick) {
    const f = this.f;
    const b = a * R_FIELDS;
    this.scan(a, tick, true);
    const view = this.viewRadius(tick);
    for (let c = 0; c < CLASSES; c++) {
      for (let s = 0; s < 8; s++) {
        const d2 = this.sec[c * 8 + s];
        out[c * 8 + s] = d2 >= view * view ? 0 : (view - realmIsqrt(d2)) * (1 / 2048);
      }
    }
    this.nearestTown(a);
    const l0 = levelOf(f[b + F.XP0]), l1 = levelOf(f[b + F.XP1]), l2 = levelOf(f[b + F.XP2]), l3 = levelOf(f[b + F.XP3]);
    out[72] = f[b + F.HP] * (1 / 256);
    out[73] = f[b + F.FOOD] * (1 / 128);
    out[74] = f[b + F.WATER] * (1 / 128);
    out[75] = l0 * (1 / 16);
    out[76] = l1 * (1 / 16);
    out[77] = l2 * (1 / 16);
    out[78] = l3 * (1 / 16);
    out[79] = f[b + F.WOOD] * (1 / 8);
    out[80] = f[b + F.ORE] * (1 / 8);
    out[81] = Math.min(f[b + F.GOLD], 64) * (1 / 64);
    out[82] = f[b + F.RAT] * (1 / 4);
    out[83] = f[b + F.TOOL] * (1 / 4);
    out[84] = f[b + F.WEAP] * (1 / 4);
    out[85] = f[b + F.CD] === 0 ? 1 : 0;
    out[86] = isNight(tick) ? 1 : 0;
    out[87] = this.town.d2 < REACH_TOWN * REACH_TOWN ? 1 : 0;
    const len = realmIsqrt(this.town.dx * this.town.dx + this.town.dy * this.town.dy);
    out[88] = len === 0 ? 0 : Math.trunc((this.town.dx * 128) / len) * (1 / 128);
    out[89] = len === 0 ? 0 : Math.trunc((this.town.dy * 128) / len) * (1 / 128);
    out[90] = this.allyCount(a) * (1 / 8);
    out[91] = f[b + F.RARE] * (1 / 8);
    out[92] = this.nd2[1] !== R_FAR && isGreat(this.nidx[1]) ? 1 : 0;
    out[93] = this.nd2[2] !== R_FAR && isGreat(this.nidx[2]) ? 1 : 0;
    out[94] = this.inSafeZone(f[b + F.X], f[b + F.Y]) ? 1 : 0;
    out[95] = f[b + F.LAST_HIT] < 40 ? 1 : 0;
    let bossD2 = R_FAR, bossIdx = -1;
    for (let j = R_BOSS0; j < R_ANIMAL0; j++) {
      if (!this.alive(j)) continue;
      const dx = f[j * R_FIELDS + F.X] - f[b + F.X], dy = f[j * R_FIELDS + F.Y] - f[b + F.Y];
      const d2 = dx * dx + dy * dy;
      if (d2 < bossD2) { bossD2 = d2; bossIdx = j; }
    }
    const bossSeen = bossD2 < view * view;
    out[96] = bossSeen ? (view - realmIsqrt(bossD2)) * (1 / 2048) : 0;
    const resist = bossSeen ? bossResist(bossIdx) : -1;
    out[97] = resist === 0 ? 1 : 0;
    out[98] = resist === 1 ? 1 : 0;
    out[99] = resist === 2 ? 1 : 0;
    let greatD2 = R_FAR, greatIdx = -1;
    for (let j = R_GREAT0; j < R_GREAT_END; j++) {
      if (this.nodeTimer[j] !== 0) continue;
      const dx = this.nodeX[j] - f[b + F.X], dy = this.nodeY[j] - f[b + F.Y];
      const d2 = dx * dx + dy * dy;
      if (d2 < greatD2) { greatD2 = d2; greatIdx = j; }
    }
    out[100] = greatD2 < view * view ? this.gprog[greatIdx - R_GREAT0] * (1 / 64) : 0;
    out[101] = ((tick % R_DAY_TICKS) >> 4) * (1 / 256);
  }

  toward(e, dx, dy) {
    const len = realmIsqrt(dx * dx + dy * dy);
    this.ctlX[e] = len === 0 ? 0 : Math.trunc((dx * 128) / len);
    this.ctlY[e] = len === 0 ? 0 : Math.trunc((dy * 128) / len);
  }

  wander(e, tick) {
    const dir = R_DIR8[mix(this.seed, tick >>> 5, e, 22) & 7];
    this.ctlX[e] = dir[0];
    this.ctlY[e] = dir[1];
  }

  inSafeZone(x, y) {
    for (let j = NODE_FIRST_TOWN; j < NODE_FIRST_TOWN + 4; j++) {
      const dx = this.nodeX[j] - x, dy = this.nodeY[j] - y;
      if (dx * dx + dy * dy < SAFE_RADIUS * SAFE_RADIUS) return true;
    }
    return false;
  }

  attackDamage(e, style) {
    const b = e * R_FIELDS;
    if (e >= R_PLAYERS) {
      const base = isBoss(e) ? BOSS_DAMAGE_BASE : 4 + MOB_DAMAGE_STEP * this.f[b + F.LEVEL];
      return Math.trunc((base * this.cfg.mobPct * damageScale(this.difficulty)) / 10000);
    }
    const weap = this.f[b + F.WEAP];
    if (style === 0) return 6 + 2 * levelOf(this.f[b + F.XP0]) + 4 * weap;
    if (style === 1) return 4 + 2 * levelOf(this.f[b + F.XP1]) + 3 * weap;
    return 5 + 2 * levelOf(this.f[b + F.XP2]) + 3 * weap;
  }

  tryAttack(e, style, allowPlayers) {
    const f = this.f;
    const b = e * R_FIELDS;
    if (f[b + F.CD] !== 0) return false;
    const range = e >= R_PLAYERS ? MOB_STYLE_RANGE[style] : STYLE_RANGE[style];
    const ex = f[b + F.X], ey = f[b + F.Y];
    let best = -1, bestD2 = R_FAR;
    if (e < R_PLAYERS && this.inSafeZone(ex, ey)) return false;
    for (let j = 0; j < R_ENT; j++) {
      if (j === e || !this.alive(j)) continue;
      if (e < R_PLAYERS) {
        if (j < R_PLAYERS && (!allowPlayers || this.cfg.pvp === 0)) continue;
      } else if (j >= R_PLAYERS) continue;
      const jb = j * R_FIELDS;
      const dx = f[jb + F.X] - ex, dy = f[jb + F.Y] - ey;
      const d2 = dx * dx + dy * dy;
      if (d2 > range * range || d2 >= bestD2) continue;
      if (j < R_PLAYERS && this.inSafeZone(f[jb + F.X], f[jb + F.Y])) continue;
      best = j;
      bestD2 = d2;
    }
    if (best < 0) return false;
    let damage = this.attackDamage(e, style);
    const defenderStyle = f[best * R_FIELDS + F.STYLE];
    if (styleBeats(style, defenderStyle)) damage = (damage * 3) >> 1;
    else if (styleBeats(defenderStyle, style)) damage = (damage * 3) >> 2;
    if (e < R_PLAYERS && best < R_PLAYERS) damage >>= 1;
    if (isBoss(best) && bossResist(best) === style) damage >>= 2;
    this.atkTgt[e] = best;
    this.atkDmg[e] = damage;
    this.atkStyle[e] = style;
    this.act[e] = ACT.ATTACK;
    f[b + F.CD] = isBoss(e) ? BOSS_COOLDOWN : e >= R_PLAYERS ? 20 : STYLE_COOLDOWN[style];
    return true;
  }

  nearestInteractNode(e, allowGreat) {
    const f = this.f;
    const b = e * R_FIELDS;
    const ex = f[b + F.X], ey = f[b + F.Y];
    let best = -1, bestD2 = REACH_TOWN * REACH_TOWN + 1;
    for (let j = 0; j < R_NODES; j++) {
      const type = nodeTypeOf(j);
      if (type === N_NONE || this.nodeTimer[j] !== 0) continue;
      if (!allowGreat && isGreat(j)) continue;
      const reach = type === N_TOWN ? REACH_TOWN : REACH_NODE;
      const dx = this.nodeX[j] - ex, dy = this.nodeY[j] - ey;
      const d2 = dx * dx + dy * dy;
      if (d2 <= reach * reach && d2 < bestD2) { best = j; bestD2 = d2; }
    }
    return best;
  }

  learnerDecide(e, outputs, tick) {
    const base = e * 12;
    const ox = Math.fround(outputs[base] - outputs[base + 1]);
    const oy = Math.fround(outputs[base + 2] - outputs[base + 3]);
    this.ctlX[e] = Math.max(-128, Math.min(128, Math.floor(ox * 128)));
    this.ctlY[e] = Math.max(-128, Math.min(128, Math.floor(oy * 128)));
    this.f[e * R_FIELDS + F.SPRINT] = outputs[base + 4] > 0.5 ? 1 : 0;
    let style = 0;
    if (outputs[base + 7] > outputs[base + 6]) style = 1;
    if (outputs[base + 8] > outputs[base + 6 + style]) style = 2;
    if (outputs[base + 6 + style] > 0.5 && this.tryAttack(e, style, true)) return;
    if (outputs[base + 5] > 0.5) { this.act[e] = ACT.INTERACT; this.tgtNode[e] = this.nearestInteractNode(e, true); }
    else if (outputs[base + 9] > 0.5) this.act[e] = ACT.CRAFT_TOOL;
    else if (outputs[base + 10] > 0.5) this.act[e] = ACT.CRAFT_WEAP;
    else if (outputs[base + 11] > 0.5) this.act[e] = ACT.BUY;
  }

  canCraftTool(e) {
    const b = e * R_FIELDS;
    const t = this.f[b + F.TOOL];
    return t < 3 && this.f[b + F.WOOD] >= 2 + 2 * t && this.f[b + F.ORE] >= 1 + t && (t < 2 || this.f[b + F.RARE] >= RARE_TIER3);
  }

  canCraftWeap(e) {
    const b = e * R_FIELDS;
    const w = this.f[b + F.WEAP];
    return w < 3 && this.f[b + F.ORE] >= 2 + 2 * w && this.f[b + F.WOOD] >= 1 + w && (w < 2 || this.f[b + F.RARE] >= RARE_TIER3);
  }

  openAt(x, y) {
    return x >= 0 && x < R_WORLD && y >= 0 && y < R_WORLD && this.passableAt(x, y);
  }

  clearAhead(px, py, dx, dy, len) {
    const sx = Math.trunc((dx * STEER_AHEAD) / len), sy = Math.trunc((dy * STEER_AHEAD) / len);
    return this.openAt(px + (sx >> 1), py + (sy >> 1)) && this.openAt(px + sx, py + sy);
  }

  botToward(e, dx, dy) {
    this.toward(e, dx, dy);
    const px = this.f[e * R_FIELDS + F.X], py = this.f[e * R_FIELDS + F.Y];
    const len = realmIsqrt(dx * dx + dy * dy);
    if (len === 0 || this.clearAhead(px, py, dx, dy, len)) return;
    let best = -1, bestScore = R_FAR;
    for (let k = 0; k < 8; k++) {
      let reach = 0;
      while (reach < STEER_RAY && this.openAt(px + R_DIR8[k][0] * (reach + 1), py + R_DIR8[k][1] * (reach + 1))) reach++;
      if (reach === 0) continue;
      const ex = dx - R_DIR8[k][0] * reach, ey = dy - R_DIR8[k][1] * reach;
      const score = Math.trunc((ex * ex + ey * ey) / 64);
      if (score < bestScore) { bestScore = score; best = k; }
    }
    if (best >= 0) { this.ctlX[e] = R_DIR8[best][0]; this.ctlY[e] = R_DIR8[best][1]; }
  }

  botDecide(e, tick) {
    this.botAct(e, tick);
  }

  botAct(e, tick) {
    const f = this.f;
    const b = e * R_FIELDS;
    this.scan(e, tick, false);
    this.nearestTown(e);
    this.ctlX[e] = 0;
    this.ctlY[e] = 0;
    const nd2 = this.nd2;
    const inTown = this.town.d2 <= REACH_TOWN * REACH_TOWN;
    const l0 = levelOf(f[b + F.XP0]), l1 = levelOf(f[b + F.XP1]), l2 = levelOf(f[b + F.XP2]);
    let style = 0;
    if (l1 > l0) style = 1;
    if (l2 > (style === 0 ? l0 : l1)) style = 2;
    const allowGreat = false;
    const desperate = f[b + F.WATER] < 50 || (f[b + F.FOOD] < 50 && f[b + F.RAT] === 0);
    const bossHold = nd2[CLASS_BOSS] < BOSS_FLEE * BOSS_FLEE && !desperate;
    if (bossHold && !inTown) {
      this.botToward(e, this.town.dx, this.town.dy);
      return;
    }
    const hostile = nd2[5] < 500 * 500 && f[b + F.HP] > 50 && !this.inSafeZone(f[b + F.X], f[b + F.Y]);
    if (hostile) {
      if (this.tryAttack(e, style, false)) return;
      this.botToward(e, this.ndx[5], this.ndy[5]);
      return;
    }
    const goTo = (cls) => {
      if (nd2[cls] === R_FAR) { this.wander(e, tick); return; }
      if (nd2[cls] <= REACH_NODE * REACH_NODE) { this.act[e] = ACT.INTERACT; this.tgtNode[e] = this.nearestInteractNode(e, allowGreat); return; }
      this.botToward(e, this.ndx[cls], this.ndy[cls]);
    };
    if (f[b + F.WATER] < 50) { goTo(3); return; }
    if (f[b + F.FOOD] < 50 && f[b + F.RAT] === 0) { goTo(0); return; }
    if (inTown) {
      if (this.canCraftTool(e)) { this.act[e] = ACT.CRAFT_TOOL; return; }
      if (this.canCraftWeap(e)) { this.act[e] = ACT.CRAFT_WEAP; return; }
      if (f[b + F.WOOD] + f[b + F.ORE] > 0) { this.act[e] = ACT.INTERACT; this.tgtNode[e] = this.nearestInteractNode(e, allowGreat); return; }
      if (f[b + F.GOLD] >= 15 && f[b + F.RAT] < 2) { this.act[e] = ACT.BUY; return; }
    }
    if (bossHold) return;
    if (f[b + F.WOOD] + f[b + F.ORE] >= 8 || (f[b + F.WOOD] >= 4 && f[b + F.ORE] >= 3)) {
      if (this.town.d2 > REACH_TOWN * REACH_TOWN) this.botToward(e, this.town.dx, this.town.dy);
      else { this.act[e] = ACT.INTERACT; this.tgtNode[e] = this.nearestInteractNode(e, allowGreat); }
      return;
    }
    goTo(f[b + F.ORE] < f[b + F.WOOD] ? 2 : 1);
  }

  teacherOutputs(a, tick, out) {
    const f = this.f;
    const b = a * R_FIELDS;
    const cooldown = f[b + F.CD];
    const style = f[b + F.STYLE];
    this.act[a] = 0;
    this.tgtNode[a] = -1;
    this.botDecide(a, tick);
    out.fill(0);
    out[0] = Math.max(this.ctlX[a], 0) / 128;
    out[1] = Math.max(-this.ctlX[a], 0) / 128;
    out[2] = Math.max(this.ctlY[a], 0) / 128;
    out[3] = Math.max(-this.ctlY[a], 0) / 128;
    const action = this.act[a];
    if (action === ACT.ATTACK) out[6 + this.atkStyle[a]] = 1;
    else if (action === ACT.INTERACT) out[5] = 1;
    else if (action === ACT.CRAFT_TOOL) out[9] = 1;
    else if (action === ACT.CRAFT_WEAP) out[10] = 1;
    else if (action === ACT.BUY) out[11] = 1;
    f[b + F.CD] = cooldown;
    f[b + F.STYLE] = style;
    this.act[a] = 0;
    this.tgtNode[a] = -1;
  }

  mobDecide(e, tick) {
    const f = this.f;
    const b = e * R_FIELDS;
    const ex = f[b + F.X], ey = f[b + F.Y];
    this.ctlX[e] = 0;
    this.ctlY[e] = 0;
    let best = -1, bestD2 = R_FAR;
    for (let j = 0; j < R_PLAYERS; j++) {
      if (!this.alive(j)) continue;
      const dx = f[j * R_FIELDS + F.X] - ex, dy = f[j * R_FIELDS + F.Y] - ey;
      const d2 = dx * dx + dy * dy;
      if (d2 < bestD2) { bestD2 = d2; best = j; }
    }
    if (e >= R_ANIMAL0) {
      if (best >= 0 && bestD2 < 500 * 500) this.toward(e, ex - f[best * R_FIELDS + F.X], ey - f[best * R_FIELDS + F.Y]);
      else if (((tick >>> 4) + e) % 3 === 0) this.wander(e, tick);
      return;
    }
    const aggro = isNight(tick) ? (this.homeNearTown(e) ? 900 + Math.trunc((900 * this.difficulty) / 100) : 1200) : 900;
    const style = f[b + F.STYLE];
    const range = MOB_STYLE_RANGE[style];
    const hx = f[b + F.HX] - ex, hy = f[b + F.HY] - ey;
    if (best >= 0 && bestD2 < aggro * aggro && hx * hx + hy * hy < 3000 * 3000) {
      const dx = f[best * R_FIELDS + F.X] - ex, dy = f[best * R_FIELDS + F.Y] - ey;
      if (bestD2 > (range * 3 / 4 | 0) * (range * 3 / 4 | 0)) this.toward(e, dx, dy);
      this.tryAttack(e, style, false);
    } else if (hx * hx + hy * hy > 600 * 600) this.toward(e, hx, hy);
    else if (((tick >>> 5) + e) % 2 === 0) this.wander(e, tick);
  }

  moveEntity(e, tick) {
    const f = this.f;
    const b = e * R_FIELDS;
    const sprint = e < R_PLAYERS && f[b + F.SPRINT] === 1;
    let base = e < R_PLAYERS ? 26 : e < R_ANIMAL0 ? 20 : 24;
    if (e >= R_MOB0 && e < R_ANIMAL0 && isNight(tick) && this.homeNearTown(e)) base += Math.trunc((12 * this.difficulty) / 100);
    const speed = sprint ? (base * 8) >> 2 : base;
    const ox = this.ctlX[e], oy = this.ctlY[e];
    const denom = Math.max(realmIsqrt(ox * ox + oy * oy), 128);
    const tx = Math.trunc((ox * speed) / denom);
    const ty = Math.trunc((oy * speed) / denom);
    f[b + F.VX] += (tx - f[b + F.VX]) >> 2;
    f[b + F.VY] += (ty - f[b + F.VY]) >> 2;
    let nx = f[b + F.X] + f[b + F.VX];
    let ny = f[b + F.Y] + f[b + F.VY];
    if (nx < 0 || nx > R_WORLD - 1 || !this.passableAt(nx, f[b + F.Y])) { nx = f[b + F.X]; f[b + F.VX] = 0; }
    if (ny < 0 || ny > R_WORLD - 1 || !this.passableAt(nx, ny)) { ny = f[b + F.Y]; f[b + F.VY] = 0; }
    f[b + F.X] = nx;
    f[b + F.Y] = ny;
  }

  harvestNeed(type, tool) {
    if (type === N_BERRY) return 4;
    if (type === N_SPRING) return 2;
    return Math.max(2, (type === N_TREE ? 10 : 14) - 2 * tool);
  }

  channelPhase(e) {
    const f = this.f;
    const b = e * R_FIELDS;
    this.complete[e] = 0;
    const node = this.act[e] === ACT.INTERACT ? this.tgtNode[e] : -1;
    const type = node >= 0 ? nodeTypeOf(node) : N_NONE;
    if (node < 0 || type === N_TOWN || isGreat(node)) {
      f[b + F.CHAN_NODE] = -1;
      f[b + F.CHAN] = 0;
      return;
    }
    if (f[b + F.CHAN_NODE] === node) f[b + F.CHAN]++;
    else { f[b + F.CHAN_NODE] = node; f[b + F.CHAN] = 1; }
    if (f[b + F.CHAN] >= this.harvestNeed(type, f[b + F.TOOL])) this.complete[e] = 1;
  }

  award(e, gold, craftUp) {
    this.rew[e] += gold * 32 + craftUp * 1024;
    this.stats[R_STAT.CRAFT_UPS] += craftUp;
  }

  gainXp(e, field, xp) {
    const slot = e * R_FIELDS + field;
    const before = this.f[slot];
    this.f[slot] = before + xp;
    this.rew[e] += (Math.min(before + xp, XP_CAP) - Math.min(before, XP_CAP)) * 4;
  }

  milestone(e, which) {
    const slot = e * R_FIELDS + F.ACH;
    if ((this.f[slot] >> which) & 1) return;
    this.f[slot] |= 1 << which;
    this.rew[e] += MILESTONE_REWARD[which];
  }

  restore(e, field, amount) {
    const slot = e * R_FIELDS + field;
    const before = this.f[slot];
    const after = Math.min(100, before + amount);
    this.f[slot] = after;
    if (before < NEED_LEVEL) this.rew[e] += (after - before) * NEED_REWARD;
  }

  greatPhase(g) {
    const node = R_GREAT0 + g;
    let harvesters = 0;
    for (let p = 0; p < R_PLAYERS; p++) if (this.alive(p) && this.act[p] === ACT.INTERACT && this.tgtNode[p] === node) harvesters++;
    let progress = this.gprog[g];
    progress = harvesters > 0 ? Math.min(GREAT_PROGRESS_CAP, progress + harvesters) : Math.max(0, progress - 2);
    this.gdone[g] = 0;
    this.gcount[g] = harvesters;
    if (harvesters >= 1 + Math.trunc((2 * this.difficulty) / 100) && progress >= GREAT_PROGRESS_NEED) {
      this.gdone[g] = 1;
      progress = 0;
      this.nodeTimer[node] = GREAT_TIMER;
    }
    this.gprog[g] = progress;
  }

  marketPhase(i, tick) {
    let sold = 0;
    for (let p = 0; p < R_PLAYERS; p++) if (this.tgtNode[p] === NODE_FIRST_TOWN + i) sold += this.sunit[p];
    let level = Math.min(MARKET_CAP, this.sat[i] + sold);
    if ((tick + i) % MARKET_DECAY === 0 && level > 0) level--;
    this.sat[i] = level;
  }

  participant(p, v, tick) {
    const b = p * R_FIELDS;
    if (!this.alive(p)) return false;
    if (this.f[b + F.BOSS_ID] === v && tick - this.f[b + F.BOSS_TICK] <= BOSS_WINDOW) return true;
    return this.act[p] === ACT.ATTACK && this.atkTgt[p] === v;
  }

  interactPhase(e) {
    const f = this.f;
    const b = e * R_FIELDS;
    const action = this.act[e];
    if (action === ACT.INTERACT) {
      const node = this.tgtNode[e];
      if (node < 0) return;
      const type = nodeTypeOf(node);
      if (type === N_TOWN) {
        const price = Math.max(MARKET_FLOOR, 100 - this.sat[node - NODE_FIRST_TOWN]);
        const gold = Math.trunc(((f[b + F.WOOD] * 2 + f[b + F.ORE] * 5) * price) / 100);
        this.sunit[e] = f[b + F.WOOD] + f[b + F.ORE];
        f[b + F.WOOD] = 0;
        f[b + F.ORE] = 0;
        f[b + F.GOLD] += gold;
        this.award(e, gold, 0);
        return;
      }
      if (isGreat(node)) {
        if (this.gdone[node - R_GREAT0] === 0) {
          if (this.gcount[node - R_GREAT0] >= 2) this.rew[e] += GREAT_TOGETHER;
          return;
        }
        f[b + F.RARE] = Math.min(RARE_CAP, f[b + F.RARE] + RARE_PER_GREAT);
        this.gainXp(e, F.XP3, GREAT_XP);
        f[b + F.GOLD] += GREAT_GOLD;
        this.award(e, GREAT_GOLD, 0);
        this.rew[e] += GREAT_REWARD;
        this.milestone(e, MILESTONE.GREAT);
        if (e < R_LEARNERS) this.stats[R_STAT.GREAT_HARVESTS]++;
        return;
      }
      if (!this.complete[e]) return;
      for (let j = 0; j < e; j++) if (this.complete[j] && this.tgtNode[j] === node && this.act[j] === ACT.INTERACT) { f[b + F.CHAN] = 0; f[b + F.CHAN_NODE] = -1; return; }
      f[b + F.CHAN] = 0;
      f[b + F.CHAN_NODE] = -1;
      const regrow = { [N_SPRING]: SPRING_REGROW, [N_BERRY]: BERRY_REGROW, [N_TREE]: 600, [N_ORE]: 900 }[type];
      this.nodeTimer[node] = Math.trunc((regrow * this.cfg.regrowPct) / 100);
      if (e < R_LEARNERS) this.stats[R_STAT.HARVESTS]++;
      if (type === N_SPRING) this.restore(e, F.WATER, 40);
      else if (type === N_BERRY) { this.restore(e, F.FOOD, 30); this.gainXp(e, F.XP3, 6); }
      else if (type === N_TREE) { f[b + F.WOOD] = Math.min(8, f[b + F.WOOD] + 1); this.gainXp(e, F.XP3, 10); }
      else { f[b + F.ORE] = Math.min(8, f[b + F.ORE] + 1); this.gainXp(e, F.XP3, 14); }
      return;
    }
    this.nearestTown(e);
    if (this.town.d2 > REACH_TOWN * REACH_TOWN) return;
    if (action === ACT.CRAFT_TOOL && this.canCraftTool(e)) {
      const t = f[b + F.TOOL];
      f[b + F.WOOD] -= 2 + 2 * t;
      f[b + F.ORE] -= 1 + t;
      if (t >= 2) f[b + F.RARE] -= RARE_TIER3;
      f[b + F.TOOL]++;
      this.award(e, 0, 1);
    } else if (action === ACT.CRAFT_WEAP && this.canCraftWeap(e)) {
      const w = f[b + F.WEAP];
      f[b + F.ORE] -= 2 + 2 * w;
      f[b + F.WOOD] -= 1 + w;
      if (w >= 2) f[b + F.RARE] -= RARE_TIER3;
      f[b + F.WEAP]++;
      this.award(e, 0, 1);
    } else if (action === ACT.BUY && f[b + F.GOLD] >= 10 && f[b + F.RAT] < 4) {
      f[b + F.GOLD] -= 10;
      f[b + F.RAT]++;
    }
  }

  damagePhase(v, tick) {
    const f = this.f;
    this.incoming[v] = 0;
    this.killer[v] = -1;
    this.dies[v] = 0;
    this.part[v] = 0;
    this.victimGold[v] = f[v * R_FIELDS + F.GOLD];
    this.victimLevel[v] = f[v * R_FIELDS + F.LEVEL];
    if (!this.alive(v)) return;
    let best = 0;
    let fromMobs = 0;
    for (let a = 0; a < R_ENT; a++) {
      if (this.act[a] !== ACT.ATTACK || this.atkTgt[a] !== v || !this.alive(a)) continue;
      if (a >= R_PLAYERS) fromMobs += this.atkDmg[a];
      else this.incoming[v] += this.atkDmg[a];
      if (this.atkDmg[a] > best) { best = this.atkDmg[a]; this.killer[v] = a; }
    }
    const guard = v < R_PLAYERS ? GUARD_PCT * Math.min(GUARD_CAP, this.allyCount(v)) : 0;
    const armor = v < R_PLAYERS ? ARMOR_PCT * f[v * R_FIELDS + F.TOOL] : 0;
    this.incoming[v] += Math.trunc((Math.trunc((fromMobs * (100 - guard)) / 100) * (100 - armor)) / 100);
    this.dies[v] = f[v * R_FIELDS + F.HP] - this.incoming[v] <= 0 ? 1 : 0;
    if (isBoss(v)) for (let p = 0; p < R_PLAYERS; p++) if (this.participant(p, v, tick)) this.part[v]++;
  }

  shapePhase(e, tick) {
    const f = this.f;
    const b = e * R_FIELDS;
    if (!this.alive(e)) return;
    const view = this.viewRadius(tick);
    const view2 = view * view;
    const ex = f[b + F.X], ey = f[b + F.Y];
    let springD2 = R_FAR, berryD2 = R_FAR;
    for (let j = 0; j < NODE_FIRST_TOWN; j++) {
      if (this.nodeTimer[j] !== 0) continue;
      const type = nodeTypeOf(j);
      if (type !== N_SPRING && type !== N_BERRY) continue;
      const dx = this.nodeX[j] - ex, dy = this.nodeY[j] - ey;
      const d2 = dx * dx + dy * dy;
      if (d2 >= view2) continue;
      if (type === N_SPRING) { if (d2 < springD2) springD2 = d2; } else if (d2 < berryD2) berryD2 = d2;
    }
    const water = springD2 === R_FAR ? 0 : realmIsqrt(springD2);
    const food = berryD2 === R_FAR ? 0 : realmIsqrt(berryD2);
    const prevWater = f[b + F.HX], prevFood = f[b + F.HY];
    let gain = 0;
    if (f[b + F.WATER] < SHAPE_NEED && prevWater > 0 && water > 0) gain += Math.max(-SHAPE_CLAMP, Math.min(SHAPE_CLAMP, (prevWater - water) >> 3));
    if (f[b + F.FOOD] < SHAPE_NEED && f[b + F.RAT] === 0 && prevFood > 0 && food > 0) gain += Math.max(-SHAPE_CLAMP, Math.min(SHAPE_CLAMP, (prevFood - food) >> 3));
    f[b + F.HX] = water;
    f[b + F.HY] = food;
    this.rew[e] += gain;
  }

  sharePhase(e) {
    const f = this.f;
    if (e >= R_PLAYERS || !this.alive(e)) return;
    this.allies[e] = this.allyCount(e);
    let pack = 0;
    for (let v = R_MOB0; v < R_BOSS0; v++) {
      const k = this.killer[v];
      if (!this.dies[v] || k < 0 || k >= R_PLAYERS || k === e) continue;
      const dx = f[k * R_FIELDS + F.X] - f[e * R_FIELDS + F.X], dy = f[k * R_FIELDS + F.Y] - f[e * R_FIELDS + F.Y];
      if (dx * dx + dy * dy < ALLY_RADIUS * ALLY_RADIUS) pack += (this.victimLevel[v] * SHARE_XP) << (10 * this.atkStyle[k]);
    }
    this.part[e] = pack;
  }

  applyPhase(e, tick) {
    const f = this.f;
    const b = e * R_FIELDS;
    if (e < 4) this.marketPhase(e, tick);
    if (!this.alive(e)) {
      f[b + F.DEAD]--;
      if (f[b + F.DEAD] === 0) this.spawn(e, tick);
      return;
    }
    if (this.act[e] === ACT.ATTACK && e < R_PLAYERS) {
      const style = this.atkStyle[e];
      const xp = Math.max(1, this.atkDmg[e] >> 1);
      f[b + F.STYLE] = style;
      this.gainXp(e, F.XP0 + style, xp);
    }
    if (e < R_PLAYERS) {
      for (let v = R_BOSS0; v < R_ANIMAL0; v++) {
        if (!this.dies[v] || !this.participant(e, v, tick)) continue;
        const gold = Math.trunc(BOSS_GOLD / this.part[v]);
        const xp = Math.trunc(BOSS_XP / this.part[v]);
        this.gainXp(e, F.XP0 + f[b + F.STYLE], xp);
        f[b + F.GOLD] += gold;
        f[b + F.RARE] = Math.min(RARE_CAP, f[b + F.RARE] + BOSS_RARE);
        this.award(e, gold, 0);
        this.rew[e] += BOSS_REWARD;
        this.milestone(e, MILESTONE.BOSS);
        if (e < R_LEARNERS) this.stats[R_STAT.BOSS_KILLS]++;
      }
      if (this.act[e] === ACT.ATTACK && isBoss(this.atkTgt[e])) {
        f[b + F.BOSS_ID] = this.atkTgt[e];
        f[b + F.BOSS_TICK] = tick;
      }
    }
    if (e < R_PLAYERS) for (let s = 0; s < 3; s++) {
      const amount = (this.part[e] >> (10 * s)) & 1023;
      if (amount > 0) this.gainXp(e, F.XP0 + s, amount);
    }
    for (let v = 0; v < R_ENT; v++) {
      if (!this.dies[v] || e >= R_PLAYERS || isBoss(v)) continue;
      if (this.killer[v] !== e) continue;
      if (v >= R_ANIMAL0) {
        f[b + F.RAT] = Math.min(4, f[b + F.RAT] + 1);
        f[b + F.GOLD] += 1;
        this.gainXp(e, F.XP3, 4);
        this.award(e, 1, 0);
        this.stats[R_STAT.MOB_KILLS]++;
      } else if (v >= R_MOB0) {
        const level = this.victimLevel[v];
        const gold = 2 + 2 * level;
        this.gainXp(e, F.XP0 + this.atkStyle[e], level * 12);
        f[b + F.GOLD] += gold;
        this.award(e, gold, 0);
        this.milestone(e, MILESTONE.MOB);
        this.stats[R_STAT.MOB_KILLS]++;
      } else {
        const loot = Math.min(this.victimGold[v], 20);
        this.gainXp(e, F.XP0 + this.atkStyle[e], 20);
        f[b + F.GOLD] += loot;
        this.award(e, loot, 0);
      }
    }
    f[b + F.HP] -= this.incoming[e];
    f[b + F.LAST_HIT] = this.incoming[e] > 0 ? 0 : f[b + F.LAST_HIT] + 1;
    if (f[b + F.CD] > 0) f[b + F.CD]--;
    f[b + F.AGE]++;
    const decay = 200 - this.difficulty;
    const period = Math.trunc((Math.trunc(2400 / this.cfg.hungerPct) * decay) / 100);
    const wperiod = Math.trunc((Math.trunc(1600 / this.cfg.hungerPct) * decay) / 100);
    if ((tick + e) % period === 0 && f[b + F.FOOD] > 0) f[b + F.FOOD]--;
    if ((tick + e) % wperiod === 0 && f[b + F.WATER] > 0) f[b + F.WATER]--;
    if (e < R_PLAYERS && f[b + F.SPRINT] === 1 && (tick + e) % 8 === 0 && f[b + F.WATER] > 0) f[b + F.WATER]--;
    if (e < R_PLAYERS && f[b + F.FOOD] < 40 && f[b + F.RAT] > 0) { f[b + F.RAT]--; this.restore(e, F.FOOD, 50); }
    if (e < R_PLAYERS) {
      const allies = this.allies[e];
      if (e < R_LEARNERS && allies > 0) this.stats[R_STAT.ALLY_TICKS]++;
      const fed = f[b + F.FOOD] >= 40 && f[b + F.WATER] >= 40;
      if ((tick + e) % 12 === 0 && fed && f[b + F.HP] < this.maxHp(e)) f[b + F.HP]++;
      if ((tick + e) % 6 === 0 && allies > 0 && fed && f[b + F.HP] < this.maxHp(e)) f[b + F.HP] = Math.min(this.maxHp(e), f[b + F.HP] + Math.min(AURA_CAP, allies));
      if ((tick + e) % 8 === 0 && (f[b + F.FOOD] === 0 || f[b + F.WATER] === 0)) f[b + F.HP] -= 2;
      if (f[b + F.HP] > 0) this.rew[e] += ALIVE_REWARD;
    }
    if (f[b + F.HP] <= 0) {
      if (e < R_PLAYERS) {
        this.rew[e] -= 1024;
        this.stats[R_STAT.PLAYER_DEATHS]++;
        if (e < R_LEARNERS) this.dead[e] = 1;
        else this.spawn(e, tick);
      } else {
        const b2 = e * R_FIELDS;
        f.fill(0, b2, b2 + R_FIELDS);
        f[b2 + F.DEAD] = isBoss(e) ? BOSS_RESPAWN : e < R_ANIMAL0 ? 200 : 150;
        f[b2 + F.CHAN_NODE] = -1;
      }
    }
  }

  step(actions, outputs, tick) {
    const f = this.f;
    this.act.fill(0);
    this.tgtNode.fill(-1);
    this.rew.fill(0);
    this.dead.fill(0);
    this.sunit.fill(0);
    for (let e = 0; e < R_PLAYERS; e++) this.shapePhase(e, tick);
    for (let e = 0; e < R_ENT; e++) {
      this.ctlX[e] = 0;
      this.ctlY[e] = 0;
      if (!this.alive(e)) continue;
      f[e * R_FIELDS + F.SPRINT] = 0;
      if (e < R_LEARNERS) this.learnerDecide(e, outputs, tick);
      else if (e < R_PLAYERS) this.botDecide(e, tick);
      else this.mobDecide(e, tick);
    }
    for (let e = 0; e < R_ENT; e++) if (this.alive(e)) this.moveEntity(e, tick);
    for (let j = 0; j < R_NODES; j++) if (this.nodeTimer[j] > 0) this.nodeTimer[j]--;
    for (let e = 0; e < R_ENT; e++) this.channelPhase(e);
    for (let g = 0; g < R_GREATS; g++) this.greatPhase(g);
    for (let e = 0; e < R_PLAYERS; e++) if (this.alive(e)) this.interactPhase(e);
    for (let v = 0; v < R_ENT; v++) this.damagePhase(v, tick);
    for (let e = 0; e < R_ENT; e++) this.sharePhase(e);
    for (let e = 0; e < R_ENT; e++) this.applyPhase(e, tick);
    for (let e = 0; e < R_PLAYERS; e++) {
      if (e < R_LEARNERS) this.reward[e] = this.rew[e];
      else {
        this.stats[STAT.REW_BASE] += this.rew[e];
        this.stats[STAT.TICKS_BASE]++;
      }
    }
  }

  pack() {
    const words = new Uint32Array(R_WORDS);
    for (let j = 0; j < R_NODES; j++) {
      words[j] = (this.nodeX[j] | (this.nodeY[j] << 16)) >>> 0;
      words[R_NODE_TIMER_OFF + j] = this.nodeTimer[j];
    }
    for (let i = 0; i < R_ENT * R_FIELDS; i++) words[R_ENT_OFF + i] = this.f[i] >>> 0;
    words[R_CFG_OFF] = this.cfg.regrowPct;
    words[R_CFG_OFF + 1] = this.cfg.mobPct;
    words[R_CFG_OFF + 2] = this.cfg.hungerPct;
    words[R_CFG_OFF + 3] = this.cfg.pvp;
    for (let i = 0; i < 4; i++) words[R_SAT_OFF + i] = this.sat[i];
    for (let g = 0; g < R_GREATS; g++) words[R_GPROG_OFF + g] = this.gprog[g];
    return words;
  }

  snapshot() {
    return realmSnapshotFromWords(this.pack(), this.seed);
  }
}

export function realmSnapshotFromWords(words, seed) {
  const cells = new Uint8Array(R_GRID * R_GRID);
  for (let cy = 0; cy < R_GRID; cy++) for (let cx = 0; cx < R_GRID; cx++) cells[cy * R_GRID + cx] = realmTerrain(seed, cx, cy);
  const nodes = [];
  for (let j = 0; j < R_NODES; j++) {
    const type = nodeTypeOf(j);
    if (type === N_NONE || words[R_NODE_TIMER_OFF + j] !== 0) continue;
    nodes.push({ x: words[j] & 0xffff, y: words[j] >>> 16, type, great: isGreat(j) });
  }
  const agents = [];
  for (let e = 0; e < R_ENT; e++) {
    const base = R_ENT_OFF + e * R_FIELDS;
    if (words[base + F.DEAD] !== 0) continue;
    agents.push({ x: words[base + F.X] | 0, y: words[base + F.Y] | 0, kind: e < R_LEARNERS ? 0 : e < R_PLAYERS ? 1 : isBoss(e) ? 4 : e < R_ANIMAL0 ? 2 : 3, hp: words[base + F.HP] | 0, level: e < R_PLAYERS ? 1 + Math.min(9, realmIsqrt(Math.max(words[base + F.XP0], words[base + F.XP1], words[base + F.XP2]) >> 4)) : words[base + F.LEVEL] | 0, evo: e < R_LEARNERS, alive: true });
  }
  return { kind: 'realm', width: R_WORLD, height: R_WORLD, grid: R_GRID, cells, nodes, agents };
}

export function realmValidate(words) {
  const problems = [];
  for (let e = 0; e < R_ENT; e++) {
    const base = R_ENT_OFF + e * R_FIELDS;
    if (words[base + F.DEAD] !== 0) continue;
    const x = words[base + F.X] | 0, y = words[base + F.Y] | 0;
    if (x < 0 || x >= R_WORLD || y < 0 || y >= R_WORLD) problems.push('entity ' + e + ' out of bounds');
    const hp = words[base + F.HP] | 0;
    if (hp > (isBoss(e) ? BOSS_HP : 300) || (e >= R_LEARNERS && hp <= 0)) problems.push('entity ' + e + ' hp ' + hp);
    if ((words[base + F.FOOD] | 0) < 0 || (words[base + F.WATER] | 0) < 0) problems.push('entity ' + e + ' negative meter');
    const rare = words[base + F.RARE] | 0;
    if (rare < 0 || rare > RARE_CAP) problems.push('entity ' + e + ' rare ' + rare);
  }
  for (let g = 0; g < R_GREATS; g++) if (words[R_GPROG_OFF + g] > GREAT_PROGRESS_CAP) problems.push('great node ' + g + ' progress ' + words[R_GPROG_OFF + g]);
  for (let i = 0; i < 4; i++) if (words[R_SAT_OFF + i] > MARKET_CAP) problems.push('town ' + i + ' saturation ' + words[R_SAT_OFF + i]);
  return problems;
}

export function realmDensify(env, rng) {
  for (let e = 0; e < R_ENT; e++) {
    const b = e * R_FIELDS;
    env.f[b + F.X] = 1800 + Math.floor(rng.next() * 900);
    env.f[b + F.Y] = 1800 + Math.floor(rng.next() * 900);
    if (!env.passableAt(env.f[b + F.X], env.f[b + F.Y])) { env.f[b + F.X] = 2048; env.f[b + F.Y] = 2048; }
    if (e < R_PLAYERS) {
      env.f[b + F.XP0] = Math.floor(rng.next() * 400);
      env.f[b + F.XP1] = Math.floor(rng.next() * 400);
      env.f[b + F.XP2] = Math.floor(rng.next() * 400);
      env.f[b + F.WOOD] = Math.floor(rng.next() * 6);
      env.f[b + F.ORE] = Math.floor(rng.next() * 6);
      env.f[b + F.GOLD] = Math.floor(rng.next() * 30);
      env.f[b + F.FOOD] = 20 + Math.floor(rng.next() * 80);
      env.f[b + F.WATER] = 20 + Math.floor(rng.next() * 80);
      env.f[b + F.HP] = 40 + Math.floor(rng.next() * 60);
      env.f[b + F.RARE] = Math.floor(rng.next() * 5);
      env.f[b + F.TOOL] = Math.floor(rng.next() * 3);
      env.f[b + F.WEAP] = Math.floor(rng.next() * 3);
      if (rng.next() < 0.3) { env.f[b + F.BOSS_ID] = R_BOSS0 + Math.floor(rng.next() * 4); env.f[b + F.BOSS_TICK] = 0; }
    } else {
      env.f[b + F.HX] = env.f[b + F.X];
      env.f[b + F.HY] = env.f[b + F.Y];
      if (isBoss(e)) env.f[b + F.HP] = 40 + Math.floor(rng.next() * 80);
    }
  }
  for (let j = 0; j < NODE_FIRST_TOWN; j += 3) {
    env.nodeX[j] = 1900 + Math.floor(rng.next() * 700);
    env.nodeY[j] = 1900 + Math.floor(rng.next() * 700);
    if (!env.passableAt(env.nodeX[j], env.nodeY[j])) { env.nodeX[j] = 2048; env.nodeY[j] = 2048; }
  }
  for (let g = 0; g < R_GREATS; g += 2) {
    const j = R_GREAT0 + g;
    env.nodeX[j] = 1900 + Math.floor(rng.next() * 700);
    env.nodeY[j] = 1900 + Math.floor(rng.next() * 700);
    if (!env.passableAt(env.nodeX[j], env.nodeY[j])) { env.nodeX[j] = 2048; env.nodeY[j] = 2048; }
    env.gprog[g] = Math.floor(rng.next() * 80);
  }
  for (let i = 0; i < 4; i++) env.sat[i] = Math.floor(rng.next() * 90);
}

export const REALM_INFO = { F, ACT, R_STAT, ENT: R_ENT, PLAYERS: R_PLAYERS, LEARNERS: R_LEARNERS, MOB0: R_MOB0, BOSS0: R_BOSS0, ANIMAL0: R_ANIMAL0, GREAT0: R_GREAT0, GREAT_END: R_GREAT_END, FIELDS: R_FIELDS, SAFE_RADIUS, ALLY_RADIUS, NODE_FIRST_TOWN, nodeTypeOf, levelOf, isGreat, isBoss, inSafeZone: (env, x, y) => env.inSafeZone(x, y) };

const REALM_COLORS =['#3b7a3b', '#245a2c', '#6b6b6b', '#1f4f8f', '#c2b26b'];
const REALM_NODE_COLORS = ['#e05a8a', '#8b5a2b', '#d0d0e8', '#4fc3f7', '#f0e130'];

export function realmRender(ctx, snap, w, h) {
  const scale = Math.min(w, h) / snap.width;
  const cell = (snap.width / snap.grid) * scale;
  for (let cy = 0; cy < snap.grid; cy++) for (let cx = 0; cx < snap.grid; cx++) {
    ctx.fillStyle = REALM_COLORS[snap.cells[cy * snap.grid + cx]];
    ctx.fillRect(cx * cell, cy * cell, cell + 1, cell + 1);
  }
  for (const n of snap.nodes) {
    ctx.fillStyle = REALM_NODE_COLORS[n.type];
    const size = n.type === 4 ? 9 : n.great ? 6 : 3;
    ctx.fillRect(n.x * scale - size / 2, n.y * scale - size / 2, size, size);
  }
  const agentColors = ['#58a6ff', '#ffa657', '#f85149', '#d2a8ff', '#ff00ff'];
  for (const a of snap.agents) {
    ctx.beginPath();
    ctx.arc(a.x * scale, a.y * scale, 2 + a.level * 0.5, 0, Math.PI * 2);
    ctx.fillStyle = agentColors[a.kind];
    ctx.fill();
  }
}

export const REALM = {
  id: 'realm3a',
  name: 'Realm (MMO world)',
  tickHz: 30,
  dims: { nIn: 102, nOut: 12, nNodes: 144 },
  agents: R_ENT,
  learners: R_LEARNERS,
  maxAge: R_MAX_AGE,
  worldWords: R_WORDS,
  workgroupBytes: 15792,
  inputNames: [].concat(...['berry', 'tree', 'ore', 'spring', 'town', 'hostile', 'animal', 'weakerPlayer', 'strongerPlayer'].map((c) => ['E', 'SE', 'S', 'SW', 'W', 'NW', 'N', 'NE'].map((d) => c + d))).concat(['hp', 'food', 'water', 'lvlMelee', 'lvlRange', 'lvlMage', 'lvlGather', 'wood', 'ore', 'gold', 'rations', 'toolTier', 'weaponTier', 'canAttack', 'night', 'inTown', 'townDx', 'townDy', 'allies', 'rare', 'nearTreeGreat', 'nearOreGreat', 'inSafe', 'recentlyHit', 'bossNear', 'bossResistsMelee', 'bossResistsRange', 'bossResistsMage', 'greatProgress', 'dayPhase']),
  actionNames: ['+x', '-x', '+y', '-y', 'sprint', 'interact', 'melee', 'range', 'mage', 'craftTool', 'craftWeapon', 'buyRation'],
  statNames: ['mobKills', 'greatHarvests', 'bossKills', 'craftUps', 'harvests', 'allyTicks', 'playerDeaths'],
  defaultCfg: () => ({ regrowPct: 100, mobPct: 100, hungerPct: 100, pvp: 1 }),
  randomCfg: (seed) => ({ regrowPct: 60 + (mix(seed, 0, 0, 8) % 81), mobPct: 70 + (mix(seed, 1, 0, 8) % 61), hungerPct: 75 + (mix(seed, 2, 0, 8) % 51), pvp: mix(seed, 3, 0, 8) % 4 === 0 ? 1 : 0 }),
  createEnv: (seed, cfg, isEval, stats) => new RealmEnv(seed, cfg, isEval, stats),
  teacher: (env, a, tick, out) => env.teacherOutputs(a, tick, out),
  packEnv: (env) => env.pack(),
  snapshotFromWords: realmSnapshotFromWords,
  validateWords: realmValidate,
  densify: realmDensify,
  render: realmRender,
  wgsl: REALM_WGSL
};
