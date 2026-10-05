export const REALM_WGSL = `
const RW_WORLD: i32 = 8192;
const RW_PLAYERS: u32 = 32u;
const RW_MOB0: u32 = 32u;
const RW_ANIMAL0: u32 = 56u;
const RW_FAR: i32 = 1073741824;
const RW_BOSS0: u32 = 52u;
const RW_GREAT0: u32 = 204u;
const RW_GREAT_END: u32 = 236u;
const RW_NODE_TIMER_OFF: u32 = 256u;
const RW_ENT_OFF: u32 = 512u;
const RW_CFG_OFF: u32 = 2560u;
const RW_SAT_OFF: u32 = 2564u;
const RW_GPROG_OFF: u32 = 2568u;
const RW_GREAT_NEED: i32 = 60;
const RW_GREAT_CAP: i32 = 90;
const RW_GREAT_TIMER: u32 = 1500u;
const RW_BOSS_HP: i32 = 500;
const RW_BOSS_LEVEL: i32 = 10;
const RW_BOSS_RESPAWN: i32 = 1200;
const RW_BOSS_DAMAGE: i32 = 44;
const RW_BOSS_COOLDOWN: i32 = 30;
const RW_BOSS_WINDOW: i32 = 60;
const RW_BOSS_GOLD: i32 = 60;
const RW_BOSS_XP: i32 = 200;
const RW_BOSS_FLEE: i32 = 1300;
const RW_GREAT_GOLD: i32 = 15;
const RW_GREAT_TOGETHER: i32 = 2;
const RW_GUARD_PCT: i32 = 10;
const RW_GUARD_CAP: i32 = 5;
const RW_RARE_PER_GREAT: i32 = 3;
const RW_RARE_CAP: i32 = 8;
const RW_RARE_TIER3: i32 = 2;
const RW_GREAT_XP: i32 = 40;
const RW_ALLY_RADIUS: i32 = 800;
const RW_RAID_RADIUS: i32 = 2600;
const RW_MARKET_FLOOR: i32 = 30;
const RW_MARKET_CAP: i32 = 150;
const RW_MARKET_DECAY: u32 = 6u;
const RW_ALIVE_REWARD: i32 = 1;
const RW_GREAT_REWARD: i32 = 1024;
const RW_BOSS_RARE: i32 = 2;
const RW_BOSS_REWARD: i32 = 2048;
const RW_XP_CAP: i32 = 1296;
const RW_ARMOR_PCT: i32 = 10;
const RW_AURA_CAP: i32 = 3;
const RW_SHARE_XP: i32 = 3;
const RW_SPRING_REGROW: i32 = 700;
const RW_BERRY_REGROW: i32 = 700;
const RW_MOB_DAMAGE_STEP: i32 = 4;
const RW_STEER_AHEAD: i32 = 192;
const RW_STEER_RAY: i32 = 20;
const RW_SHAPE_NEED: i32 = 50;
const RW_PLAYER_KEY: i32 = 1000000;
const RW_DAY_AGGRO: i32 = 700;
const RW_SHAPE_CLAMP: i32 = 8;
const RW_MS_GREAT: u32 = 0u;
const RW_MS_BOSS: u32 = 1u;
const RW_MS_MOB: u32 = 2u;
const RW_NEED_LEVEL: i32 = 60;
const RW_NEED_REWARD: i32 = 4;
const RW_DAY_TICKS: u32 = 3600u;
const RW_NIGHT_FROM: u32 = 2400u;
const RW_REACH_NODE: i32 = 320;
const RW_REACH_TOWN: i32 = 500;
const RW_SAFE_RADIUS: i32 = 600;
const RW_NONE: u32 = 7u;
const RW_TOWN: u32 = 4u;
const R_S: u32 = 33u;
const RW_OBS_EMPTY: u32 = 1000000000u;
const RW_KEY_NONE: u32 = 0xffffffffu;

const RX_X: u32 = 0u;
const RX_Y: u32 = 1u;
const RX_VX: u32 = 2u;
const RX_VY: u32 = 3u;
const RX_HP: u32 = 4u;
const RX_FOOD: u32 = 5u;
const RX_WATER: u32 = 6u;
const RX_CD: u32 = 7u;
const RX_CHAN: u32 = 8u;
const RX_STYLE: u32 = 9u;
const RX_XP0: u32 = 10u;
const RX_XP1: u32 = 11u;
const RX_XP2: u32 = 12u;
const RX_XP3: u32 = 13u;
const RX_WOOD: u32 = 14u;
const RX_ORE: u32 = 15u;
const RX_GOLD: u32 = 16u;
const RX_RAT: u32 = 17u;
const RX_TOOL: u32 = 18u;
const RX_WEAP: u32 = 19u;
const RX_AGE: u32 = 20u;
const RX_DEAD: u32 = 21u;
const RX_HX: u32 = 22u;
const RX_HY: u32 = 23u;
const RX_CHAN_NODE: u32 = 24u;
const RX_LAST_HIT: u32 = 25u;
const RX_LEVEL: u32 = 26u;
const RX_SPRINT: u32 = 27u;
const RX_RARE: u32 = 28u;
const RX_BOSS_ID: u32 = 29u;
const RX_BOSS_TICK: u32 = 30u;
const RX_ACH: u32 = 31u;

const RA_NONE: i32 = 0;
const RA_INTERACT: i32 = 1;
const RA_ATTACK: i32 = 2;
const RA_CRAFT_TOOL: i32 = 3;
const RA_CRAFT_WEAP: i32 = 4;
const RA_BUY: i32 = 5;

const RS_MOB_KILLS: u32 = STAT_GAME0;
const RS_GREAT: u32 = STAT_GAME0 + 1u;
const RS_BOSS: u32 = STAT_GAME0 + 2u;
const RS_CRAFT: u32 = STAT_GAME0 + 3u;
const RS_HARVEST: u32 = STAT_GAME0 + 4u;
const RS_ALLY: u32 = STAT_GAME0 + 5u;
const RS_DEATHS: u32 = STAT_GAME0 + 6u;

var<workgroup> r_f: array<i32, 2112>;
var<workgroup> r_nxy: array<u32, 256>;
var<workgroup> r_ntimer: array<u32, 256>;
var<workgroup> r_terr: array<u32, 128>;
var<workgroup> r_ctlx: array<i32, 64>;
var<workgroup> r_ctly: array<i32, 64>;
var<workgroup> r_act: array<i32, 64>;
var<workgroup> r_tgt: array<i32, 64>;
var<workgroup> r_atkt: array<i32, 64>;
var<workgroup> r_atkd: array<i32, 64>;
var<workgroup> r_atks: array<i32, 64>;
var<workgroup> r_comp: array<i32, 64>;
var<workgroup> r_inc: array<i32, 64>;
var<workgroup> r_kill: array<i32, 64>;
var<workgroup> r_dies: array<i32, 64>;
var<workgroup> r_rew: array<i32, 64>;
var<workgroup> r_vgold: array<i32, 64>;
var<workgroup> r_vlevel: array<i32, 64>;
var<workgroup> r_pow: array<i32, 64>;
var<workgroup> r_part: array<i32, 64>;
var<workgroup> r_gprog: array<i32, 32>;
var<workgroup> r_gdone: array<i32, 32>;
var<workgroup> r_gcount: array<i32, 32>;
var<workgroup> r_sunit: array<i32, 32>;
var<workgroup> r_ally: array<i32, 32>;
var<workgroup> r_sat: array<i32, 4>;
var<workgroup> r_regrow: i32;
var<workgroup> r_mobpct: i32;
var<workgroup> r_hunger: i32;
var<workgroup> r_pvp: i32;

var<private> r_nd2: array<i32, 10>;
var<private> r_ndx: array<i32, 10>;
var<private> r_ndy: array<i32, 10>;
var<private> r_tree_d2: i32;
var<private> r_tree_great: bool;
var<private> r_ore_d2: i32;
var<private> r_ore_great: bool;
var<private> r_td2: i32;
var<private> r_tdx: i32;
var<private> r_tdy: i32;
var<private> RW_DIRX: array<i32, 8> = array<i32, 8>(128, 90, 0, -90, -128, -90, 0, 90);
var<private> RW_DIRY: array<i32, 8> = array<i32, 8>(0, 90, 128, 90, 0, -90, -128, -90);

fn r_isqrt(x: i32) -> i32 {
  var s = i32(sqrt(f32(x)));
  loop { if (s * s > x) { s = s - 1; } else { break; } }
  loop { if ((s + 1) * (s + 1) <= x) { s = s + 1; } else { break; } }
  return s;
}

fn r_sector(dx: i32, dy: i32) -> u32 {
  let ax = abs(dx);
  let ay = abs(dy);
  if (ay * 12 <= ax * 5) { return select(4u, 0u, dx >= 0); }
  if (ax * 12 <= ay * 5) { return select(6u, 2u, dy >= 0); }
  if (dx > 0) { return select(7u, 1u, dy > 0); }
  return select(5u, 3u, dy > 0);
}

fn r_level(xp: i32) -> i32 { return 1 + min(9, r_isqrt(xp >> 4u)); }

fn r_beats(a: i32, b: i32) -> bool { return (a == 0 && b == 1) || (a == 1 && b == 2) || (a == 2 && b == 0); }

fn r_night(tick: u32) -> bool { return tick % RW_DAY_TICKS >= RW_NIGHT_FROM; }

fn r_is_great(j: u32) -> bool { return j >= RW_GREAT0 && j < RW_GREAT_END; }

fn r_is_boss(e: u32) -> bool { return e >= RW_BOSS0 && e < RW_ANIMAL0; }

fn r_boss_resist(e: u32) -> u32 { return (e - RW_BOSS0) % 3u; }

fn r_damage_scale() -> i32 { return 30 + (70 * world_difficulty()) / 100; }

fn r_view(tick: u32) -> i32 { return select(2048, 1400, r_night(tick)); }

fn r_town_cx(i: u32) -> i32 { return i32((i & 1u) * 16u + 8u); }
fn r_town_cy(i: u32) -> i32 { return i32((i >> 1u) * 16u + 8u); }

fn r_terrain_raw(cx: u32, cy: u32) -> u32 {
  for (var i = 0u; i < 4u; i++) {
    if (abs(i32(cx) - r_town_cx(i)) <= 3 && abs(i32(cy) - r_town_cy(i)) <= 3) { return 0u; }
  }
  let n3 = mix4(w_seed, cx >> 3u, cy >> 3u, 30u) & 255u;
  let n1 = mix4(w_seed, cx >> 1u, cy >> 1u, 31u) & 255u;
  let n0 = mix4(w_seed, cx, cy, 32u) & 255u;
  let h = (n3 * 4u + n1 * 2u + n0) / 7u;
  if (h < 60u) { return 3u; }
  if (h < 80u) { return 4u; }
  if (h < 170u) { return 0u; }
  if (h < 215u) { return 1u; }
  return 2u;
}

fn r_terrain(cx: u32, cy: u32) -> u32 {
  let cell = (cy << 5u) | cx;
  return (r_terr[cell >> 3u] >> ((cell & 7u) * 4u)) & 15u;
}

fn r_fill_terrain(t: u32) {
  for (var h = 0u; h < 2u; h++) {
    let w = t + 64u * h;
    var packed = 0u;
    for (var k = 0u; k < 8u; k++) {
      let cell = w * 8u + k;
      packed = packed | (r_terrain_raw(cell & 31u, cell >> 5u) << (k * 4u));
    }
    r_terr[w] = packed;
  }
}

fn r_passable_at(x: i32, y: i32) -> bool {
  let t = r_terrain(u32(x >> 8u), u32(y >> 8u));
  return t != 3u && t != 2u;
}

fn r_near_rock(cx: u32, cy: u32) -> bool {
  return r_terrain((cx + 1u) & 31u, cy) == 2u || r_terrain((cx + 31u) & 31u, cy) == 2u || r_terrain(cx, (cy + 1u) & 31u) == 2u || r_terrain(cx, (cy + 31u) & 31u) == 2u;
}

fn r_node_type(j: u32) -> u32 {
  if (j < 60u) { return 0u; }
  if (j < 120u) { return 1u; }
  if (j < 168u) { return 2u; }
  if (j < 200u) { return 3u; }
  if (j < 204u) { return 4u; }
  if (j < 220u) { return 1u; }
  if (j < 236u) { return 2u; }
  return 7u;
}

fn r_place_node(j: u32) {
  let ty = r_node_type(j);
  r_ntimer[j] = 0u;
  if (ty == RW_NONE) { r_nxy[j] = 0u; return; }
  if (ty == RW_TOWN) {
    let ti = j - 200u;
    r_nxy[j] = u32(r_town_cx(ti) * 256 + 128) | (u32(r_town_cy(ti) * 256 + 128) << 16u);
    return;
  }
  var x = 0;
  var y = 0;
  for (var k = 0u; k < 8u; k++) {
    x = i32(mix4(w_seed, j, k, 33u) % 8192u);
    y = i32(mix4(w_seed, j, k, 34u) % 8192u);
    let t = r_terrain(u32(x >> 8u), u32(y >> 8u));
    if (t == 3u || t == 2u) { continue; }
    var preferred = false;
    if (ty == 0u) { preferred = t == 0u; }
    else if (ty == 1u) { preferred = t == 1u; }
    else if (ty == 2u) { preferred = r_near_rock(u32(x >> 8u), u32(y >> 8u)); }
    else { preferred = t == 4u; }
    if (preferred || k >= 6u) { break; }
  }
  if (!r_passable_at(x, y)) {
    x = 2048 + i32(j & 7u) * 64;
    y = 2048 + i32(j >> 3u) * 16;
  }
  r_nxy[j] = u32(x) | (u32(y) << 16u);
}

fn r_nearest_town(x: i32, y: i32) {
  r_td2 = RW_FAR;
  r_tdx = 0;
  r_tdy = 0;
  for (var i = 0u; i < 4u; i++) {
    let dx = r_town_cx(i) * 256 + 128 - x;
    let dy = r_town_cy(i) * 256 + 128 - y;
    let d2 = dx * dx + dy * dy;
    if (d2 < r_td2) { r_td2 = d2; r_tdx = dx; r_tdy = dy; }
  }
}

fn r_in_safe(x: i32, y: i32) -> bool {
  for (var i = 0u; i < 4u; i++) {
    let dx = r_town_cx(i) * 256 + 128 - x;
    let dy = r_town_cy(i) * 256 + 128 - y;
    if (dx * dx + dy * dy < RW_SAFE_RADIUS * RW_SAFE_RADIUS) { return true; }
  }
  return false;
}

fn r_alive(e: u32) -> bool { return r_f[e * R_S + RX_DEAD] == 0; }

fn r_power(e: u32) -> i32 {
  let b = e * R_S;
  return r_level(r_f[b + RX_XP0]) + r_level(r_f[b + RX_XP1]) + r_level(r_f[b + RX_XP2]) + 2 * r_f[b + RX_WEAP];
}

fn r_max_hp(e: u32) -> i32 {
  let b = e * R_S;
  if (e >= RW_PLAYERS) { return 30 + 15 * r_f[b + RX_LEVEL]; }
  return 90 + 10 * max(r_level(r_f[b + RX_XP0]), max(r_level(r_f[b + RX_XP1]), r_level(r_f[b + RX_XP2])));
}

fn r_spawn(e: u32, tick: u32) {
  let b = e * R_S;
  for (var i = 0u; i < 32u; i++) { r_f[b + i] = 0; }
  r_f[b + RX_CHAN_NODE] = -1;
  if (e < RW_PLAYERS) {
    let ti = mix4(w_seed, tick, e, 40u) & 3u;
    r_f[b + RX_X] = r_town_cx(ti) * 256 + 128 + i32(mix4(w_seed, tick, e, 41u) % 601u) - 300;
    r_f[b + RX_Y] = r_town_cy(ti) * 256 + 128 + i32(mix4(w_seed, tick, e, 42u) % 601u) - 300;
    r_f[b + RX_HP] = 100;
    r_f[b + RX_FOOD] = 80;
    r_f[b + RX_WATER] = 80;
    r_f[b + RX_LAST_HIT] = 1000;
    return;
  }
  var x = 0;
  var y = 0;
  var placed = false;
  if (e < RW_BOSS0 && r_night(tick) && mix4(w_seed, tick, e, 45u) % 100u < u32(world_difficulty())) {
    let ti = mix4(w_seed, tick, e, 46u) & 3u;
    let dir = mix4(w_seed, tick, e, 47u) & 7u;
    let reach = 9 + i32(mix4(w_seed, tick, e, 48u) % 5u);
    x = r_town_cx(ti) * 256 + 128 + RW_DIRX[dir] * reach;
    y = r_town_cy(ti) * 256 + 128 + RW_DIRY[dir] * reach;
    placed = r_passable_at(x, y);
  }
  if (!placed) {
    let minTown = select(1200, 2400, e >= RW_BOSS0);
    for (var k = 0u; k < 8u; k++) {
      x = i32(mix4(w_seed, tick, e * 8u + k, 43u) % 8192u);
      y = i32(mix4(w_seed, tick, e * 8u + k, 44u) % 8192u);
      if (!r_passable_at(x, y)) { continue; }
      r_nearest_town(x, y);
      if (e >= RW_ANIMAL0 || r_td2 > minTown * minTown) { break; }
    }
    if (!r_passable_at(x, y)) { x = 4096; y = 4096 + i32(e); }
  }
  r_f[b + RX_X] = x;
  r_f[b + RX_Y] = y;
  r_f[b + RX_HX] = x;
  r_f[b + RX_HY] = y;
  r_nearest_town(x, y);
  if (r_is_boss(e)) {
    r_f[b + RX_LEVEL] = RW_BOSS_LEVEL;
    r_f[b + RX_STYLE] = 1 + i32(e & 1u);
    r_f[b + RX_HP] = RW_BOSS_HP;
  } else if (e < RW_ANIMAL0) {
    r_f[b + RX_LEVEL] = 1 + min(7, r_isqrt(r_td2) / 600);
    r_f[b + RX_STYLE] = i32(e % 3u);
    r_f[b + RX_HP] = 30 + 15 * r_f[b + RX_LEVEL];
  } else {
    r_f[b + RX_LEVEL] = 1;
    r_f[b + RX_HP] = 20;
  }
  r_f[b + RX_FOOD] = 100;
  r_f[b + RX_WATER] = 100;
}

fn g_respawn(a: u32, tick: u32) { r_spawn(a, tick); }

fn r_consider_sector(e: u32, cls: u32, dx: i32, dy: i32, d2: i32) {
  atomicMin(&obsBuf[e * OSTR + cls * 8u + r_sector(dx, dy)], u32(d2));
}

fn r_scan_part(e: u32, q: u32, tick: u32) {
  let view = r_view(tick);
  let view2 = view * view;
  let ex = r_f[e * R_S + RX_X];
  let ey = r_f[e * R_S + RX_Y];
  let j0 = q * 59u;
  for (var j = j0; j < j0 + 59u; j++) {
    let ty = r_node_type(j);
    if (r_ntimer[j] != 0u) { continue; }
    let pos = r_nxy[j];
    let dx = i32(pos & 0xffffu) - ex;
    let dy = i32(pos >> 16u) - ey;
    let d2 = dx * dx + dy * dy;
    if (d2 >= view2) { continue; }
    r_consider_sector(e, ty, dx, dy, d2);
  }
  let myPower = r_pow[e];
  let k0 = q * 16u;
  for (var j = k0; j < k0 + 16u; j++) {
    if (j == e || !r_alive(j)) { continue; }
    let dx = r_f[j * R_S + RX_X] - ex;
    let dy = r_f[j * R_S + RX_Y] - ey;
    let d2 = dx * dx + dy * dy;
    if (d2 >= view2) { continue; }
    var cls = 6u;
    if (j < RW_PLAYERS) { cls = select(7u, 8u, r_pow[j] >= myPower); }
    else if (j < RW_ANIMAL0) { cls = 5u; }
    r_consider_sector(e, cls, dx, dy, d2);
  }
}

fn r_bot_prescan(t: u32, tick: u32) {
  let k = t & 15u;
  let q = t >> 4u;
  let e = LEARNERS + k;
  if (!r_alive(e)) { return; }
  let view = r_view(tick);
  let view2 = view * view;
  let ex = r_f[e * R_S + RX_X];
  let ey = r_f[e * R_S + RX_Y];
  let j0 = q * 50u;
  for (var j = j0; j < j0 + 50u; j++) {
    if (r_ntimer[j] != 0u) { continue; }
    let pos = r_nxy[j];
    let dx = i32(pos & 0xffffu) - ex;
    let dy = i32(pos >> 16u) - ey;
    let d2 = dx * dx + dy * dy;
    if (d2 < view2) { atomicMin(&obsBuf[k * 6u + r_node_type(j)], (u32(d2) << 8u) | j); }
  }
  let m0 = RW_MOB0 + q * 5u;
  for (var j = m0; j < m0 + 5u; j++) {
    if (!r_alive(j)) { continue; }
    let dx = r_f[j * R_S + RX_X] - ex;
    let dy = r_f[j * R_S + RX_Y] - ey;
    let d2 = dx * dx + dy * dy;
    if (d2 < view2) { atomicMin(&obsBuf[k * 6u + 4u], (u32(d2) << 8u) | j); }
  }
  if (q == 0u) {
    for (var j = RW_BOSS0; j < RW_ANIMAL0; j++) {
      if (!r_alive(j)) { continue; }
      let dx = r_f[j * R_S + RX_X] - ex;
      let dy = r_f[j * R_S + RX_Y] - ey;
      let d2 = dx * dx + dy * dy;
      if (d2 < view2) { atomicMin(&obsBuf[k * 6u + 5u], u32(d2)); }
    }
  }
}

fn r_load_bot_scan(e: u32) {
  for (var i = 0u; i < 10u; i++) { r_nd2[i] = RW_FAR; }
  let k = e - LEARNERS;
  let ex = r_f[e * R_S + RX_X];
  let ey = r_f[e * R_S + RX_Y];
  for (var ty = 0u; ty < 5u; ty++) {
    let key = atomicLoad(&obsBuf[k * 6u + ty]);
    var dx = 0;
    var dy = 0;
    var d2 = RW_FAR;
    if (key != RW_KEY_NONE) {
      let j = key & 255u;
      d2 = i32(key >> 8u);
      if (ty < 4u) {
        let pos = r_nxy[j];
        dx = i32(pos & 0xffffu) - ex;
        dy = i32(pos >> 16u) - ey;
      } else {
        dx = r_f[j * R_S + RX_X] - ex;
        dy = r_f[j * R_S + RX_Y] - ey;
      }
    }
    let slot = select(ty, 5u, ty == 4u);
    r_nd2[slot] = d2;
    r_ndx[slot] = dx;
    r_ndy[slot] = dy;
  }
  let bossKey = atomicLoad(&obsBuf[k * 6u + 5u]);
  if (bossKey != RW_KEY_NONE) { r_nd2[9] = i32(bossKey); }
}

fn r_ally_count(e: u32) -> i32 {
  let ex = r_f[e * R_S + RX_X];
  let ey = r_f[e * R_S + RX_Y];
  var count = 0;
  for (var j = 0u; j < RW_PLAYERS; j++) {
    if (j == e || !r_alive(j)) { continue; }
    let dx = r_f[j * R_S + RX_X] - ex;
    let dy = r_f[j * R_S + RX_Y] - ey;
    if (dx * dx + dy * dy < RW_ALLY_RADIUS * RW_ALLY_RADIUS) { count++; }
  }
  return count;
}

fn r_near_value(d2: f32, view: i32) -> f32 {
  if (d2 >= f32(view * view)) { return 0.0; }
  return f32(view - r_isqrt(i32(d2))) * (1.0 / 2048.0);
}

fn g_observe(t: u32, tick: u32) {
  let l = t % LEARNERS;
  let q = t / LEARNERS;
  if (t < RW_PLAYERS) { r_pow[t] = r_power(t); }
  for (var i = q * 18u; i < q * 18u + 18u; i++) { atomicStore(&obsBuf[l * OSTR + i], RW_OBS_EMPTY); }
  workgroupBarrier();
  r_scan_part(l, q, tick);
  workgroupBarrier();
  let view = r_view(tick);
  for (var i = q * 18u; i < q * 18u + 18u; i++) { set_obs(l, i, r_near_value(f32(atomicLoad(&obsBuf[l * OSTR + i])), view)); }
  if (q == 0u) {
    let b = l * R_S;
    r_nearest_town(r_f[b + RX_X], r_f[b + RX_Y]);
    set_obs(l, 72u, f32(r_f[b + RX_HP]) * (1.0 / 256.0));
    set_obs(l, 73u, f32(r_f[b + RX_FOOD]) * (1.0 / 128.0));
    set_obs(l, 74u, f32(r_f[b + RX_WATER]) * (1.0 / 128.0));
    set_obs(l, 75u, f32(r_level(r_f[b + RX_XP0])) * (1.0 / 16.0));
    set_obs(l, 76u, f32(r_level(r_f[b + RX_XP1])) * (1.0 / 16.0));
    set_obs(l, 77u, f32(r_level(r_f[b + RX_XP2])) * (1.0 / 16.0));
    set_obs(l, 79u, f32(r_f[b + RX_WOOD]) * (1.0 / 8.0));
    set_obs(l, 80u, f32(r_f[b + RX_ORE]) * (1.0 / 8.0));
    set_obs(l, 81u, f32(min(r_f[b + RX_GOLD], 64)) * (1.0 / 64.0));
    set_obs(l, 82u, f32(r_f[b + RX_RAT]) * (1.0 / 4.0));
    set_obs(l, 83u, f32(r_f[b + RX_TOOL]) * (1.0 / 4.0));
    set_obs(l, 84u, f32(r_f[b + RX_WEAP]) * (1.0 / 4.0));
    set_obs(l, 86u, select(0.0, 1.0, r_night(tick)));
    set_obs(l, 87u, select(0.0, 1.0, r_td2 < RW_REACH_TOWN * RW_REACH_TOWN));
    let len = r_isqrt(r_tdx * r_tdx + r_tdy * r_tdy);
    let safe = max(len, 1);
    set_obs(l, 88u, select(f32((r_tdx * 128) / safe) * (1.0 / 128.0), 0.0, len == 0));
    set_obs(l, 89u, select(f32((r_tdy * 128) / safe) * (1.0 / 128.0), 0.0, len == 0));
    set_obs(l, 91u, f32(r_f[b + RX_RARE]) * (1.0 / 8.0));
  } else if (q == 1u) {
    set_obs(l, 90u, f32(r_ally_count(l)) * (1.0 / 8.0));
  } else if (q == 2u) {
    let b = l * R_S;
    let ex = r_f[b + RX_X];
    let ey = r_f[b + RX_Y];
    var springD2 = RW_FAR;
    var springDx = 0;
    var springDy = 0;
    var berryD2 = RW_FAR;
    var berryDx = 0;
    var berryDy = 0;
    for (var j = 0u; j < 200u; j++) {
      if (r_ntimer[j] != 0u) { continue; }
      let ty = r_node_type(j);
      if (ty != 3u && ty != 0u) { continue; }
      let pos = r_nxy[j];
      let dx = i32(pos & 0xffffu) - ex;
      let dy = i32(pos >> 16u) - ey;
      let d2 = dx * dx + dy * dy;
      if (d2 >= view * view) { continue; }
      if (ty == 3u) {
        if (d2 < springD2) { springD2 = d2; springDx = dx; springDy = dy; }
      } else if (d2 < berryD2) { berryD2 = d2; berryDx = dx; berryDy = dy; }
    }
    let thirst = 100 - r_f[b + RX_WATER];
    var hunger = 0;
    if (r_f[b + RX_RAT] == 0) { hunger = 100 - r_f[b + RX_FOOD]; }
    let urge = max(0, min(64, max(thirst, hunger) - 20));
    let wantSpring = thirst >= hunger;
    let needD2 = select(berryD2, springD2, wantSpring);
    let needDx = select(berryDx, springDx, wantSpring);
    let needDy = select(berryDy, springDy, wantSpring);
    var needLen = 0;
    if (needD2 != RW_FAR) { needLen = r_isqrt(needD2); }
    let safeLen = max(needLen, 1);
    set_obs(l, 78u, select(f32(((needDx * 128) / safeLen) * urge) * (1.0 / 8192.0), 0.0, needLen == 0));
    set_obs(l, 92u, select(f32(((needDy * 128) / safeLen) * urge) * (1.0 / 8192.0), 0.0, needLen == 0));
    set_obs(l, 93u, f32(urge) * (1.0 / 64.0));
  } else {
    let b = l * R_S;
    let ex = r_f[b + RX_X];
    let ey = r_f[b + RX_Y];
    set_obs(l, 94u, select(0.0, 1.0, r_in_safe(ex, ey)));
    set_obs(l, 95u, select(0.0, 1.0, r_f[b + RX_LAST_HIT] < 40));
    var bossD2 = RW_FAR;
    var bossIdx = 0u;
    for (var j = RW_BOSS0; j < RW_ANIMAL0; j++) {
      if (!r_alive(j)) { continue; }
      let dx = r_f[j * R_S + RX_X] - ex;
      let dy = r_f[j * R_S + RX_Y] - ey;
      let d2 = dx * dx + dy * dy;
      if (d2 < bossD2) { bossD2 = d2; bossIdx = j; }
    }
    let bossSeen = bossD2 < view * view;
    let resist = select(3u, r_boss_resist(bossIdx), bossSeen);
    set_obs(l, 96u, select(0.0, r_near_value(f32(bossD2), view), bossSeen));
    set_obs(l, 97u, select(0.0, 1.0, resist == 0u));
    set_obs(l, 98u, select(0.0, 1.0, resist == 1u));
    set_obs(l, 99u, select(0.0, 1.0, resist == 2u));
    var greatD2 = RW_FAR;
    var greatIdx = RW_GREAT0;
    for (var j = RW_GREAT0; j < RW_GREAT_END; j++) {
      if (r_ntimer[j] != 0u) { continue; }
      let pos = r_nxy[j];
      let dx = i32(pos & 0xffffu) - ex;
      let dy = i32(pos >> 16u) - ey;
      let d2 = dx * dx + dy * dy;
      if (d2 < greatD2) { greatD2 = d2; greatIdx = j; }
    }
    set_obs(l, 100u, select(0.0, f32(r_gprog[greatIdx - RW_GREAT0]) * (1.0 / 64.0), greatD2 < view * view));
    var greatLen = 0;
    if (greatD2 < view * view) { greatLen = r_isqrt(greatD2); }
    let gpos = r_nxy[greatIdx];
    let greatSafe = max(greatLen, 1);
    set_obs(l, 85u, select(f32(((i32(gpos & 0xffffu) - ex) * 128) / greatSafe) * (1.0 / 128.0), 0.0, greatLen == 0));
    set_obs(l, 101u, select(f32(((i32(gpos >> 16u) - ey) * 128) / greatSafe) * (1.0 / 128.0), 0.0, greatLen == 0));
  }
}

fn r_toward(e: u32, dx: i32, dy: i32) {
  let len = r_isqrt(dx * dx + dy * dy);
  let safe = max(len, 1);
  r_ctlx[e] = select((dx * 128) / safe, 0, len == 0);
  r_ctly[e] = select((dy * 128) / safe, 0, len == 0);
}

fn r_wander(e: u32, tick: u32) {
  let dir = mix4(w_seed, tick >> 5u, e, 22u) & 7u;
  r_ctlx[e] = RW_DIRX[dir];
  r_ctly[e] = RW_DIRY[dir];
}

fn r_style_range(s: i32) -> i32 {
  if (s == 0) { return 200; }
  if (s == 1) { return 800; }
  return 500;
}

fn r_mob_range(s: i32) -> i32 {
  if (s == 0) { return 200; }
  if (s == 1) { return 700; }
  return 450;
}

fn r_style_cooldown(s: i32) -> i32 {
  if (s == 0) { return 10; }
  if (s == 1) { return 18; }
  return 24;
}

fn r_attack_damage(e: u32, style: i32) -> i32 {
  let b = e * R_S;
  if (e >= RW_PLAYERS) {
    let base = select(4 + RW_MOB_DAMAGE_STEP * r_f[b + RX_LEVEL], RW_BOSS_DAMAGE, r_is_boss(e));
    return (base * r_mobpct * r_damage_scale()) / 10000;
  }
  let weap = r_f[b + RX_WEAP];
  if (style == 0) { return 6 + 2 * r_level(r_f[b + RX_XP0]) + 4 * weap; }
  if (style == 1) { return 4 + 2 * r_level(r_f[b + RX_XP1]) + 3 * weap; }
  return 5 + 2 * r_level(r_f[b + RX_XP2]) + 3 * weap;
}

fn r_try_attack(e: u32, style: i32, allowPlayers: bool) -> bool {
  let b = e * R_S;
  if (r_f[b + RX_CD] != 0) { return false; }
  let range = select(r_style_range(style), r_mob_range(style), e >= RW_PLAYERS);
  let ex = r_f[b + RX_X];
  let ey = r_f[b + RX_Y];
  var best = -1;
  var bestKey = RW_FAR;
  if (e < RW_PLAYERS && r_in_safe(ex, ey)) { return false; }
  for (var j = 0u; j < 64u; j++) {
    if (j == e || !r_alive(j)) { continue; }
    if (e < RW_PLAYERS) {
      if (j < RW_PLAYERS && (!allowPlayers || r_pvp == 0)) { continue; }
    } else if (j >= RW_PLAYERS) { continue; }
    let jx = r_f[j * R_S + RX_X];
    let jy = r_f[j * R_S + RX_Y];
    let dx = jx - ex;
    let dy = jy - ey;
    let d2 = dx * dx + dy * dy;
    if (d2 > range * range) { continue; }
    let key = d2 + select(0, RW_PLAYER_KEY, j < RW_PLAYERS);
    if (key >= bestKey) { continue; }
    if (j < RW_PLAYERS && r_in_safe(jx, jy)) { continue; }
    best = i32(j);
    bestKey = key;
  }
  if (best < 0) { return false; }
  var damage = r_attack_damage(e, style);
  let defenderStyle = r_f[u32(best) * R_S + RX_STYLE];
  if (r_beats(style, defenderStyle)) { damage = (damage * 3) >> 1u; }
  else if (r_beats(defenderStyle, style)) { damage = (damage * 3) >> 2u; }
  if (e < RW_PLAYERS && u32(best) < RW_PLAYERS) { damage = damage >> 2u; }
  if (r_is_boss(u32(best)) && i32(r_boss_resist(u32(best))) == style) { damage = damage >> 2u; }
  r_atkt[e] = best;
  r_atkd[e] = damage;
  r_atks[e] = style;
  r_act[e] = RA_ATTACK;
  var cooldown = r_style_cooldown(style);
  if (e >= RW_PLAYERS) { cooldown = select(20, RW_BOSS_COOLDOWN, r_is_boss(e)); }
  r_f[b + RX_CD] = cooldown;
  return true;
}

fn r_nearest_interact(e: u32, allowGreat: bool) -> i32 {
  let ex = r_f[e * R_S + RX_X];
  let ey = r_f[e * R_S + RX_Y];
  var best = -1;
  var bestD2 = RW_REACH_TOWN * RW_REACH_TOWN + 1;
  for (var j = 0u; j < RW_GREAT_END; j++) {
    if (r_ntimer[j] != 0u) { continue; }
    if (!allowGreat && r_is_great(j)) { continue; }
    let reach = select(RW_REACH_NODE, RW_REACH_TOWN, j >= 200u && j < RW_GREAT0);
    let pos = r_nxy[j];
    let dx = i32(pos & 0xffffu) - ex;
    let dy = i32(pos >> 16u) - ey;
    let d2 = dx * dx + dy * dy;
    if (d2 <= reach * reach && d2 < bestD2) { best = i32(j); bestD2 = d2; }
  }
  return best;
}

fn r_learner_decide(e: u32) {
  let ox = out_of(e, 0u) - out_of(e, 1u);
  let oy = out_of(e, 2u) - out_of(e, 3u);
  r_ctlx[e] = clamp(i32(floor(ox * 128.0)), -128, 128);
  r_ctly[e] = clamp(i32(floor(oy * 128.0)), -128, 128);
  r_f[e * R_S + RX_SPRINT] = select(0, 1, out_of(e, 4u) > 0.5);
  var style = 0u;
  if (out_of(e, 7u) > out_of(e, 6u)) { style = 1u; }
  if (out_of(e, 8u) > out_of(e, 6u + style)) { style = 2u; }
  if (out_of(e, 6u + style) > 0.5 && r_try_attack(e, i32(style), true)) { return; }
  if (out_of(e, 5u) > 0.5) { r_act[e] = RA_INTERACT; r_tgt[e] = r_nearest_interact(e, true); }
  else if (out_of(e, 9u) > 0.5) { r_act[e] = RA_CRAFT_TOOL; }
  else if (out_of(e, 10u) > 0.5) { r_act[e] = RA_CRAFT_WEAP; }
  else if (out_of(e, 11u) > 0.5) { r_act[e] = RA_BUY; }
}

fn r_can_craft_tool(e: u32) -> bool {
  let b = e * R_S;
  let t = r_f[b + RX_TOOL];
  return t < 3 && r_f[b + RX_WOOD] >= 2 + 2 * t && r_f[b + RX_ORE] >= 1 + t && (t < 2 || r_f[b + RX_RARE] >= RW_RARE_TIER3);
}

fn r_can_craft_weap(e: u32) -> bool {
  let b = e * R_S;
  let w = r_f[b + RX_WEAP];
  return w < 3 && r_f[b + RX_ORE] >= 2 + 2 * w && r_f[b + RX_WOOD] >= 1 + w && (w < 2 || r_f[b + RX_RARE] >= RW_RARE_TIER3);
}

fn r_open_at(x: i32, y: i32) -> bool {
  return x >= 0 && x < RW_WORLD && y >= 0 && y < RW_WORLD && r_passable_at(x, y);
}

fn r_clear_ahead(px: i32, py: i32, dx: i32, dy: i32, len: i32) -> bool {
  let sx = (dx * RW_STEER_AHEAD) / len;
  let sy = (dy * RW_STEER_AHEAD) / len;
  return r_open_at(px + (sx >> 1u), py + (sy >> 1u)) && r_open_at(px + sx, py + sy);
}

fn r_bot_toward(e: u32, dx: i32, dy: i32) {
  r_toward(e, dx, dy);
  let px = r_f[e * R_S + RX_X];
  let py = r_f[e * R_S + RX_Y];
  let len = r_isqrt(dx * dx + dy * dy);
  if (len == 0 || r_clear_ahead(px, py, dx, dy, len)) { return; }
  var best = -1;
  var bestScore = RW_FAR;
  for (var k = 0u; k < 8u; k++) {
    var reach = 0;
    loop {
      if (reach < RW_STEER_RAY && r_open_at(px + RW_DIRX[k] * (reach + 1), py + RW_DIRY[k] * (reach + 1))) { reach = reach + 1; } else { break; }
    }
    if (reach == 0) { continue; }
    let ex = dx - RW_DIRX[k] * reach;
    let ey = dy - RW_DIRY[k] * reach;
    let score = (ex * ex + ey * ey) / 64;
    if (score < bestScore) { bestScore = score; best = i32(k); }
  }
  if (best >= 0) { r_ctlx[e] = RW_DIRX[best]; r_ctly[e] = RW_DIRY[best]; }
}

fn r_go_to(e: u32, cls: u32, tick: u32, allowGreat: bool) {
  if (r_nd2[cls] == RW_FAR) { r_wander(e, tick); return; }
  if (r_nd2[cls] <= RW_REACH_NODE * RW_REACH_NODE) { r_act[e] = RA_INTERACT; r_tgt[e] = r_nearest_interact(e, allowGreat); return; }
  r_bot_toward(e, r_ndx[cls], r_ndy[cls]);
}

fn r_bot_decide(e: u32, tick: u32) {
  let b = e * R_S;
  r_load_bot_scan(e);
  r_nearest_town(r_f[b + RX_X], r_f[b + RX_Y]);
  let inTown = r_td2 <= RW_REACH_TOWN * RW_REACH_TOWN;
  let l0 = r_level(r_f[b + RX_XP0]);
  let l1 = r_level(r_f[b + RX_XP1]);
  let l2 = r_level(r_f[b + RX_XP2]);
  var style = 0;
  if (l1 > l0) { style = 1; }
  if (l2 > select(l1, l0, style == 0)) { style = 2; }
  let allowGreat = false;
  let desperate = r_f[b + RX_WATER] < 50 || (r_f[b + RX_FOOD] < 50 && r_f[b + RX_RAT] == 0);
  let bossHold = r_nd2[9] < RW_BOSS_FLEE * RW_BOSS_FLEE && !desperate;
  if (bossHold && !inTown) {
    r_bot_toward(e, r_tdx, r_tdy);
    return;
  }
  if (r_nd2[5] < 250000 && r_f[b + RX_HP] > 50 && !r_in_safe(r_f[b + RX_X], r_f[b + RX_Y])) {
    if (r_try_attack(e, style, false)) { return; }
    r_bot_toward(e, r_ndx[5], r_ndy[5]);
    return;
  }
  if (r_f[b + RX_WATER] < 50) { r_go_to(e, 3u, tick, allowGreat); return; }
  if (r_f[b + RX_FOOD] < 50 && r_f[b + RX_RAT] == 0) { r_go_to(e, 0u, tick, allowGreat); return; }
  if (inTown) {
    if (r_can_craft_tool(e)) { r_act[e] = RA_CRAFT_TOOL; return; }
    if (r_can_craft_weap(e)) { r_act[e] = RA_CRAFT_WEAP; return; }
    if (r_f[b + RX_WOOD] + r_f[b + RX_ORE] > 0) { r_act[e] = RA_INTERACT; r_tgt[e] = r_nearest_interact(e, allowGreat); return; }
    if (r_f[b + RX_GOLD] >= 15 && r_f[b + RX_RAT] < 2) { r_act[e] = RA_BUY; return; }
  }
  if (bossHold) { return; }
  if (r_f[b + RX_WOOD] + r_f[b + RX_ORE] >= 8 || (r_f[b + RX_WOOD] >= 4 && r_f[b + RX_ORE] >= 3)) {
    if (r_td2 > RW_REACH_TOWN * RW_REACH_TOWN) { r_bot_toward(e, r_tdx, r_tdy); }
    else { r_act[e] = RA_INTERACT; r_tgt[e] = r_nearest_interact(e, allowGreat); }
    return;
  }
  r_go_to(e, select(1u, 2u, r_f[b + RX_ORE] < r_f[b + RX_WOOD]), tick, allowGreat);
}

fn r_mob_decide(e: u32, tick: u32) {
  let b = e * R_S;
  let ex = r_f[b + RX_X];
  let ey = r_f[b + RX_Y];
  var best = -1;
  var bestD2 = RW_FAR;
  for (var j = 0u; j < RW_PLAYERS; j++) {
    if (!r_alive(j)) { continue; }
    let dx = r_f[j * R_S + RX_X] - ex;
    let dy = r_f[j * R_S + RX_Y] - ey;
    let d2 = dx * dx + dy * dy;
    if (d2 < bestD2) { bestD2 = d2; best = i32(j); }
  }
  if (e >= RW_ANIMAL0) {
    if (best >= 0 && bestD2 < 250000) {
      r_toward(e, ex - r_f[u32(best) * R_S + RX_X], ey - r_f[u32(best) * R_S + RX_Y]);
    } else if (((tick >> 4u) + e) % 3u == 0u) { r_wander(e, tick); }
    return;
  }
  var aggro = RW_DAY_AGGRO;
  if (r_night(tick)) {
    r_nearest_town(r_f[b + RX_HX], r_f[b + RX_HY]);
    aggro = select(1200, 900 + (900 * world_difficulty()) / 100, r_td2 < RW_RAID_RADIUS * RW_RAID_RADIUS);
  }
  let style = r_f[b + RX_STYLE];
  let range = r_mob_range(style);
  let hx = r_f[b + RX_HX] - ex;
  let hy = r_f[b + RX_HY] - ey;
  if (best >= 0 && bestD2 < aggro * aggro && hx * hx + hy * hy < select(9000000, 1440000, r_is_boss(e))) {
    let dx = r_f[u32(best) * R_S + RX_X] - ex;
    let dy = r_f[u32(best) * R_S + RX_Y] - ey;
    let close = (range * 3) / 4;
    if (bestD2 > close * close) { r_toward(e, dx, dy); }
    r_try_attack(e, style, false);
  } else if (hx * hx + hy * hy > 360000) {
    r_toward(e, hx, hy);
  } else if (((tick >> 5u) + e) % 2u == 0u) {
    r_wander(e, tick);
  }
}

fn r_move(e: u32, tick: u32) {
  let b = e * R_S;
  let sprint = e < RW_PLAYERS && r_f[b + RX_SPRINT] == 1;
  var base = 24;
  if (e < RW_PLAYERS) { base = 26; }
  else if (e < RW_ANIMAL0) { base = 20; }
  if (e >= RW_MOB0 && e < RW_ANIMAL0 && r_night(tick)) {
    r_nearest_town(r_f[b + RX_HX], r_f[b + RX_HY]);
    if (r_td2 < RW_RAID_RADIUS * RW_RAID_RADIUS) { base = base + (12 * world_difficulty()) / 100; }
  }
  let speed = select(base, (base * 8) >> 2u, sprint);
  let ox = r_ctlx[e];
  let oy = r_ctly[e];
  let denom = max(r_isqrt(ox * ox + oy * oy), 128);
  let tx = (ox * speed) / denom;
  let ty = (oy * speed) / denom;
  r_f[b + RX_VX] = r_f[b + RX_VX] + ((tx - r_f[b + RX_VX]) >> 2u);
  r_f[b + RX_VY] = r_f[b + RX_VY] + ((ty - r_f[b + RX_VY]) >> 2u);
  var nx = r_f[b + RX_X] + r_f[b + RX_VX];
  var ny = r_f[b + RX_Y] + r_f[b + RX_VY];
  if (nx < 0 || nx > RW_WORLD - 1 || !r_passable_at(nx, r_f[b + RX_Y])) { nx = r_f[b + RX_X]; r_f[b + RX_VX] = 0; }
  if (ny < 0 || ny > RW_WORLD - 1 || !r_passable_at(nx, ny)) { ny = r_f[b + RX_Y]; r_f[b + RX_VY] = 0; }
  r_f[b + RX_X] = nx;
  r_f[b + RX_Y] = ny;
}

fn r_harvest_need(ty: u32, tool: i32) -> i32 {
  if (ty == 0u) { return 4; }
  if (ty == 3u) { return 2; }
  return max(2, select(14, 10, ty == 1u) - 2 * tool);
}

fn r_channel(e: u32) {
  let b = e * R_S;
  r_comp[e] = 0;
  let node = select(-1, r_tgt[e], r_act[e] == RA_INTERACT);
  var ty = RW_NONE;
  if (node >= 0) { ty = r_node_type(u32(node)); }
  if (node < 0 || ty == RW_TOWN || r_is_great(u32(node))) {
    r_f[b + RX_CHAN_NODE] = -1;
    r_f[b + RX_CHAN] = 0;
    return;
  }
  if (r_f[b + RX_CHAN_NODE] == node) { r_f[b + RX_CHAN] = r_f[b + RX_CHAN] + 1; }
  else { r_f[b + RX_CHAN_NODE] = node; r_f[b + RX_CHAN] = 1; }
  if (r_f[b + RX_CHAN] >= r_harvest_need(ty, r_f[b + RX_TOOL])) { r_comp[e] = 1; }
}

fn r_award(e: u32, gold: i32, craftUp: i32) {
  r_rew[e] = r_rew[e] + gold * 32 + craftUp * 1024;
  if (craftUp != 0) { stat_add(RS_CRAFT, craftUp); }
}

fn r_gain_xp(e: u32, field: u32, xp: i32) {
  let slot = e * R_S + field;
  let before = r_f[slot];
  r_f[slot] = before + xp;
  r_rew[e] = r_rew[e] + (min(before + xp, RW_XP_CAP) - min(before, RW_XP_CAP)) * 4;
}

fn r_milestone(e: u32, which: u32) {
  let slot = e * R_S + RX_ACH;
  if (((r_f[slot] >> which) & 1) != 0) { return; }
  r_f[slot] = r_f[slot] | (1 << which);
  var reward = 256;
  if (which == 0u) { reward = 1024; }
  else if (which == 1u) { reward = 2048; }
  r_rew[e] = r_rew[e] + reward;
}

fn r_restore(e: u32, slot: u32, amount: i32) {
  let i = e * R_S + slot;
  let before = r_f[i];
  let after = min(100, before + amount);
  r_f[i] = after;
  if (before < RW_NEED_LEVEL) { r_rew[e] = r_rew[e] + (after - before) * RW_NEED_REWARD; }
}

fn r_great(t: u32) {
  if (t >= 32u) { return; }
  let node = RW_GREAT0 + t;
  var harvesters = 0;
  for (var p = 0u; p < RW_PLAYERS; p++) {
    if (r_alive(p) && r_act[p] == RA_INTERACT && r_tgt[p] == i32(node)) { harvesters++; }
  }
  var progress = r_gprog[t];
  if (harvesters > 0) { progress = min(RW_GREAT_CAP, progress + harvesters); }
  else { progress = max(0, progress - 2); }
  r_gdone[t] = 0;
  r_gcount[t] = harvesters;
  if (harvesters >= 1 + (2 * world_difficulty()) / 100 && progress >= RW_GREAT_NEED) {
    r_gdone[t] = 1;
    progress = 0;
    r_ntimer[node] = RW_GREAT_TIMER;
  }
  r_gprog[t] = progress;
}

fn r_participant(p: u32, v: u32, tick: u32) -> bool {
  let b = p * R_S;
  if (!r_alive(p)) { return false; }
  if (r_f[b + RX_BOSS_ID] == i32(v) && i32(tick) - r_f[b + RX_BOSS_TICK] <= RW_BOSS_WINDOW) { return true; }
  return r_act[p] == RA_ATTACK && r_atkt[p] == i32(v);
}

fn r_interact(e: u32) {
  let b = e * R_S;
  let action = r_act[e];
  if (action == RA_INTERACT) {
    let node = r_tgt[e];
    if (node < 0) { return; }
    let ty = r_node_type(u32(node));
    if (ty == RW_TOWN) {
      let price = max(RW_MARKET_FLOOR, 100 - r_sat[u32(node) - 200u]);
      let gold = ((r_f[b + RX_WOOD] * 2 + r_f[b + RX_ORE] * 5) * price) / 100;
      r_sunit[e] = r_f[b + RX_WOOD] + r_f[b + RX_ORE];
      r_f[b + RX_WOOD] = 0;
      r_f[b + RX_ORE] = 0;
      r_f[b + RX_GOLD] = r_f[b + RX_GOLD] + gold;
      r_award(e, gold, 0);
      return;
    }
    if (r_is_great(u32(node))) {
      if (r_gdone[u32(node) - RW_GREAT0] == 0) {
        if (r_gcount[u32(node) - RW_GREAT0] >= 2) { r_rew[e] = r_rew[e] + RW_GREAT_TOGETHER; }
        return;
      }
      r_f[b + RX_RARE] = min(RW_RARE_CAP, r_f[b + RX_RARE] + RW_RARE_PER_GREAT);
      r_gain_xp(e, RX_XP3, RW_GREAT_XP);
      r_f[b + RX_GOLD] = r_f[b + RX_GOLD] + RW_GREAT_GOLD;
      r_award(e, RW_GREAT_GOLD, 0);
      r_rew[e] = r_rew[e] + RW_GREAT_REWARD;
      r_milestone(e, RW_MS_GREAT);
      if (e < LEARNERS) { stat_add(RS_GREAT, 1); }
      return;
    }
    if (r_comp[e] == 0) { return; }
    for (var j = 0u; j < e; j++) {
      if (r_comp[j] != 0 && r_tgt[j] == node && r_act[j] == RA_INTERACT) {
        r_f[b + RX_CHAN] = 0;
        r_f[b + RX_CHAN_NODE] = -1;
        return;
      }
    }
    r_f[b + RX_CHAN] = 0;
    r_f[b + RX_CHAN_NODE] = -1;
    var regrow = RW_SPRING_REGROW;
    if (ty == 0u) { regrow = RW_BERRY_REGROW; }
    else if (ty == 1u) { regrow = 600; }
    else if (ty == 2u) { regrow = 900; }
    r_ntimer[u32(node)] = u32((regrow * r_regrow) / 100);
    if (e < LEARNERS) { stat_add(RS_HARVEST, 1); }
    if (ty == 3u) {
      r_restore(e, RX_WATER, 40);
    } else if (ty == 0u) {
      r_restore(e, RX_FOOD, 30);
      r_gain_xp(e, RX_XP3, 6);
    } else if (ty == 1u) {
      r_f[b + RX_WOOD] = min(8, r_f[b + RX_WOOD] + 1);
      r_gain_xp(e, RX_XP3, 10);
    } else {
      r_f[b + RX_ORE] = min(8, r_f[b + RX_ORE] + 1);
      r_gain_xp(e, RX_XP3, 14);
    }
    return;
  }
  r_nearest_town(r_f[b + RX_X], r_f[b + RX_Y]);
  if (r_td2 > RW_REACH_TOWN * RW_REACH_TOWN) { return; }
  if (action == RA_CRAFT_TOOL && r_can_craft_tool(e)) {
    let t = r_f[b + RX_TOOL];
    r_f[b + RX_WOOD] = r_f[b + RX_WOOD] - (2 + 2 * t);
    r_f[b + RX_ORE] = r_f[b + RX_ORE] - (1 + t);
    if (t >= 2) { r_f[b + RX_RARE] = r_f[b + RX_RARE] - RW_RARE_TIER3; }
    r_f[b + RX_TOOL] = t + 1;
    r_award(e, 0, 1);
  } else if (action == RA_CRAFT_WEAP && r_can_craft_weap(e)) {
    let w = r_f[b + RX_WEAP];
    r_f[b + RX_ORE] = r_f[b + RX_ORE] - (2 + 2 * w);
    r_f[b + RX_WOOD] = r_f[b + RX_WOOD] - (1 + w);
    if (w >= 2) { r_f[b + RX_RARE] = r_f[b + RX_RARE] - RW_RARE_TIER3; }
    r_f[b + RX_WEAP] = w + 1;
    r_award(e, 0, 1);
  } else if (action == RA_BUY && r_f[b + RX_GOLD] >= 10 && r_f[b + RX_RAT] < 4) {
    r_f[b + RX_GOLD] = r_f[b + RX_GOLD] - 10;
    r_f[b + RX_RAT] = r_f[b + RX_RAT] + 1;
  }
}

fn r_damage(v: u32, tick: u32) {
  let vb = v * R_S;
  r_inc[v] = 0;
  r_kill[v] = -1;
  r_dies[v] = 0;
  r_part[v] = 0;
  r_vgold[v] = r_f[vb + RX_GOLD];
  r_vlevel[v] = r_f[vb + RX_LEVEL];
  if (!r_alive(v)) { return; }
  var best = 0;
  var incoming = 0;
  var fromMobs = 0;
  var killer = -1;
  for (var a = 0u; a < 64u; a++) {
    if (r_act[a] != RA_ATTACK || r_atkt[a] != i32(v) || !r_alive(a)) { continue; }
    if (a >= RW_PLAYERS) { fromMobs += r_atkd[a]; }
    else { incoming += r_atkd[a]; }
    if (r_atkd[a] > best) { best = r_atkd[a]; killer = i32(a); }
  }
  var guard = 0;
  if (v < RW_PLAYERS) { guard = RW_GUARD_PCT * min(RW_GUARD_CAP, r_ally_count(v)); }
  var armor = 0;
  if (v < RW_PLAYERS) { armor = RW_ARMOR_PCT * r_f[vb + RX_TOOL]; }
  incoming += (((fromMobs * (100 - guard)) / 100) * (100 - armor)) / 100;
  r_inc[v] = incoming;
  r_kill[v] = killer;
  r_dies[v] = select(0, 1, r_f[vb + RX_HP] - incoming <= 0);
  if (r_is_boss(v)) {
    var participants = 0;
    for (var p = 0u; p < RW_PLAYERS; p++) {
      if (r_participant(p, v, tick)) { participants++; }
    }
    r_part[v] = participants;
  }
}

fn r_market(i: u32, tick: u32) {
  var sold = 0;
  for (var p = 0u; p < RW_PLAYERS; p++) {
    if (r_tgt[p] == 200 + i32(i)) { sold += r_sunit[p]; }
  }
  var level = min(RW_MARKET_CAP, r_sat[i] + sold);
  if ((tick + i) % RW_MARKET_DECAY == 0u && level > 0) { level = level - 1; }
  r_sat[i] = level;
}

fn r_shape(e: u32, tick: u32) {
  let b = e * R_S;
  let view = r_view(tick);
  let view2 = view * view;
  let ex = r_f[b + RX_X];
  let ey = r_f[b + RX_Y];
  var springD2 = RW_FAR;
  var berryD2 = RW_FAR;
  for (var j = 0u; j < 200u; j++) {
    if (r_ntimer[j] != 0u) { continue; }
    let ty = r_node_type(j);
    if (ty != 3u && ty != 0u) { continue; }
    let pos = r_nxy[j];
    let dx = i32(pos & 0xffffu) - ex;
    let dy = i32(pos >> 16u) - ey;
    let d2 = dx * dx + dy * dy;
    if (d2 >= view2) { continue; }
    if (ty == 3u) { springD2 = min(springD2, d2); } else { berryD2 = min(berryD2, d2); }
  }
  var water = 0;
  if (springD2 != RW_FAR) { water = r_isqrt(springD2); }
  var food = 0;
  if (berryD2 != RW_FAR) { food = r_isqrt(berryD2); }
  let prevWater = r_f[b + RX_HX];
  let prevFood = r_f[b + RX_HY];
  var gain = 0;
  if (r_f[b + RX_WATER] < RW_SHAPE_NEED && prevWater > 0 && water > 0) { gain = gain + clamp((prevWater - water) >> 3u, -RW_SHAPE_CLAMP, RW_SHAPE_CLAMP); }
  if (r_f[b + RX_FOOD] < RW_SHAPE_NEED && r_f[b + RX_RAT] == 0 && prevFood > 0 && food > 0) { gain = gain + clamp((prevFood - food) >> 3u, -RW_SHAPE_CLAMP, RW_SHAPE_CLAMP); }
  r_f[b + RX_HX] = water;
  r_f[b + RX_HY] = food;
  r_rew[e] = r_rew[e] + gain;
}

fn r_share(e: u32) {
  if (e >= RW_PLAYERS || !r_alive(e)) { return; }
  r_ally[e] = r_ally_count(e);
  var pack = 0;
  for (var v = RW_MOB0; v < RW_BOSS0; v++) {
    let k = r_kill[v];
    if (r_dies[v] == 0 || k < 0 || k >= i32(RW_PLAYERS) || k == i32(e)) { continue; }
    let dx = r_f[u32(k) * R_S + RX_X] - r_f[e * R_S + RX_X];
    let dy = r_f[u32(k) * R_S + RX_Y] - r_f[e * R_S + RX_Y];
    if (dx * dx + dy * dy < RW_ALLY_RADIUS * RW_ALLY_RADIUS) { pack = pack + ((r_vlevel[v] * RW_SHARE_XP) << (10u * u32(r_atks[u32(k)]))); }
  }
  r_part[e] = pack;
}

fn r_apply(e: u32, tick: u32) {
  let b = e * R_S;
  if (e < 4u) { r_market(e, tick); }
  if (!r_alive(e)) {
    r_f[b + RX_DEAD] = r_f[b + RX_DEAD] - 1;
    if (r_f[b + RX_DEAD] == 0) { r_spawn(e, tick); }
    return;
  }
  if (r_act[e] == RA_ATTACK && e < RW_PLAYERS) {
    let style = r_atks[e];
    let xp = max(1, r_atkd[e] >> 1u);
    r_f[b + RX_STYLE] = style;
    r_gain_xp(e, RX_XP0 + u32(style), xp);
  }
  if (e < RW_PLAYERS) {
    for (var v = RW_BOSS0; v < RW_ANIMAL0; v++) {
      if (r_dies[v] == 0 || !r_participant(e, v, tick)) { continue; }
      let gold = RW_BOSS_GOLD / r_part[v];
      let xp = RW_BOSS_XP / r_part[v];
      r_gain_xp(e, RX_XP0 + u32(r_f[b + RX_STYLE]), xp);
      r_f[b + RX_GOLD] = r_f[b + RX_GOLD] + gold;
      r_f[b + RX_RARE] = min(RW_RARE_CAP, r_f[b + RX_RARE] + RW_BOSS_RARE);
      r_award(e, gold, 0);
      r_rew[e] = r_rew[e] + RW_BOSS_REWARD;
      r_milestone(e, RW_MS_BOSS);
      if (e < LEARNERS) { stat_add(RS_BOSS, 1); }
    }
    for (var st = 0u; st < 3u; st++) {
      let amount = (r_part[e] >> (10u * st)) & 1023;
      if (amount > 0) { r_gain_xp(e, RX_XP0 + st, amount); }
    }
    if (r_act[e] == RA_ATTACK && r_is_boss(u32(r_atkt[e]))) {
      r_f[b + RX_BOSS_ID] = r_atkt[e];
      r_f[b + RX_BOSS_TICK] = i32(tick);
    }
    for (var v = 0u; v < 64u; v++) {
      if (r_dies[v] == 0 || r_kill[v] != i32(e) || r_is_boss(v)) { continue; }
      let xpField = RX_XP0 + u32(r_atks[e]);
      if (v >= RW_ANIMAL0) {
        r_f[b + RX_RAT] = min(4, r_f[b + RX_RAT] + 1);
        r_f[b + RX_GOLD] = r_f[b + RX_GOLD] + 1;
        r_gain_xp(e, RX_XP3, 4);
        r_award(e, 1, 0);
        stat_add(RS_MOB_KILLS, 1);
      } else if (v >= RW_MOB0) {
        let level = r_vlevel[v];
        let gold = 2 + 2 * level;
        r_gain_xp(e, xpField, level * 12);
        r_f[b + RX_GOLD] = r_f[b + RX_GOLD] + gold;
        r_award(e, gold, 0);
        r_milestone(e, RW_MS_MOB);
        stat_add(RS_MOB_KILLS, 1);
      } else {
        let loot = min(r_vgold[v], 10);
        r_gain_xp(e, xpField, 10);
        r_f[b + RX_GOLD] = r_f[b + RX_GOLD] + loot;
        r_award(e, loot, 0);
      }
    }
  }
  r_f[b + RX_HP] = r_f[b + RX_HP] - r_inc[e];
  r_f[b + RX_LAST_HIT] = select(r_f[b + RX_LAST_HIT] + 1, 0, r_inc[e] > 0);
  if (r_f[b + RX_CD] > 0) { r_f[b + RX_CD] = r_f[b + RX_CD] - 1; }
  r_f[b + RX_AGE] = r_f[b + RX_AGE] + 1;
  let decay = 200 - world_difficulty();
  let period = u32(((2400 / r_hunger) * decay) / 100);
  let wperiod = u32(((1600 / r_hunger) * decay) / 100);
  if ((tick + e) % period == 0u && r_f[b + RX_FOOD] > 0) { r_f[b + RX_FOOD] = r_f[b + RX_FOOD] - 1; }
  if ((tick + e) % wperiod == 0u && r_f[b + RX_WATER] > 0) { r_f[b + RX_WATER] = r_f[b + RX_WATER] - 1; }
  if (e < RW_PLAYERS) {
    if (r_f[b + RX_SPRINT] == 1 && (tick + e) % 8u == 0u && r_f[b + RX_WATER] > 0) { r_f[b + RX_WATER] = r_f[b + RX_WATER] - 1; }
    if (r_f[b + RX_FOOD] < 40 && r_f[b + RX_RAT] > 0) {
      r_f[b + RX_RAT] = r_f[b + RX_RAT] - 1;
      r_restore(e, RX_FOOD, 50);
    }
    let allies = r_ally[e];
    if (e < LEARNERS && allies > 0) { stat_add(RS_ALLY, 1); }
    let fed = r_f[b + RX_FOOD] >= 40 && r_f[b + RX_WATER] >= 40;
    if ((tick + e) % 12u == 0u && fed && r_f[b + RX_HP] < r_max_hp(e)) { r_f[b + RX_HP] = r_f[b + RX_HP] + 1; }
    if ((tick + e) % 6u == 0u && allies > 0 && fed && r_f[b + RX_HP] < r_max_hp(e)) { r_f[b + RX_HP] = min(r_max_hp(e), r_f[b + RX_HP] + min(RW_AURA_CAP, allies)); }
    if ((tick + e) % 8u == 0u && (r_f[b + RX_FOOD] == 0 || r_f[b + RX_WATER] == 0)) { r_f[b + RX_HP] = r_f[b + RX_HP] - 2; }
    if (r_f[b + RX_HP] > 0) { r_rew[e] = r_rew[e] + RW_ALIVE_REWARD; }
  }
  if (r_f[b + RX_HP] <= 0) {
    if (e < RW_PLAYERS) {
      r_rew[e] = r_rew[e] - 1024;
      stat_add(RS_DEATHS, 1);
      if (e < LEARNERS) { set_dead(e); }
      else { r_spawn(e, tick); }
    } else {
      for (var i = 0u; i < 32u; i++) { r_f[b + i] = 0; }
      var respawn = select(150, 200, e < RW_ANIMAL0);
      if (r_is_boss(e)) { respawn = RW_BOSS_RESPAWN; }
      r_f[b + RX_DEAD] = respawn;
      r_f[b + RX_CHAN_NODE] = -1;
    }
  }
  if (e < LEARNERS) { set_reward(e, r_rew[e]); }
  else if (e < RW_PLAYERS) {
    stat_add(STAT_REW_BASE, r_rew[e]);
    stat_add(STAT_TICKS_BASE, 1);
  }
}

fn g_step(t: u32, tick: u32) {
  r_act[t] = RA_NONE;
  r_tgt[t] = -1;
  r_rew[t] = 0;
  r_ctlx[t] = 0;
  r_ctly[t] = 0;
  if (t < RW_PLAYERS) { r_sunit[t] = 0; }
  if (t < RW_PLAYERS && r_alive(t)) { r_shape(t, tick); }
  atomicStore(&obsBuf[t], RW_KEY_NONE);
  if (t < 32u) { atomicStore(&obsBuf[64u + t], RW_KEY_NONE); }
  workgroupBarrier();
  r_bot_prescan(t, tick);
  workgroupBarrier();
  if (r_alive(t)) {
    r_f[t * R_S + RX_SPRINT] = 0;
    if (t < LEARNERS) { r_learner_decide(t); }
    else if (t < RW_PLAYERS) { r_bot_decide(t, tick); }
    else { r_mob_decide(t, tick); }
  }
  workgroupBarrier();
  if (r_alive(t)) { r_move(t, tick); }
  for (var j = 0u; j < 4u; j++) {
    let p = t + 64u * j;
    if (r_ntimer[p] > 0u) { r_ntimer[p] = r_ntimer[p] - 1u; }
  }
  r_channel(t);
  workgroupBarrier();
  r_great(t);
  workgroupBarrier();
  if (t < RW_PLAYERS && r_alive(t)) { r_interact(t); }
  workgroupBarrier();
  r_damage(t, tick);
  workgroupBarrier();
  r_share(t);
  workgroupBarrier();
  r_apply(t, tick);
}

fn g_load(wi: u32, t: u32) {
  for (var j = 0u; j < 4u; j++) {
    let p = t + 64u * j;
    r_nxy[p] = game_word(wi, p);
    r_ntimer[p] = game_word(wi, RW_NODE_TIMER_OFF + p);
  }
  for (var i = 0u; i < 32u; i++) {
    let p = t + 64u * i;
    r_f[(p >> 5u) * R_S + (p & 31u)] = bitcast<i32>(game_word(wi, RW_ENT_OFF + p));
  }
  if (t < 4u) { r_sat[t] = i32(game_word(wi, RW_SAT_OFF + t)); }
  if (t < 32u) { r_gprog[t] = i32(game_word(wi, RW_GPROG_OFF + t)); }
  r_fill_terrain(t);
  if (t == 0u) {
    r_regrow = i32(game_word(wi, RW_CFG_OFF));
    r_mobpct = i32(game_word(wi, RW_CFG_OFF + 1u));
    r_hunger = i32(game_word(wi, RW_CFG_OFF + 2u));
    r_pvp = i32(game_word(wi, RW_CFG_OFF + 3u));
  }
}

fn g_save(wi: u32, t: u32) {
  for (var j = 0u; j < 4u; j++) {
    let p = t + 64u * j;
    game_set(wi, p, r_nxy[p]);
    game_set(wi, RW_NODE_TIMER_OFF + p, r_ntimer[p]);
  }
  for (var i = 0u; i < 32u; i++) {
    let p = t + 64u * i;
    game_set(wi, RW_ENT_OFF + p, bitcast<u32>(r_f[(p >> 5u) * R_S + (p & 31u)]));
  }
  if (t < 4u) { game_set(wi, RW_SAT_OFF + t, u32(r_sat[t])); }
  if (t < 32u) { game_set(wi, RW_GPROG_OFF + t, u32(r_gprog[t])); }
}

fn g_init(wi: u32, seed: u32, is_eval: u32, randomize: u32, t: u32) {
  if (t == 0u) {
    var regrow = 100u;
    var mobpct = 100u;
    var hunger = 100u;
    var pvp = 1u;
    if (randomize == 1u && is_eval == 0u) {
      regrow = 60u + mix4(seed, 0u, 0u, 8u) % 81u;
      mobpct = 70u + mix4(seed, 1u, 0u, 8u) % 61u;
      hunger = 75u + mix4(seed, 2u, 0u, 8u) % 51u;
      pvp = select(1u, 0u, mix4(seed, 3u, 0u, 8u) % 4u == 0u);
    }
    game_set(wi, RW_CFG_OFF, regrow);
    game_set(wi, RW_CFG_OFF + 1u, mobpct);
    game_set(wi, RW_CFG_OFF + 2u, hunger);
    game_set(wi, RW_CFG_OFF + 3u, pvp);
  }
  if (t < 4u) { r_sat[t] = 0; }
  if (t < 32u) { r_gprog[t] = 0; }
  r_fill_terrain(t);
  workgroupBarrier();
  for (var j = 0u; j < 4u; j++) { r_place_node(t + 64u * j); }
  r_spawn(t, 0u);
  workgroupBarrier();
  g_save(wi, t);
}
`;
