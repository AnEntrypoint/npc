export const BLOB_WGSL = `
const B_WORLD: i32 = 8192;
const B_PELLETS: u32 = 256u;
const B_START_MASS: i32 = 160;
const B_PELLET_MASS: i32 = 8;
const B_VIEW: i32 = 2048;
const B_VIEW2: i32 = 4194304;
const B_RADIUS_K: i32 = 6;
const B_SPEED_NUM: i32 = 540;
const B_BOOST_MIN: i32 = 300;
const B_FLEE2: i32 = 2560000;
const B_CHASE2: i32 = 3240000;
const B_FAR: i32 = 1073741824;
const B_STAT_PELLETS: u32 = STAT_GAME0;
const B_STAT_KILLS: u32 = STAT_GAME0 + 1u;
const B_STAT_BOOST: u32 = STAT_GAME0 + 2u;
const B_STAT_MASS: u32 = STAT_GAME0 + 3u;
const B_STAT_EATEN: u32 = STAT_GAME0 + 4u;
const B_PEL_TIMER_OFF: u32 = 256u;
const B_AGENT_OFF: u32 = 512u;
const B_CFG_OFF: u32 = 896u;

struct BNear { food: i32, fx: i32, fy: i32, threat: i32, tx: i32, ty: i32, prey: i32, qx: i32, qy: i32 }

var<workgroup> b_pel_pos: array<u32, 256>;
var<workgroup> b_pel_timer: array<u32, 256>;
var<workgroup> b_x: array<i32, 48>;
var<workgroup> b_y: array<i32, 48>;
var<workgroup> b_vx: array<i32, 48>;
var<workgroup> b_vy: array<i32, 48>;
var<workgroup> b_mass: array<i32, 48>;
var<workgroup> b_age: array<i32, 48>;
var<workgroup> b_radius: array<i32, 48>;
var<workgroup> b_ctlx: array<i32, 48>;
var<workgroup> b_ctly: array<i32, 48>;
var<workgroup> b_boost: array<i32, 48>;
var<workgroup> b_bcost: array<i32, 48>;
var<workgroup> b_mass0: array<i32, 48>;
var<workgroup> b_gain: array<atomic<i32>, 48>;
var<workgroup> b_eatenBy: array<i32, 48>;
var<workgroup> b_respawn: u32;
var<workgroup> b_decay: u32;
var<workgroup> b_aggro: u32;

var<private> sec_food: array<i32, 8>;
var<private> sec_threat: array<i32, 8>;
var<private> sec_prey: array<i32, 8>;
var<private> near: BNear;

fn isqrt(x: i32) -> i32 {
  var s = i32(sqrt(f32(x)));
  loop { if (s * s > x) { s = s - 1; } else { break; } }
  loop { if ((s + 1) * (s + 1) <= x) { s = s + 1; } else { break; } }
  return s;
}

fn sector_of(dx: i32, dy: i32) -> u32 {
  let ax = abs(dx);
  let ay = abs(dy);
  if (ay * 12 <= ax * 5) { return select(4u, 0u, dx >= 0); }
  if (ax * 12 <= ay * 5) { return select(6u, 2u, dy >= 0); }
  if (dx > 0) { return select(7u, 1u, dy > 0); }
  return select(5u, 3u, dy > 0);
}

fn radius_of(m: i32) -> i32 { return isqrt(m) * B_RADIUS_K; }

fn near_value(d2: i32) -> f32 {
  if (d2 >= B_VIEW2) { return 0.0; }
  return f32(B_VIEW - isqrt(d2)) * (1.0 / 2048.0);
}

fn place_pellet(p: u32, tick: u32) {
  let x = mix4(w_seed, tick, p, 21u) % 8192u;
  let y = mix4(w_seed, tick, p, 22u) % 8192u;
  b_pel_pos[p] = x | (y << 16u);
  b_pel_timer[p] = 0u;
}

fn place_agent(a: u32, tick: u32) {
  b_x[a] = i32(mix4(w_seed, tick, a, 4u) % 8192u);
  b_y[a] = i32(mix4(w_seed, tick, a, 5u) % 8192u);
  b_vx[a] = 0;
  b_vy[a] = 0;
  b_mass[a] = B_START_MASS;
  b_age[a] = 0;
  b_radius[a] = radius_of(B_START_MASS);
}

fn g_respawn(a: u32, tick: u32) { place_agent(a, tick); }

fn scan(a: u32) {
  for (var s = 0u; s < 8u; s++) { sec_food[s] = B_FAR; sec_threat[s] = B_FAR; sec_prey[s] = B_FAR; }
  near = BNear(B_FAR, 0, 0, B_FAR, 0, 0, B_FAR, 0, 0);
  let ax = b_x[a];
  let ay = b_y[a];
  let ar = b_radius[a];
  for (var p = 0u; p < B_PELLETS; p++) {
    if (b_pel_timer[p] != 0u) { continue; }
    let pos = b_pel_pos[p];
    let dx = i32(pos & 0xffffu) - ax;
    let dy = i32(pos >> 16u) - ay;
    let d2 = dx * dx + dy * dy;
    if (d2 >= B_VIEW2) { continue; }
    let s = sector_of(dx, dy);
    if (d2 < sec_food[s]) { sec_food[s] = d2; }
    if (d2 < near.food) { near.food = d2; near.fx = dx; near.fy = dy; }
  }
  for (var j = 0u; j < AGENTS; j++) {
    if (j == a) { continue; }
    let dx = b_x[j] - ax;
    let dy = b_y[j] - ay;
    let d2 = dx * dx + dy * dy;
    if (d2 >= B_VIEW2) { continue; }
    let s = sector_of(dx, dy);
    let jr = b_radius[j];
    if (jr * 100 > ar * 115) {
      if (d2 < sec_threat[s]) { sec_threat[s] = d2; }
      if (d2 < near.threat) { near.threat = d2; near.tx = dx; near.ty = dy; }
    } else if (ar * 100 > jr * 115) {
      if (d2 < sec_prey[s]) { sec_prey[s] = d2; }
      if (d2 < near.prey) { near.prey = d2; near.qx = dx; near.qy = dy; }
    }
  }
}

fn g_observe(t: u32, tick: u32) {
  if (t >= LEARNERS) { return; }
  scan(t);
  for (var s = 0u; s < 8u; s++) {
    set_obs(t, s, near_value(sec_food[s]));
    set_obs(t, 8u + s, near_value(sec_threat[s]));
    set_obs(t, 16u + s, near_value(sec_prey[s]));
  }
  set_obs(t, 24u, f32(isqrt(b_mass[t])) * (1.0 / 128.0));
  set_obs(t, 25u, select(0.0, 1.0, b_mass[t] > B_BOOST_MIN));
  set_obs(t, 26u, f32(b_x[t]) * (1.0 / 8192.0));
  set_obs(t, 27u, f32(B_WORLD - b_x[t]) * (1.0 / 8192.0));
  set_obs(t, 28u, f32(b_y[t]) * (1.0 / 8192.0));
  set_obs(t, 29u, f32(B_WORLD - b_y[t]) * (1.0 / 8192.0));
}

fn normalize_into(dx: i32, dy: i32, a: u32) {
  let len = isqrt(dx * dx + dy * dy);
  let safe = max(len, 1);
  b_ctlx[a] = select((dx * 128) / safe, 0, len == 0);
  b_ctly[a] = select((dy * 128) / safe, 0, len == 0);
}

fn bot_control(a: u32, tick: u32) {
  scan(a);
  b_boost[a] = 0;
  if (near.threat < B_FLEE2) {
    normalize_into(-near.tx, -near.ty, a);
    b_boost[a] = select(0, 1, b_mass[a] > B_BOOST_MIN);
  } else if (b_aggro == 1u && near.prey < B_CHASE2) {
    normalize_into(near.qx, near.qy, a);
  } else if (near.food < B_FAR) {
    normalize_into(near.fx, near.fy, a);
  } else {
    let dir = mix4(w_seed, tick >> 5u, a, 22u) & 7u;
    var dx = 128;
    var dy = 0;
    if (dir == 1u) { dx = 90; dy = 90; }
    else if (dir == 2u) { dx = 0; dy = 128; }
    else if (dir == 3u) { dx = -90; dy = 90; }
    else if (dir == 4u) { dx = -128; dy = 0; }
    else if (dir == 5u) { dx = -90; dy = -90; }
    else if (dir == 6u) { dx = 0; dy = -128; }
    else if (dir == 7u) { dx = 90; dy = -90; }
    b_ctlx[a] = dx;
    b_ctly[a] = dy;
  }
}

fn learner_control(a: u32) {
  let ox = out_of(a, 0u) - out_of(a, 1u);
  let oy = out_of(a, 2u) - out_of(a, 3u);
  b_ctlx[a] = clamp(i32(floor(ox * 128.0)), -128, 128);
  b_ctly[a] = clamp(i32(floor(oy * 128.0)), -128, 128);
  b_boost[a] = select(0, 1, out_of(a, 4u) > 0.5 && b_mass[a] > B_BOOST_MIN);
}

fn move_agent(a: u32) {
  let boost = b_boost[a];
  let speed = (B_SPEED_NUM / (isqrt(b_mass[a]) + 8)) * select(1, 2, boost == 1);
  let ox = b_ctlx[a];
  let oy = b_ctly[a];
  let denom = max(isqrt(ox * ox + oy * oy), 128);
  let tx = (ox * speed) / denom;
  let ty = (oy * speed) / denom;
  b_vx[a] = b_vx[a] + ((tx - b_vx[a]) >> 2u);
  b_vy[a] = b_vy[a] + ((ty - b_vy[a]) >> 2u);
  var nx = b_x[a] + b_vx[a];
  var ny = b_y[a] + b_vy[a];
  if (nx < 0) { nx = 0; b_vx[a] = 0; }
  if (nx > B_WORLD - 1) { nx = B_WORLD - 1; b_vx[a] = 0; }
  if (ny < 0) { ny = 0; b_vy[a] = 0; }
  if (ny > B_WORLD - 1) { ny = B_WORLD - 1; b_vy[a] = 0; }
  b_x[a] = nx;
  b_y[a] = ny;
  b_bcost[a] = select(0, b_mass[a] >> 6u, boost == 1);
  b_mass[a] = b_mass[a] - b_bcost[a];
  b_radius[a] = radius_of(b_mass[a]);
  if (boost == 1) { stat_add(B_STAT_BOOST, 1); }
}

fn pellet_phase(t: u32, tick: u32) {
  for (var j = 0u; j < 4u; j++) {
    let p = t + 64u * j;
    if (b_pel_timer[p] > 0u) {
      b_pel_timer[p] = b_pel_timer[p] - 1u;
      if (b_pel_timer[p] == 0u) { place_pellet(p, tick); }
    } else {
      let pos = b_pel_pos[p];
      let px = i32(pos & 0xffffu);
      let py = i32(pos >> 16u);
      for (var a = 0u; a < AGENTS; a++) {
        let dx = px - b_x[a];
        let dy = py - b_y[a];
        let r = b_radius[a];
        if (dx * dx + dy * dy < r * r) {
          atomicAdd(&b_gain[a], B_PELLET_MASS);
          b_pel_timer[p] = b_respawn;
          stat_add(B_STAT_PELLETS, 1);
          break;
        }
      }
    }
  }
}

fn victim_phase(j: u32) {
  var eater = -1;
  var bestRadius = -1;
  let rj = b_radius[j];
  for (var i = 0u; i < AGENTS; i++) {
    if (i == j) { continue; }
    let ri = b_radius[i];
    if (ri * 100 <= rj * 115) { continue; }
    let reach = ri - (rj >> 2u);
    let dx = b_x[j] - b_x[i];
    let dy = b_y[j] - b_y[i];
    if (reach > 0 && dx * dx + dy * dy < reach * reach && ri > bestRadius) {
      bestRadius = ri;
      eater = i32(i);
    }
  }
  b_eatenBy[j] = eater;
  if (eater >= 0) {
    atomicAdd(&b_gain[u32(eater)], (b_mass[j] * 3) >> 2u);
    stat_add(B_STAT_KILLS, 1);
    if (j < LEARNERS) { stat_add(B_STAT_EATEN, 1); }
  }
}

fn apply_agent(a: u32) {
  let gained = atomicLoad(&b_gain[a]);
  stat_add(B_STAT_MASS, gained);
  var m = b_mass[a] + gained;
  m = m - (m >> b_decay);
  b_mass[a] = m;
  b_age[a] = b_age[a] + 1;
  var rewardFx = (isqrt(m * 64) - isqrt(b_mass0[a] * 64)) * 16;
  if (b_eatenBy[a] >= 0) { rewardFx = rewardFx - 1024; }
  if (a < LEARNERS) {
    set_reward(a, rewardFx);
    if (b_eatenBy[a] >= 0) { set_dead(a); }
  } else {
    stat_add(STAT_REW_BASE, rewardFx);
    stat_add(STAT_TICKS_BASE, 1);
  }
  b_radius[a] = radius_of(m);
}

fn g_step(t: u32, tick: u32) {
  if (t < AGENTS) {
    atomicStore(&b_gain[t], 0);
    b_eatenBy[t] = -1;
    b_mass0[t] = b_mass[t];
    if (t < LEARNERS) { learner_control(t); } else { bot_control(t, tick); }
  }
  workgroupBarrier();
  if (t < AGENTS) { move_agent(t); }
  workgroupBarrier();
  pellet_phase(t, tick);
  if (t < AGENTS) { victim_phase(t); }
  workgroupBarrier();
  if (t < AGENTS) { apply_agent(t); }
  workgroupBarrier();
  if (t == 0u) {
    for (var a = LEARNERS; a < AGENTS; a++) { if (b_eatenBy[a] >= 0) { place_agent(a, tick); } }
  }
  workgroupBarrier();
}

fn g_load(wi: u32, t: u32) {
  for (var j = 0u; j < 4u; j++) {
    let p = t + 64u * j;
    b_pel_pos[p] = game_word(wi, p);
    b_pel_timer[p] = game_word(wi, B_PEL_TIMER_OFF + p);
  }
  if (t < AGENTS) {
    let base = B_AGENT_OFF + t * 8u;
    b_x[t] = bitcast<i32>(game_word(wi, base));
    b_y[t] = bitcast<i32>(game_word(wi, base + 1u));
    b_vx[t] = bitcast<i32>(game_word(wi, base + 2u));
    b_vy[t] = bitcast<i32>(game_word(wi, base + 3u));
    b_mass[t] = bitcast<i32>(game_word(wi, base + 4u));
    b_age[t] = bitcast<i32>(game_word(wi, base + 5u));
    b_radius[t] = radius_of(b_mass[t]);
    b_boost[t] = 0;
    b_bcost[t] = 0;
    b_mass0[t] = b_mass[t];
  }
  if (t == 0u) {
    b_respawn = game_word(wi, B_CFG_OFF);
    b_decay = game_word(wi, B_CFG_OFF + 1u);
    b_aggro = game_word(wi, B_CFG_OFF + 2u);
  }
}

fn g_save(wi: u32, t: u32) {
  for (var j = 0u; j < 4u; j++) {
    let p = t + 64u * j;
    game_set(wi, p, b_pel_pos[p]);
    game_set(wi, B_PEL_TIMER_OFF + p, b_pel_timer[p]);
  }
  if (t < AGENTS) {
    let base = B_AGENT_OFF + t * 8u;
    game_set(wi, base, bitcast<u32>(b_x[t]));
    game_set(wi, base + 1u, bitcast<u32>(b_y[t]));
    game_set(wi, base + 2u, bitcast<u32>(b_vx[t]));
    game_set(wi, base + 3u, bitcast<u32>(b_vy[t]));
    game_set(wi, base + 4u, bitcast<u32>(b_mass[t]));
    game_set(wi, base + 5u, bitcast<u32>(b_age[t]));
  }
}

fn g_init(wi: u32, seed: u32, is_eval: u32, randomize: u32, t: u32) {
  if (t == 0u) {
    var respawn = 60u;
    var decay = 10u;
    var aggro = 1u;
    if (randomize == 1u && is_eval == 0u) {
      respawn = 30u + mix4(seed, 0u, 0u, 8u) % 91u;
      decay = 9u + mix4(seed, 1u, 0u, 8u) % 3u;
      aggro = select(0u, 1u, mix4(seed, 2u, 0u, 8u) % 4u == 0u);
    }
    game_set(wi, B_CFG_OFF, respawn);
    game_set(wi, B_CFG_OFF + 1u, decay);
    game_set(wi, B_CFG_OFF + 2u, aggro);
  }
  for (var j = 0u; j < 4u; j++) { place_pellet(t + 64u * j, 0u); }
  if (t < AGENTS) { place_agent(t, 0u); }
  g_save(wi, t);
}
`;
