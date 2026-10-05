import http from 'node:http';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Rng, mix, randomGenome, genomeToJSON } from '../src/core.js';
import { GAMES } from '../src/games/index.js';
import { RealtimeNpc, FixedStepLoop, REWARD_SCALE } from './npc.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const WS_GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11';
const ACTION_TTL_TICKS = 2;
const MAX_CLIENT_FRAME = 4096;
const DEFAULT_TICK_HZ = 10;
const STATS_WORDS = 16;
const DEFAULT_PALETTE = ['#1d3a22', '#59606b', '#c8a13a', '#6fa8dc', '#b5651d', '#a05cc0', '#c0392b', '#2ecc71'];

function parseArgs(argv) {
  const out = { port: 8080, host: '0.0.0.0', seed: 1234, champions: null, game: null, hz: 0, speed: 1, maxEdges: 256, assign: 'roundrobin' };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    if (a === '--port') out.port = Number(next());
    else if (a === '--host') out.host = next();
    else if (a === '--seed') out.seed = Number(next());
    else if (a === '--champions') out.champions = next();
    else if (a === '--game') out.game = next();
    else if (a === '--hz') out.hz = Number(next());
    else if (a === '--speed') out.speed = Number(next());
    else if (a === '--max-edges') out.maxEdges = Number(next());
    else if (a === '--ac') out.ac = Object.assign(out.ac || {}, {});
    else if (a === '--ac-opt') {
      const target = out.ac || (out.ac = {});
      for (const part of String(next()).split(',')) {
        const kv = part.split('=');
        if (kv.length === 2 && kv[0].trim()) target[kv[0].trim()] = Number(kv[1]);
      }
    }
    else if (a === '--assign') out.assign = next();
    else if (!a.startsWith('--') && !out.champions) out.champions = a;
  }
  return out;
}

function populationMembers(parsed) {
  const members = Array.isArray(parsed) ? parsed : parsed && Array.isArray(parsed.policies) ? parsed.policies : null;
  return members && members.length > 0 && members.every((m) => m && typeof m === 'object' && m.genome) ? members : null;
}

function championSlot(slot, count, champions, mode) {
  const fractions = champions.fractions;
  if (mode !== 'fraction' || !fractions) return slot % champions.length;
  const total = fractions.reduce((sum, v) => sum + v, 0);
  const shares = total > 0 ? fractions.map((v) => v / total) : fractions.map(() => 1 / fractions.length);
  let cumulative = 0;
  for (let k = 0; k < shares.length; k++) {
    cumulative += shares[k];
    if ((slot + 0.5) / count < cumulative && shares[k] > 0) return k;
  }
  return shares.map((v, k) => (v > 0 ? k : -1)).filter((k) => k >= 0).pop();
}

function loadChampionJson(file) {
  const parsed = JSON.parse(fs.readFileSync(file, 'utf8'));
  const members = populationMembers(parsed);
  if (members) {
    const list = members.map((m) => m.genome);
    list.roles = members.map((m, i) => (typeof m.role === 'string' && m.role ? m.role : 'p' + i));
    list.fractions = members.map((m) => (Number.isFinite(m.fraction) && m.fraction > 0 ? m.fraction : 0));
    return list;
  }
  if (Array.isArray(parsed)) return parsed;
  if (parsed.champions) return parsed.champions;
  if (parsed.genomes) return parsed.genomes;
  return [parsed];
}

function randomChampions(game, opts) {
  const out = [];
  for (let i = 0; i < 8; i++) out.push(genomeToJSON(randomGenome(new Rng(mix(opts.seed, i, 0, 77)), opts.maxEdges, game.dims), game));
  return out;
}

class Session {
  constructor(game, opts) {
    this.game = game;
    this.opts = opts;
    this.stats = new Int32Array(STATS_WORDS);
    this.env = game.createEnv(opts.seed, game.defaultCfg(), false, this.stats);
    this.champions = opts.champions ? loadChampionJson(opts.champions) : [];
    if (this.champions.length === 0) this.champions = randomChampions(game, opts);
    for (const c of this.champions) if (c.game && c.game !== game.id) throw new Error('champion is for game ' + c.game + ', server runs ' + game.id);
    this.npcs = [];
    for (let a = 0; a < game.learners; a++) this.npcs.push(this.buildNpc(a));
    this.humans = new Map();
    this.scores = new Float64Array(game.learners);
    this.obs = new Float32Array(game.dims.nIn);
    this.actions = new Int32Array(game.learners);
    this.outputs = new Float32Array(game.learners * game.dims.nOut);
    this.tick = 0;
  }

  championIndex(a) {
    return championSlot(a, this.game.learners, this.champions, this.opts.assign);
  }

  roleCounts() {
    const counts = {};
    for (let a = 0; a < this.game.learners; a++) {
      const index = this.championIndex(a);
      const role = this.champions.roles ? this.champions.roles[index] : '#' + index;
      counts[role] = (counts[role] || 0) + 1;
    }
    return counts;
  }

  buildNpc(a) {
    if (this.champions.length === 0) return null;
    const genome = this.champions[this.championIndex(a)];
    return new RealtimeNpc(genome, { seed: this.opts.seed, agent: a, maxEdges: this.opts.maxEdges, learnMode: this.opts.ac ? 'ac' : undefined, ac: this.opts.ac || undefined });
  }

  freeSlot() {
    for (let a = this.game.learners - 1; a >= 0; a--) if (!this.humans.has(a)) return a;
    return -1;
  }

  seat(slot) {
    this.humans.set(slot, { action: 0, outputs: new Float32Array(this.game.dims.nOut), ttl: 0 });
    this.env.respawn(slot, this.tick);
    this.scores[slot] = 0;
  }

  release(slot) {
    this.humans.delete(slot);
    this.env.respawn(slot, this.tick);
    if (this.npcs[slot]) this.npcs[slot].rebirth();
  }

  setHumanAction(slot, action, outputs) {
    const human = this.humans.get(slot);
    if (!human) return;
    human.action = action;
    human.outputs.fill(0);
    if (outputs) for (let k = 0; k < human.outputs.length; k++) human.outputs[k] = Math.max(-1, Math.min(1, Number(outputs[k]) || 0));
    else human.outputs[action] = 1;
    human.ttl = ACTION_TTL_TICKS;
  }

  step(tick) {
    const { game, env } = this;
    const nOut = game.dims.nOut;
    for (let a = 0; a < game.learners; a++) {
      const human = this.humans.get(a);
      if (human) {
        const live = human.ttl > 0;
        this.actions[a] = live ? human.action : 0;
        for (let k = 0; k < nOut; k++) this.outputs[a * nOut + k] = live ? human.outputs[k] : 0;
        if (live) human.ttl--;
        continue;
      }
      const npc = this.npcs[a];
      if (!npc) {
        this.actions[a] = 0;
        continue;
      }
      env.observe(a, this.obs, tick);
      this.actions[a] = npc.decide(this.obs, tick);
      for (let k = 0; k < nOut; k++) this.outputs[a * nOut + k] = npc.outputs[k];
    }
    env.step(this.actions, this.outputs, tick);
    for (let a = 0; a < game.learners; a++) {
      const rewardFx = env.reward[a];
      const human = this.humans.get(a);
      this.scores[a] += rewardFx / REWARD_SCALE;
      const npc = this.npcs[a];
      const npcLifeOver = npc && !human && npc.lifeTicks + 1 >= game.maxAge;
      const died = env.dead[a] !== 0 || npcLifeOver;
      if (npc && !human) npc.reward(rewardFx / REWARD_SCALE, tick, died);
      if (died) {
        env.respawn(a, tick);
        if (npc && !human) npc.rebirth();
        if (!human) this.scores[a] = 0;
      }
    }
    this.tick = tick + 1;
  }

  wireSnapshot() {
    const snap = this.env.snapshot();
    const out = { t: 'snap', tick: this.tick };
    for (const key of Object.keys(snap)) out[key] = ArrayBuffer.isView(snap[key]) ? Array.from(snap[key]) : snap[key];
    out.agents = snap.agents.map((ag, a) => Object.assign({}, ag, { human: this.humans.has(a) }));
    out.scores = Array.from(this.scores, (s) => Math.round(s * 10) / 10);
    return out;
  }

  helloFor(slot) {
    const g = this.game;
    return {
      t: 'hello',
      slot,
      game: { id: g.id, name: g.name, tickHz: this.hz, actionNames: g.actionNames, inputNames: g.inputNames, palette: g.palette || DEFAULT_PALETTE, hasRender: typeof g.render === 'function', learners: g.learners, agents: g.agents },
      champions: this.champions.length
    };
  }
}

function encodeFrame(opcode, payload) {
  const len = payload.length;
  let header;
  if (len < 126) header = Buffer.from([0x80 | opcode, len]);
  else if (len < 65536) {
    header = Buffer.alloc(4);
    header[0] = 0x80 | opcode;
    header[1] = 126;
    header.writeUInt16BE(len, 2);
  } else {
    header = Buffer.alloc(10);
    header[0] = 0x80 | opcode;
    header[1] = 127;
    header.writeBigUInt64BE(BigInt(len), 2);
  }
  return Buffer.concat([header, payload]);
}

class Peer {
  constructor(socket, onMessage, onClose) {
    this.socket = socket;
    this.buffer = Buffer.alloc(0);
    this.fragments = [];
    this.fragmentOpcode = 0;
    this.onMessage = onMessage;
    this.onClose = onClose;
    this.closed = false;
    this.slot = -1;
    socket.on('data', (chunk) => this.receive(chunk));
    socket.on('close', () => this.finish());
    socket.on('error', () => this.finish());
  }

  finish() {
    if (this.closed) return;
    this.closed = true;
    this.onClose(this);
  }

  sendText(text) {
    if (this.closed || this.socket.destroyed) return;
    this.socket.write(encodeFrame(0x1, Buffer.from(text)));
  }

  close(code = 1000) {
    if (this.closed) return;
    const body = Buffer.alloc(2);
    body.writeUInt16BE(code, 0);
    this.socket.write(encodeFrame(0x8, body));
    this.socket.end();
    this.finish();
  }

  receive(chunk) {
    this.buffer = Buffer.concat([this.buffer, chunk]);
    if (this.buffer.length > MAX_CLIENT_FRAME * 4) return this.close(1009);
    while (this.parseFrame()) {
      if (this.closed) return;
    }
  }

  parseFrame() {
    const buf = this.buffer;
    if (buf.length < 2) return false;
    const fin = (buf[0] & 0x80) !== 0;
    const opcode = buf[0] & 0x0f;
    const masked = (buf[1] & 0x80) !== 0;
    let len = buf[1] & 0x7f;
    let offset = 2;
    if (len === 126) {
      if (buf.length < 4) return false;
      len = buf.readUInt16BE(2);
      offset = 4;
    } else if (len === 127) {
      if (buf.length < 10) return false;
      const big = buf.readBigUInt64BE(2);
      if (big > BigInt(MAX_CLIENT_FRAME)) {
        this.close(1009);
        return false;
      }
      len = Number(big);
      offset = 10;
    }
    if (len > MAX_CLIENT_FRAME) {
      this.close(1009);
      return false;
    }
    if (!masked) {
      this.close(1002);
      return false;
    }
    if (buf.length < offset + 4 + len) return false;
    const mask = buf.subarray(offset, offset + 4);
    const payload = Buffer.from(buf.subarray(offset + 4, offset + 4 + len));
    for (let i = 0; i < len; i++) payload[i] ^= mask[i & 3];
    this.buffer = buf.subarray(offset + 4 + len);
    this.handleFrame(fin, opcode, payload);
    return true;
  }

  handleFrame(fin, opcode, payload) {
    if (opcode === 0x8) return this.close(1000);
    if (opcode === 0x9) {
      this.socket.write(encodeFrame(0xa, payload));
      return;
    }
    if (opcode === 0xa) return;
    if (opcode === 0x1 || opcode === 0x2) {
      this.fragments = [payload];
      this.fragmentOpcode = opcode;
    } else if (opcode === 0x0) this.fragments.push(payload);
    else return;
    if (!fin) return;
    const whole = Buffer.concat(this.fragments);
    this.fragments = [];
    if (this.fragmentOpcode === 0x1) this.onMessage(this, whole.toString('utf8'));
  }
}

function startServer(opts) {
  const game = GAMES[opts.game || (GAMES.arena ? 'arena' : Object.keys(GAMES)[0])];
  if (!game) throw new Error('unknown game ' + opts.game + ', have ' + Object.keys(GAMES).join(','));
  const session = new Session(game, opts);
  session.hz = opts.hz || game.tickHz || DEFAULT_TICK_HZ;
  const peers = new Set();

  function onMessage(peer, text) {
    let msg;
    try {
      msg = JSON.parse(text);
    } catch {
      return;
    }
    if (msg.t !== 'act' || peer.slot < 0) return;
    const action = msg.a | 0;
    if (action < 0 || action >= game.dims.nOut) return;
    const outputs = Array.isArray(msg.o) && msg.o.length === game.dims.nOut ? msg.o : null;
    session.setHumanAction(peer.slot, action, outputs);
  }

  function onClose(peer) {
    peers.delete(peer);
    if (peer.slot >= 0) session.release(peer.slot);
  }

  const mime = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8', '.json': 'application/json' };
  const staticRoutes = { '/': 'play.html', '/play.html': 'play.html', '/npc.js': 'npc.js' };

  const server = http.createServer((req, res) => {
    const url = req.url.split('?')[0];
    if (url === '/health') {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ ok: true, game: game.id, tick: session.tick, hz: session.hz, humans: session.humans.size, peers: peers.size, champions: session.champions.length, roles: session.roleCounts() }));
      return;
    }
    if (url === '/render.js' && typeof game.render === 'function') {
      res.writeHead(200, { 'content-type': mime['.js'], 'cache-control': 'no-store' });
      res.end('export default ' + game.render.toString());
      return;
    }
    const file = staticRoutes[url];
    if (!file) {
      res.writeHead(404);
      res.end('not found');
      return;
    }
    fs.readFile(path.join(here, file), (err, data) => {
      if (err) {
        res.writeHead(500);
        res.end('read error');
        return;
      }
      res.writeHead(200, { 'content-type': mime[path.extname(file)] || 'application/octet-stream', 'cache-control': 'no-store' });
      res.end(data);
    });
  });

  server.on('upgrade', (req, socket) => {
    const key = req.headers['sec-websocket-key'];
    if (req.url.split('?')[0] !== '/ws' || !key || String(req.headers.upgrade).toLowerCase() !== 'websocket' || req.headers['sec-websocket-version'] !== '13') {
      socket.end('HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n');
      return;
    }
    const accept = crypto.createHash('sha1').update(key + WS_GUID).digest('base64');
    socket.write('HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ' + accept + '\r\n\r\n');
    socket.setNoDelay(true);
    const peer = new Peer(socket, onMessage, onClose);
    peers.add(peer);
    const slot = session.freeSlot();
    if (slot < 0) {
      peer.sendText(JSON.stringify({ t: 'full' }));
      return;
    }
    peer.slot = slot;
    session.seat(slot);
    peer.sendText(JSON.stringify(session.helloFor(slot)));
  });

  const loop = new FixedStepLoop({
    hz: session.hz,
    speed: opts.speed,
    step: (tick) => session.step(tick),
    render: () => {
      if (peers.size === 0) return;
      const text = JSON.stringify(session.wireSnapshot());
      for (const peer of peers) peer.sendText(text);
    }
  });

  server.on('close', () => loop.stop());
  server.listen(opts.port, opts.host, () => {
    console.log('listening ' + server.address().port + ' game=' + game.id + ' hz=' + session.hz + ' champions=' + session.champions.length + (opts.ac ? ' learn=ac ' + JSON.stringify(opts.ac) : ''));
    loop.start();
  });
  return server;
}

startServer(parseArgs(process.argv.slice(2)));
