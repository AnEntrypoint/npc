// Emit the fully assembled realm WGSL (core template + realm chunk) to stdout
// or a file, so it can be validated offline with naga (`cargo install naga-cli`)
// instead of waiting for a GPU:
//
//   node tools/emitwgsl.mjs > /tmp/realm.wgsl && naga /tmp/realm.wgsl
//
// naga is a full WGSL front-end: it catches reserved keywords, type errors and
// syntax. It does NOT check device limits (workgroup storage bytes, binding
// counts), so it complements rather than replaces `tests=1` on real hardware.
//
// cfg mirrors src/engine.js:193 -> buildShader(game, { worlds, islands,
// evalStart, maxEdges, migrateCount, evo }).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildShader } from '../src/shader.js';
import { REALM } from '../src/games/realm.js';
import { DEFAULT_OPTS, evoConfig } from '../src/core.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const arg = process.argv[2];
const worlds = Number(process.env.WORLDS || 512);

const o = { ...DEFAULT_OPTS, worlds, islands: 2, maxEdges: 256, migrateCount: 4 };
const evalWorlds = Math.max(1, Math.round(worlds * o.evalFraction));
const cfg = {
  worlds,
  islands: o.islands,
  evalStart: worlds - evalWorlds,
  maxEdges: o.maxEdges,
  migrateCount: o.migrateCount,
  evo: evoConfig(o),
};

const out = buildShader(REALM, cfg);
if (arg) {
  fs.writeFileSync(arg, out);
  const undef = out.match(/undefined|NaN/g);
  process.stderr.write(
    `wrote ${arg} ${out.length} bytes; undefined/NaN tokens: ${undef ? undef.length : 0}\n`,
  );
} else {
  process.stdout.write(out);
}
