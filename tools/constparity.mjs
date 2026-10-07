// Static JS<->WGSL constant parity check for realm.
//
// The GPU lockstep test is the real authority, but it needs a GPU. This is the
// cheap substitute that catches the "string replace hit the wrong occurrence"
// bug class (AGENTS.md: a literal select(18u, 16u, ..) survived the 16->20
// stat growth and zeroed every h2h tick for 10/21 tests).
//
// Two shapes are checked:
//   1. const twin  -- `const X = N;`            <->  `const RW_X: i32 = N;`
//   2. fn twin     -- `const X = [a, b, c];`    <->  `fn r_x(s: i32) -> i32 { if (s==0) {return a;} ... }`
// Shape 2 exists because WGSL const arrays of i32 are awkward to index from
// non-uniform control flow, so several tunables are written as if-chains. Those
// are the ones a const-only scan silently misses, which is why the alias map
// below is explicit rather than derived.
//
// Run: node tools/constparity.mjs
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const js = fs.readFileSync(path.join(ROOT, 'src/games/realm.js'), 'utf8');
const wg = fs.readFileSync(path.join(ROOT, 'src/games/realm.wgsl.js'), 'utf8');

// JS const name -> WGSL if-chain function name. Add here when a new tunable is
// written as a WGSL function.
const FN_TWINS = {
  STYLE_RANGE: 'r_style_range',
  MOB_STYLE_RANGE: 'r_mob_range',
  STYLE_COOLDOWN: 'r_style_cooldown',
};

function collect(src, re) {
  const out = {};
  const r = new RegExp(re.source, 'gm');
  let m;
  while ((m = r.exec(src))) out[m[1]] = m[2];
  return out;
}

const JS_SCALAR = /^const ([A-Z][A-Z0-9_]*)\s*=\s*(-?\d+)\s*;/;
const WG_SCALAR = /^const RW_([A-Z0-9_]*)\s*:\s*(?:i32|u32|f32)\s*=\s*(-?\d+)/;
const JS_ARRAY = /^const ([A-Z][A-Z0-9_]*)\s*=\s*\[([^\]]*)\]/;

// `fn r_x(s: i32) -> i32 { if (s == 0) { return A; } if (s == 1) { return B; } return C; }`
const WG_IFCHAIN =
  /^fn (r_[a-z0-9_]+)\s*\(\s*s:\s*i32\s*\)\s*->\s*i32\s*\{((?:\s*if\s*\(s == \d+\)\s*\{\s*return\s*-?\d+;\s*\})*\s*return\s*-?\d+;\s*\})/;
// `fn r_x() -> i32 { return N; }`
const WG_ZEROARG = /^fn (r_[a-z0-9_]+)\s*\(\s*\)\s*->\s*i32\s*\{\s*return\s*(-?\d+);\s*\}/;

const norm = (s) => s.replace(/\s+/g, '').replace(/,$/, '');

function ifChainValues(name, body) {
  const vals = [];
  const re = /if\s*\(s == (\d+)\)\s*\{\s*return\s*(-?\d+);\s*\}/g;
  let m;
  let next = 0;
  while ((m = re.exec(body))) {
    const idx = Number(m[1]);
    if (idx !== next) throw new Error(`${name}: if-chain skips index ${next}`);
    vals.push(m[2]);
    next++;
  }
  const tail = /\}\s*return\s*(-?\d+);\s*\}/.exec(body);
  if (!tail) throw new Error(`${name}: if-chain has no final return`);
  vals.push(tail[1]);
  return vals;
}

let ok = 0;
const bad = [];

const J = collect(js, JS_SCALAR);
const W = collect(wg, WG_SCALAR);
const JA = collect(js, JS_ARRAY);
const zeroArg = collect(wg, WG_ZEROARG);

for (const k of Object.keys(J)) {
  if (k in W) {
    if (J[k] !== W[k]) bad.push(`${k}: js=${J[k]} wgsl=${W[k]}`);
    else ok++;
  } else if (FN_TWINS[k] && FN_TWINS[k] in zeroArg) {
    if (J[k] !== zeroArg[FN_TWINS[k]]) bad.push(`${k}: js=${J[k]} wgsl ${FN_TWINS[k]}()=${zeroArg[FN_TWINS[k]]}`);
    else ok++;
  }
}

const chains = {};
{
  const r = new RegExp(WG_IFCHAIN.source, 'gm');
  let m;
  while ((m = r.exec(wg))) chains[m[1]] = ifChainValues(m[1], m[2]);
}
for (const k of Object.keys(JA)) {
  const fn = FN_TWINS[k];
  if (!fn) continue;
  if (!(fn in chains)) { bad.push(`${k}: no WGSL if-chain ${fn}`); continue; }
  const a = norm(JA[k]);
  const b = norm(chains[fn].join(','));
  if (a !== b) bad.push(`${k}: js=[${a}] wgsl ${fn}=[${b}]`);
  else ok++;
}

for (const k of Object.keys(FN_TWINS)) {
  if (!(k in J) && !(k in JA)) bad.push(`${k}: alias named in FN_TWINS but absent from realm.js`);
}

console.log(`constparity: pairs ok=${ok}  MISMATCH=${bad.length}`);
for (const b of bad) console.log('  MISMATCH  ' + b);
process.exit(bad.length ? 1 : 0);
