import { readFileSync, writeFileSync, existsSync, readdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const root = dirname(fileURLToPath(import.meta.url));
const order = ['core.js', 'games/*', 'games/index.js', 'shader.js', 'engine.js', 'rl.js', 'rlshader.js', 'rlengine.js', 'app.js'];
const placeholder = '__APP_SCRIPT__';

function inline(source) {
  return source
    .replace(/^import\s[^;]*?from\s*['"][^'"]+['"];?[ \t]*$/gm, '')
    .replace(/^import\s*['"][^'"]+['"];?[ \t]*$/gm, '')
    .replace(/^export\s+default\s+/gm, '')
    .replace(/^export\s*\{[^}]*\};?[ \t]*$/gm, '')
    .replace(/^export\s+/gm, '')
    .replace(/<\/script/gi, '<\/script');
}

function expand(entry) {
  if (entry !== 'games/*') return [entry];
  const listed = readdirSync(join(root, 'src', 'games')).filter((f) => f.endsWith('.js')).map((f) => 'games/' + f);
  return listed.filter((f) => f !== 'games/index.js').sort((x, y) => Number(y.endsWith('.wgsl.js')) - Number(x.endsWith('.wgsl.js')) || x.localeCompare(y));
}

const parts = [];
const seen = new Set();
for (const name of order.flatMap(expand)) {
  if (seen.has(name)) continue;
  seen.add(name);
  const path = join(root, 'src', name);
  if (!existsSync(path)) {
    process.stderr.write(`missing src/${name}, skipped
`);
    continue;
  }
  parts.push(inline(readFileSync(path, 'utf8')));
}

const template = readFileSync(join(root, 'template.html'), 'utf8');
const body = parts.join('\n');
const builder = readFileSync(fileURLToPath(import.meta.url), 'utf8');
const hash = createHash('sha256').update(template).update('\n').update(body).update('\n').update(builder).digest('hex').slice(0, 16);
const stamp = '<meta name="build-hash" content="' + hash + '">';
const html = template.replace(placeholder, () => body).replace(/<head([^>]*)>/i, (tag) => tag + stamp);

if (process.argv.includes('--check')) {
  const current = readFileSync(join(root, 'index.html'), 'utf8');
  const recorded = /<meta name="build-hash" content="([0-9a-f]+)">/.exec(current);
  if (!recorded) {
    process.stdout.write('index.html has no build-hash, rebuild before trusting it\n');
    process.exit(1);
  }
  if (recorded[1] !== hash) {
    process.stdout.write('index.html is stale: build-hash ' + recorded[1] + ' but sources hash to ' + hash + '\n');
    process.exit(1);
  }
  process.stdout.write('index.html current, build-hash ' + hash + '\n');
} else {
  writeFileSync(join(root, 'index.html'), html);
  process.stdout.write(`index.html ${html.length} bytes from ${parts.length} modules, build-hash ${hash}\n`);
}
