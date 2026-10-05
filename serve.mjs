import { createServer } from 'node:http';
import { readFile, mkdir, appendFile, writeFile } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('.', import.meta.url));
const port = Number(process.argv[2] || 8123);
const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.css': 'text/css', '.wgsl': 'text/plain' };

const runsDir = join(root, 'runs');

async function readBody(request) {
  const chunks = [];
  for await (const chunk of request) chunks.push(chunk);
  return Buffer.concat(chunks);
}

async function handleApi(request, response, url) {
  const name = (url.searchParams.get('name') || 'run').replace(/[^a-zA-Z0-9_.-]/g, '_');
  await mkdir(runsDir, { recursive: true });
  const body = await readBody(request);
  if (url.pathname === '/api/log') await appendFile(join(runsDir, name + '.log'), body.toString() + String.fromCharCode(10));
  else await writeFile(join(runsDir, name), body);
  response.writeHead(204).end();
}

createServer(async (request, response) => {
  const url = new URL(request.url, 'http://x');
  if (request.method === 'POST' && (url.pathname === '/api/log' || url.pathname === '/api/save')) { await handleApi(request, response, url); return; }
  const path = normalize(decodeURIComponent(new URL(request.url, 'http://x').pathname));
  const file = join(root, path.endsWith('/') ? path + 'index.html' : path);
  if (!file.startsWith(root)) { response.writeHead(403).end(); return; }
  try {
    const body = await readFile(file);
    response.writeHead(200, { 'content-type': types[extname(file)] || 'application/octet-stream', 'cache-control': 'no-store' }).end(body);
  } catch {
    response.writeHead(404).end('not found');
  }
}).listen(port, () => console.log('serving ' + root + ' on http://localhost:' + port));
