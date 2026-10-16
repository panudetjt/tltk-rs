#!/usr/bin/env node

/**
 * scripts/node_bench.mjs — benchmark the shipped browser wasm under Node.js.
 *
 * The wasm lane of `mise run bench-report`: drives the same tltk_web glue the
 * browser toolbox uses (wasm-bindgen call boundary included), so the numbers
 * are what a browser actually pays per query. Same protocol as the python and
 * native-rust lanes: per function group, warmup W then repeat R timed passes,
 * one pass = every corpus case of that function once, median µs/pass.
 *
 * State semantics: one persistent warm engine for the whole run — that is the
 * browser reality (the engine is built once per page after first paint and
 * reused for every query). The native lane resets its Session per pass instead;
 * the difference is the colloc counters accumulating, which does not move the
 * timing. Model decode and wasm instantiation happen before timing, matching
 * load_models() placement in the other lanes.
 *
 * The artifact records its own environment, including the sha256 of the exact
 * tltk_web.wasm binary that was measured (build provenance: scalar or simd128
 * flavor, wasm32-unknown-unknown).
 *
 * Usage: node scripts/node_bench.mjs --wasm-dir web/wasm/simd --corpus corpus/corpus.jsonl
 *          [--label wasm-simd] [--models-dir crates/tltk-core/data]
 *          [--warmup 3] [--repeat 10] [--json]
 */

import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';

function printHelp() {
  console.log(`Usage: node scripts/node_bench.mjs [options]

Options:
  --wasm-dir <dir>      Directory with tltk_web.js/.wasm (e.g. web/wasm/simd)
  --corpus <file>       Corpus jsonl file (e.g. corpus/corpus.jsonl)
  --models-dir <dir>    Models directory (default: crates/tltk-core/data)
  --label <name>        Flavor label for the report (default: wasm dir basename)
  --warmup <n>          Warmup passes (default: 3)
  --repeat <n>          Timed passes (default: 10)
  --json                JSON to stdout, human table to stderr
  -h, --help            Show this help
`);
}

const args = process.argv.slice(2);
let wasmDir = null;
let corpusPath = null;
let modelsDir = 'crates/tltk-core/data';
let label = null;
let warmup = 3;
let repeat = 10;
let json = false;

for (let i = 0; i < args.length; i++) {
  const arg = args[i];
  if (arg === '--help' || arg === '-h') {
    printHelp();
    process.exit(0);
  } else if (arg === '--wasm-dir' && i + 1 < args.length) {
    wasmDir = args[++i];
  } else if (arg === '--corpus' && i + 1 < args.length) {
    corpusPath = args[++i];
  } else if (arg === '--models-dir' && i + 1 < args.length) {
    modelsDir = args[++i];
  } else if (arg === '--label' && i + 1 < args.length) {
    label = args[++i];
  } else if (arg === '--warmup' && i + 1 < args.length) {
    warmup = parseInt(args[++i], 10);
  } else if (arg === '--repeat' && i + 1 < args.length) {
    repeat = parseInt(args[++i], 10);
  } else if (arg === '--json') {
    json = true;
  } else {
    console.error(`Unknown or incomplete option: ${arg}`);
    printHelp();
    process.exit(2);
  }
}
if (!wasmDir || !corpusPath) {
  console.error('Error: --wasm-dir and --corpus are required.');
  printHelp();
  process.exit(2);
}
if (!label) label = path.basename(wasmDir);

function log(line) {
  (json ? process.stderr : process.stdout).write(line + '\n');
}

function gitState() {
  try {
    const sha = execFileSync('git', ['rev-parse', '--short', 'HEAD'], { encoding: 'utf8' }).trim();
    if (!sha) return null;
    const dirty = execFileSync('git', ['status', '--porcelain'], { encoding: 'utf8' })
      .split('\n').filter((l) => l.trim()).length;
    return dirty > 0 ? `${sha}+${dirty} dirty files` : sha;
  } catch {
    return null;
  }
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  if (n % 2 === 1) return sorted[(n - 1) / 2];
  return Math.round((sorted[n / 2 - 1] + sorted[n / 2]) / 2);
}

// ---- environment (field names mirror the python/rust lanes) ----------------
const wasmFile = path.resolve(wasmDir, 'tltk_web.wasm');
const wasmBytes = fs.readFileSync(wasmFile);
const environment = {
  timestamp_utc: new Date().toISOString().replace(/\.\d+Z$/, 'Z'),
  host: os.hostname(),
  os: process.platform,
  kernel: os.release(),
  cpu_model: os.cpus()[0]?.model?.trim() ?? null,
  cpu_cores_available: os.cpus().length,
  mem_total_kb: Math.round(os.totalmem() / 1024),
  load_avg_1m_5m_15m: os.loadavg().map((x) => Math.round(x * 100) / 100),
  git: gitState(),
  node: process.version,
  v8: process.versions.v8,
  wasm: {
    flavor: label,
    target: 'wasm32-unknown-unknown',
    file: path.basename(wasmFile),
    bytes: wasmBytes.length,
    sha256: crypto.createHash('sha256').update(wasmBytes).digest('hex'),
  },
};
log(`# env: ${JSON.stringify(environment)}`);

// ---- load corpus ------------------------------------------------------------
const cases = [];
for (const raw of fs.readFileSync(path.resolve(corpusPath), 'utf-8').split('\n')) {
  const line = raw.trim();
  if (!line || line.startsWith('#')) continue;
  cases.push(JSON.parse(line));
}
const fnOrder = [];
const groups = new Map();
for (const c of cases) {
  if (!groups.has(c.function)) {
    fnOrder.push(c.function);
    groups.set(c.function, []);
  }
  groups.get(c.function).push(c);
}

// ---- engine init (outside the timed region, like load_models in the lanes) --
const { default: init, Tltk } = await import(
  pathToFileURL(path.resolve(wasmDir, 'tltk_web.js')).href
);
await init({ module_or_path: wasmBytes });

const tier1Names = [
  'thdict.bin.gz', 'sylseg.bin.gz', 'sylvar.bin.gz', 'phs.bin.gz',
  'pos.bin.gz', 'ner.bin.gz', 'wordtier.bin.gz', 'textclass.bin.gz',
  'sylrule.txt', 'thaisyl.txt',
];
const tier1Datas = tier1Names.map((name) => {
  const p = path.resolve(modelsDir, name);
  if (!fs.existsSync(p)) {
    console.error(`Error: required tier-1 model file missing: ${p}`);
    process.exit(1);
  }
  return new Uint8Array(fs.readFileSync(p));
});
const tltk = new Tltk(tier1Names, tier1Datas);

const tier2Names = ['edu.bin.gz', 'w2v.bin.gz'];
const tier2Datas = tier2Names
  .filter((name) => fs.existsSync(path.resolve(modelsDir, name)))
  .map((name) => new Uint8Array(fs.readFileSync(path.resolve(modelsDir, name))));
if (tier2Datas.length > 0) tltk.attach(tier2Names.slice(0, tier2Datas.length), tier2Datas);

const rssAfterLoad = process.memoryUsage().rss;

// one persistent warm engine — browser semantics (see header note)
function runCase(c) {
  try {
    return tltk.process(c.function, c.text, c.cutoff || 'medium') ?? '';
  } catch {
    return ''; // upstream crash cases are results for the other lanes too
  }
}

// smoke every function once so lazy inits (glyph atlas etc.) land pre-timing
for (const fn of fnOrder) {
  for (const c of groups.get(fn)) runCase(c);
}

const sink = { n: 0 }; // keep results alive the way black_box does natively
const fnResults = {};
for (const fn of fnOrder) {
  const group = groups.get(fn);
  for (let w = 0; w < warmup; w++) {
    for (const c of group) sink.n += runCase(c).length;
  }
  const passUs = [];
  for (let r = 0; r < repeat; r++) {
    const t0 = performance.now();
    for (const c of group) sink.n += runCase(c).length;
    passUs.push(Math.round((performance.now() - t0) * 1000));
  }
  fnResults[fn] = { cases: group.length, median_us_per_pass: median(passUs) };
  log(`${fn.padEnd(19)}cases=${String(group.length).padEnd(2)}  median_us_per_pass=${fnResults[fn].median_us_per_pass}`);
}

const total = Object.values(fnResults).reduce((acc, f) => acc + f.median_us_per_pass, 0);
const rssAfterPass = process.memoryUsage().rss;
log(`${'total'.padEnd(19)}cases=${String(cases.length).padEnd(2)}  median_us_per_pass=${total}`);
log(`sink=${sink.n} (prevents dead-code elimination of results)`);

if (json) {
  process.stdout.write(
    JSON.stringify(
      {
        label,
        protocol: { warmup, repeat, statistic: 'median_us_per_pass', engine: 'persistent warm (browser semantics)' },
        environment,
        functions: fnResults,
        total_us_per_pass: total,
        memory: {
          after_load_kb: Math.round(rssAfterLoad / 1024),
          after_pass_kb: Math.round(rssAfterPass / 1024),
        },
      },
      null,
      2,
    ) + '\n',
  );
}
