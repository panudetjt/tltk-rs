#!/usr/bin/env node
// web_e2e.mjs — real-browser loop validator for the toolbox (playwright).
//
// The 2026-10-17 "no options" production incident: app.js carried a
// merged-in stray `}` — `node --check` (CJS mode) passed it, but the browser
// parses it as an ES module and the whole file died, leaving an empty
// function dropdown. This harness exists so that class of bug can never
// ship again: it opens the REAL page in a REAL chromium and
//
//   1. fails on any pageerror / console error / failed request;
//   2. asserts the function dropdown is populated and matches the cards
//      declared in web/js/app.js exactly (same ids, same order);
//   3. loops over EVERY function: selects it, fills the input with the
//      card's placeholder (or a per-function override below), presses
//      ประมวลผล, and requires the success stats line — PASS/FAIL table,
//      non-zero exit on any failure.
//
// Playwright resolves from web/e2e/node_modules (self-healed by
// `mise run web-e2e`, which pins the version to the box's cached chromium).
//
// Usage:
//   node scripts/web_e2e.mjs                 # serves ./web on 127.0.0.1:8123
//   node scripts/web_e2e.mjs --base <url>    # validate an existing deploy
//   node scripts/web_e2e.mjs --prod          # = --base https://tltk.panudet.dev/
//   node scripts/web_e2e.mjs --only chunk,w2v
//   node scripts/web_e2e.mjs --timeout <ms>  # per-function cap (default 240000;
//                                            # first tier-2 run fetches ~17.8MB)

import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import { existsSync } from 'node:fs';
import { extname, join, normalize, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO = dirname(dirname(fileURLToPath(import.meta.url)));
const WEB = join(REPO, 'web');

function parseArgs(argv) {
  const a = { base: null, only: null, timeout: 240000, loadOnly: false };
  for (let i = 2; i < argv.length; i++) {
    if (argv[i] === '--base') a.base = argv[++i];
    else if (argv[i] === '--prod') a.base = 'https://tltk.panudet.dev/';
    else if (argv[i] === '--only') a.only = new Set(argv[++i].split(','));
    else if (argv[i] === '--timeout') a.timeout = Number(argv[++i]);
    else if (argv[i] === '--load-only') a.loadOnly = true;
    else if (argv[i] === '--help') a.help = true;
  }
  if (a.help) {
    console.log('usage: web_e2e.mjs [--base URL | --prod] [--only fn1,fn2] [--timeout ms]');
    process.exit(0);
  }
  return a;
}

async function loadPlaywright() {
  const candidates = [
    join(REPO, 'web', 'e2e', 'node_modules', 'playwright', 'index.mjs'),
    'playwright',
  ];
  for (const c of candidates) {
    try {
      if (c.startsWith('/') && !existsSync(c)) continue;
      return await import(c.startsWith('/') ? pathToFileURL(c).href : c);
    } catch { /* try next */ }
  }
  console.error('[e2e] playwright not installed — run: mise run web-e2e');
  process.exit(2);
}

// --- static server (no deps; correct MIME for wasm streaming compile) -------
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.wasm': 'application/wasm',
  '.gz': 'application/gzip',
  '.txt': 'text/plain; charset=utf-8',
};

async function serveStatic(root, port) {
  const server = createServer(async (req, res) => {
    try {
      const url = new URL(req.url, 'http://x');
      let p = normalize(decodeURIComponent(url.pathname)).replace(/^(\.\.[/\\])+/, '');
      if (p === '/' || p === '') p = '/index.html';
      const file = join(root, p);
      if (!file.startsWith(root)) { res.writeHead(403); res.end(); return; }
      const body = await readFile(file);
      res.writeHead(200, {
        'content-type': MIME[extname(file)] ?? 'application/octet-stream',
        'content-length': body.length,
      });
      res.end(body);
    } catch {
      res.writeHead(404); res.end('not found');
    }
  });
  await new Promise((r) => server.listen(port, '127.0.0.1', r));
  return { url: `http://127.0.0.1:${port}/`, close: () => server.close() };
}

// --- expected function cards, evaluated from the UI source itself ------------
// (a regex bridge across one malformed card is exactly how the 2026-10-17
// seam stayed invisible — evaluate the real FUNCTIONS array literal instead)
function parseCards() {
  const text = readFileSync(join(WEB, 'js', 'app.js'), 'utf8');
  const start = text.indexOf('const FUNCTIONS = [');
  if (start < 0) throw new Error('FUNCTIONS array not found in app.js');
  const open = text.indexOf('[', start);
  let depth = 0, end = -1, inStr = null;
  for (let i = open; i < text.length; i++) {
    const ch = text[i], prev = text[i - 1];
    if (inStr) { if (ch === inStr && prev !== '\\') inStr = null; continue; }
    if (ch === "'" || ch === '"' || ch === '`') { inStr = ch; continue; }
    if (ch === '[') depth++;
    else if (ch === ']') { depth--; if (depth === 0) { end = i; break; } }
  }
  if (end < 0) throw new Error('FUNCTIONS array not terminated in app.js');
  const cards = new Function(`return ${text.slice(open, end + 1)};`)();
  for (const c of cards) {
    if (!c.id || typeof c.placeholder !== 'string' || !c.tier) {
      throw new Error(`malformed card in app.js: ${JSON.stringify(c).slice(0, 120)}`);
    }
  }
  return cards;
}

// Inputs where the card placeholder is not itself a good demo input
// (textana's placeholder hits the upstream single-EDU StatisticsError crash
// case — ok=false by design; smoke inputs must be the surviving kind).
const SMOKE_TEXT = 'ความสะดวกสบายของประชาชนในเขตเมือง\nความสะดวกสบายของประชาชนในเขตชนบท\nการพัฒนาคุณภาพชีวิตที่ดีกว่าเดิมเป็นเป้าหมายสำคัญของประเทศ';
const INPUT_OVERRIDES = {
  textana: SMOKE_TEXT,
  text_class: SMOKE_TEXT,
  txt2feat: SMOKE_TEXT,
  malt_parser: 'เขานั่งดูหนังอยู่ที่บ้าน',
  malt_parser_wordlist: 'เขา\nนั่ง\nดู\nหนัง',
  w2v_diffplot: 'กิน นอน เดิน',
  w2v_dimplot: 'แมว ปลา',
  w2v_compare_color: 'แมว ปลา นก',
  known: 'แมว หมา บ้า',
  edits2: 'ก', // full alphabet edits1∘edits1 — 'กก' renders ~16s of main-thread block
  check_thaidict: 'แมว',
  read_thaidict: 'คำวิ่ง\nทดสอบ.\nแมว',
  reset_thaidict: 'reset',
  wrd_len: 'ทดสอบ',
};

async function main() {
  const args = parseArgs(process.argv);
  const { chromium } = await loadPlaywright();

  let closer = null;
  let base = args.base;
  if (!base) {
    const s = await serveStatic(WEB, 8123);
    base = s.url; closer = s.close;
    console.log(`[e2e] serving ${WEB} at ${base}`);
  }
  const cards = parseCards();
  if (cards.length === 0) { console.error('[e2e] FATAL: no cards parsed from app.js'); process.exit(2); }
  console.log(`[e2e] app.js declares ${cards.length} functions`);

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  const problems = [];
  page.on('pageerror', (e) => problems.push(`[pageerror] ${e}`));
  page.on('console', (m) => { if (m.type() === 'error') problems.push(`[console.error] ${m.text()}`); });
  page.on('requestfailed', (r) => problems.push(`[requestfailed] ${r.url()} ${r.failure()?.errorText ?? ''}`));

  await page.goto(base, { waitUntil: 'load', timeout: 60000 });
  await page.waitForTimeout(1500);

  // Gate 1: the page must be alive — the "no options" incident gate.
  const loadErrors = problems.splice(0);
  if (loadErrors.length) {
    console.error('[e2e] FAIL: page produced errors on load:');
    loadErrors.forEach((e) => console.error('  ' + e.slice(0, 400)));
    await browser.close(); closer?.(); process.exit(1);
  }

  // Gate 2: dropdown must match the declared cards exactly (ids + order).
  const uiIds = await page.locator('#functionSelect option').evaluateAll(
    (os) => os.map((o) => o.value),
  );
  if (!args.only) {
    const mismatch = uiIds.length !== cards.length || cards.some((c, i) => uiIds[i] !== c.id);
    if (mismatch || uiIds.length === 0) {
      console.error(`[e2e] FAIL: dropdown has ${uiIds.length} options, app.js declares ${cards.length} (or ids/order differ)`);
      console.error('  dropdown:', uiIds.join(','));
      console.error('  app.js  :', cards.map((c) => c.id).join(','));
      await browser.close(); closer?.(); process.exit(1);
    }
  }
  console.log(`[e2e] dropdown OK (${uiIds.length} options)`);

  if (args.loadOnly) {
    // deploy-time gate: the page loads and the dropdown matches — the loop
    // over every function lives in the CI web-e2e job, not on the deploy path
    console.log('[e2e] load gate passed (--load-only)');
    await browser.close(); closer?.(); process.exit(0);
  }

  // Gate 3: run every function through the real UI.
  const expected = args.only ? cards.filter((c) => args.only.has(c.id)) : cards;
  if (expected.length === 0) { console.error('[e2e] --only matched nothing'); process.exit(2); }
  const results = [];
  for (const card of expected) {
    const row = { id: card.id, ok: false, note: '' };
    try {
      await page.locator('#functionSelect').selectOption(card.id);
      await page.waitForTimeout(150); // let onFunctionChange settle
      const input = page.locator('#inputText');
      // deterministic: never reuse whatever the previous loop iteration left
      const inputText = INPUT_OVERRIDES[card.id] ?? card.placeholder;
      if (!inputText || !inputText.trim()) throw new Error('no input available (empty placeholder)');
      await input.fill(inputText);
      // dispatchEvent skips actionability handshakes — big-output renders
      // (edits2) block the main thread for seconds and would time out a click
      await page.locator('#runBtn').dispatchEvent('click');

      const stats = page.locator('#execStats');
      await page.waitForFunction(
        () => /ประมวลผลสำเร็จ|เกิดข้อผิดพลาด/.test(document.getElementById('execStats').textContent),
        undefined,
        { timeout: args.timeout },
      );
      const statText = await stats.textContent();
      if (statText.includes('เกิดข้อผิดพลาด')) {
        const err = await page.locator('#errorMessage').textContent().catch(() => '');
        row.note = (err || 'unknown error').trim().slice(0, 140);
      } else {
        row.ok = true;
        row.note = statText.trim();
      }
    } catch (e) {
      row.note = 'timeout/crash: ' + String(e).split('\n')[0].slice(0, 140);
    }
    // A page that died mid-loop invalidates everything after it.
    const died = problems.splice(0);
    if (died.length) row.note += ' | ' + died[0].slice(0, 160);
    results.push(row);
    console.log(`${row.ok ? 'PASS' : 'FAIL'}  ${card.id.padEnd(20)} ${row.note}`);
  }

  await browser.close();
  closer?.();

  const failed = results.filter((r) => !r.ok);
  console.log(`\n[e2e] ${results.length - failed.length}/${results.length} passed`);
  if (failed.length) {
    console.error('[e2e] FAILURES: ' + failed.map((f) => f.id).join(', '));
    process.exit(1);
  }
}

main().catch((e) => { console.error('[e2e] fatal:', e); process.exit(2); });
