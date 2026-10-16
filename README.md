# tltk-rs

Rust rewrite of [TLTK](https://pypi.org/project/tltk/) (Thai Language
Toolkit, Chulalongkorn University) — the Python package behind
[app.stichula.org/py/tltk](https://app.stichula.org/py/tltk/) — served as a
JSON API on a **Cloudflare Worker**.

Parity is **defined by the stichula validator** and enforced by the
`tltk-verifier` harness: every corpus case is sent to the live validator, the
response is stored as a golden row, and the Rust implementation must produce
byte-identical output.

```
corpus.jsonl ──▶ scripts/capture_stichula.py ──▶ golden/stichula.jsonl
      │                                                 │
      └──▶ tltk-verifier check ◀─────────────────────────┘
```

## Layout

| path | what |
|---|---|
| `crates/tltk-core/` | the library: all 41 wire functions, no network, no async |
| `crates/tltk-worker/` | Cloudflare Worker (workers-rs) — thin HTTP shell over the core |
| `crates/tltk-verifier/` | parity harness (`run` / `check` / `selftest`) |
| `tools/convert/` | Python scripts: pickle → portable binary models, local oracle |
| `corpus/` | test inputs (`{function, text, cutoff}` JSONL) |
| `golden/` | captured responses (stichula + local Python oracle) |
| `scripts/` | corpus gen, stichula capture, web e2e validator |

Core and server are **separate crates**: `tltk-worker` depends on `tltk-core`,
which never touches HTTP so it can also be used as a plain library.

## API (mirrors the validator)

`POST /process` with `{"text": "...", "function": "...", "cutoff": "medium"}` →
`{"success": true, "function": ..., "input": ..., "result": "..."}` (or
`success: false` + `error`). 41 functions: the 16 validator names
(`th2ipa`, `th2roman`, `word_segment`, `syl_segment`, `segment`, `g2p`,
`th2read`, `spell_variants`, `spoonerism`, `pos_tag`, `ner_tag`,
`similar_words`, `similarity`, `outofgroup`, `analogy`, `plotwords`) plus
`textana`, `text_class`, `txt2feat`, `chunk`, `tnc_tag`, `th2ipa_all`,
`sylparse_all`, `wrd_len`, `match_engtext`, `known`, `edits2`,
`check_thaidict`, `reset_thaidict`, `read_thaidict`, `dg_print`, `dg_save`,
`print_dtree`, `scan_dtree`, `get_tree_depth`, `tree_max_depth`, `w2v`,
`compound`, `w2v_diffplot`, `w2v_dimplot`, `w2v_compare_color`. The port
surface is **tltk 1.11** (repinned 2026-10-17; 1.11's only functional change
vs 1.10 is TextAna's LdvTTR ×1000 → ×100, which the `text_class`/`txt2feat`
feature path reads through TextAna). The 16 stichula functions are
byte-identical to the validator (captured at 1.10 and unchanged in 1.11);
the rest are byte-identical to the 1.11 Python oracle (the three
`w2v_*plot` renderers are disclosed-diff PNG, like `plotwords`) — see
[PORTING-GAPS.md](PORTING-GAPS.md) for the full inventory, including what
stays blocked (TNC.3g data, MaltParser's JVM).

## Model pipeline

The Python package ships pickled models (nltk PerceptronTagger, sklearn-crfsuite
CRF, sklearn RandomForest, gensim Word2Vec, trigram dicts). `tools/convert/convert.py`
converts each into a compact sorted-entry binary format (`*.bin.gz`), decoded
with binary search at load:

| blob | size (gz) | source | served from |
|---|---|---|---|
| thdict | 0.4 MB | thdict pickle | embedded in WASM |
| sylseg | 2.5 MB | sylseg.3g + read_stat derivations | embedded |
| sylvar | 0.06 MB | sylform_var.pklz | embedded |
| phs | 1.2 MB | PhSTrigram.sts (8 derived tables) | embedded |
| pos | 1.1 MB | pos_tagger.pklz (PerceptronTagger) | embedded |
| wordtier | 0.4 MB | word.pklz (TextAna tier lexicon) | embedded |
| ner | 0.09 MB | crfsuite model (see NER note) | embedded |
| textclass | 1.0 MB | chklevel_rf.pklz + chkgenre_rf.pklz | embedded |
| edu | 6.2 MB | sent_segment.pklz (DictVectorizer+RF) | web asset (`data/edu.bin.gz`) |
| w2v | 11.6 MB | TNCc5model3.bin (gensim KeyedVectors) | web asset (`data/w2v.bin.gz`) |
All blobs ship as site assets under `web/data/` (committed symlink →
`crates/tltk-core/data/`) and are fetched once per tier by the loader —
tier-1 upfront, edu/w2v lazily on the first tier-2 function. Nothing rides
in R2 any more.

## Parity status (verifier, corpus = corpus/corpus.jsonl, 244 cases)

- **vs the live stichula validator: 241/244 (98.8%)**
- vs the local Python oracle (tltk 1.11 wheel in a venv): 240/241
  (one case is the disclosed NER tag-boundary flip above; the local wrapper's
  OOV check and `selftest` cover the anchor case 9/9)

Known, documented divergences:

1. **NER weight precision** — the crfsuite model artifact is recoverable only
   through `Tagger.dump()` (6-decimal weights; the original
   `ner_tagger.pklz` carries the model in a FileResource temp file that only
   materialises at unpickle time). One tag-boundary case in the corpus flips
   on a razor-thin margin. TODO: parse the crfsuite binary for full f32
   weights.
2. **plotwords** — same `IMAGE:<base64 png>` wire shape (a Rust-rendered PCA
   scatter); the pixels differ from the server's matplotlib rendering, which
   is out of scope.
3. **Stateful colloc counters** — Python mutates module-global BiCount/Count /
   TotalWord once per new syllable pair per PROCESS; we replicate per request
   (= a fresh Python process). A long-lived server process drifts slightly
   from this; the golden capture tolerates it.

## Commands

```bash
# build the verifier
cargo build -p tltk-verifier

# sanity checks on the anchor case
./target/debug/verifier selftest crates/tltk-core/data

# run the core over the corpus
./target/debug/verifier run corpus/corpus.jsonl crates/tltk-core/data

# parity check vs golden (pass/fail per case; exit 1 on any mismatch)
./target/debug/verifier check corpus/corpus.jsonl golden/stichula.jsonl crates/tltk-core/data

# regenerate models from the Python sources (needs the venv in tools/convert)
python3 tools/convert/convert.py <tltk-package-dir> crates/tltk-core/data \
    --ner-model tools/convert/ner.crfsuite
```

## wasm parity lane

Run parity verification under wasm32-WASI (`wasm32-wasip1`) using `wasmtime`:

```bash
bash scripts/wasm_parity.sh
```

- **What it proves**: The full verifier — all 16 functions incl. w2v — byte-identical inside wasm32-WASI.
- **What it does NOT prove**: The deployed worker target is `wasm32-unknown-unknown` with its own bindings; that crate is compile-checked per round and now also simd128-armed — the runtime lane uses `wasip1` because it needs std fs.
- **How to run it**: Run `bash scripts/wasm_parity.sh` (or `./scripts/wasm_parity.sh`). It compiles `tltk-verifier` for `wasm32-wasip1`, runs all three golden sets (`stichula`, `local`, `extra`) and `selftest` under `wasmtime`, verifies output numbers match expected parity baselines, and exits 0 only when all pass.

## Deploy

**Toolbox (production)** — assets-only Workers, no bucket and no bindings.
Push to main runs `deploy-cloudflare.yml`: `mise run build-web`, stage
`web/data` as real copies (the committed symlink becomes plain files for
the upload), parity-gate the shipped wasm, then
`npx -y wrangler@4 deploy --config web/wrangler.jsonc`. The models are part
of the site assets. Manual reproduction of just the deploy:
`mise run build-web` then the same wrangler line (staging the data dir the
same way first).

Real-browser regression gate: `mise run web-e2e` (playwright; also a CI
job) — fails on any page error, asserts the dropdown matches the declared
functions, then runs every one of them through the actual UI.

**Standalone worker crate (optional, legacy)** — `crates/tltk-worker` keeps
the R2 `MODELS` binding for edu/w2v:

```bash
npx wrangler r2 bucket create tltk-models          # once
./scripts/upload_r2.sh tltk-models                 # edu + w2v
cd crates/tltk-worker && npx wrangler deploy       # builds wasm + uploads
```

Without the binding, the embedded models serve the tier-1 functions; the
edu/w2v ones (segment, chunk, the w2v family) return an explicit error.

## toolbox (client-only)

A client-only web toolbox page usable identically to the stichula original with **zero server**: static hosting only (free unlimited), with all inference executing directly in the browser via WebAssembly.

### What it is

- **Full 16-function Thai NLP suite**: `th2ipa`, `th2roman`, `word_segment`, `syl_segment`, `segment`, `g2p`, `th2read`, `spell_variants`, `spoonerism`, `pos_tag`, `ner_tag`, `similar_words`, `similarity`, `outofgroup`, `analogy`, and `plotwords` (PCA scatter plot rendered as inline PNG).
- **Byte-identical results**: strictly follows tltk 1.11 / stichula parity.
- **Zero external dependencies**: no frameworks, no bundler, no CDN fonts or scripts — works completely offline once cached.

### How to run locally

```bash
bash scripts/serve_toolbox.sh
```

The script checks for required model files (including gitignored `edu.bin.gz` and `w2v.bin.gz`), builds the WebAssembly glue via `scripts/build_web.sh` if needed, maps model assets to `/data`, and serves the application at `http://localhost:8080/`.

### How to deploy (static-only)

Deploy to any static web host (Cloudflare Pages / Workers Static Assets, Netlify, Vercel, GitHub Pages, or S3/R2). No server-side worker script or container is required.

Example Cloudflare Workers static assets configuration (`web/wrangler.jsonc`):

```toml
name = "tltk-toolbox"
compatibility_date = "2026-10-16"

[assets]
directory = "./static"
binding = "ASSETS"
```

Deploy with:

```bash
npx wrangler deploy --config static/wrangler.jsonc
# Or via Cloudflare Pages:
npx wrangler pages deploy static
```

### Browser support & runtime SIMD selection

- **Runtime probe**: On first page load, an inlined ~29-byte probe module evaluates `WebAssembly.validate` for 128-bit SIMD (`simd128`) capability.
- **Dynamic dual-build loading**: SIMD-capable browsers dynamically load `web/wasm/simd/tltk_web.js` for accelerated dot-product vector operations, while other runtimes seamlessly fall back to `web/wasm/scalar/tltk_web.js`.
- **Cached probing**: The detection result is stored in `localStorage` to avoid redundant validation on future visits.

### First-load sizes & tiered model loading

- **Tier-1 models (~5.4 MB)**: Loaded on first tool execution. Fetches the 8 core models and tables (`thdict.bin.gz`, `sylseg.bin.gz`, `sylvar.bin.gz`, `phs.bin.gz`, `pos.bin.gz`, `ner.bin.gz`, `sylrule.txt`, `thaisyl.txt`).
- **Tier-2 models (~17.8 MB)**: Loaded on-demand only when calling discourse segmentation or Word2Vec operations (`similar_words`, `similarity`, `outofgroup`, `analogy`, `plotwords`, or `segment`). Fetches `edu.bin.gz` (~6.2 MB) and `w2v.bin.gz` (~11.6 MB).
- **Cache-first semantics**: All model assets are fetched with `cache: 'force-cache'`, persisting in HTTP/browser cache across sessions.
- **Asset footprint**: Total UI overhead (HTML + CSS + JS, excluding WASM binaries and model blobs) is under 45 KB.
## Credits

TLTK by Wirote Aroonmanakun, Chulalongkorn University (PyPI `tltk` 1.11).
This project reimplements the inference pipelines in Rust for the
[stichula.org](https://app.stichula.org/py/tltk/) API surface; model artifacts
remain © their original authors.

## License

Code: MIT (see [LICENSE](LICENSE)). This project derives from and redistributes
converted artifacts of TLTK 1.11 (BSD-3-Clause) and a Thai font atlas — see
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for the notices that travel
with distributions.

## Performance

**Four lanes on the same corpus** (the 16-function `corpus/corpus.jsonl`,
244 cases, same protocol, same box): the Python oracle (tltk 1.10), native
Rust (`tltk-core --release`), and — because that is what the browser actually
runs — the **shipped wasm engine** (`tltk-web` glue, `wasm32-unknown-unknown`)
in both flavors the toolbox serves: simd128 and scalar. One command reproduces
everything below:

```bash
mise run bench-report          # ROUNDS=3 default; ROUNDS=n to change
```

It runs interleaved rounds of all four lanes plus a memory profile and writes
raw per-round JSON + `summary.md` to `target/bench/<UTC-stamp>/`. Every
artifact embeds a recorded environment block — timestamp, host, kernel, CPU
model, available cores, total RAM, load average, **git revision + dirty-file
count**, toolchain versions; the wasm lanes add the **sha256 of the exact
`tltk_web.wasm` measured** — so the numbers and their measurement conditions
always travel together. Protocol (all lanes): models warm, warmup 3 × repeat
10 timed passes where one pass = every corpus case of that function once,
median reported, best of 3 interleaved rounds. State semantics: native resets
its `Session` per pass and Python restores fresh counter state per pass; the
wasm lanes keep one persistent warm engine — browser semantics, the engine is
built once per page and reused for every query.

The table cites `target/bench/20261016T134149Z/` (measured 2026-10-16 at git
`e458358`, rustc 1.99.1 `--release`, tltk 1.10 on CPython 3.9.25, wasm under
Node — full environment at the bottom).

| function | tltk 1.10 (Python) | native Rust | wasm simd128 | wasm scalar | py→native | simd→native |
|---|---:|---:|---:|---:|---:|---:|
| segment | 4,994,760 | 45,904 | 62,279 | 64,607 | 108.8× | 1.4× |
| syl_segment | 3,521,125 | 66,145 | 104,578 | 111,623 | 53.2× | 1.6× |
| word_segment | 3,481,343 | 67,391 | 105,811 | 112,658 | 51.7× | 1.6× |
| ner_tag | 2,939,044 | 84,358 | 129,656 | 132,390 | 34.8× | 1.5× |
| g2p | 1,598,522 | 33,785 | 52,485 | 55,626 | 47.3× | 1.6× |
| th2roman | 1,355,584 | 26,662 | 41,164 | 43,853 | 50.8× | 1.5× |
| th2ipa | 1,350,445 | 26,169 | 40,729 | 43,797 | 51.6× | 1.6× |
| pos_tag | 1,121,584 | 24,467 | 39,050 | 40,739 | 45.8× | 1.6× |
| th2read | 1,082,867 | 21,523 | 32,381 | 33,800 | 50.3× | 1.5× |
| spell_variants | 525,729 | 6,358 | 9,693 | 10,379 | 82.7× | 1.5× |
| spoonerism | 418,059 | 5,429 | 8,265 | 8,726 | 77.0× | 1.5× |
| plotwords | 155,894 | 837 | 1,349 | 1,561 | 186.3× | 1.6× |
| similar_words | 74,233 | 2,958 | 4,109 | 11,459 | 25.1× | 1.4× |
| analogy | 42,340 | 583 | 949 | 2,505 | 72.6× | 1.6× |
| similarity | 2,026 | 1 | 8 | 8 | see note | — |
| outofgroup | 1,598 | 2 | 9 | 11 | see note | — |
| **whole corpus pass** | **22,665,153** | **412,572** | **632,515** | **673,742** | **54.9×** | **1.5×** |

All four columns are µs/pass; "py→native" is the python/native speedup and
"simd→native" is what the browser's wasm runtime adds over native (simd128
flavor — the honest end-to-end number for a wasm-capable browser query).

- The "whole corpus pass" row is the sum of the per-function medians (the same
  statistic `verifier bench` reports as `total`).
- The wasm runtime cost is a flat **~1.5×** over native across the suite — the
  scalar fallback costs about the same except on the w2v functions, where the
  simd128 build is what pays (similar_words 4.1 ms vs 11.5 ms scalar).
- `similarity`/`outofgroup` are near-instant in every lane; their rows measure
  tltk's OOV idiom (a ~90k-word `list(index_to_key)` scan per call) against a
  hash lookup, so the "speedup" is a µs-scale curiosity, not a headline.
- `plotwords`: all lanes render a real chart, but the python renderer differs
  by design (disclosed divergence #2 — matplotlib vs the Rust rasterizer).
- Memory, warm models: peak RSS **~253 MiB** native Rust (`verifier mem`) vs
  **~730–735 MiB** python vs **~285 MiB** node (wasm engine incl. node process
  overhead, not browser-isolated) — recorded in the artifact's `mem.json` /
  per-round JSON.
- The optimization history of tltk-core itself — regex hoisting, CHD perfect
  hashing, allocation-free chart edges, AVX2/simd128 w2v (~2.6× on the whole
  corpus) — lives in the `perf*` commit series (`git log --grep '^perf'`).
- In the browser: simd128 is runtime-probed (scalar fallback), the service
  worker precaches tier 1 → tier 2 assets and **pre-decodes the engine in the
  background after first paint**, so the first query is instant on a warm
  visit, and the toolbox works fully offline after the first load.

<details>
<summary>Recorded environment of the cited run (also embedded in every artifact)</summary>

- box: `DESKTOP-KIBRVR7` — Intel(R) Core(TM) i5-14400 (16 cores available),
  28,064,072 KiB RAM, Linux kernel `5.15.167.4-microsoft-standard-WSL2`
- git: `e458358` + 11 dirty files (the bench tooling itself, measured from a
  pinned worktree so in-flight core changes could not contaminate the run)
- rust (native lane): `rustc 1.99.1 (58cf8f9c3 2026-10-13)`, `release` profile
- wasm lanes: `wasm32-unknown-unknown` via the `tltk-web` glue under Node
  `v22.21.1` (V8 `12.4.254.21-node.33`); simd128 blob 1,684,169 bytes,
  sha256 `da781394…c9d5b7`; scalar blob 1,687,395 bytes,
  sha256 `fbe607f9…62eb5c30` (full hashes in the artifacts)
- python: CPython `3.9.25` (era-correct oracle venv), tltk `1.10`, deps:
  numpy 1.23.5, pandas 1.5.3, scikit-learn 1.1.1, scipy 1.12.0, gensim 4.3.2,
  nltk 3.9.2, sklearn-crfsuite 0.5.0, matplotlib 3.9.4, requests 2.32.5
- load average at round start (py/rs/wasm): `[3.14, 1.19, 1.25]` /
  `[2.13, 1.73, 1.45]` / `[1.96, 1.71, 1.45]`

</details>

## Run on a new machine

Toolchains are pinned by [mise](https://mise.jdx.dev) (`rust 1.99.1 / python 3.12 / node 24 LTS / uv`),
and every entry point is a mise task — `mise.toml` is the single source of truth
for the environment and for producing/running every artifact (`mise tasks` to list):

```bash
mise install             # installs the pinned toolchain
mise run fetch-models    # tltk==1.11 from PyPI + TNCc5 w2v -> edu/w2v blobs (1st run only)
mise run build-web       # builds web/wasm/{scalar,simd}
mise run serve           # serves the toolbox at http://localhost:8080
```

Gates: `mise run gate` (clippy + tests), `mise run parity` (the three golden
sets), `mise run node-parity` (shipped browser wasm, both flavors),
`mise run wasm-parity` (wasm32-WASI under wasmtime; auto-downloads on first run).

## CI/CD (GitHub Actions)

- **ci.yml** — every push/PR: clippy zero, tests, parity ×3, both wasm arms,
  node parity both flavors, the wasm32-WASI lane.
- **publish-crates.yml** — tag `v*`: publishes `tltk-core` to crates.io
  (needs the `CARGO_REGISTRY_TOKEN` secret; the other crates are app-side and
  stay repo-only).
- **deploy-cloudflare.yml** — push to `main` touching code/static: rebuilds
  the web artifacts, runs the parity gate on the shipped wasm, then deploys
  the static toolbox via wrangler (needs `CLOUDFLARE_API_TOKEN` +
  `CLOUDFLARE_ACCOUNT_ID` secrets). Static-asset serving on Cloudflare is
  free and unmetered.

## Staying current with upstream

The port targets **tltk 1.11** (repinned 2026-10-17; the stichula validator
goldens were captured against its 1.10-era service and stay green because
1.11 touches nothing in that surface). When upstream moves, `mise run
upstream-check` fetches the latest wheel from PyPI, diffs `nlp.py`/
`corpus.py` against the pinned version (file level + function level), and
prints the re-port checklist (bump pins, re-convert models, regenerate oracle
goldens via `tools/convert/ref_dump.py` under the era venv, run all parity
lanes). Every parity diff after a bump is a real signal, not noise.

## Releasing

`mise run release -- 0.1.0` bumps the workspace version, tags `v0.1.0`, pushes
(CI publishes `tltk-core` to crates.io), watches the publish workflow, then
cuts a GitHub Release with a grouped conventional-commit changelog.
