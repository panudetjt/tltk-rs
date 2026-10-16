# AGENTS.md — tltk-rs

Rust rewrite of TLTK (Thai NLP toolkit) — **41 wire functions**: the 16
validator functions **byte-identical to tltk 1.10**, `textana` at **tltk 1.11
semantics** (1.11's only functional change is inside TextAna: LdvTTR ×100),
and 24 more of upstream's inference surface ported per `PORTING-GAPS.md`
(pipelines, tree helpers, translit variants, dict/spell/text utils, w2v
extras, RF text classifier — byte-identical to the Python oracle except the
disclosed-diff plot renderers). The verifier corpus decides parity, not
opinions.

## Environment: mise.toml is the single source of truth

Toolchain pins (rust 1.99.1 / python 3.12 / node 24 LTS / uv) and every
entry point live there — run `mise tasks` and use the tasks; do not
hand-type the commands they wrap.

## Parity contract — read before touching tltk-core

- Golden sets: `corpus/corpus.jsonl` × `golden/stichula.jsonl` (baseline
  **241/3/0**) and `golden/local_python.jsonl` (**228/7/9**) — the 3+7 are
  disclosed upstream diffs (1 ner_tag CRF float flip, 2 plotwords renderer,
  + 6 plot-renderer rows in `corpus_fn_plots` — separate file, fail-by-design);
  and `corpus/corpus_extra.jsonl` × `golden/local_python_extra.jsonl`
  (**100/0/0**, must stay zero — these are python-oracle goldens; the 13
  `textana` rows are captured from the tltk **1.11** oracle, the rest from
  1.10 — legitimate because 1.11 changes nothing outside TextAna).
- Goldens are captured from the Python oracle or the stichula validator —
  **never from the Rust binary** (circular evidence). Porting a new function:
  **probe the oracle first** — Python hides wire traps (dict insertion-order
  keys, `statistics.mean` int/float flips, repr floats, conditional key
  presence); `textana.rs`'s module doc is the worked example, and
  PORTING-GAPS.md §4 carries the checklist.
- Algorithm changes are allowed only when outputs stay byte-identical;
  verify with `mise run parity`, `mise run node-parity`, `mise run wasm-parity`.
- Perf work: `mise run bench-report` (4 lanes — python oracle, native rust,
  shipped wasm simd128 + scalar under Node — plus memory; artifacts with
  recorded environments under `target/bench/<stamp>/`); sides: `mise run bench`
  / `mise run bench-python` / `mise run bench-wasm` / `mise run mem`. Every
  bench artifact carries its system environment block (wasm lanes add the
  sha256 of the measured tltk_web.wasm) — keep it that way. A number without a
  same-box interleaved A/B is a measurement of the box, not the change.

## Gotchas

- `crates/tltk-core/data/edu.bin.gz` + `w2v.bin.gz` are **gitignored** —
  fresh checkouts must `mise run fetch-models` or parity silently shows a
  wrong-looking ~215/29 shape (w2v rows failing, models incomplete).
  `wordtier.bin.gz` (textana's lexicon, from `word.pklz`) is committed —
  the same pickle in 1.10 and 1.11, regenerate with `tools/convert/convert.py`.
- wasm ships in two flavors (simd128 probed at runtime in the browser);
  `wasm-parity` runs the wasip1 lane and auto-downloads wasmtime.
- Edition 2024: SIMD code in `w2v.rs` keeps inner `unsafe {}` blocks
  (`unsafe_op_in_unsafe_fn` is a hard error now).
- `mise run build-web` changes engine bytes — bump `DEFAULT_VERSION` in
  `web/js/loader.js` (content hash) in the same change, or returning browsers
  serve the stale engine from the SW cache.
- This checkout often carries in-flight work from parallel sessions: check
  `git status` + `git diff` before editing shared docs (AGENTS.md/README —
  your context may be stale) and keep commits scoped to one concern; when a
  shared file mixes both, stage hunk-wise (`git apply --cached`).

## Upstream

Port surface is **1.10 + textana@1.11** — the 16 validator functions track
the version app.stichula.org runs; `textana` carries 1.11 semantics (its only
functional change: LdvTTR ×100 — and upstream's "single-paragraph now works"
release note was never implemented in code: crash cases stay `ok=false`
goldens); everything else ported tracks 1.10 oracle behaviour byte-for-byte.
`mise run upstream-check` prints PyPI release notes + a file/def
diff of latest vs the ported pin + the re-port checklist. As of 2026-10:
nothing functional in 1.11 remains unported; the only upstream names still
absent are data/JVM-blocked (TNC.3g n-grams + mm/nbest segmenters,
MaltParser, similar_edu — PORTING-GAPS.md §1a/§1c/§2a/§2b carries the
evidence).
Upstream truth for behaviour questions and any port: the Python oracle at
`~/.cache/tltk-rs/tltk-1.10-pkg/tltk/` (`nlp.py` + `corpus.py` are the
entire upstream API; `mise run oracle-setup` creates the era-correct env).
Upstream-only names (the Ana family, all of corpus.py) grep in the oracle,
not the repo — the repo carries only the validator's snake_case wire names.
A weekly `upstream-watch` workflow runs the same check
on GitHub and opens one issue per new upstream version — Dependabot can't
watch it: the pin lives in shell scripts, not a manifest.

## Shipping

- crates.io: pushing a `v*` tag publishes `tltk-core` (needs the
  `CARGO_REGISTRY_TOKEN` secret; other crates are app-side, repo-only).
- Toolbox: push to main → `deploy-cloudflare.yml` (needs CF secrets);
  deploy config is `web/wrangler.jsonc` (`$schema`-validated).
- `LICENSE` (MIT) + `THIRD_PARTY_LICENSES.md` (TLTK BSD-3 notice) must
  travel with distributions.
