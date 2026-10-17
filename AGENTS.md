# AGENTS.md — tltk-rs

Rust rewrite of TLTK (Thai NLP toolkit) — **41 wire functions**, port surface
**tltk 1.11** (repinned 2026-10-17; 1.11's only functional change vs 1.10 is
TextAna's LdvTTR ×1000 → ×100): the 16 validator functions **byte-identical
to the stichula validator / Python oracle**, and 25 more of upstream's
inference surface ported per `PORTING-GAPS.md` (pipelines, tree helpers,
translit variants, dict/spell/text utils, w2v extras, RF text classifier —
byte-identical to the Python oracle except the disclosed-diff plot
renderers). The verifier corpus decides parity, not opinions.

## Environment: mise.toml is the single source of truth

Toolchain pins (rust 1.99.1 / python 3.12 / node 24 LTS / uv) and every
entry point live there — run `mise tasks` and use the tasks; do not
hand-type the commands they wrap.

## Parity contract — read before touching tltk-core

- Golden sets: `corpus/corpus.jsonl` × `golden/stichula.jsonl` (baseline
  **241/3/0**) and `golden/local_python.jsonl` (**240/1/3**) — the 3+1 are
  disclosed upstream diffs (1 ner_tag CRF float flip — the same row in both
  lanes, 2 plotwords renderer, + 6 plot-renderer rows in `corpus_fn_plots` —
  separate file, fail-by-design); and `corpus/corpus_extra.jsonl` ×
  `golden/local_python_extra.jsonl` (**100/0/0**, must stay zero — these are
  python-oracle goldens). All oracle goldens are captured from the tltk
  **1.11** oracle under the era venv (2026-10-17 repin; the repin's re-capture
  also replaced the local set's segment/similarity rows that the original
  capture had recorded as sklearn unpickle errors — the Rust's real
  segmentations match the fixed goldens, which moved the local baseline
  228/7/9 → 240/1/3).
- Goldens are captured from the Python oracle or the stichula validator —
  **never from the Rust binary** (circular evidence). Porting a new function:
  **probe the oracle first** — Python hides wire traps (dict insertion-order
  keys, `statistics.mean` int/float flips, repr floats, conditional key
  presence); `textana.rs`'s module doc is the worked example, and
  PORTING-GAPS.md §4 carries the checklist. The wire↔oracle call mapping for
  parity/bench work lives in `tools/convert/ref_dump.py` (corpus mode) and
  `scripts/bench_python.py` (full 16).
- Algorithm changes are allowed only when outputs stay byte-identical;
  verify with `mise run parity`, `mise run node-parity`, `mise run wasm-parity`
  — and `mise run web-e2e` for anything touching `web/js/` (the browser parses
  those files as ES modules; a CJS `node --check` passing proves nothing).
- Perf work: `mise run bench-report` (4 lanes — python oracle, native rust,
  shipped wasm simd128 + scalar under Node — plus memory; artifacts with
  recorded environments under `target/bench/<stamp>/`); sides: `mise run bench`
  / `mise run bench-python` / `mise run bench-wasm` / `mise run mem`.
- Every bench artifact carries its system environment block (wasm lanes add
  the sha256 of the measured tltk_web.wasm) — keep it that way.
- A number without a same-box interleaved A/B is a measurement of the box,
  not the change.
- Cite only numbers measured at a **pinned revision** — canonical runs happen
  in a worktree at a known commit, and the artifact's `git` field is the
  receipt.

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
- tltk 1.11's METADATA pins `scikit-learn~=1.2.0`, which cannot unpickle its
  own `sent_segment.pklz` (needs <1.2) — irrelevant here because the era venv
  installs deps directly and PYTHONPATHs the tree; never `pip install tltk`.
- This checkout often carries in-flight work from parallel sessions: check
  `git status` + `git diff` before editing shared docs (AGENTS.md/README —
  your context may be stale) and keep commits scoped to one concern; when a
  shared file mixes both, stage hunk-wise (`git apply --cached`).
- Land worker branches with `mise run land -- worker/<slug>` — it merges
  without committing, runs gate + parity on the merged tree, and only a green
  tree reaches the commit. Keep-both conflict seams (2026-10-16/17: a missing
  comma ×2 and a stray `}` that blanked the production dropdown) all came
  from plain merges committed before any check ran.
- Filesystem executability probes lie on some dev boxes (ruff's EXE001 never
  fires under WSL2) — the gate's `scripts/check_exec_bits.py` audits the git
  index instead: a shebang means 100755 in the index. Verify CI with
  `mise run ci-green` — its exit status comes from the run's API conclusion,
  never from piped `gh run watch` output.

## Upstream

Port surface is **tltk 1.11** (repinned 2026-10-17; before that
1.10 + textana@1.11) — 1.11's only functional change vs 1.10 is LdvTTR ×100
inside TextAna, and the TextClass/txt2feat feature path reads that same
field through TextAna, so one ported value covers both. The stichula
validator goldens stay captured from app.stichula.org (still 1.10) and stay
green because 1.11 touches nothing in that surface. Upstream's
"single-paragraph now works" release note was never implemented in code:
crash cases stay `ok=false` goldens.
`mise run upstream-check` prints PyPI release notes + a file/def
diff of latest vs the ported pin + the re-port checklist (step 4's golden
regen = `tools/convert/ref_dump.py` corpus mode under the era venv, one
process in corpus order). As of 2026-10:
nothing functional in 1.11 remains unported; the only upstream names still
absent are data/JVM-blocked (TNC.3g n-grams + mm/nbest segmenters,
MaltParser, similar_edu — PORTING-GAPS.md §1a/§1c/§2a/§2b carries the
evidence).
Upstream truth for behaviour questions and any port: the Python oracle at
`~/.cache/tltk-rs/tltk-1.11-pkg/tltk/` (`nlp.py` + `corpus.py` are the
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
