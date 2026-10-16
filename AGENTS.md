# AGENTS.md — tltk-rs

Rust rewrite of TLTK 1.10 (Thai NLP toolkit) — 16 functions, **byte-identical
to tltk 1.10**. The verifier corpus decides parity, not opinions.

## Environment: mise.toml is the single source of truth

Toolchain pins (rust 1.99.1 / python 3.12 / node 24 LTS / uv) and every
entry point live there — run `mise tasks` and use the tasks; do not
hand-type the commands they wrap.

## Parity contract — read before touching tltk-core

- Golden sets: `corpus/corpus.jsonl` × `golden/stichula.jsonl` (baseline
  **241/3/0**) and `golden/local_python.jsonl` (**228/7/9**) — the 3+7 are
  disclosed upstream diffs (1 ner_tag CRF float flip, 2 plotwords renderer);
  and `corpus/corpus_extra.jsonl` × `golden/local_python_extra.jsonl`
  (**87/0/0**, must stay zero — these are python-oracle goldens).
- Goldens are captured from the Python oracle or the stichula validator —
  **never from the Rust binary** (circular evidence).
- Algorithm changes are allowed only when outputs stay byte-identical;
  verify with `mise run parity`, `mise run node-parity`, `mise run wasm-parity`.
- Perf work: `mise run bench` / `mise run mem`. A number without a
  same-box interleaved A/B is a measurement of the box, not the change.

## Gotchas

- `crates/tltk-core/data/edu.bin.gz` + `w2v.bin.gz` are **gitignored** —
  fresh checkouts must `mise run fetch-models` or parity silently shows a
  wrong-looking ~215/29 shape (w2v rows failing, models incomplete).
- wasm ships in two flavors (simd128 probed at runtime in the browser);
  `wasm-parity` runs the wasip1 lane and auto-downloads wasmtime.
- Edition 2024: SIMD code in `w2v.rs` keeps inner `unsafe {}` blocks
  (`unsafe_op_in_unsafe_fn` is a hard error now).

## Upstream

Port target is **1.10** — the version app.stichula.org runs (not PyPI
latest). `mise run upstream-check` prints PyPI release notes + a file/def
diff of latest vs 1.10 + the re-port checklist. As of 2026-10: 1.11 exists,
TextAna-module fixes only — outside our 16-function port surface. A weekly
`upstream-watch` workflow runs the same check on GitHub (Mondays 03:17 UTC,
plus manual dispatch) and opens one issue per new upstream version —
Dependabot can't watch it: the pin lives in shell scripts, not a manifest.

## Shipping

- crates.io: pushing a `v*` tag publishes `tltk-core` (needs the
  `CARGO_REGISTRY_TOKEN` secret; other crates are app-side, repo-only).
- Toolbox: push to main → `deploy-cloudflare.yml` (needs CF secrets);
  deploy config is `web/wrangler.jsonc` (`$schema`-validated).
- `LICENSE` (MIT) + `THIRD_PARTY_LICENSES.md` (TLTK BSD-3 notice) must
  travel with distributions.
