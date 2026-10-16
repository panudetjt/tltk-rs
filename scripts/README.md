# Segment Oracle Environment Setup

## Background

TLTK 1.11's `nlp.segment` function uses an EDU segmentation model (`sent_segment.pklz`) containing a scikit-learn `RandomForestClassifier` pickled with scikit-learn 1.0.2.

On modern environments (Python 3.12+ with scikit-learn >= 1.2), unpickling this estimator fails because the underlying tree `node` struct dtype changed in scikit-learn 1.2 (`node array from the pickle has an incompatible dtype`).

To unlock the segment oracle without retraining or modifying the upstream model, a period-correct environment is used:
- **Python**: 3.9.x (managed via `mise`)
- **scikit-learn**: 1.1.1
- **numpy**: < 1.24 (e.g. 1.23.5)
- **scipy**: < 1.13 (e.g. 1.12.0, required for gensim 4.3.2)
- **pandas**: 1.5.3
- **gensim**: 4.3.2
- **nltk**, **sklearn-crfsuite**, **requests**
Default venv path: `~/.local/share/tltk-rs/venv-segment-oracle` (persistent, survives /tmp wipes)

## Usage

### Check existing environment (dry run)
```bash
bash scripts/setup_segment_oracle.sh --check
```
This checks Python 3.9, the package versions, and runs a test probe on `segment('แมวกินปลาทะเล')` without modifying the environment.

### Reproduce environment from scratch
```bash
# 1. Ensure Python 3.9 is installed (via mise)
mise install python@3.9

# 2. Run the setup script to create $HOME/.local/share/tltk-rs/venv-segment-oracle with pinned packages
bash scripts/setup_segment_oracle.sh
```

To force recreate an existing environment:
```bash
bash scripts/setup_segment_oracle.sh --force
```

### Running with the oracle
```bash
PYTHONPATH=~/.cache/tltk-rs/tltk-1.11-pkg ~/.local/share/tltk-rs/venv-segment-oracle/bin/python your_script.py
```

## Benchmark the Python oracle

`mise run bench-python` runs `scripts/bench_python.py` in this venv over
`corpus/corpus.jsonl` with the same protocol as `verifier bench`: per function
group, warmup 3 then repeat 10 timed passes (one pass = every corpus case of
that function once), reporting the median µs per pass. The mutable colloc
counters (`BiCount`/`Count`/`TotalWord`/... in `nlp` and `th2ipa`) are
snapshotted and restored between passes — the fresh-process state the Rust
bench reproduces with a new `Session`. Models (pos/ner/edu/w2v) load once
before timing.

### Reproducing the published numbers end-to-end

`mise run bench-report` (env `ROUNDS=n` to change round count, default 3)
runs ROUNDS interleaved rounds of all four lanes — python oracle, native rust,
shipped browser wasm (simd128 + scalar flavors) under Node — plus a rust
memory profile, and writes everything to `target/bench/<UTC-stamp>/`. Before
the rounds it preflights every engine (native `selftest`; `--smoke-only` on
both `node_bench.mjs` flavors and on `bench_python.py` — load all models and
touch every corpus case once), so a broken model blob fails in seconds with
its `preflight_*.log` instead of mid-run:

- `bench_py_<n>.json` / `bench_rs_<n>.json` — raw per-round results, each
  embedding a recorded `environment` block (timestamp, host, kernel, CPU
  model, available cores, total RAM, load average, git revision + dirty-file
  count; plus the full python dep versions / the rustc version and build
  profile)
- `bench_wasm_simd_<n>.json` / `bench_wasm_scalar_<n>.json` — the wasm lanes
  (`scripts/node_bench.mjs`, also standalone as `mise run bench-wasm`): they
  drive the real `tltk_web` glue the browser toolbox uses (wasm-bindgen
  boundary included) with a persistent warm engine — browser semantics, the
  engine is built once per page and reused for every query. Their environment
  block adds the node/v8 versions and the sha256 + byte size of the exact
  `tltk_web.wasm` measured
- `mem.json` — `verifier mem --json` (phase RSS + peak VmHWM + environment)
- `summary.md` — best-of-rounds table across lanes (py→native speedup,
  simd→native wasm overhead), environment sections, memory section (generated
  by `scripts/bench_summary.py`, which can also be run standalone on any
  artifacts directory)

The Performance table in the top-level README cites one of these
`summary.md` reports verbatim.

Notes:

- `setup_segment_oracle.sh` installs `matplotlib` too (needed by the
  `plotwords` render path); existing venvs created before that need
  `uv pip install --python <venv>/bin/python matplotlib` or a `--force` setup.
- `TNCc5model3.bin` is not in the wheel; the bench seeds it into the package
  dir from `$TLTK_W2V_BIN` or `~/.cache/tltk-bench/TNCc5model3.bin`, and
  downloads it from the official URL (`corpus.py download_TNCw2v`) on first use.
- Per-case exceptions are results, not bench failures — tltk 1.11 has
  disclosed crash cases (e.g. `spoonerism('สวัสดี')` raises `KeyError: 'dat'`;
  the validator 500s on it and the goldens record it with `ok: false`).
