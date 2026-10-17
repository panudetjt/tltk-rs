#!/usr/bin/env bash
# scripts/bench_report.sh — full reproducible comparison across all four lanes.
#
# Lanes: python oracle (mise run bench-python), native rust (mise run bench),
# shipped browser wasm simd128 + scalar flavors under Node (node_bench.mjs),
# plus a native memory profile. ROUNDS interleaved rounds are summarized into
# target/bench/<UTC-stamp>/summary.md. Every artifact embeds its recorded
# environment block (timestamp, host, kernel, CPU, cores, RAM, load average,
# git revision + dirty count, toolchain versions; wasm lanes add the sha256 of
# the exact tltk_web.wasm binary measured), so the numbers and their
# measurement conditions travel together.
#
# Usage: bash scripts/bench_report.sh [ROUNDS]   (default 3; or env ROUNDS=n)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ROUNDS="${1:-${ROUNDS:-3}}"
OUT="target/bench/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"

if [ ! -s web/wasm/simd/tltk_web.wasm ] || [ ! -s web/wasm/scalar/tltk_web.wasm ]; then
    echo "==> building web wasm (both flavors)"
    bash scripts/build_web.sh > "$OUT/build_web.log" 2>&1
fi

# Preflight every engine before spending rounds: a broken model blob or a
# decode panic fails here in seconds instead of mid-run (each check loads all
# models and touches every corpus case once; artifacts keep the logs).
echo "==> preflight: native model load + anchors"
cargo run --release -q -p tltk-verifier -- selftest crates/tltk-core/data \
    > "$OUT/preflight_native.log" 2>&1
echo "==> preflight: wasm engines (both flavors)"
node scripts/node_bench.mjs --wasm-dir web/wasm/simd --corpus corpus/corpus.jsonl \
    --label wasm-simd --smoke-only > "$OUT/preflight_wasm_simd.log" 2>&1
node scripts/node_bench.mjs --wasm-dir web/wasm/scalar --corpus corpus/corpus.jsonl \
    --label wasm-scalar --smoke-only > "$OUT/preflight_wasm_scalar.log" 2>&1
echo "==> preflight: python oracle"
PY_BIN="$HOME/.local/share/tltk-rs/venv-segment-oracle/bin/python"
[ -x "$PY_BIN" ] || bash scripts/setup_segment_oracle.sh > "$OUT/preflight_venv.log" 2>&1
"$PY_BIN" scripts/bench_python.py corpus/corpus.jsonl --warmup 0 --repeat 1 --smoke-only \
    > "$OUT/preflight_py.log" 2>&1

echo "==> bench-report: $ROUNDS interleaved round(s) -> $OUT"
for i in $(seq 1 "$ROUNDS"); do
    echo "==> round $i/$ROUNDS: python oracle (tltk 1.11)"
    mise run bench-python > "$OUT/bench_py_$i.json" 2> "$OUT/bench_py_$i.log"
    echo "==> round $i/$ROUNDS: rust tltk-core (--release)"
    mise run bench > "$OUT/bench_rs_$i.json" 2> "$OUT/bench_rs_$i.log"
    echo "==> round $i/$ROUNDS: wasm simd128 (browser engine)"
    node scripts/node_bench.mjs --wasm-dir web/wasm/simd --corpus corpus/corpus.jsonl \
        --label wasm-simd --json > "$OUT/bench_wasm_simd_$i.json" 2> "$OUT/bench_wasm_simd_$i.log"
    echo "==> round $i/$ROUNDS: wasm scalar (fallback flavor)"
    node scripts/node_bench.mjs --wasm-dir web/wasm/scalar --corpus corpus/corpus.jsonl \
        --label wasm-scalar --json > "$OUT/bench_wasm_scalar_$i.json" 2> "$OUT/bench_wasm_scalar_$i.log"
done

echo "==> rust memory profile"
mise run mem -- --json > "$OUT/mem.json" 2> "$OUT/mem.log"

python3 scripts/bench_summary.py "$OUT"
echo "OK: artifacts in $OUT (summary.md + raw JSON per round)"
