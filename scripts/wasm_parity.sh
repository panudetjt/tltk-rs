#!/usr/bin/env bash
set -euo pipefail

# Parity baseline numbers under wasm32-wasip1:
# Note: These numbers must be updated in lockstep with the native baselines
# if goldens ever change.
# Expected parity baselines:
#   stichula: pass=241 fail=3 missing_golden=0
#   local:    pass=240 fail=1 missing_golden=3
#   extra:    pass=100 fail=0 missing_golden=0 (all oracle goldens captured from the tltk 1.11 oracle)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# 1. Locate wasmtime: PATH first, else auto-download a pinned build into a persistent cache
WASMTIME=""
if command -v wasmtime >/dev/null 2>&1; then
    WASMTIME="$(command -v wasmtime)"
else
    # Auto-download a pinned wasmtime into a persistent cache (survives /tmp wipes)
    WASMTIME_CACHE="${TLTK_WASMTIME_CACHE:-$HOME/.cache/tltk-rs/wasmtime}"
    WASMTIME_VER="v49.0.3"
    case "$(uname -m)-$(uname -s)" in
        x86_64-Linux)  WASMTIME_ARCH="x86_64-linux" ;;
        aarch64-Linux) WASMTIME_ARCH="aarch64-linux" ;;
        arm64-Darwin)  WASMTIME_ARCH="aarch64-macos" ;;
        x86_64-Darwin) WASMTIME_ARCH="x86_64-macos" ;;
        *) echo "ERROR: unsupported platform for wasmtime auto-download: $(uname -m)-$(uname -s)" >&2; exit 1 ;;
    esac
    WASMTIME="$WASMTIME_CACHE/wasmtime"
    if [[ ! -x "$WASMTIME" ]]; then
        echo "==> Downloading wasmtime $WASMTIME_VER ($WASMTIME_ARCH) to $WASMTIME_CACHE"
        mkdir -p "$WASMTIME_CACHE"
        TARBALL="wasmtime-$WASMTIME_VER-$WASMTIME_ARCH.tar.xz"
        curl -fsSL "https://github.com/bytecodealliance/wasmtime/releases/download/$WASMTIME_VER/$TARBALL" \
            -o "$WASMTIME_CACHE/$TARBALL" || { echo "ERROR: wasmtime download failed" >&2; exit 1; }
        tar -xJf "$WASMTIME_CACHE/$TARBALL" -C "$WASMTIME_CACHE" \
            "wasmtime-$WASMTIME_VER-$WASMTIME_ARCH/wasmtime"
        mv "$WASMTIME_CACHE/wasmtime-$WASMTIME_VER-$WASMTIME_ARCH/wasmtime" "$WASMTIME"
        chmod +x "$WASMTIME"
        rm -rf "$WASMTIME_CACHE/$TARBALL" "$WASMTIME_CACHE/wasmtime-$WASMTIME_VER-$WASMTIME_ARCH"
    fi
fi
echo "Using wasmtime at: $WASMTIME"

# 2. Check required model files (gitignored, but must exist on disk)
for model in "crates/tltk-core/data/edu.bin.gz" "crates/tltk-core/data/w2v.bin.gz"; do
    if [[ ! -f "$model" ]]; then
        echo "ERROR: Missing required model file: $model" >&2
        exit 1
    fi
done

# 3. Build verifier for wasm32-wasip1
# Select cargo toolchain with wasm32-wasip1 installed if default lacks it
CARGO_CMD=("cargo")
if ! rustup target list --installed 2>/dev/null | grep -qx "wasm32-wasip1"; then
    for tc in $(rustup toolchain list 2>/dev/null | awk '{print $1}'); do
        if rustup target list --installed --toolchain "$tc" 2>/dev/null | grep -qx "wasm32-wasip1"; then
            CARGO_CMD=("cargo" "+$tc")
            break
        fi
    done
fi

echo "Building verifier for wasm32-wasip1 using: ${CARGO_CMD[*]}..."
"${CARGO_CMD[@]}" build --release -p tltk-verifier --target wasm32-wasip1

WASM_BIN="$REPO_ROOT/target/wasm32-wasip1/release/verifier.wasm"
if [[ ! -f "$WASM_BIN" ]]; then
    echo "ERROR: Compiled WASM binary not found at $WASM_BIN" >&2
    exit 1
fi

# Helper to run a parity check set
run_check() {
    local name="$1"
    local corpus="$2"
    local golden="$3"
    local expected="$4"

    echo "==> Running $name parity check under wasmtime..."
    local raw_output
    # verifier check exits 1 when fail > 0 (expected for sets with known mismatches)
    raw_output="$("$WASMTIME" run --dir .::. "$WASM_BIN" check "$corpus" "$golden" crates/tltk-core/data 2>&1 || true)"

    local summary
    summary="$(echo "$raw_output" | grep -E '^verifier: pass=[0-9]+ fail=[0-9]+ missing_golden=[0-9]+' || true)"

    if [[ -z "$summary" ]]; then
        echo "FAIL: $name did not produce a verifier summary line" >&2
        echo "$raw_output" >&2
        return 1
    fi

    echo "$summary"
    if [[ "$summary" != "$expected" ]]; then
        echo "FAIL: $name parity mismatch!" >&2
        echo "  Expected: $expected" >&2
        echo "  Got     : $summary" >&2
        echo "$raw_output" >&2
        return 1
    fi
    echo "PASS: $name"
}

# 4. Run checks for all three golden sets
run_check "stichula" "corpus/corpus.jsonl" "golden/stichula.jsonl" "verifier: pass=241 fail=3 missing_golden=0"
run_check "local" "corpus/corpus.jsonl" "golden/local_python.jsonl" "verifier: pass=240 fail=1 missing_golden=3"
run_check "extra" "corpus/corpus_extra.jsonl" "golden/local_python_extra.jsonl" "verifier: pass=100 fail=0 missing_golden=0"

# 5. Run selftest
echo "==> Running selftest under wasmtime..."
selftest_output="$("$WASMTIME" run --dir .::. "$WASM_BIN" selftest crates/tltk-core/data 2>&1 || true)"
echo "$selftest_output"
if echo "$selftest_output" | grep -q "FAIL" || ! echo "$selftest_output" | grep -q "ok   pos_tag"; then
    echo "FAIL: selftest encountered errors" >&2
    exit 1
fi
echo "selftest ok"
echo "PASS: selftest"

echo "All wasm parity checks PASSED."
exit 0
