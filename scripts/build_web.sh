#!/usr/bin/env bash
set -euo pipefail

# scripts/build_web.sh — Dual-flavor WebAssembly builder for tltk-web
#
# Emits both scalar and simd128 wasm artifacts into web/wasm/
#
# Target requirements:
#   simd128: WebAssembly SIMD support (Node.js 16+, Chrome 91+, Firefox 89+, Safari 16.4+)
#   scalar:  Standard WebAssembly MVP fallback for any browser/runtime

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

print_help() {
    cat <<'EOF'
Usage: scripts/build_web.sh [options]

Builds crates/tltk-web for wasm32-unknown-unknown twice:
  1. Default scalar flavor -> web/wasm/scalar/tltk_web.{js,wasm}
  2. simd128 flavor         -> web/wasm/simd/tltk_web.{js,wasm}

Options:
  -h, --help    Show this help message and exit

Artifacts produced:
  web/wasm/scalar/tltk_web.js
  web/wasm/scalar/tltk_web.wasm
  web/wasm/simd/tltk_web.js
  web/wasm/simd/tltk_web.wasm

Runtime requirements:
  - scalar: Works in any environment with basic WebAssembly support.
  - simd128: Requires WebAssembly SIMD (v128) support:
      * Node.js >= 16 (Node 22 supported natively)
      * Chrome >= 91, Firefox >= 89, Safari >= 16.4
      * Can be probed at runtime via WebAssembly.validate() on a tiny SIMD module.

JS Calling Convention:
  import init, { Tltk } from './web/wasm/<flavor>/tltk_web.js';
  await init();

  // Constructor (tier-1 models):
  // 1. Array-parallel form:
  //    const tltk = new Tltk(names, datas);
  //    where names: string[], datas: Uint8Array[]
  // 2. Object wrapper form:
  //    const tltk = new Tltk({ names, datas });
  // 3. Dictionary form:
  //    const tltk = new Tltk({ 'thdict.bin.gz': uint8Array, ... });

  // Attach (tier-2 models edu.bin.gz / w2v.bin.gz):
  // 1. Array-parallel:
  //    tltk.attach(names, datas);
  // 2. Object wrapper:
  //    tltk.attach({ names, datas });
  // 3. Dictionary:
  //    tltk.attach({ 'edu.bin.gz': uint8Array, ... });

  // Process function:
  //    const result = tltk.process(functionName, text, cutoff);
  //    where functionName is one of 16 supported functions,
  //    text is the input Thai text, and cutoff is 'low' | 'medium' | 'high'.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    print_help
    exit 0
fi

# 1. Locate wasm-bindgen CLI
WASM_BINDGEN=""
if [[ -x "$HOME/.cargo/bin/wasm-bindgen" ]]; then
    WASM_BINDGEN="$HOME/.cargo/bin/wasm-bindgen"
elif command -v wasm-bindgen >/dev/null 2>&1; then
    WASM_BINDGEN="$(command -v wasm-bindgen)"
else
    echo "ERROR: wasm-bindgen CLI not found at ~/.cargo/bin/wasm-bindgen or on PATH." >&2
    exit 1
fi
echo "Using wasm-bindgen: $WASM_BINDGEN ($("$WASM_BINDGEN" --version))"

# 2. Ensure output directories exist
mkdir -p "$REPO_ROOT/web/wasm/scalar"
mkdir -p "$REPO_ROOT/web/wasm/simd"

# 3. Build scalar flavor
echo "==> Building scalar flavor (wasm32-unknown-unknown)..."
cargo build --release -p tltk-web --target wasm32-unknown-unknown
"$WASM_BINDGEN" --target web --out-dir "$REPO_ROOT/web/wasm/scalar" "$REPO_ROOT/target/wasm32-unknown-unknown/release/tltk_web.wasm"
cp "$REPO_ROOT/web/wasm/scalar/tltk_web_bg.wasm" "$REPO_ROOT/web/wasm/scalar/tltk_web.wasm"

# 4. Build simd128 flavor
echo "==> Building simd128 flavor (wasm32-unknown-unknown +simd128)..."
RUSTFLAGS="-C target-feature=+simd128" cargo build --release -p tltk-web --target wasm32-unknown-unknown
"$WASM_BINDGEN" --target web --out-dir "$REPO_ROOT/web/wasm/simd" "$REPO_ROOT/target/wasm32-unknown-unknown/release/tltk_web.wasm"
cp "$REPO_ROOT/web/wasm/simd/tltk_web_bg.wasm" "$REPO_ROOT/web/wasm/simd/tltk_web.wasm"

# 5. Verify all four artifacts exist and are non-empty
ARTIFACTS=(
    "web/wasm/scalar/tltk_web.js"
    "web/wasm/scalar/tltk_web.wasm"
    "web/wasm/simd/tltk_web.js"
    "web/wasm/simd/tltk_web.wasm"
)

for art in "${ARTIFACTS[@]}"; do
    if [[ ! -s "$REPO_ROOT/$art" ]]; then
        echo "ERROR: Artifact missing or empty: $art" >&2
        exit 1
    fi
    echo "  OK: $art ($(du -h "$REPO_ROOT/$art" | cut -f1))"
done

echo "Build complete. All web wasm artifacts ready."
