#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PORT="${PORT:-8080}"

# 1. Check required model files (gitignored, but must exist on disk)
TIER1_MODELS=(
    "crates/tltk-core/data/thdict.bin.gz"
    "crates/tltk-core/data/sylseg.bin.gz"
    "crates/tltk-core/data/sylvar.bin.gz"
    "crates/tltk-core/data/phs.bin.gz"
    "crates/tltk-core/data/pos.bin.gz"
    "crates/tltk-core/data/ner.bin.gz"
    "crates/tltk-core/data/sylrule.txt"
    "crates/tltk-core/data/thaisyl.txt"
)
TIER2_MODELS=(
    "crates/tltk-core/data/edu.bin.gz"
    "crates/tltk-core/data/w2v.bin.gz"
)

for model in "${TIER1_MODELS[@]}" "${TIER2_MODELS[@]}"; do
    if [[ ! -f "$model" ]]; then
        echo "ERROR: Missing required model file: $model" >&2
        exit 1
    fi
done

# 2. Build web glue if missing
if [[ ! -f "static/wasm/scalar/tltk_web.js" || ! -f "static/wasm/simd/tltk_web.js" ]]; then
    if [[ -f "scripts/build_web.sh" ]]; then
        echo "Building WebAssembly glue via scripts/build_web.sh..."
        bash scripts/build_web.sh
    else
        echo "NOTE: scripts/build_web.sh not found or glue worker in flight; continuing with available static assets"
    fi
fi

# 3. Prepare data symlink / mapping under static/data
if [[ ! -e "$REPO_ROOT/static/data" ]]; then
    ln -sfn ../crates/tltk-core/data "$REPO_ROOT/static/data"
fi

# 4. Print URL and start static server
echo "Serving TLTK toolbox at http://localhost:${PORT}/"
echo "Press Ctrl+C to stop"

exec python3 -m http.server "$PORT" --directory "$REPO_ROOT/static"
