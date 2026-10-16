#!/usr/bin/env bash
# Fetch the two gitignored model blobs (edu.bin.gz, w2v.bin.gz) by downloading
# the tltk==1.10 wheel from PyPI and converting it with tools/convert/convert.py.
# Re-runnable; the conversion venv is cached under ~/.local/share/tltk-rs/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${TLTK_VENV_HOME:-$HOME/.local/share/tltk-rs/venv-convert}"
DATA_DIR="crates/tltk-core/data"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

need() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: $1 not on PATH (run via mise: https://mise.jdx.dev)" >&2; exit 1; }; }
PY="python3"
need "$PY"

# 1. Persistent conversion venv (reused across runs)
if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "==> Creating conversion venv at $VENV_DIR"
    "$PY" -m venv "$VENV_DIR"
fi
echo "==> Installing conversion deps (cached after first run)"
"$VENV_DIR/bin/pip" install -q -r tools/requirements.txt

# 2. Download the tltk 1.10 wheel (no deps — we only want its data files)
echo "==> Downloading tltk==1.10 wheel from PyPI"
"$VENV_DIR/bin/pip" download -q tltk==1.10 --no-deps -d "$WORK_DIR/wheel"
WHEEL="$(ls "$WORK_DIR"/wheel/tltk-1.10*.whl | head -1)"
[ -n "$WHEEL" ] || { echo "ERROR: tltk wheel not found after download" >&2; exit 1; }

# 3. Unzip the wheel to get the package dir (models live inside tltk/)
echo "==> Extracting wheel"
"$PY" -m zipfile -e "$WHEEL" "$WORK_DIR/pkg"
TLTK_DIR="$WORK_DIR/pkg/tltk"

# The w2v model is NOT in the PyPI wheel — the tltk package itself downloads it
# from Chula's server on first use (tltk/corpus.py W2V_load). Fetch from the
# same official URL.
W2V_SRC="$TLTK_DIR/TNCc5model3.bin"
W2V_ARG=()
if [ ! -f "$W2V_SRC" ]; then
    echo "==> TNCc5model3.bin not in wheel; downloading from arts.chula.ac.th (official URL from tltk corpus.py)"
    W2V_SRC="$WORK_DIR/TNCc5model3.bin"
    curl -fsSL --max-time 600 "https://www.arts.chula.ac.th/~ling/contents/Upload/TNCc5model3.bin" -o "$W2V_SRC" \
        || { echo "ERROR: TNCc5model3.bin download failed (Chula server unreachable?)" >&2; exit 1; }
    W2V_ARG=(--w2v "$W2V_SRC")
fi

# 4. Convert into a temp dir, copy ONLY the two gitignored blobs
#    (the six committed blobs are not touched — regenerated gzip would churn)
echo "==> Converting (this needs the heavy deps; first run takes a few minutes)"
"$VENV_DIR/bin/python" tools/convert/convert.py "$TLTK_DIR" "$WORK_DIR/out" "${W2V_ARG[@]}"

for blob in edu.bin.gz w2v.bin.gz; do
    [ -f "$WORK_DIR/out/$blob" ] || { echo "ERROR: converter produced no $blob" >&2; exit 1; }
    cp "$WORK_DIR/out/$blob" "$DATA_DIR/$blob"
    echo "  $DATA_DIR/$blob ($(stat -c%s "$DATA_DIR/$blob") bytes)"
done

echo "OK: model blobs fetched."
