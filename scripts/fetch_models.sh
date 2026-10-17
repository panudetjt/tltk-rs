#!/usr/bin/env bash
# Fetch the two gitignored model blobs (edu.bin.gz, w2v.bin.gz) by downloading
# the tltk==1.11 wheel from PyPI and converting it with tools/convert/convert.py
# (deps are PEP 723 inline — uv resolves and caches them; no manual venv).
# Re-runnable on any machine with mise (rust/python/node/uv) installed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="crates/tltk-core/data"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

need() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: $1 not on PATH (run via mise: https://mise.jdx.dev)" >&2; exit 1; }; }
need uv
need curl
need python3

# 1. Resolve the tltk==1.11 wheel URL from the PyPI JSON API and download it
echo "==> Downloading tltk==1.11 wheel from PyPI"
WHEEL_URL="$(curl -fsSL https://pypi.org/pypi/tltk/1.11/json | python3 -c '
import json, sys
for f in json.load(sys.stdin)["urls"]:
    if f["filename"].endswith("py3-none-any.whl"):
        print(f["url"]); break
')"
[ -n "$WHEEL_URL" ] || { echo "ERROR: no universal wheel found for tltk==1.11" >&2; exit 1; }
curl -fsSL "$WHEEL_URL" -o "$WORK_DIR/tltk-1.11.whl"

# 2. Extract: edu source lives in the wheel; the w2v model does NOT ship in
#    the wheel — the tltk package itself downloads TNCc5model3.bin from
#    Chula's server on first use (tltk/corpus.py W2V_load). Same official URL.
echo "==> Extracting wheel"
python3 -m zipfile -e "$WORK_DIR/tltk-1.11.whl" "$WORK_DIR/pkg"
TLTK_DIR="$WORK_DIR/pkg/tltk"

W2V_SRC="$TLTK_DIR/TNCc5model3.bin"
W2V_ARG=()
if [ ! -f "$W2V_SRC" ]; then
    echo "==> TNCc5model3.bin not in wheel; downloading from arts.chula.ac.th (official URL from tltk corpus.py)"
    W2V_SRC="$WORK_DIR/TNCc5model3.bin"
    curl -fsSL --max-time 600 "https://www.arts.chula.ac.th/~ling/contents/Upload/TNCc5model3.bin" -o "$W2V_SRC" \
        || { echo "ERROR: TNCc5model3.bin download failed (Chula server unreachable?)" >&2; exit 1; }
    W2V_ARG=(--w2v "$W2V_SRC")
fi

# 3. Convert into a temp dir (uv caches the PEP 723 env after the first run),
#    copy ONLY the two gitignored blobs — the six committed blobs stay untouched
echo "==> Converting (first run resolves deps; cached afterwards)"
uv run --quiet tools/convert/convert.py "$TLTK_DIR" "$WORK_DIR/out" "${W2V_ARG[@]}"

# MaltParser engine blob: NOT in the wheel — regenerate deterministically
# from the wheel's thamalt.mco via the official maltparser-1.9.2 dist
# (BSD-3, maltparser.org) + a JDK. Skipped gracefully: malt_parser /
# malt_parser_wordlist error with a clear message when the blob is absent.
if [ ! -f "$DATA_DIR/malt.bin.gz" ]; then
  if command -v java >/dev/null 2>&1 && command -v javac >/dev/null 2>&1; then
    MALTPARSER_DIR="${MALTPARSER_PATH:-$HOME/.cache/tltk-rs/maltparser-1.9.2}"
    if [ ! -f "$MALTPARSER_DIR/maltparser-1.9.2.jar" ]; then
      echo "==> Fetching maltparser-1.9.2 dist (BSD-3) into $MALTPARSER_DIR"
      mkdir -p "$(dirname "$MALTPARSER_DIR")"
      curl -fsSL -o "$WORK_DIR/maltparser-1.9.2.zip" https://maltparser.org/dist/maltparser-1.9.2.zip
      unzip -q -o "$WORK_DIR/maltparser-1.9.2.zip" -d "$(dirname "$MALTPARSER_DIR")"
    fi
    echo "==> Converting thamalt.mco -> malt.bin.gz (deterministic; needs JDK)"
    MALTPARSER_PATH="$MALTPARSER_DIR" python3 "$REPO_ROOT/tools/convert/convert.py" malt
  else
    echo "[fetch-models] java/javac not found — skipping malt.bin.gz (malt_parser/malt_parser_wordlist will error until it is generated)"
  fi
fi

for blob in edu.bin.gz w2v.bin.gz; do
    [ -f "$WORK_DIR/out/$blob" ] || { echo "ERROR: converter produced no $blob" >&2; exit 1; }
    cp "$WORK_DIR/out/$blob" "$DATA_DIR/$blob"
    echo "  $DATA_DIR/$blob ($(stat -c%s "$DATA_DIR/$blob") bytes)"
done

echo "OK: model blobs fetched."
