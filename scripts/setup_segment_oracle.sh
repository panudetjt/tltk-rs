#!/usr/bin/env bash
# scripts/setup_segment_oracle.sh
#
# Reproduces or checks the period-correct Python 3.9 environment required for
# the TLTK 1.11 segment oracle (RandomForest pickle incompatibility with
# sklearn >= 1.2). Fully self-contained: uv downloads a managed Python 3.9,
# and the tltk 1.11 reference tree is fetched from PyPI into a local cache —
# no personal paths involved.
#
# Usage:
#   bash scripts/setup_segment_oracle.sh [--check] [--force]

set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/.local/share/tltk-rs/venv-segment-oracle}"
TLTK_REF="${TLTK_REF:-$HOME/.cache/tltk-rs/tltk-1.11-pkg}"

CHECK_ONLY=false
FORCE=false

for arg in "$@"; do
    case "$arg" in
        --check) CHECK_ONLY=true ;;
        --force) FORCE=true ;;
        -h|--help)
            echo "Usage: $0 [--check] [--force]"; exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

need() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: $1 not on PATH (run via mise)" >&2; exit 1; }; }
need uv
need curl
need python3

fetch_tltk_ref() {
    if [ -d "$TLTK_REF/tltk" ]; then return 0; fi
    echo "==> Fetching tltk==1.11 reference tree from PyPI into $TLTK_REF"
    local work; work="$(mktemp -d)"
    local url
    url="$(curl -fsSL https://pypi.org/pypi/tltk/1.11/json | python3 -c '
import json, sys
for f in json.load(sys.stdin)["urls"]:
    if f["filename"].endswith("py3-none-any.whl"):
        print(f["url"]); break
')"
    [ -n "$url" ] || { echo "ERROR: no universal wheel on PyPI" >&2; exit 1; }
    curl -fsSL "$url" -o "$work/tltk.whl"
    mkdir -p "$TLTK_REF"
    python3 -m zipfile -e "$work/tltk.whl" "$TLTK_REF"
    rm -rf "$work"
}

verify_env() {
    local py_bin="$VENV_DIR/bin/python"
    echo "Checking segment oracle environment at $VENV_DIR..."
    [ -x "$py_bin" ] || { echo "FAIL: $py_bin missing (run without --check first)" >&2; return 1; }

    "$py_bin" - <<'EOF'
import importlib.metadata
import numpy, sklearn, pandas, gensim, nltk, requests, matplotlib
assert tuple(int(x) for x in numpy.__version__.split(".")[:2]) < (1, 24)
assert sklearn.__version__ == "1.1.1"
assert pandas.__version__.startswith("1.5.")
assert gensim.__version__ == "4.3.2"
print(f"  [OK] py{__import__('sys').version.split()[0]} numpy {numpy.__version__} "
      f"sklearn {sklearn.__version__} pandas {pandas.__version__} gensim {gensim.__version__} "
      f"crfsuite {importlib.metadata.version('sklearn-crfsuite')}")
EOF

    fetch_tltk_ref
    echo "  Testing segment oracle probe..."
    PYTHONPATH="$TLTK_REF" "$py_bin" - <<'EOF'
import warnings; warnings.filterwarnings("ignore")
from tltk.nlp import segment
res = segment("แมวกินปลาทะเล")
assert res == "แมว|กิน|ปลาทะเล|<s/><u/>", f"got {res!r}"
print(f"  [OK] segment('แมวกินปลาทะเล') -> {res}")
EOF
    echo "Environment check PASSED: segment oracle ready (PYTHONPATH=$TLTK_REF $VENV_DIR/bin/python)."
}

if [ "$CHECK_ONLY" = true ]; then verify_env; exit 0; fi

if [ "$FORCE" = true ] && [ -d "$VENV_DIR" ]; then
    echo "Removing existing venv (--force)"; rm -rf "$VENV_DIR"
fi

echo "==> Creating Python 3.9 venv via uv (downloads a managed CPython if needed)"
uv venv --quiet --python 3.9 "$VENV_DIR"

echo "==> Installing pinned era-correct deps"
uv pip install --quiet --python "$VENV_DIR/bin/python" \
    "numpy<1.24" "scipy<1.13" "scikit-learn==1.1.1" "pandas==1.5.3" \
    "gensim==4.3.2" "nltk" "sklearn-crfsuite" "requests" "matplotlib"

fetch_tltk_ref
verify_env
