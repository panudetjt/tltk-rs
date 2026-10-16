#!/usr/bin/env bash
# scripts/setup_segment_oracle.sh
#
# Reproduces or checks the period-correct Python 3.9 environment required for
# the TLTK 1.10 segment oracle (RandomForest pickle incompatibility with sklearn >= 1.2).
#
# Usage:
#   bash scripts/setup_segment_oracle.sh [--check] [--force]
#
# Options:
#   --check    Verify the existing environment without modifying or destroying it.
#   --force    Recreate the virtual environment from scratch.
#   --help     Display this help message.

set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/.local/share/tltk-rs/venv-segment-oracle}"
TLTK_REF="${TLTK_REF:-~/work/my-project/tltk-ref/tltk-110}"

CHECK_ONLY=false
FORCE=false

for arg in "$@"; do
    case "$arg" in
        --check)
            CHECK_ONLY=true
            ;;
        --force)
            FORCE=true
            ;;
        -h|--help)
            echo "Usage: $0 [--check] [--force]"
            echo "  --check  Dry check of existing env without modifying anything"
            echo "  --force  Recreate virtualenv from scratch"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

export PATH="$HOME/.local/bin:$PATH"

find_python39() {
    if command -v mise >/dev/null 2>&1; then
        local mise_dir
        mise_dir="$(mise where python@3.9 2>/dev/null || true)"
        if [[ -n "$mise_dir" && -x "$mise_dir/bin/python" ]]; then
            echo "$mise_dir/bin/python"
            return 0
        fi
        local mise_py
        mise_py="$(mise which python@3.9 2>/dev/null || true)"
        if [[ -n "$mise_py" && -x "$mise_py" ]]; then
            echo "$mise_py"
            return 0
        fi
    fi
    if command -v python3.9 >/dev/null 2>&1; then
        which python3.9
        return 0
    fi
    for cand in "$HOME/.local/share/mise/installs/python"/3.9*/bin/python; do
        if [[ -x "$cand" ]]; then
            echo "$cand"
            return 0
        fi
    done
    return 1
}

verify_env() {
    local py_bin="$VENV_DIR/bin/python"
    echo "Checking segment oracle environment at $VENV_DIR..."

    if [[ ! -x "$py_bin" ]]; then
        echo "FAIL: $py_bin does not exist or is not executable" >&2
        return 1
    fi

    # 1. Check Python version
    local py_ver
    py_ver="$("$py_bin" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")"
    if [[ ! "$py_ver" =~ ^3\.9\. ]]; then
        echo "FAIL: Expected Python 3.9.x, got $py_ver" >&2
        return 1
    fi
    echo "  [OK] Python version: $py_ver"

    # 2. Check key package versions
    "$py_bin" - <<'EOF'
import sys
import importlib.metadata
import numpy
import sklearn
import pandas
import gensim
import nltk
import sklearn_crfsuite
import requests

np_ver = tuple(int(x) for x in numpy.__version__.split(".")[:2])
assert np_ver < (1, 24), f"numpy < 1.24 required, got {numpy.__version__}"

skl_ver = sklearn.__version__
assert skl_ver == "1.1.1", f"scikit-learn == 1.1.1 required, got {skl_ver}"

pd_ver = pandas.__version__
assert pd_ver.startswith("1.5."), f"pandas 1.5.x required, got {pd_ver}"

gensim_ver = gensim.__version__
assert gensim_ver == "4.3.2", f"gensim == 4.3.2 required, got {gensim_ver}"

crf_ver = importlib.metadata.version("sklearn-crfsuite")

print(f"  [OK] numpy: {numpy.__version__}")
print(f"  [OK] scikit-learn: {sklearn.__version__}")
print(f"  [OK] pandas: {pandas.__version__}")
print(f"  [OK] gensim: {gensim.__version__}")
print(f"  [OK] nltk: {nltk.__version__}")
print(f"  [OK] sklearn-crfsuite: {crf_ver}")
print(f"  [OK] requests: {requests.__version__}")
EOF

    # 3. Test segment oracle probe
    if [[ -d "$TLTK_REF" ]]; then
        echo "Testing TLTK 1.10 segment probe..."
        PYTHONPATH="$TLTK_REF" "$py_bin" - <<'EOF'
import warnings
warnings.filterwarnings("ignore")
from tltk.nlp import segment
res = segment("แมวกินปลาทะเล")
expected = "แมว|กิน|ปลาทะเล|<s/><u/>"
assert res == expected, f"Expected {expected!r}, got {res!r}"
print(f"  [OK] segment('แมวกินปลาทะเล') -> {res}")
EOF
    else
        echo "  [WARN] TLTK reference dir $TLTK_REF not found; skipped live oracle probe"
    fi

    echo "Environment check PASSED: $VENV_DIR is ready for segment oracle."
    return 0
}

if [ "$CHECK_ONLY" = true ]; then
    verify_env
    exit 0
fi

# Setup / build mode
if [ "$FORCE" = true ] && [ -d "$VENV_DIR" ]; then
    echo "Removing existing venv at $VENV_DIR (--force specified)..."
    rm -rf "$VENV_DIR"
fi

if [ ! -d "$VENV_DIR" ]; then
    PY39="$(find_python39 || true)"
    if [[ -z "$PY39" ]]; then
        if command -v mise >/dev/null 2>&1; then
            echo "Installing Python 3.9 via mise..."
            mise install python@3.9
            PY39="$(find_python39)"
        else
            echo "FAIL: python 3.9 not found and mise not available" >&2
            exit 1
        fi
    fi
    echo "Creating virtualenv at $VENV_DIR using $PY39..."
    "$PY39" -m venv "$VENV_DIR"
fi

echo "Installing pinned dependencies into $VENV_DIR..."
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet \
    "numpy<1.24" \
    "scipy<1.13" \
    "scikit-learn==1.1.1" \
    "pandas==1.5.3" \
    "gensim==4.3.2" \
    "nltk" \
    "sklearn-crfsuite" \
    "requests"

verify_env
