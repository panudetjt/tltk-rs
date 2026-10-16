#!/usr/bin/env bash
# Check PyPI for a newer tltk release and diff it against the version we ported
# (1.11). Prints function-level changes in nlp.py/corpus.py so the re-port
# surface is obvious. Exit 0 with "up to date", exit 0 with a report if newer.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PORTED_VERSION="1.11"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

need() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: $1 not on PATH (run via mise)" >&2; exit 1; }; }
need curl; need python3; need uv

latest="$(curl -fsSL https://pypi.org/pypi/tltk/json | python3 -c 'import json,sys; print(json.load(sys.stdin)["info"]["version"])')"
echo "ported: $PORTED_VERSION   latest on PyPI: $latest"
if [ "$latest" = "$PORTED_VERSION" ]; then echo "up to date"; exit 0; fi

get_wheel() { # $1=ver $2=dest
    curl -fsSL "https://pypi.org/pypi/tltk/$1/json" | python3 -c '
import json, sys
for f in json.load(sys.stdin)["urls"]:
    if f["filename"].endswith("py3-none-any.whl"): print(f["url"]); break
' | xargs -I{} curl -fsSL {} -o "$2"
}
get_wheel "$PORTED_VERSION" "$WORK/old.whl"
get_wheel "$latest" "$WORK/new.whl"
python3 -m zipfile -e "$WORK/old.whl" "$WORK/old"
python3 -m zipfile -e "$WORK/new.whl" "$WORK/new"

echo; echo "== release notes from PyPI (Updates section mentioning $latest)"
curl -fsSL "https://pypi.org/pypi/tltk/$latest/json" | python3 -c '
import json, sys, re
desc = json.load(sys.stdin)["info"].get("description") or ""
lines = desc.splitlines()
for i, l in enumerate(lines):
    if re.search(r"^\s*Version\s+%s\s*[:.]" % re.escape(sys.argv[1]), l, re.I):
        print("\n".join(lines[i:i+6])); print("---"); break
else:
    for i, l in enumerate(lines):
        if re.search(r"version\s*%s" % re.escape(sys.argv[1]), l, re.I):
            print("\n".join(lines[i:i+6])); print("---"); break
    else:
        print("(no %s mention in description; full description on pypi.org)" % sys.argv[1])
' "$latest"

echo; echo "== changed files (tltk/*.py + data manifests)"
( cd "$WORK" && diff -rq old/tltk new/tltk 2>/dev/null | head -40 ) || true

for f in nlp.py corpus.py; do
    echo; echo "== def-level changes in $f"
    diff <(grep -n "^def \|^    def " "$WORK/old/tltk/$f" | sed 's/^[0-9]*://') \
         <(grep -n "^def \|^    def " "$WORK/new/tltk/$f" | sed 's/^[0-9]*://') || true
    echo "-- full diff: $WORK ignored; regenerate with:"
    echo "   curl -s https://pypi.org/pypi/tltk/$latest/json | jq -r '.urls[].url' # then unzip & diff"
done

cat <<REPORT

== re-port checklist when upstream moves
1. Bump PORTED_VERSION here + in scripts/fetch_models.sh (tltk==<ver>) and
   scripts/setup_segment_oracle.sh.
2. mise run fetch-models  (re-convert blobs; compare sizes vs the committed six)
3. mise run oracle-setup  (era env may need re-pinning if deps moved)
4. Regenerate goldens from the new oracle (corpus worker pattern, R5) — never
   from the Rust binary.
5. mise run parity / node-parity / wasm-parity — every diff is a real signal.
REPORT
