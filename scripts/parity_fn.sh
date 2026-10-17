#!/usr/bin/env bash
# Parity for ONE fn corpus: parity_fn.sh <name> (e.g. malt) — non-plots fn
# corpora only (corpus_fn_plots is disclosed-diff by design; see mise parity).
set -euo pipefail
cd "$(dirname "$0")/.."

name="${1:-}"
[ -n "$name" ] || { echo "usage: mise run parity-fn -- <corpus_fn suffix, e.g. malt>" >&2; exit 2; }
c="corpus/corpus_fn_${name}.jsonl"
[ -f "$c" ] || { echo "no such corpus file: $c" >&2; exit 2; }
out=$(cargo run -q -p tltk-verifier -- check "$c" "golden/local_python_fn_${name}.jsonl" crates/tltk-core/data || true)
echo "$out"
echo "$out" | grep -q "fail=0 missing_golden=0" || exit 1
