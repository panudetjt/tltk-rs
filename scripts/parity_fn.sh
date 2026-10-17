#!/usr/bin/env bash
# The parity gate — single source of truth shared by `mise run parity` and the
# CI parity step. Kept in one place because the two copies drifted twice on
# 2026-10-16 (the plots disclosed-diff special-case and the `|| true` capture
# each existed in only one of them).
#
# Asserts, in order:
#   1. stichula goldens   — exact baseline 241/3/0
#   2. local python       — exact baseline 240/1/3
#   3. extra oracle       — fail=0 required (grows with new functions)
#   4. every corpus/corpus_fn_*.jsonl slice — fail=0, except corpus_fn_plots
#      which is disclosed-diff by design (hand-rolled renderer ≠ matplotlib
#      bytes): every row must fail on PNG bytes with none missing.
#
# Baselines move when a golden set changes — THIS FILE is the only place to
# update them (mise and CI both run this script). 240/1/3 came with the
# 2026-10-17 local_python golden refresh (was 228/7/9).
set -uo pipefail
cd "$(dirname "$0")/.."

out=$(cargo run -q -p tltk-verifier -- check corpus/corpus.jsonl golden/stichula.jsonl crates/tltk-core/data || true); echo "$out"; echo "$out" | grep -q "pass=241 fail=3 missing_golden=0" || exit 1
out=$(cargo run -q -p tltk-verifier -- check corpus/corpus.jsonl golden/local_python.jsonl crates/tltk-core/data || true); echo "$out"; echo "$out" | grep -q "pass=240 fail=1 missing_golden=3" || exit 1
out=$(cargo run -q -p tltk-verifier -- check corpus/corpus_extra.jsonl golden/local_python_extra.jsonl crates/tltk-core/data || true); echo "$out"; echo "$out" | grep -q "fail=0 missing_golden=0" || exit 1

shopt -s nullglob
for c in corpus/corpus_fn_*.jsonl; do
  g="golden/local_python_fn_${c##*corpus_fn_}"
  out=$(cargo run -q -p tltk-verifier -- check "$c" "$g" crates/tltk-core/data || true); echo "$out"
  n=$(wc -l < "$c")
  if [ "${c##*/}" = "corpus_fn_plots.jsonl" ]; then
    echo "$out" | grep -q "pass=0 fail=$n missing_golden=0" || exit 1
  else
    echo "$out" | grep -q "fail=0 missing_golden=0" || exit 1
  fi
done
