#!/usr/bin/env bash
# Land a worker branch with the gate enforced BEFORE the merge commit exists.
#
# Every keep-both merge seam of 2026-10-16/17 (missing comma in the worker
# functions array ×2, stray `}` in app.js — the "no options" production
# incident) landed because the merge was committed before clippy or a browser
# saw it. This script makes that order impossible: the merge is created
# without a commit, the full gate + parity run against the merged tree, and
# only a green tree is allowed to commit.
#
# Usage:
#   mise run land -- worker/<slug>     # merge + gate + parity + commit
#   bash scripts/land.sh --continue    # after fixing a red gate or conflicts
#   git merge --abort                  # to discard an in-progress landing
#
# Notes:
# - the index must be clean (parallel-session WIP in the working tree is fine:
#   unstaged files are never swept into the merge commit)
# - `--continue` re-runs the same gates; conflicts must be resolved and
#   `git add`ed first
set -euo pipefail
cd "$(dirname "$0")/.."

if git diff --cached --quiet; then :; else
  echo "land: the index is not clean — commit or unstage it first"; exit 1
fi

if [ "${1:-}" = "--continue" ]; then
  if git diff --name-only --diff-filter=U | grep -q .; then
    echo "land: unresolved conflicts:"; git diff --name-only --diff-filter=U; exit 1
  fi
else
  branch="${1:?usage: land.sh worker/<branch> | --continue}"
  git rev-parse --verify --quiet "$branch" >/dev/null || { echo "land: no such branch: $branch"; exit 1; }
  if git merge-base --is-ancestor "$branch" HEAD; then
    echo "land: $branch is already merged"; exit 1
  fi
  if ! git merge --no-ff --no-commit "$branch"; then
    echo "land: conflicts — resolve, git add, then: bash scripts/land.sh --continue"
    exit 1
  fi
fi

fail() { echo "land: $1 — fix in the working tree, then: bash scripts/land.sh --continue (or git merge --abort to discard)"; exit 1; }
mise run gate || fail "gate is red on the merged tree"
bash scripts/parity_fn.sh || fail "parity is red on the merged tree"

git commit --no-edit
git log --oneline -1
echo "land: done"
