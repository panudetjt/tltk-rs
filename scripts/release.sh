#!/usr/bin/env bash
# Release tltk-rs: bump version -> tag -> push (CI publishes the library
# crates — malt-parser, then tltk-core — to crates.io) -> GitHub Release with
# a conventional-commit changelog.
#
# Usage: bash scripts/release.sh <version>     e.g. 0.1.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

VERSION="${1:-}"
case "$VERSION" in
  ""|*-*) echo "Usage: $0 <version>   (e.g. 0.1.0)" >&2; exit 2 ;;
esac
TAG="v$VERSION"

command -v gh >/dev/null 2>&1 || { echo "ERROR: gh CLI required" >&2; exit 1; }

[ -z "$(git status --short)" ] || { echo "ERROR: dirty working tree — commit first" >&2; exit 1; }
git rev-parse --abbrev-ref HEAD | grep -qx main || { echo "ERROR: not on main" >&2; exit 1; }
git fetch -q origin main
[ "$(git rev-parse main)" = "$(git rev-parse origin/main)" ] || { echo "ERROR: main is ahead/behind origin — sync first" >&2; exit 1; }
git rev-parse -q --verify "refs/tags/$TAG" >/dev/null && { echo "ERROR: $TAG already exists" >&2; exit 1; }

# 1. Bump the workspace version (member crates inherit it)
sed -i "s/^version = \"[^\"]*\"$/version = \"$VERSION\"/" Cargo.toml
grep -q "^version = \"$VERSION\"" Cargo.toml || { echo "ERROR: version bump failed" >&2; exit 1; }
git add Cargo.toml && git commit -q -m "chore: release $TAG"

# 2. Tag + push (the v* tag triggers publish-crates.yml)
git tag -a "$TAG" -m "tltk-rs $TAG"
git push -q origin main "$TAG"

# 3. Wait for the crates.io publish workflow
echo "==> waiting for publish-crates workflow…"
RUN_ID=$(gh run list -R panudetjt/tltk-rs --workflow publish-crates.yml --limit 1 --json databaseId -q '.[0].databaseId')
gh run watch "$RUN_ID" -R panudetjt/tltk-rs --exit-status >/dev/null
echo "  crates.io publish green"

# 4. GitHub Release with a grouped conventional-commit changelog
PREV_TAG="$(git describe --tags --abbrev=0 "$TAG^" 2>/dev/null || true)"
RANGE="${PREV_TAG:+$PREV_TAG..}HEAD"
NOTES=$(git log "$RANGE" --oneline --no-decorate | python3 -c '
import sys, re
groups = {}
for line in sys.stdin:
    m = re.match(r"([0-9a-f]+) (\w+)(\([^)]*\))?: (.*)", line.strip())
    if not m: continue
    sha, kind, _, msg = m.groups()
    bucket = {"feat": "Features", "fix": "Fixes", "perf": "Performance",
              "refactor": "Refactoring", "docs": "Docs", "test": "Tests"}.get(kind, "Chores")
    groups.setdefault(bucket, []).append(f"- {msg} ({sha})")
order = ["Features", "Fixes", "Performance", "Refactoring", "Tests", "Docs", "Chores"]
return "\n\n".join(f"### {b}\n" + "\n".join(groups[b]) for b in order if b in groups)
' 2>/dev/null || git log "$RANGE" --oneline)
gh release create "$TAG" -R panudetjt/tltk-rs --title "tltk-rs $TAG" --notes "## Changes

$NOTES

**crates.io**: https://crates.io/crates/tltk-core/$VERSION
**crates.io**: https://crates.io/crates/malt-parser/$VERSION
**toolbox**: https://tltk.panudet.dev/"
echo "==> released $TAG (tag pushed, crates published, GitHub release created)"
