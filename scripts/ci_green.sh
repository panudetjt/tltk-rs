#!/usr/bin/env bash
# The single sanctioned way to verify CI for a commit.
#
# `gh run watch --exit-status` piped through anything (tail, head, tee) reports
# the PIPE's exit code — the "green CI" at 2026-10-17 00:0x was a `tail` exit
# code while the gate job was red. This script ends on exactly one exit
# decision, taken from `gh run view --json conclusion` — never from streamed
# output.
#
# Usage:
#   bash scripts/ci_green.sh [commit-sha]     # default: HEAD
#   mise run ci-green -- <sha>
set -euo pipefail
cd "$(dirname "$0")/.."

command -v gh >/dev/null || { echo "ci-green: gh CLI not on PATH"; exit 2; }

SHA="${1:-$(git rev-parse HEAD)}"
RUN_ID="$(gh run list --workflow=ci.yml --commit "$SHA" --limit 1 \
    --json databaseId --jq '.[0].databaseId // empty')"
[ -n "$RUN_ID" ] || { echo "ci-green: no ci.yml run found for $SHA"; exit 1; }

echo "ci-green: waiting for run $RUN_ID (commit $SHA)"
# watch's own exit status is advisory; the decision below re-reads the
# conclusion from the API
gh run watch "$RUN_ID" --exit-status >/dev/null 2>&1 || true
CONCLUSION="$(gh run view "$RUN_ID" --json conclusion --jq '.conclusion')"
echo "ci-green: run $RUN_ID conclusion=$CONCLUSION"
[ "$CONCLUSION" = "success" ]
