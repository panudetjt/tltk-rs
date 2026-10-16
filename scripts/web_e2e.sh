#!/usr/bin/env bash
# Real-browser e2e loop validator (playwright) — see scripts/web_e2e.mjs header.
# Self-healing: installs the pinned playwright into web/e2e on first run.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -d web/e2e/node_modules/playwright ]; then
  echo "[web-e2e] installing pinned playwright into web/e2e (browsers come from ~/.cache/ms-playwright)"
  (cd web/e2e && npm install --no-fund --no-audit --silent)
fi
exec node scripts/web_e2e.mjs "$@"
