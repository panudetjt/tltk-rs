#!/usr/bin/env bash
# Upload the big model files to the R2 bucket bound as MODELS.
# Prereqs: `npx wrangler login` already done; bucket created (see below).
set -euo pipefail

BUCKET="${1:-tltk-models}"
DATA_DIR="$(cd "$(dirname "$0")/../crates/tltk-core/data" && pwd)"

for f in edu.bin.gz w2v.bin.gz ner.bin.gz; do
  if [[ -f "$DATA_DIR/$f" ]]; then
    echo "uploading $f ($(du -h "$DATA_DIR/$f" | cut -f1)) -> r2://$BUCKET/$f"
    npx wrangler r2 object put "$BUCKET/$f" --file "$DATA_DIR/$f" --remote
  else
    echo "skip $f (not present)"
  fi
done
echo "done. bucket contents:"
npx wrangler r2 object get "$BUCKET" --remote >/dev/null 2>&1 || true
npx wrangler r2 bucket list 2>/dev/null | head -5 || true
