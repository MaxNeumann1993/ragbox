#!/usr/bin/env bash
# Ingest the bundled sample_docs/ into Qdrant via the running API.
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"

curl -sf -X POST "$API_URL/ingest" \
  -H "Content-Type: application/json" \
  -d '{"directory": "sample_docs"}'
echo
