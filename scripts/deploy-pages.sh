#!/usr/bin/env bash
# Build web/site and deploy it to the Cloudflare Pages project `h3-rs`.
# Needs CLOUDFLARE_API_TOKEN (Pages: Edit) and CLOUDFLARE_ACCOUNT_ID. The
# first deploy of a new project needs `wrangler pages project create h3-rs`.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CLOUDFLARE_API_TOKEN:?set CLOUDFLARE_API_TOKEN}"
: "${CLOUDFLARE_ACCOUNT_ID:?set CLOUDFLARE_ACCOUNT_ID}"
python3 web/build.py
npx -y wrangler@4 pages deploy web/site --project-name h3-rs --branch main --commit-dirty=true
