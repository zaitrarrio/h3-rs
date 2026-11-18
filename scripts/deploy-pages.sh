#!/usr/bin/env bash
# Build web/site and deploy it to the Cloudflare Pages project `h3-rs`.
# Needs CLOUDFLARE_API_TOKEN (Pages: Edit) and CLOUDFLARE_ACCOUNT_ID. The
# project exists (created with `wrangler pages project create h3-rs --force`, classic
# Pages); it serves https://h3-video.gialabs.sh and https://h3-rs.pages.dev.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CLOUDFLARE_API_TOKEN:?set CLOUDFLARE_API_TOKEN}"
: "${CLOUDFLARE_ACCOUNT_ID:?set CLOUDFLARE_ACCOUNT_ID}"
python3 web/build.py
npx -y wrangler@4 pages deploy web/site --project-name h3-rs --branch main --commit-dirty=true
