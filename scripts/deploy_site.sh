#!/usr/bin/env bash
# Deploys site/ to blolabel.ai. The auto-article bot runs this after every blog
# post, from a fresh clone, so it refuses to ship a site with pages missing.
# Without this, the gitignored homepage was dropped by every content deploy and
# blolabel.ai/ served a 404 from 2026-09-21 to 2026-09-30.
set -euo pipefail
cd "$(dirname "$0")/.."

missing=0
for f in site/index.html site/results.json site/blog/index.html site/style.css; do
  if [ ! -s "$f" ]; then echo "deploy_site: refusing to deploy, $f is missing or empty" >&2; missing=1; fi
done
[ "$missing" -eq 0 ] || exit 1

npx wrangler pages deploy site --project-name blolabel --branch main --commit-dirty=true

# Check the live homepage, not just that the upload succeeded.
for i in 1 2 3 4 5 6; do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 20 https://blolabel.ai/ || true)
  [ "$code" = "200" ] && { echo "deploy_site: https://blolabel.ai/ is 200"; exit 0; }
  sleep $((i * 5))
done
echo "deploy_site: deployed, but https://blolabel.ai/ returned $code" >&2
exit 1
