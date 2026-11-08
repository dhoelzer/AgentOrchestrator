#!/usr/bin/env bash
# Publish docs/wiki/*.md to this repository's GitHub wiki.
#
#   ./scripts/publish-wiki.sh
#
# GitHub creates a repository's wiki git remote lazily: <repo>.wiki.git does not
# exist until the wiki has at least one page, and a token-authenticated first
# push cannot create it. If the wiki is still empty this script prints the one
# manual step and exits; after that, re-running publishes every page.

set -euo pipefail

REPO="${WIKI_REPO:-dhoelzer/AgentOrchestrator}"
SRC="$(cd "$(dirname "$0")/../docs/wiki" && pwd)"
TOKEN="${GITHUB_TOKEN:?GITHUB_TOKEN must be set}"
WIKI_URL="https://github.com/${REPO}.wiki.git"

git_auth() {
  git -c credential.helper="!f() { echo username=x-access-token; echo password=${TOKEN}; }; f" "$@"
}

if ! git_auth ls-remote --exit-code "$WIKI_URL" >/dev/null 2>&1; then
  cat >&2 <<EOF
The wiki for ${REPO} has no pages yet, so ${WIKI_URL} does not exist, and
GitHub will not let a token create it.

One-time step: open
    https://github.com/${REPO}/wiki/_new
create any page (call it "Home"; the content does not matter), then re-run:
    ./scripts/publish-wiki.sh
EOF
  exit 1
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

git_auth clone --quiet "$WIKI_URL" "$work"
cd "$work"

# A fresh clone carries no identity of its own, so commit would fail outright.
git config user.name "${GIT_AUTHOR_NAME:-dhoelzer}"
git config user.email "${GIT_AUTHOR_EMAIL:-openhands@all-hands.dev}"

# GitHub wikis use `master`; keep whatever branch the remote already has rather
# than forcing a rename on an existing wiki.
branch="$(git symbolic-ref --short HEAD)"

cp "$SRC"/*.md "$work"/
git add -A

if git diff --cached --quiet; then
  echo "Wiki already up to date on ${branch}."
  exit 0
fi

src_sha="$(git -C "$SRC/../.." rev-parse --short HEAD 2>/dev/null || echo unknown)"
git -c commit.gpgsign=false commit -q -m "Update wiki from docs/wiki

Generated from harness source at ${src_sha}.

Co-authored-by: openhands <openhands@all-hands.dev>"
git_auth push --quiet origin "$branch"
echo "Published $(ls "$SRC"/*.md | wc -l | tr -d ' ') pages to the ${REPO} wiki (${branch})."
