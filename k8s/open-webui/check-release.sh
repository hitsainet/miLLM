#!/usr/bin/env bash
# Pin Open WebUI to its latest upstream release, if newer than what's deployed.
#
#   k8s/open-webui/check-release.sh <checkout of gitops/pilot>
#
# Run nightly by .github/workflows/owui-release-check.yml. Writes the pin to
# k8s/open-webui/.argocd-source-open-webui.yaml (the parameter-override file
# ArgoCD applies on top of kustomization.yaml); never commits. Prints the
# outcome and sets changed=true|false in $GITHUB_OUTPUT when present.
set -euo pipefail

root=${1:?usage: check-release.sh <repo root>}
dir="$root/k8s/open-webui"
pin="$dir/.argocd-source-open-webui.yaml"
image=ghcr.io/open-webui/open-webui

out() { [ -n "${GITHUB_OUTPUT:-}" ] && echo "$1" >> "$GITHUB_OUTPUT"; return 0; }

# "Latest release" excludes drafts and prereleases.
latest=$(curl -fsSL https://api.github.com/repos/open-webui/open-webui/releases/latest \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])')
if ! [[ $latest =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "::error::unexpected release tag '$latest'" >&2
  exit 1
fi

# The image is published by upstream CI some time after the release; if it
# isn't there yet, try again next run rather than pin a tag that can't pull.
token=$(curl -fsSL "https://ghcr.io/token?scope=repository:open-webui/open-webui:pull" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])')
if ! curl -fsSI -o /dev/null -H "Authorization: Bearer $token" \
  -H 'Accept: application/vnd.oci.image.index.v1+json' \
  -H 'Accept: application/vnd.docker.distribution.manifest.list.v2+json' \
  "https://ghcr.io/v2/open-webui/open-webui/manifests/$latest"; then
  echo "Release $latest has no image on ghcr.io yet — skipping."
  out changed=false
  exit 0
fi

# Deployed tag: the pin file if present, else the kustomization baseline.
if [ -f "$pin" ]; then
  current=$(sed -nE "s|.*$image=$image:(v[0-9.]+).*|\1|p" "$pin")
else
  current=$(sed -nE 's/^ *newTag: *(v[0-9.]+).*/\1/p' "$dir/kustomization.yaml")
fi
if [ -z "$current" ]; then
  echo "::error::could not determine the deployed Open WebUI tag" >&2
  exit 1
fi

# Only ever move forward (also a no-op when equal).
if [ "$(printf '%s\n%s\n' "$current" "$latest" | sort -V | tail -1)" = "$current" ]; then
  echo "Open WebUI $current is current (latest release $latest)."
  out changed=false
  exit 0
fi

cat > "$pin" <<EOF
kustomize:
  images:
  - $image=$image:$latest
EOF
echo "Open WebUI $current -> $latest"
out changed=true
out from="$current"
out to="$latest"
