#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: $0 IMAGE BUILD_SHA [SOURCE_FINGERPRINT] [IMAGE_REFERENCE]" >&2
  exit 2
fi

image=$1
build_sha=$2
portal_grant=ci-portal-grant-000000000000000000000000000000000000000000
access_token=ci-standalone-token-000000000000000000000000000000000000000000
source_fingerprint=${3:-development}
image_reference=${4:-development}
active_container=

cleanup() {
  if [[ -n "$active_container" ]]; then
    docker rm -f "$active_container" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

for mode in portal standalone; do
  active_container="google-mcp-smoke-$mode"
  cleanup
  active_container="google-mcp-smoke-$mode"

  mode_env=(-e "MCP_MODE=$mode")
  if [[ "$mode" == portal ]]; then
    mode_env+=(-e "MCP_PORTAL_GRANT_TOKEN=$portal_grant")
  else
    mode_env+=(-e "MCP_ACCESS_TOKEN=$access_token")
  fi

  docker run -d --rm --name "$active_container" \
    --init --network none --read-only --user 10001:10001 \
    --cap-drop ALL --security-opt no-new-privileges --pids-limit 256 \
    --tmpfs /tmp:rw,noexec,nosuid,nodev,size=64m,mode=1777 \
    "${mode_env[@]}" \
    -e "MCP_BUILD_SHA=$build_sha" \
    -e "MCP_SERVER_VERSION=$build_sha" \
    -e "MCP_SOURCE_FINGERPRINT=$source_fingerprint" \
    -e "MCP_IMAGE_REFERENCE=$image_reference" \
    -e "MCP_ALLOWED_HOSTS=127.0.0.1,localhost" \
    -e "MCP_DISABLE_DEFAULT_GOOGLE_FALLBACK=true" \
    -e "MCP_BYOK_CLIENT_CACHE_SIZE=0" \
    -e "MCP_BYOK_CLIENT_CACHE_TTL_SECONDS=0" \
    "$image" >/dev/null

  passed=false
  smoke_output=
  for _ in {1..30}; do
    if smoke_output=$(docker exec "$active_container" python /app/scripts/runtime_smoke.py 2>&1); then
      printf '%s\n' "$smoke_output"
      passed=true
      break
    fi
    sleep 1
  done
  if [[ "$passed" != true ]]; then
    printf '%s\n' "$smoke_output" >&2
    docker logs "$active_container" >&2 || true
    exit 1
  fi
  cleanup
  active_container=
done
