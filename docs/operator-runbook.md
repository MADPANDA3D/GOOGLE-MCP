# Google MCP Operator Runbook

This runbook covers an independent source deployment and the optional Portal
profile. Every value in angle brackets is a placeholder. Never paste provider
credentials into a server `.env` file.

## Prerequisites

- Docker Engine with Compose v2
- Git
- `curl` for health checks
- Python 3.12 or 3.13 plus `uv` only when running source verification locally
- a TLS reverse proxy or private network for any non-loopback client

## Verify the source first

For a release, check out the exact stable tag and compare its commit with the
release page. Then run the same provider-free gates used by CI:

```sh
uv sync --frozen --group dev
uv run python -m compileall -q fastmcp scripts
uv run pytest -q -p no:cacheprovider
uv run ruff check .
uv run ruff format --check .
uv run isort --check-only fastmcp scripts
uv run mypy
uv run python scripts/check_source_safety.py
uv run bandit -q -r fastmcp -x fastmcp/tests -lll
uv run pip-audit -r requirements.lock
docker compose -f docker-compose.yml config --quiet
docker compose -f docker-compose.portal.yml config --quiet
```

These checks use synthetic credentials and must not call Google. A provider
smoke test is a separate, explicitly authorized step.

## Standalone deployment

### 1. Create configuration

```sh
cp .env.example .env
chmod 600 .env
```

Set at least:

```dotenv
MCP_MODE=standalone
MCP_ACCESS_TOKEN=<STRONG_UNIQUE_RANDOM_VALUE_OF_AT_LEAST_32_CHARACTERS>
MCP_BUILD_SHA=<FULL_SOURCE_COMMIT_SHA>
MCP_SOURCE_FINGERPRINT=<SHA256_OF_THE_EXACT_SOURCE_ARCHIVE>
MCP_IMAGE_REFERENCE=<LOCAL_BUILD_OR_RELEASE_REFERENCE>
MCP_ALLOWED_HOSTS=localhost,127.0.0.1,<TRUSTED_PROXY_HOST>
```

Keep the public defaults, including request-only BYOK, zero client-cache size,
explicit confirmation for all 49 destructive or billable tools plus six
confirmed non-destructive writes, fixed GET/HEAD-only raw reads, browser-Origin
rejection, bounded batch/provider fan-out, and request logging disabled.

### 2. Build and start

```sh
docker compose --env-file .env -f docker-compose.yml config --quiet
docker compose --env-file .env -f docker-compose.yml up -d --build
docker compose --env-file .env -f docker-compose.yml ps
curl --fail --silent --show-error http://127.0.0.1:8086/health
```

The standalone profile binds to loopback by default. Keep it there unless a
firewall and explicit network design require otherwise.

### 3. Run provider-free runtime smoke

```sh
docker compose --env-file .env -f docker-compose.yml exec -T google-mcp \
  python scripts/runtime_smoke.py
```

The smoke verifies health, authentication-first rejection, browser-Origin
rejection, exact tool discovery, a local navigation call, and missing-BYOK
rejection. It does not call Google.

### 4. Connect the client

Use the MCP URL `http://127.0.0.1:8086/mcp` for a same-host client. Supply the
standalone bearer token and request-scoped Google OAuth headers through the
client's protected secret facility. Use TLS before crossing a host boundary.

## Optional Portal deployment

Portal mode is service-to-service. It expects an existing private Docker
network and a trusted broker that injects request-scoped provider credentials.

Set:

```dotenv
MCP_MODE=portal
MCP_PORTAL_GRANT_TOKEN=<UNIQUE_PORTAL_TO_SERVICE_GRANT>
MCP_PORTAL_NETWORK=<PRIVATE_DOCKER_NETWORK_NAME>
MCP_BUILD_SHA=<FULL_SOURCE_COMMIT_SHA>
MCP_SOURCE_FINGERPRINT=<SHA256_OF_THE_EXACT_SOURCE_ARCHIVE>
MCP_ALLOWED_HOSTS=google-mcp,<TRUSTED_PROXY_HOST>
```

Then:

```sh
docker network inspect <PRIVATE_DOCKER_NETWORK_NAME>
docker compose --env-file .env -f docker-compose.portal.yml config --quiet
docker compose --env-file .env -f docker-compose.portal.yml up -d --build
docker compose --env-file .env -f docker-compose.portal.yml ps
docker compose --env-file .env -f docker-compose.portal.yml exec -T google-mcp \
  python scripts/runtime_smoke.py
```

The Portal profile exposes port 8086 only to the selected Docker network; it
does not publish a host port. Complete the broker mapping described in
[portal-compat.md](portal-compat.md).

## Reverse-proxy requirements

- Terminate TLS with a certificate valid for the client-facing hostname.
- Forward the selected service-auth header unchanged.
- Forward the four documented Google BYOK headers only from trusted clients or
  the trusted broker; never log their values.
- Preserve MCP streaming behavior and session headers.
- Set request-size, connection, idle-timeout, and rate-limit policies that fit
  the deployment.
- Restrict direct access to the container port.
- Send a Host value included in `MCP_ALLOWED_HOSTS`.
- Do not add broad CORS access. Browser-origin traffic is rejected by default.

## Health and readiness

`GET /health` is the only unauthenticated readiness endpoint. Check at least:

- `status` is `healthy`
- `service` identifies Google MCP
- `tool_count` matches the release contract
- `mode` matches the intended startup mode
- the catalog/build/source fields match the deployed release
- configuration fields report presence only, never values

A healthy process does not prove Google credentials, scopes, quota, provider
availability, or account access. Those properties are request-specific.

## Upgrade procedure

1. Read the release notes and note tool, schema, mode, and environment changes.
2. Fetch the exact release tag or immutable image digest from that release.
3. Verify source/package attestations and the published digest when available.
4. Back up the current `.env` through the normal secret-management process.
5. Run provider-free CI and Compose validation against the candidate.
6. Build or pull the candidate without replacing the running instance.
7. Start it on an isolated loopback port or private network.
8. Run health and `runtime_smoke.py`.
9. Switch the trusted proxy or client mapping.
10. Observe authentication failures, latency, restarts, and error rates before
    any provider test.

Do not infer an image digest from a tag. The release page is the authority for
an image only after the workflow publishes and records that digest.

## Rollback procedure

1. Stop routing new sessions to the failed candidate.
2. Restore the previous verified tag or release digest and its matching
   configuration contract.
3. Run provider-free health and smoke checks.
4. Restore routing and monitor.
5. Preserve sanitized logs and exact build identities for diagnosis.

The server intentionally persists no provider credentials or application
database. Rollback is primarily code, image, and configuration rollback; it
cannot reverse Google-side mutations already accepted by a provider.

## Troubleshooting

| Symptom | First checks |
|---|---|
| Process exits during startup | `MCP_MODE` value and the selected service credential |
| `401` on `/mcp` | Bearer format in standalone mode, or exact Portal grant header/token in Portal mode |
| Host or Origin rejected | `MCP_ALLOWED_HOSTS`, proxy Host preservation, and browser-Origin policy |
| Local navigation works but provider tool fails | Complete OAuth header tuple, enabled API, OAuth scope, provider account permission |
| Maps tool fails | Maps key header, billing/API enablement, and explicit confirmation |
| Raw request rejected | GET/HEAD only, approved Google host, no body/headers, no credential-bearing query key |
| Output-limit error | Narrow the query, page size, time range, or download; do not disable bounds casually |
| Container unhealthy | `/health`, exact expected tool count, container logs after sanitization, memory/CPU limits |
| Portal cannot connect | External Docker network name, service membership, broker route, Host allowlist |

If diagnostic output could contain credentials or provider data, redact it
before sharing. Follow [SUPPORT.md](../SUPPORT.md) for public reports and
[SECURITY.md](../SECURITY.md) for vulnerabilities.
