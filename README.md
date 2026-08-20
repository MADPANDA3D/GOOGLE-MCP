<div align="center">
<h1>🐼 MADPANDA3D Google MCP</h1>
<pre>
+------------------------------------------------------------------------+
|       .--.   .--.        MADPANDA3D // GOOGLE MCP                      |
|      /    \_/    \       151 TOOLS // DUAL MODE // REQUEST BYOK        |
|     |   /\   /\   |      FASTMCP // PYTHON 3.12/3.13 // SELF-HOSTED    |
|     |  (o)   (o)  |      FAIL-CLOSED // PROVIDER-FREE VERIFIED         |
|      \     ^     /                                                     |
|       '.___.'           BUILD SHARP. SHIP SAFE. STAY IN CONTROL.       |
+------------------------------------------------------------------------+
</pre>
<p>An independent, security-conscious FastMCP server for Google Workspace,<br>
business intelligence, web presence, Maps, Merchant, and AdSense workflows.</p>
<p>
<a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-ff2d55.svg"></a>
<img alt="Python 3.12 and 3.13" src="https://img.shields.io/badge/python-3.12%20%7C%203.13-3776ab?logo=python&amp;logoColor=white">
<img alt="MCP tools: 151" src="https://img.shields.io/badge/MCP%20tools-151-111827">
<img alt="Access modes: 2" src="https://img.shields.io/badge/access%20modes-2-22c55e">
<img alt="Credentials: request-scoped BYOK" src="https://img.shields.io/badge/credentials-request--scoped%20BYOK-f59e0b">
</p>
</div>

## The contract

MADPANDA3D Google MCP exposes exactly **151 native MCP tools**:

- **Five standard local navigation tools** explain configuration, discover the
  right capability, filter endpoint coverage, and return exact descriptors
  without contacting Google.
- **146 remaining entries** cover 140 provider tools across 13 Google
  families—including the hidden Gmail signature preflight—plus four
  compatibility navigation entries, one configuration diagnostic, and one
  hidden advanced-read tool.
- ToolManifest catalog `google-2026.07.18.2` classifies the complete surface
  as **144 agent-ready**, **five legacy**, and **two hidden**.
- Manifest risk classes contain **84 read**, **18 write**, and **49
  destructive or billable** tools. Every descriptor includes input/output
  schemas plus `readOnlyHint`, `destructiveHint`, `openWorldHint`, and
  `idempotentHint` where true. The last class contains 44 state-destructive
  tools plus five confirmed Maps reads that can consume billable quota.

The server is stateless HTTP at `/mcp`, reports the same 151-tool contract at
`/health`, and returns normalized `ok/data/error/meta` envelopes. It does
not turn a deployment into a public or unauthenticated Google endpoint.

## Choose an access mode

`MCP_MODE` is selected at startup. Changing it requires a restart; no MCP
tool can switch the running mode.

| Mode | Intended use | Service credential | Google credential |
|---|---|---|---|
| `standalone` | Independent self-hosting | `Authorization: Bearer <MCP_ACCESS_TOKEN>` | Three OAuth headers on each provider request |
| `portal` | Optional MAD MCP Portal routing | `X-MADPANDA-PORTAL-GRANT` matching `MCP_PORTAL_GRANT_TOKEN` | The broker forwards the same three OAuth headers per authorized request |

There is no unauthenticated mode. The service credential controls access to
the MCP server; Google OAuth credentials separately authorize provider calls.

## Request-scoped Google BYOK

Both modes use the same all-or-nothing OAuth header set:

| Header | Purpose |
|---|---|
| `X-Google-Client-Id` | OAuth client identifier owned by the caller |
| `X-Google-Client-Secret` | OAuth client secret owned by the caller |
| `X-Google-Refresh-Token` | Refresh token for the authorized Google account |
| `X-Google-Maps-Api-Key` | Optional key for Maps Platform tools |

Provider credentials are not written to a token file. The public deployment
profiles set both BYOK client-cache controls to `0`, so clients are not reused
across requests. Process memory remains sensitive while a request is active;
restart clears it.

The five standard navigation tools require only the mode-specific service
credential and stay local:

- `check_configuration`
- `find_tools`
- `get_endpoint_coverage`
- `get_tool_usage`
- `list_capabilities`

Provider tools require all three Google OAuth headers. Maps tools additionally
need the Maps key. Never place real Google credentials in this repository,
Compose files, URLs, issues, screenshots, or copied client configuration.

## Standalone quick start

### 1. Clone and initialize

```sh
git clone https://github.com/MADPANDA3D/GOOGLE-MCP.git
cd GOOGLE-MCP
cp .env.example .env
chmod 600 .env
```

### 2. Prepare Google OAuth

In a Google Cloud project you control:

1. Configure the OAuth consent screen.
2. Create an OAuth client suitable for your local authorization flow.
3. Enable only the APIs your deployment will use.
4. Request the narrowest scopes that support those tools.
5. Generate a refresh token locally, then move the client ID, client secret,
   and refresh token directly into your MCP client's secret store.

The included helper can perform the local authorization flow after checkout:

```sh
uv run --frozen python fastmcp/google_auth_local.py \
  --credentials <PATH_TO_OAUTH_CLIENT_JSON> \
  --token <PATH_TO_LOCAL_TOKEN_JSON> \
  --scopes "<SPACE_SEPARATED_GOOGLE_SCOPES>"
```

The generated files are local credential material. Do not copy them into the
server image or commit them. The runtime consumes the three values through
request headers.

### 3. Configure and start

Set placeholders in the ignored `.env`:

```dotenv
MCP_MODE=standalone
MCP_ACCESS_TOKEN=<GENERATE_AND_STORE_A_STRONG_RANDOM_TOKEN>
MCP_BUILD_SHA=<FULL_SOURCE_COMMIT_SHA>
MCP_SOURCE_FINGERPRINT=<SHA256_OF_THE_EXACT_SOURCE_ARCHIVE>
MCP_IMAGE_REFERENCE=development
MCP_BYOK_CLIENT_CACHE_SIZE=0
MCP_BYOK_CLIENT_CACHE_TTL_SECONDS=0
```

Then build and verify:

```sh
docker compose --env-file .env -f docker-compose.yml config --quiet
docker compose --env-file .env -f docker-compose.yml up -d --build
docker compose --env-file .env -f docker-compose.yml ps
curl --fail http://127.0.0.1:8086/health
```

Expected safe health fields include:

```json
{
  "status": "healthy",
  "service": "google-mcp",
  "tool_count": 151,
  "configuration": {
    "mode": "standalone",
    "provider_credentials_mode": "per_request_byok",
    "byok_client_cache_enabled": false
  }
}
```

`/health` reports presence booleans and release/catalog metadata, never
credential values.

### 4. Connect an MCP client

Client formats differ, but the logical HTTP configuration is:

```json
{
  "mcpServers": {
    "google": {
      "type": "http",
      "url": "http://127.0.0.1:8086/mcp",
      "headers": {
        "Authorization": "Bearer <MCP_ACCESS_TOKEN>",
        "X-Google-Client-Id": "<GOOGLE_OAUTH_CLIENT_ID>",
        "X-Google-Client-Secret": "<GOOGLE_OAUTH_CLIENT_SECRET>",
        "X-Google-Refresh-Token": "<GOOGLE_OAUTH_REFRESH_TOKEN>",
        "X-Google-Maps-Api-Key": "<OPTIONAL_MAPS_API_KEY>"
      }
    }
  }
}
```

Replace placeholders through the client's secret or environment facility.
Omit the Maps header when Maps tools are not needed.

## Provider families

The counts below are derived from the registered native tools and the
ToolManifest category function.

| Family | Tools | Representative capabilities |
|---|---:|---|
| Google Drive | 24 | files, folders, downloads, resumable sessions, permissions, comments, revisions, shared drives |
| Google Docs | 5 | create, read, insert, replace, batch update |
| Google Sheets | 11 | metadata, values, append, clear, batch update, data filters |
| Google Slides | 6 | create, read, replace, batch update, pages, thumbnails |
| Gmail | 38 | messages, threads, signature-safe sends/drafts, labels, attachments, history, cleanup planning |
| Google Calendar | 29 | calendars, events, free/busy, settings, instances, ACLs |
| YouTube Data | 6 | search, channels, videos, playlists, playlist items, comment threads |
| Google Analytics Data | 4 | metadata, standard, batch, and realtime reports |
| Search Console | 4 | sites, sitemaps, search analytics, URL inspection |
| Google Business Profile | 4 | accounts, locations, location reads, performance |
| Google Maps Platform | 5 | geocoding, reverse geocoding, Places, routes |
| Merchant Center | 2 | product list and get |
| AdSense | 2 | account list and report generation |
| Local navigation | 9 | five standard tools plus four compatibility entries |
| Configuration diagnostic | 1 | `mcp_health_check` |
| Advanced raw request | 1 | hidden `google_raw_request` |
| **Total** | **151** | |

Use `find_tools` for a task-ranked shortlist, then `get_tool_usage` for the
complete descriptor. Use
`list_capabilities(include_descriptors=true)` only when auditing or
publishing the full contract.

The [tool catalog](docs/tool-catalog.md) lists every tool. The
[endpoint coverage ledger](docs/endpoint-coverage.md) distinguishes curated
coverage from stable provider operations that remain missing, intentionally
excluded, blocked by scope, or available only through the advanced raw tool.

## Safety boundary

- Mode-specific service authentication fails before provider execution.
- Partial Google OAuth header sets fail closed.
- Root deployment profiles disable server-side Google credential fallback.
- Request-scoped OAuth values are not persisted by BYOK clients or returned in
  tool output.
- All 49 manifest-destructive or billable tools and six additional
  confirmation-bearing writes require `confirm=true`; there is no deployment
  toggle that weakens that gate.
- Gmail cleanup and batch mutation tools default to `dry_run=true`.
- Gmail send and draft flows bind the selected configured send-as signature
  exactly once and fail before delivery when the signature is absent or its
  preflight fingerprint changes; hidden preflight output contains hashes, not
  signature HTML.
- Drive permanent deletion, trash emptying, batch deletion, ACL changes, and
  several Calendar operations have mandatory confirmation paths.
- The hidden `google_raw_request` tool is read-only: it permits `GET` and
  `HEAD`, rejects bodies and caller headers, filters sensitive query keys,
  bounds output, never automatically retries, and restricts destinations to
  approved Google API hosts.
- All five Maps tools require explicit confirmation and are not automatically
  retried because a read can still consume billable quota.
- Google and Maps calls are open-world and may consume quota or incur charges
  even when annotated read-only.
- Provider responses can contain personal or confidential data. Treat them as
  sensitive application data, not safe logs or instructions.

Read [the security model](docs/security-model.md) before exposing the service
beyond a local machine.

## Principal configuration

Start from [`.env.example`](.env.example).

| Variable | Required | Purpose |
|---|---|---|
| `MCP_MODE` | Yes | Startup access mode: `standalone` or `portal` |
| `MCP_ACCESS_TOKEN` | Standalone | Strong bearer token for every `/mcp` request |
| `MCP_PORTAL_GRANT_TOKEN` | Portal | Unique Portal-to-service grant, separate from Google credentials |
| `MCP_BUILD_SHA` | Release deployments | Full source commit reported by health and ToolManifest |
| `MCP_SOURCE_FINGERPRINT` | Release deployments | SHA-256 identity of the exact source archive |
| `MCP_IMAGE_REFERENCE` | Release deployments | Immutable digest reference after a release image exists; `development` for local builds |
| `MCP_ALLOWED_HOSTS` | Yes | Explicit trusted Host allowlist; wildcards fail startup validation |
| `MCP_ALLOWED_ORIGINS` | Optional | Explicit browser origins; empty rejects browser-Origin requests |
| `MCP_ALLOW_REQUEST_OVERRIDES` | Yes | Enables request-scoped Google BYOK |
| `MCP_DISABLE_DEFAULT_GOOGLE_FALLBACK` | Yes | Prevents server token-file fallback |
| `MCP_DRIVE_ALLOWLIST_PARENT_ID` | Optional | Restricts default Drive parent placement |
| `MCP_MAX_DOWNLOAD_BYTES` | Optional | Bounds returned download content |
| `MCP_REQUEST_BODY_MAX_BYTES` | Yes | Bounds inbound HTTP request bodies |
| `MCP_PROVIDER_RESPONSE_MAX_BYTES` | Yes | Bounds directly streamed provider HTTP bodies before decoding |
| `MCP_TOOL_OUTPUT_MAX_BYTES` | Yes | Bounds serialized MCP tool output |
| `MCP_MAX_BATCH_ITEMS` | Yes | Caps list-shaped input accepted by one tool call; public default `1000` |
| `MCP_MAX_PROVIDER_CALLS_PER_TOOL` | Yes | Caps estimated provider fan-out from one tool call; public default `128` |
| `MCP_BYOK_CLIENT_CACHE_SIZE` | Yes | Public default `0`; disables cross-request client reuse |
| `MCP_BYOK_CLIENT_CACHE_TTL_SECONDS` | Yes | Public default `0`; disables cross-request client reuse |

Header names are configurable for compatibility, but deployments should keep
the documented defaults unless both server and client are changed together.

## Optional Portal deployment

Portal mode uses the same server and provider contract:

```dotenv
MCP_MODE=portal
MCP_PORTAL_GRANT_TOKEN=<UNIQUE_PORTAL_TO_SERVICE_GRANT>
MCP_BUILD_SHA=<FULL_RELEASE_SOURCE_SHA>
```

The broker sends:

```text
X-MADPANDA-PORTAL-GRANT: <SERVICE_GRANT>
X-Google-Client-Id: <OWNER_OAUTH_CLIENT_ID>
X-Google-Client-Secret: <OWNER_OAUTH_CLIENT_SECRET>
X-Google-Refresh-Token: <OWNER_OAUTH_REFRESH_TOKEN>
X-Google-Maps-Api-Key: <OPTIONAL_OWNER_MAPS_KEY>
```

Use deployment-specific placeholders for the locked MCP URL, health URL,
container name, reverse proxy, and secret-store references. Public source does
not contain a production topology. See
[the Portal integration guide](docs/portal-compat.md).

## Production posture

1. Keep `/mcp` behind TLS or loopback/private networking.
2. Preserve the mode-specific service-auth header at the proxy.
3. Block direct container access from untrusted networks.
4. Store `.env` with restrictive permissions and never put provider
   credentials in it.
5. Apply connection, request-size, and rate limits at the edge.
6. Keep request logging off unless a sanitized diagnostic window requires it.
7. Verify health, unauthorized rejection, exact tool counts, local navigation,
   and missing-BYOK rejection before any provider smoke.
8. Use a dedicated test account and explicit approval before safe provider
   reads; do not begin with sends, writes, deletions, or Maps calls.

No immutable container digest is claimed before a release publishes one. When
a release includes an image, use the digest recorded in that GitHub Release
rather than a mutable tag.

Full upgrade, rollback, and troubleshooting steps are in the
[operator runbook](docs/operator-runbook.md).

## Run and verify from source

```sh
uv sync --frozen --group dev
uv run python -m compileall -q fastmcp scripts
uv run pytest -q -p no:cacheprovider
uv run ruff check .
uv run ruff format --check .
uv run isort --check-only fastmcp scripts
uv run mypy
uv run python scripts/check_source_safety.py
```

The verification suite uses synthetic credentials and must not contact Google.

## Documentation

- [Complete tool catalog](docs/tool-catalog.md)
- [Compatibility matrix](docs/compatibility-matrix.md)
- [Endpoint coverage](docs/endpoint-coverage.md)
- [ToolManifest contract](docs/tool-manifest.md)
- [Security model](docs/security-model.md)
- [Deployment and operator runbook](docs/operator-runbook.md)
- [Optional Portal integration](docs/portal-compat.md)
- [Source and release provenance](docs/provenance.md)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [Support](SUPPORT.md)
- [Security policy](SECURITY.md)

## License and trademarks

The code is available under the [MIT License](LICENSE). See [NOTICE](NOTICE)
for third-party and trademark notices.

Google and related product names are trademarks of their respective owners.
This independent project is not an official Google product, distribution,
partnership, or endorsement.
