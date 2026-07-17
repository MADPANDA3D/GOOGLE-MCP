# Google MCP (FastMCP)

Unified Google Workspace MCP server for Drive, Docs, Sheets, and Slides.

## What you get

- FastMCP server in Python
- OAuth token bootstrap script for local browser login
- Curated tools for common Drive/Docs/Sheets/Slides operations
- `google_raw_request` advanced/debug passthrough for Google API endpoints

## Hosted MCP (Strict BYOK)

Canonical hosted endpoint:

```
https://google-mcp.madpanda3d.com/mcp
```

Deprecated endpoint:

```
https://google-mcp.madpanda3d.com/mcp/
```

`/mcp/` returns `410 Gone` with a migration message.

Required request headers for hosted mode:

- MCP transport:
  - `Content-Type: application/json` (POST)
  - `Accept: application/json` (POST)
  - `Accept: text/event-stream` (GET stream)
- MAD MCP Portal access gate:
  - `X-MADPANDA-PORTAL-GRANT`
- BYOK credential headers (required for protocol discovery and provider tools):
  - `X-Google-Client-Id`
  - `X-Google-Client-Secret`
  - `X-Google-Refresh-Token`

The Portal grant is validated before any MCP JSON body is parsed. The five local
navigation tools `check_configuration`, `list_capabilities`,
`get_endpoint_coverage`, `get_tool_usage`, and `find_tools` require only the
valid Portal grant; they never contact Google. `initialize`, `tools/list`, all
legacy navigation tools, and every provider operation still require all Google
BYOK headers.

## Setup

### 1) Create OAuth credentials

1. Create a Google Cloud project.
2. Enable these APIs:
   - Google Drive API
   - Google Docs API
   - Google Sheets API
   - Google Slides API
   - Gmail API
   - Google Calendar API
   - YouTube Data API
   - Google Analytics Data API
   - Search Console API
   - Google Business Profile APIs
   - Content API for Shopping / Merchant Center APIs
   - AdSense Management API
   - Google Maps Platform APIs if Maps tools are enabled
3. Configure the OAuth consent screen.
4. Create an OAuth client ID (Desktop app) and download `credentials.json`.

Place it at:

```
fastmcp/.google/credentials.json
```

### 2) Generate token locally

Run this on your local machine (the one with a browser):

```bash
export GOOGLE_SCOPES="https://www.googleapis.com/auth/drive https://www.googleapis.com/auth/documents https://www.googleapis.com/auth/spreadsheets https://www.googleapis.com/auth/presentations https://www.googleapis.com/auth/gmail.modify https://www.googleapis.com/auth/gmail.send https://www.googleapis.com/auth/calendar https://www.googleapis.com/auth/youtube.readonly https://www.googleapis.com/auth/analytics.readonly https://www.googleapis.com/auth/webmasters.readonly https://www.googleapis.com/auth/business.manage https://www.googleapis.com/auth/content https://www.googleapis.com/auth/adsense.readonly"
python fastmcp/google_auth_local.py \
  --credentials fastmcp/.google/credentials.json \
  --token fastmcp/.google/token.json \
  --scopes "$GOOGLE_SCOPES"
```

This creates `fastmcp/.google/token.json` with a refresh token.

Copy the token to your VPS:

```bash
scp fastmcp/.google/token.json user@your-vps:/root/google-mcp/fastmcp/.google/token.json
```

### 3) Configure env

Edit `fastmcp/.env` if you want to override defaults:

```
MCP_HTTP_PORT=8086
MCP_BIND_ADDRESS=0.0.0.0
MCP_BUILD_SHA=<full source commit SHA>
MCP_PORTAL_GRANT_TOKEN=...
MCP_PORTAL_GRANT_HEADER=x-madpanda-portal-grant
GOOGLE_CREDENTIALS_PATH=fastmcp/.google/credentials.json
GOOGLE_TOKEN_PATH=fastmcp/.google/token.json
GOOGLE_SCOPES=... (same as above)
MCP_RAW_STRICT=true
MCP_ALLOW_REQUEST_OVERRIDES=true
MCP_REQUIRE_REQUEST_GOOGLE_CLIENT_ID=true
MCP_REQUIRE_REQUEST_GOOGLE_CLIENT_SECRET=true
MCP_REQUIRE_REQUEST_GOOGLE_REFRESH_TOKEN=true
MCP_DISABLE_DEFAULT_GOOGLE_FALLBACK=true
MCP_GOOGLE_CLIENT_ID_HEADER=x-google-client-id
MCP_GOOGLE_CLIENT_SECRET_HEADER=x-google-client-secret
MCP_GOOGLE_REFRESH_TOKEN_HEADER=x-google-refresh-token
MCP_GOOGLE_MAPS_API_KEY_HEADER=x-google-maps-api-key
GOOGLE_MAPS_API_KEY=
MCP_BYOK_CLIENT_CACHE_SIZE=256
MCP_BYOK_CLIENT_CACHE_TTL_SECONDS=900
```

If you add or change scopes later, rerun `google_auth_local.py` and copy the new
`token.json` to the VPS.

### 4) Run the server

If Python dependencies are already installed on the host, run:

```bash
python3 fastmcp/google_mcp_server.py
```

For clean local verification without a persistent venv, use Docker as shown below.

Or with Docker:

```bash
docker compose -f fastmcp/docker-compose.yaml up --build
```

## Connect to n8n

If n8n runs in Docker on the same host, make sure the Google MCP container is
on `mcp-network`. The provided `fastmcp/docker-compose.yaml` attaches the
container to that external network and uses `restart: unless-stopped`.

In n8n:

1. Add an MCP client node (search for "MCP").  
2. Set the server URL to `http://google-mcp:8086/mcp`.
3. Set server transport to `HTTP streamable`.
4. Set auth type to `Multiple Headers Auth`.
5. Add:
   - `X-MADPANDA-PORTAL-GRANT`
   - `X-Google-Client-Id`
   - `X-Google-Client-Secret`
   - `X-Google-Refresh-Token`

External callers use the canonical HTTPS endpoint. The hardened production
Compose service exposes port 8086 only inside `mcp-network`; it no longer binds
that port on the VPS host.

### VPS notes (Ubuntu/Debian)

If you see `externally-managed-environment`, install venv support first:

```bash
apt-get update && apt-get install -y python3.12-venv
```

Then create the virtual environment as shown above.

## Tools (curated)

The live catalog contains 151 native tools: 144 agent-ready tools, five legacy
compatibility tools, and two hidden internal/advanced tools. Start with the five
standard local navigation tools:

- `check_configuration`
- `list_capabilities`
- `get_endpoint_coverage`
- `get_tool_usage`
- `find_tools`

Provider categories include Drive, Docs, Sheets, Slides, Gmail, Calendar,
YouTube, Analytics, Search Console, Google Business Profile, Maps Platform,
Merchant/Shopping, and AdSense. Use the navigation tools instead of dumping the
raw catalog into an agent context.

## Agent navigation

Start with `find_tools` for a task-ranked shortlist, then call
`get_tool_usage` for the complete lossless descriptor. Call
`list_capabilities(include_descriptors=true)` only when publishing or auditing
the full provider-owned ToolManifest. `get_endpoint_coverage` reports REST
coverage, and `check_configuration` reports setup without exposing credentials.

Catalog version `google-2026.07.17.1` classifies 144 tools as `agent_ready`, five
compatibility tools as `legacy`, and two tools as `hidden`: the open-ended
`google_raw_request` plus the Portal-only Gmail signature preflight. Legacy and
hidden tools remain available through advanced broker
profiles but are not returned by default discovery. Every descriptor includes
complete input/output schemas, explicit risk/idempotency/open-world annotations,
documentation and navigation metadata, a deterministic descriptor hash, and an
out-of-band confirmation phrase for destructive execution.

`/health` preserves `ok`, `service`, `version`, `tool_count`, `tools.total`, and
configuration fields while also reporting status, full build SHA, catalog
version, descriptor hash, raw/exposed/agent-ready/documented counts, and safe
configuration readiness.

## Pagination

List/search tools accept `page_token` and return `nextPageToken` in the response (also echoed as `meta.next_page_token`).

Agents should prefer `meta.next_page_token` for paging; `data.nextPageToken` remains for compatibility.

## Gmail signatures

Normal Gmail sends and draft create/update operations resolve the selected
Gmail send-as identity and append its configured Gmail signature exactly once.
Hosted signature images are preserved in the HTML alternative, with a text
fallback for plain-text clients. Raw MIME and existing-draft sends are preserved
byte-for-byte and fail before delivery when the current configured signature is
missing. If the selected Gmail identity has no signature configured, the MCP
refuses to create or send unsigned email.

Portal previews bind the current signature fingerprint into the approved call.
If the signature changes before execution, the call fails before provider
mutation and must be previewed again. Existing-draft sends also bind the draft
MIME fingerprint so edits after preview invalidate approval.

## Large Drive uploads

For small files, `drive_upload_file` can still upload text or base64 content
directly in the MCP request. For files too large for the portal JSON request
body, call:

```text
drive_upload_file(
  name="report.pdf",
  content="",
  mime_type="application/pdf",
  upload_mode="resumable",
  file_size=23600000
)
```

The tool starts a Google Drive resumable upload session and returns an
`upload_url`. Upload the file bytes to that URL with `PUT`, `Content-Type`, and
`Content-Length`; chunked uploads should use chunk sizes that are multiples of
256 KiB except for the final chunk.

## Performance tips

- Most `get` and list tools accept `fields` for partial responses.
- `gmail_get_message` defaults to metadata; use `gmail_get_message_body` for content.
- `drive_download_file` returns a `download_url` by default; set `include_content=true` or `return_mode="base64"` to include base64 content (bounded by `MCP_MAX_DOWNLOAD_BYTES`).
- Use `mcp_health_check(run_checks=true, warm_all=true)` to validate auth/scopes and warm caches; provide `doc_id`, `sheet_id`, `slide_id` for deeper checks.
- Response meta includes `elapsed_ms`, `bytes_in`, `bytes_out`, `serialization_ms`, `request_id`, and `server_version` for performance tuning.
- Response meta includes `server_instance_id` and `server_uptime_ms` to confirm caching across calls.
- If a request parameter is malformed, tools may add `meta.warnings`; set `MCP_STRICT_PARAMS=true` to turn these into errors.

## Recommended defaults

- Use `fields` for Docs/Sheets/Slides to keep payloads small.
- Prefer `drive_download_file(return_mode="url")` unless you explicitly need file bytes.
- Use Gmail metadata tools unless you need raw MIME.
- For large inbox cleanup, start with `gmail_mailbox_overview`, then
  `gmail_sender_clusters`, then `gmail_cleanup_plan`. Apply changes only with
  `gmail_apply_cleanup_plan(dry_run=false, confirm=true)` after reviewing the
  batch summary.
- Call `mcp_health_check(run_checks=true, warm_all=true)` after restarts.
- Keep `MCP_WORKERS=1` if you want caching to persist across calls.
- `gmail_list_labels` defaults to minimal output; set `include_visibility=true` or `fields` for full label data.

## Raw request example

Use this tool to hit any Google API endpoint without adding a new wrapper:

```
method: "GET"
url: "/drive/v3/files"
params: {"pageSize": 10}
```

Note: `google_raw_request` is powerful but easy to misuse. Prefer the curated tools unless you need a specific endpoint. If you enable `MCP_RAW_STRICT=true`, requests must target a Google API host (or use a relative path); invalid hosts are rejected with a clear error.

## Live certification harness

Read-only certification:

```bash
python3 fastmcp/live_workspace_certification.py --env-file fastmcp/.env
```

Disposable Workspace write certification:

```bash
python3 fastmcp/live_workspace_certification.py --env-file fastmcp/.env --include-writes
```

Email sends are never automatic. To test sending, first run the disposable write
flow and pass an explicit recipient with `--send-test-to`.

## Notes

- In strict hosted mode (`MCP_DISABLE_DEFAULT_GOOGLE_FALLBACK=true`) the server requires BYOK headers and does not use server `token.json` for `/mcp` requests.
- For local/dev fallback mode, the server can still use the refresh token in `token.json` to auto-refresh access tokens.
- If a refresh token expires or is revoked, rerun `google_auth_local.py` locally to generate a new token.
