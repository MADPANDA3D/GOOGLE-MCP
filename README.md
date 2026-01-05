# Google MCP (FastMCP)

Unified Google Workspace MCP server for Drive, Docs, Sheets, and Slides.

## What you get

- FastMCP server in Python
- OAuth token bootstrap script for local browser login
- Curated tools for common Drive/Docs/Sheets/Slides operations
- `google_raw_request` passthrough for any Google API endpoint

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
3. Configure the OAuth consent screen.
4. Create an OAuth client ID (Desktop app) and download `credentials.json`.

Place it at:

```
fastmcp/.google/credentials.json
```

### 2) Generate token locally

Run this on your local machine (the one with a browser):

```bash
export GOOGLE_SCOPES="https://www.googleapis.com/auth/drive https://www.googleapis.com/auth/documents https://www.googleapis.com/auth/spreadsheets https://www.googleapis.com/auth/presentations https://www.googleapis.com/auth/gmail.modify https://www.googleapis.com/auth/gmail.send https://www.googleapis.com/auth/calendar"
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
GOOGLE_CREDENTIALS_PATH=fastmcp/.google/credentials.json
GOOGLE_TOKEN_PATH=fastmcp/.google/token.json
GOOGLE_SCOPES=... (same as above)
```

If you add or change scopes later, rerun `google_auth_local.py` and copy the new
`token.json` to the VPS.

### 4) Run the server

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r fastmcp/requirements.txt
python3 fastmcp/google_mcp_server.py
```

Or with Docker:

```bash
docker compose -f fastmcp/docker-compose.yaml up --build
```

## Connect to n8n

If n8n runs in Docker on the same host, make sure the Google MCP container is
on the `npm_default` network. The provided `fastmcp/docker-compose.yaml` already
attaches it there.

In n8n:

1. Add an MCP client node (search for "MCP").  
2. Set the server URL to `http://google-mcp:8086/mcp`.  

If you are connecting from outside Docker, use:

```
http://<vps-ip>:8086/mcp
```

### VPS notes (Ubuntu/Debian)

If you see `externally-managed-environment`, install venv support first:

```bash
apt-get update && apt-get install -y python3.12-venv
```

Then create the virtual environment as shown above.

## Tools (curated)

- `drive_list_files`
- `drive_search_files`
- `drive_batch_get_metadata`
- `drive_get_file`
- `drive_create_folder`
- `drive_upload_file`
- `drive_download_file`
- `drive_empty_trash`
- `drive_delete_file`
- `drive_purge_trash`
- `docs_create_document`
- `docs_get_document`
- `docs_insert_text`
- `docs_replace_text`
- `sheets_create_spreadsheet`
- `sheets_get_spreadsheet`
- `sheets_get_values`
- `sheets_batch_get_values`
- `sheets_update_values`
- `slides_create_presentation`
- `slides_get_presentation`
- `slides_replace_text`
- `gmail_list_labels`
- `gmail_create_label`
- `gmail_delete_label`
- `gmail_list_messages`
- `gmail_search_messages`
- `gmail_get_message`
- `gmail_get_message_headers`
- `gmail_get_message_body`
- `gmail_batch_get_metadata`
- `gmail_list_threads`
- `gmail_get_thread`
- `gmail_send_message`
- `gmail_send_raw_message`
- `gmail_create_draft`
- `gmail_send_draft`
- `gmail_modify_message_labels`
- `gmail_trash_message`
- `gmail_untrash_message`
- `gmail_delete_message`
- `calendar_list_calendars`
- `calendar_get_calendar`
- `calendar_create_calendar`
- `calendar_delete_calendar`
- `calendar_list_events`
- `calendar_search_events`
- `calendar_batch_get_events`
- `calendar_get_event`
- `calendar_create_event`
- `calendar_update_event`
- `calendar_delete_event`
- `calendar_quick_add`
- `google_raw_request` (passthrough to any Google API endpoint)
- `mcp_health_check`

## Pagination

List/search tools accept `page_token` and return `nextPageToken` in the response (also echoed as `meta.next_page_token`).

## Performance tips

- Most `get` and list tools accept `fields` for partial responses.
- `gmail_get_message` defaults to metadata; use `gmail_get_message_body` for content.
- `drive_download_file` returns a `download_url` by default; set `include_content=true` or `return_mode="base64"` to include base64 content (bounded by `MCP_MAX_DOWNLOAD_BYTES`).
- Use `mcp_health_check(run_checks=true, warm_all=true)` to validate auth/scopes and warm caches; provide `doc_id`, `sheet_id`, `slide_id` for deeper checks.
- Response meta includes `elapsed_ms`, `bytes_in`, `bytes_out`, and `serialization_ms` for performance tuning.

## Raw request example

Use this tool to hit any Google API endpoint without adding a new wrapper:

```
method: "GET"
url: "/drive/v3/files"
params: {"pageSize": 10}
```

## Notes

- The server uses the refresh token in `token.json` to auto-refresh access tokens.
- If the token expires without a refresh token, rerun `google_auth_local.py` locally.
