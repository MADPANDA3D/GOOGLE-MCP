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
3. Configure the OAuth consent screen.
4. Create an OAuth client ID (Desktop app) and download `credentials.json`.

Place it at:

```
fastmcp/.google/credentials.json
```

### 2) Generate token locally

Run this on your local machine (the one with a browser):

```bash
export GOOGLE_SCOPES="https://www.googleapis.com/auth/drive https://www.googleapis.com/auth/documents https://www.googleapis.com/auth/spreadsheets https://www.googleapis.com/auth/presentations"
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

### 4) Run the server

```bash
pip install -r fastmcp/requirements.txt
python fastmcp/google_mcp_server.py
```

Or with Docker:

```bash
docker compose -f fastmcp/docker-compose.yaml up --build
```

## Tools (curated)

- `drive_list_files`
- `drive_get_file`
- `drive_create_folder`
- `drive_upload_file`
- `drive_download_file`
- `docs_create_document`
- `docs_get_document`
- `docs_insert_text`
- `docs_replace_text`
- `sheets_create_spreadsheet`
- `sheets_get_spreadsheet`
- `sheets_get_values`
- `sheets_update_values`
- `slides_create_presentation`
- `slides_get_presentation`
- `slides_replace_text`
- `google_raw_request` (passthrough to any Google API endpoint)

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
