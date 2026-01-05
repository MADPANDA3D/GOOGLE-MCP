# Repository Guidelines

## Project Structure & Module Organization
- `fastmcp/google_mcp_server.py`: FastMCP server and all tool implementations.
- `fastmcp/google_auth_local.py`: local OAuth token bootstrap for browser login.
- `fastmcp/.google/`: credentials and token files (local only; do not commit).
- `fastmcp/requirements.txt`, `fastmcp/Dockerfile`, `fastmcp/docker-compose.yaml`: deps and container setup.
- `README.md`: setup, scopes, and usage notes.

## Build, Test, and Development Commands
- `python3 -m venv .venv` and `source .venv/bin/activate`: create/activate a virtualenv.
- `python3 -m pip install -r fastmcp/requirements.txt`: install Python dependencies.
- `python3 fastmcp/google_mcp_server.py`: run the server locally.
- `docker compose -f fastmcp/docker-compose.yaml up --build`: run in Docker.
- `python fastmcp/google_auth_local.py --credentials fastmcp/.google/credentials.json --token fastmcp/.google/token.json --scopes "$GOOGLE_SCOPES"`: generate a refresh token.

## Coding Style & Naming Conventions
- Python 3 with type hints, 4-space indentation, and PEP 8 spacing.
- Tool functions use `snake_case` and a service prefix (e.g., `drive_list_files`, `sheets_update_values`).
- Keep request/response payloads explicit and JSON-friendly; follow the patterns in `fastmcp/google_mcp_server.py`.

## Testing Guidelines
- No automated test suite currently. Validate changes with a manual smoke run (start the server and call a tool).
- If adding tests, prefer `pytest` and place them under `fastmcp/tests/` using `test_*.py`.

## Commit & Pull Request Guidelines
- Commit messages are short, imperative, and capitalized (e.g., "Add Gmail and Calendar tools").
- PRs should include a brief summary, test notes (even if "manual smoke"), and any config/env changes.
- Never commit secrets: keep `fastmcp/.google/` and `fastmcp/.env` local; update `fastmcp/.env.example` when adding settings.

## Security & Configuration Tips
- OAuth credentials/tokens live in `fastmcp/.google/`; re-run `google_auth_local.py` if scopes change.
- Keep `GOOGLE_SCOPES` consistent between token generation and server config.
