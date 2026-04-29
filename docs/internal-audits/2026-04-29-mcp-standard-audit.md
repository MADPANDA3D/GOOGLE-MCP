# Google MCP Standard Audit

Date: 2026-04-29  
Auditor: Codex using local `mcp-builder` standard  
Verdict: Source repaired for MAD MCP/OpenAI readiness gaps; live deployment is blocked until the production portal grant token is configured.

## Classification

- Classification: live MCP
- Framework: Python FastMCP
- Transport: streamable HTTP at `/mcp`
- Hosted endpoint: `https://google-mcp.madpanda3d.com/mcp`
- Portal service: `google` / `GOOGLE-MCP`
- Portal-visible tool count before source changes: 55
- Local source tool count after source changes: 59

## Baseline Captured

- `git status`: branch `main` ahead of `origin/main` by 1 commit; untracked `.whoami/` and `HANDOVER.md` were present before this audit.
- Docker: container `google-mcp` running from image `fastmcp-google-mcp`, attached to `mcp-network`, host port `8086` exposed.
- Docker restart policy before source changes: `no`.
- `/health` before source changes: `ok=true`, `tool_count=55`, version `3a8d1b9c8950f8bb7165bd8217730f1e7d9db49a`.
- Portal service state: configured, native adapter, `toolCount=55`.
- Portal safe smoke: `mcp_health_check(run_checks=false, warm_all=false)` returned `ok=true`.
- Open portal tickets for this service:
  - `TKT-000004`: `calendar_list_events` did not honor single-day `timeMin`/`timeMax` style arguments.
  - `TKT-000005`: `gmail_batch_get_metadata` rejected common camelCase `messageIds`.

## Blocking Failures Found

- Missing `X-MADPANDA-PORTAL-GRANT` enforcement and missing `MCP_PORTAL_GRANT_TOKEN` documentation.
- Docker Compose did not use `restart: unless-stopped`.
- Tool descriptors had no OpenAI annotations.
- Tool schemas had generated titles but no parameter descriptions.
- Large 55-tool surface had no token-fit navigation layer.
- No endpoint coverage matrix existed.
- Wrapper did not normalize common camelCase arguments before FastMCP validation.

## Repairs Made In Source

- Added fail-closed portal grant validation for `/mcp` using `MCP_PORTAL_GRANT_TOKEN` and `X-MADPANDA-PORTAL-GRANT`.
- Added health configuration readiness fields without returning secret values.
- Added generic camelCase-to-snake_case tool argument normalization for exact schema matches.
- Added navigation tools:
  - `google_mcp_welcome`
  - `google_mcp_list_capabilities`
  - `google_mcp_get_endpoint_coverage`
  - `google_mcp_get_tool_usage`
- Added OpenAI tool annotations for every tool through the FastMCP tool registry.
- Added parameter descriptions and useful enums where values are constrained.
- Added `restart: unless-stopped` to Compose.
- Added `docs/endpoint-coverage.md` with official Google REST discovery parity.
- Updated README and `.env.example` with portal grant and navigation guidance.

## Verification

- Local syntax: `.venv/bin/python -m py_compile fastmcp/google_mcp_server.py fastmcp/google_auth_local.py`
- Local lint: `.venv/bin/python -m ruff check fastmcp` -> passed.
- Local tests: `.venv/bin/python -m pytest fastmcp/tests -q` -> 12 passed.
- Docker build: `sudo -n docker-compose -f fastmcp/docker-compose.yaml build google-mcp` -> passed.
- Isolated Docker smoke: audit container on port `18086` returned `/health` with `tool_count=59`; missing portal grant returned 401; valid portal grant plus BYOK headers returned 59 tools with navigation and annotations.
- Local metadata probe: source now reports 59 tools and OpenAI annotations on sampled read, destructive, and navigation tools.
- Portal pre-deploy smoke: current live portal still reports 55 tools and `mcp_health_check(run_checks=false)` passes.

## Remaining Risks

- Production `.env` did not contain `MCP_PORTAL_GRANT_TOKEN` at audit time, and live container env did not expose that variable. Do not restart the live container with the new source until the broker and container share the same grant token.
- Portal-visible count remains 55 until deployment; expected post-deploy count is 59.
- Current OAuth scope set does not cover many Gmail settings/CSE/admin-style endpoints.
- `google_raw_request` remains intentionally high-risk. It is annotated destructive and should stay an audit/dev escape hatch, not a default agent workflow.
- Host port `8086` remains published for backward compatibility with direct/n8n usage, even though portal access should be the normal agent path.

## Recommended Commit Units

1. Runtime/auth/tool descriptor repair.
2. Endpoint coverage and audit docs.
3. Session handover/memory update.
