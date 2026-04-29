# MEMORY.md — Project Memory
> Filled in by the agent during onboarding and updated every session.
> Append-only — never delete entries. Mark outdated info with [SUPERSEDED — DATE].

---

## Project Identity

- **Project name:** Google MCP
- **What it does (one sentence):** Unified Google Workspace MCP server for Drive, Docs, Sheets, Slides, Gmail, and Calendar workflows.
- **Type:** [ ] Client deliverable  [ ] Internal tool  [x] Platform component  [x] MCP server
- **Client / owner:** MADPANDA3D
- **Current phase:** Live maintenance on VPS; `.whoami` stack seeded on 2026-04-23 so MCP Command can dispatch remediation safely.

## Tech Stack

- **Frontend:** None
- **Backend:** Python 3.10+ + FastMCP
- **Database:** None local
- **Package manager:** pip-style requirements
- **Key libraries:** `mcp`, `uvicorn`, `google-api-python-client`, `google-auth`, `google-auth-oauthlib`, `google-auth-httplib2`
- **Repo path:** `/home/services/google-mcp/fastmcp`

## Infrastructure & Deployment

- **Deployment target:** VPS Docker / hosted MCP endpoint
- **Service/container:** `google-mcp`
- **Working directory:** `/home/services/google-mcp/fastmcp`
- **Hosted endpoint:** `https://google-mcp.madpanda3d.com/mcp`
- **Auth model:** BYOK Google OAuth headers and local token bootstrap

## Key Facts

- Covers Drive, Docs, Sheets, Slides, Gmail, and Calendar.
- Hosted mode requires Google client ID, client secret, and refresh token headers on every request.
- Local OAuth bootstrap uses `fastmcp/.google/credentials.json` and `token.json`.

## Session Notes

### 2026-04-23 — Initial stack seed
- Seeded this `.whoami` stack so MCP Command can treat the service as mapped and present.
- The current live portal has Google tickets around `calendar_list_events` and `gmail_batch_get_metadata`.

### 2026-04-29 — MAD MCP/OpenAI readiness audit
- Classified this repo as a live Python FastMCP service registered in MAD MCP Portal as service ID `google`.
- Baseline before deployment: live portal and `/health` report 55 tools; source now reports 59 tools after adding `google_mcp_welcome`, `google_mcp_list_capabilities`, `google_mcp_get_endpoint_coverage`, and `google_mcp_get_tool_usage`.
- Source now enforces `X-MADPANDA-PORTAL-GRANT` on `/mcp` via `MCP_PORTAL_GRANT_TOKEN`, but live production env did not have `MCP_PORTAL_GRANT_TOKEN` configured during the audit. Do not restart the live container until portal broker and service env share the same token.
- Added source-level camelCase-to-snake_case normalization for exact schema matches, covering the known portal tickets for `calendar_list_events` timeMin/timeMax and `gmail_batch_get_metadata` messageIds.
- Added OpenAI tool annotations and parameter descriptions to all registered tools through the FastMCP registry; local tests verify annotations and schema descriptions.
- Added `docs/endpoint-coverage.md` and `docs/internal-audits/2026-04-29-mcp-standard-audit.md`.
- Validation passed locally: ruff, py_compile, pytest (12 tests), Docker build, and isolated Docker smoke on port 18086.
- Portal tickets `TKT-000004` and `TKT-000005` were commented and moved to `in_progress`; mark resolved only after live deploy and portal smoke.
