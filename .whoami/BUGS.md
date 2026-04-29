# BUGS.md — Project Bug Memory
> Append-only — never delete entries. Mark resolved bugs with [RESOLVED — DATE].
> Check this file before starting any task. The bug you're about to hit may already be documented.
> If this file is empty, that's fine — it fills up as the project grows.

---

## How to Use This File

Before writing any code, scan this file for patterns relevant to what you're about to do. If you encounter a new bug during a session, append it immediately — even if you fix it in the same session. The goal is a library of recurring failures and operational traps for this specific MCP.

---

## Bug Entry Format

```markdown
### [DATE] — Short description of the bug
**Status:** [OPEN] / [RESOLVED — DATE] / [WORKAROUND]
**Symptom:** What the developer or user sees
**Root cause:** Why it happens
**Fix:** What was done to resolve it
**Prevention:** How to avoid this in the future
```

---

## Bug Log

<!-- Agent: append new bugs below this line as they are discovered -->

### 2026-04-29 — CamelCase tool arguments bypassed intended Google filters
**Status:** [RESOLVED IN SOURCE — 2026-04-29; PENDING LIVE DEPLOY]
**Symptom:** Portal ticket `TKT-000004` showed `calendar_list_events` returning out-of-range events when called with `timeMin`, `timeMax`, and `calendarId`; `TKT-000005` showed `gmail_batch_get_metadata` rejecting `messageIds`.
**Root cause:** FastMCP schemas exposed snake_case parameters, but agents naturally used Google/JSON-style camelCase names. The wrapper did not normalize arguments before validation and dispatch.
**Fix:** Added source-level JSON-RPC argument normalization that maps camelCase to an existing snake_case schema key only when the canonical key exists and was not supplied.
**Prevention:** Keep wrapper alias tests for common Google-style arguments when adding new tools.

### 2026-04-29 — Live environment missing portal grant token
**Status:** [OPEN]
**Symptom:** New source correctly fails closed for `/mcp` when `MCP_PORTAL_GRANT_TOKEN` is missing, but the live container/env did not have that variable during audit.
**Root cause:** Older live service predated the MAD MCP Portal grant-token standard and only enforced Google BYOK headers.
**Fix:** Source and `.env.example` now require `MCP_PORTAL_GRANT_TOKEN`; live `.env` and portal broker config still need the same secret before restart.
**Prevention:** Check `grep -q '^MCP_PORTAL_GRANT_TOKEN=' fastmcp/.env` before rebuilding/restarting the live service.
