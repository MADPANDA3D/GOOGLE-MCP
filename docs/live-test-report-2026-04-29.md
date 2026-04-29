# Google MCP Live Test Report - 2026-04-29

## Status

- Classification: live MCP, currently held in maintenance for expanded Google surface review.
- Runtime: Dockerized FastMCP streamable HTTP service on `mcp-network`.
- Restart policy: `unless-stopped`.
- Health endpoint: passed at `http://127.0.0.1:8086/health`.
- Source tool count: 145.
- Portal-visible tool count: 145.
- Portal state: configured; maintenance state is tracked in the Portal database.

## Gates

- Ruff: passed.
- Python compile: passed for `google_mcp_server.py`, `google_auth_local.py`, and `live_workspace_certification.py`.
- Pytest: passed, 17 tests.
- Docker rebuild/restart: passed for `google-mcp`.
- Broker catalog listing: passed, 145 total tools across both catalog pages.
- Portal smoke: passed for `google_mcp_welcome`, `mcp_health_check`, and `gmail_mailbox_overview`.

## Live Workspace Certification

Certification command:

```bash
python3 fastmcp/live_workspace_certification.py --env-file fastmcp/.env --include-writes --report docs/live-test-report-2026-04-29.json
```

Overall result: passed.

- Prefix: `MAD_AUDIT_DELETE_ME_20260429_1777445200`
- Send test: skipped; no operator-provided recipient was approved.
- Read-only flows passed: welcome, capability navigation, health, Gmail labels, Gmail overview, Gmail sender clustering, Calendar list/events, Drive list, and raw request guardrail.
- Disposable write flows passed: Docs create/get, Sheets create/update, Slides create/get, Gmail label create/delete, Gmail draft create/delete, Calendar create event/delete event/delete calendar, and Drive file deletes for created Docs/Sheets/Slides resources.

## Created And Cleaned

All resources below were created by the audit harness and cleaned up in the same run.

| Kind | ID |
| --- | --- |
| document | `1WTcnBLq0SRrVFJNEPOWGxwzten_bc5bCgcTCehr4448` |
| spreadsheet | `15Ls_jwsc5dmreDu8YX6ldRK-erRH_P1yvZD3J-oTS24` |
| presentation | `1kGPKGHqyTn_WZu_jLGt9WtUSnRQ50r_5NP-NVYQ8j3I` |
| gmail_label | `Label_8` |
| gmail_draft | `r-1728187594242381342` |
| calendar | `c_789b5fd2f8593b8f646d72471ce194104ea62eee0a5d89f36bed6a411cec87ac@group.calendar.google.com` |
| calendar_event | `b0r9kj6m8mchcbcnob0k9u02ho` |

Cleanup summary:

- Calendar deleted: true.
- Calendar event deleted: true.
- Gmail draft deleted: true.
- Gmail label deleted: true.
- Drive-backed files deleted: 3.

## Ticket Verification

- `TKT-000004` Calendar time bounds: resolved, commented after live certification.
- `TKT-000005` Gmail `messageIds` alias: resolved, commented after regression coverage and live certification.
- Gmail regression coverage: local test confirms `messageIds` maps to `message_ids`; previous portal smoke used a real Gmail ID. The exact ticket payload used fake IDs and was not replayed literally.

## Business API Smoke

- YouTube Data API: source tool and portal route are wired. Portal smoke returned `blocked_scope` because the current OAuth token has not been re-consented for `youtube.readonly`.
- Google Maps Platform: source tool and portal route are wired. Portal smoke returned `blocked_scope` because no `x-google-maps-api-key` or `GOOGLE_MAPS_API_KEY` is configured.
- Analytics, Search Console, Business Profile, Merchant, and AdSense: source tools are present and cataloged; live calls are pending account/scope/API fixture confirmation.
- Google Ads: intentionally excluded for this wave pending developer token, customer ID model, and account prerequisites.

## Tool Surface Classification

- All 145 tools are listed by the broker and passed local schema/annotation audit.
- Navigation tools are present: welcome, capabilities, endpoint coverage, per-tool usage, health, and raw request guardrail.
- Gmail cleanup tools are present and token-fit: overview, sender clusters, cleanup plan, apply cleanup plan, batch modify, and batch delete.
- Destructive and irreversible tools remain approval-gated by schema/runtime confirmation flags and were not executed against non-disposable data.

## Remaining Blockers

blocked_fixture:

- No safe fixture calendar/document/sheet/slide exists yet for every advanced update wrapper; certification currently creates and deletes disposable fixtures for core flows.
- No safe real message/thread fixture was approved for thread trash/untrash, attachment fetch, or history replay tests.

blocked_scope:

- Current OAuth token needs re-consent for YouTube, Analytics, Search Console, Business Profile, Merchant, and AdSense scopes before those business APIs can be live-certified.
- Maps requires a configured Maps Platform API key through `x-google-maps-api-key` or `GOOGLE_MAPS_API_KEY`.
- Ads requires developer token and customer/account prerequisites before implementation.

blocked_approval:

- Real Gmail send was not executed.
- Bulk inbox cleanup mutations were not executed.
- Production sharing, ACL, Calendar settings, Drive permissions, and destructive non-disposable operations were not executed.

fails_source:

- None from the final local gates or Workspace certification run.

fails_provider:

- None for certified Workspace flows.
- YouTube returned provider `403 insufficientPermissions`, classified as `blocked_scope`.

## Release Summary

Google MCP now exposes 145 portal-visible tools with expanded Workspace depth, Gmail scale cleanup workflows, navigation helpers, endpoint coverage, and first-wave business API tools for YouTube, Analytics, Search Console, Business Profile, Maps, Merchant, and AdSense. The runtime auth path now preserves existing refresh-token grants when scopes expand, preventing new optional API scopes from breaking existing Workspace access. Local gates and live disposable Workspace certification pass; business API live certification remains blocked on OAuth re-consent, Maps API key configuration, and provider account prerequisites.
