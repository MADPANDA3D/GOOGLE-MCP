# Google MCP Live Test Report - 2026-05-01

## Status

- Classification: live MCP, still in maintenance pending the remaining provider fixtures.
- Runtime: Dockerized Python FastMCP service on `mcp-network`.
- Restart policy: `unless-stopped`.
- Health endpoint: passed at `http://127.0.0.1:8086/health`.
- Deployed version: `c49d0af`.
- Source tool count: 145.
- Portal-visible tool count: 145.
- Portal ticket: `TKT-000019` remains `in_progress`.

## Gates

- Ruff: passed in a disposable Docker Python environment.
- Python compile: passed for `google_mcp_server.py` and `google_auth_local.py`.
- Pytest: passed, 24 tests.
- Docker rebuild/restart: passed for `google-mcp`.
- Portal broker health: passed.

## OAuth And Scope Retest

- Portal runtime Google OAuth scope pack: 16 scopes with `https://www.googleapis.com/auth/content`.
- Portal audit shows successful 16-scope reconnect at `2026-05-01T01:37:54Z`.
- Stored Google `refreshToken.updated_at` advanced to `2026-05-01T01:37:54Z`.
- Merchant Content API auth smoke now passes:
  - `google_raw_request GET https://shoppingcontent.googleapis.com/content/v2.1/accounts/authinfo`
  - Provider status: HTTP 200.
  - Response contains no Merchant account identifiers, so product listing needs a real Merchant Center ID fixture.

## Source Fixes During Retest

- `gmail_mailbox_overview` now accepts `max_labels` and returns `labels_total` / `labels_returned`.
- `gmail_sender_clusters` now accepts and obeys `sample_per_cluster` with a hard cap.
- `docs_create_document`, `sheets_create_spreadsheet`, and `slides_create_presentation` now request compact provider fields instead of returning full default provider objects.

## Broker Smoke Results

Passed:

- `mcp_health_check`
- `drive_list_files`
- `calendar_list_calendars`
- `gmail_mailbox_overview`
- `gmail_sender_clusters`
- `gmail_create_label` and `gmail_delete_label`
- `gmail_create_draft` and `gmail_delete_draft`
- `gmail_create_draft`, `gmail_send_draft` to approved `madpanda3d@gmail.com`, and sender-side `gmail_trash_message`
- `docs_create_document` and `drive_delete_file`
- `sheets_create_spreadsheet`, `sheets_update_values`, and `drive_delete_file`
- `slides_create_presentation` and `drive_delete_file`
- `calendar_create_calendar`, `calendar_create_event`, `calendar_delete_event`, and `calendar_delete_calendar`
- `maps_geocode`
- `youtube_list_channels`
- `analytics_get_metadata`
- `searchconsole_list_sites`
- `adsense_list_accounts`
- Merchant `accounts/authinfo` via `google_raw_request`

Failed / blocked:

- `business_profile_list_accounts`: provider returns 429 quota exhausted with `quota_limit_value: 0` for project `910769267358`.
- `merchant_list_products`: blocked by missing Merchant Center account ID; `accounts/authinfo` returns HTTP 200 but no account identifiers.
- `gmail_delete_message`: permanent message deletion returns insufficient scopes. Gmail permanent delete requires the broader `https://mail.google.com/` scope; this was not added during retest because it requires explicit approval and re-consent.

## Disposable Resources

Created and cleaned:

- Docs document `12w-DJa0mgWnorn_oFbxr5imG-jfo26EDqM9bTtSHSTY`
- Sheets spreadsheet `1-l72Jch2mc16mqWiHT3kdRIg50Cby6o0OloC-qJYv9c`
- Slides presentation `1xFK0MoKhp2yv7Y8bhU0gZ2DSNuNAYXrYHfa4h9UkYRk`
- Gmail label `Label_9`
- Gmail draft `r1121364490927717244`
- Calendar `c_42439af5e80ee03008d890b860be198e5b0e376baa60c30ca1175eddd659a462@group.calendar.google.com`
- Calendar event `d15sbnpf6cj4g36m41domiu19s`

Approved internal send:

- Draft `r-6776342189394057534` sent to `madpanda3d@gmail.com`.
- Sent message `19de142d11edac28` was moved to sender-side trash after verification. Permanent deletion was attempted only for this disposable sender-side copy and was blocked by missing Gmail full-mail scope.

## Remaining Blockers

blocked_fixture:

- Merchant account fixture is missing. `accounts/authinfo` passes but does not return account IDs, so curated product reads need an actual Merchant Center account ID available to `leolara@madpanda3d.com`.

blocked_scope:

- Gmail permanent delete tools need `https://mail.google.com/`; current scope pack supports send/modify/trash/archive but not permanent delete.

blocked_approval:

- Bulk inbox cleanup mutations, production sharing/ACL/permission changes, production Calendar settings changes, and destructive non-disposable operations were not run.

fails_source:

- None after commits `e384679` and `c49d0af`.

fails_provider:

- Business Profile Account Management quota remains zero for project `910769267358`.

## Release Summary

Google MCP now passes the post-reconnect broker certification set for Workspace, Gmail scale planning, Maps, YouTube, Analytics, Search Console, AdSense, and Merchant OAuth readiness. Two token-fit issues discovered during live testing were repaired at source: Gmail cleanup caps are enforced, and Workspace create tools now return compact creation metadata. The only remaining release blockers are provider-side Business Profile quota and a real Merchant Center account fixture.
