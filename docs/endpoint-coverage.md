# Google MCP Endpoint Coverage

Retrieved: 2026-04-29  
Service: Google MCP (`google-mcp`)  
Implementation: Python FastMCP, Google API client library plus bounded raw REST escape hatch

Official source set consulted during this audit:

- Google Drive API v3 REST reference and discovery document: <https://developers.google.com/workspace/drive/api/reference/rest/v3>, <https://www.googleapis.com/discovery/v1/apis/drive/v3/rest>
- Google Docs API v1 REST reference and discovery document: <https://developers.google.com/docs/api/reference/rest>, <https://docs.googleapis.com/$discovery/rest?version=v1>
- Google Sheets API v4 REST reference and discovery document: <https://developers.google.com/workspace/sheets/api/reference/rest/v4/spreadsheets>, <https://sheets.googleapis.com/$discovery/rest?version=v4>
- Google Slides API v1 REST reference and discovery document: <https://developers.google.com/workspace/slides/api/reference/rest>, <https://slides.googleapis.com/$discovery/rest?version=v1>
- Gmail API v1 REST reference and discovery document: <https://developers.google.com/gmail/api/reference/rest>, <https://gmail.googleapis.com/$discovery/rest?version=v1>
- Google Calendar API v3 REST reference and discovery document: <https://developers.google.com/workspace/calendar/api/v3/reference>, <https://www.googleapis.com/discovery/v1/apis/calendar/v3/rest>

Legend:

- `implemented`: covered by a curated MCP tool.
- `missing`: stable endpoint exists but no curated MCP tool currently wraps it.
- `intentionally_excluded`: excluded for safety, callback infrastructure, or legacy provider surface.
- `blocked_scope`: requires OAuth scopes or account privileges outside the current default hosted BYOK scope set.
- `raw_escape_hatch`: available through `google_raw_request`, but not counted as curated endpoint coverage.

## Summary

| API | Discovery methods | Curated status | Notes |
|---|---:|---|---|
| Drive v3 | 58 | partial | Core file list/get/create/upload/download/delete covered. Permissions, comments, shared drives, changes, revisions, labels, watches, and app/about resources remain uncovered as curated tools. |
| Docs v1 | 3 | partial | Documents create/get covered. `documents.batchUpdate` is covered for insert and replace text only, not every request type. |
| Sheets v4 | 17 | partial | Spreadsheet create/get and values get/batchGet/update covered. Append, clear, batch update, data filters, metadata, and sheet copy are missing. |
| Slides v1 | 5 | partial | Presentation create/get and text replacement through batchUpdate covered. Page get and thumbnail endpoints are missing. |
| Gmail v1 | 79 | partial | Labels/messages/threads/drafts common workflows covered. Settings, history, CSE, forwarding, delegates, send-as, filters, attachment get, batch operations, and watches remain uncovered or blocked by scope/safety. |
| Calendar v3 | 37 | partial | Calendars, calendar list, and event list/get/create/patch/delete/quickAdd covered. ACL, freebusy, settings, colors, watches, channels, event move/import/instances/update, and CalendarList mutation are missing or intentionally excluded. |

## Drive API v3

| Resource | Official methods | Status | Curated MCP coverage | Notes |
|---|---|---|---|---|
| `files` | `generateIds`, `create`, `generateCseToken`, `emptyTrash`, `update`, `copy`, `export`, `list`, `modifyLabels`, `delete`, `listLabels`, `watch`, `download`, `get` | partial | `drive_list_files`, `drive_search_files`, `drive_get_file`, `drive_batch_get_metadata`, `drive_create_folder`, `drive_upload_file`, `drive_download_file`, `drive_delete_file`, `drive_empty_trash`, `drive_purge_trash` | `drive_download_file` covers media/export URL/content flows, not the newer long-running `files.download` operation. General `files.update`, copy, labels, watches, CSE token, and generated IDs are missing curated wrappers. |
| `permissions` | `list`, `get`, `delete`, `create`, `update` | missing | `google_raw_request` only | Sharing changes are high-impact and need explicit tool design, scopes, and confirmation. |
| `comments` | `list`, `create`, `update`, `get`, `delete` | missing | `google_raw_request` only | Missing curated comment review workflows. |
| `replies` | `list`, `delete`, `get`, `create`, `update` | missing | `google_raw_request` only | Missing curated comment reply workflows. |
| `changes` | `getStartPageToken`, `list`, `watch` | missing | `google_raw_request` only | Watch requires webhook/channel lifecycle. |
| `channels` | `stop` | intentionally_excluded | `google_raw_request` only | Channel stop is safe only when paired with channel creation/tracking. |
| `drives` | `get`, `delete`, `create`, `update`, `hide`, `unhide`, `list` | missing | `google_raw_request` only | Shared drive administration requires additional risk gates. |
| `teamdrives` | `get`, `delete`, `create`, `update`, `list` | intentionally_excluded | `google_raw_request` only | Legacy Team Drives surface; prefer `drives` wrappers if added. |
| `revisions` | `update`, `list`, `get`, `delete` | missing | `google_raw_request` only | Revision delete/update are destructive and need explicit confirmation. |
| `apps` | `list`, `get` | missing | `google_raw_request` only | Low priority metadata surface. |
| `about` | `get` | missing | `mcp_health_check` uses `about.get` internally for health only | Add a read-only curated wrapper if agents need user/storage metadata. |
| `accessproposals` | `get`, `resolve`, `list` | missing | `google_raw_request` only | Access proposal resolution affects sharing state. |
| `approvals` | `get`, `list` | missing | `google_raw_request` only | Approval metadata not yet curated. |
| `operations` | `get` | missing | `google_raw_request` only | Needed only if long-running Drive operations are exposed. |

## Docs API v1

| Resource | Official methods | Status | Curated MCP coverage | Notes |
|---|---|---|---|---|
| `documents` | `create`, `get`, `batchUpdate` | partial | `docs_create_document`, `docs_get_document`, `docs_insert_text`, `docs_replace_text` | Endpoint is covered for common text insertion/replacement only. Full batchUpdate request parity is not implemented. |

## Sheets API v4

| Resource | Official methods | Status | Curated MCP coverage | Notes |
|---|---|---|---|---|
| `spreadsheets` | `create`, `batchUpdate`, `get`, `getByDataFilter` | partial | `sheets_create_spreadsheet`, `sheets_get_spreadsheet` | General `batchUpdate` and data-filter get are missing. |
| `spreadsheets.values` | `append`, `batchUpdate`, `batchGetByDataFilter`, `batchClearByDataFilter`, `clear`, `batchUpdateByDataFilter`, `get`, `update`, `batchGet`, `batchClear` | partial | `sheets_get_values`, `sheets_batch_get_values`, `sheets_update_values` | Append, clear, batch write, and data-filter value operations are missing curated wrappers. |
| `spreadsheets.developerMetadata` | `search`, `get` | missing | `google_raw_request` only | Missing metadata lookup/search wrappers. |
| `spreadsheets.sheets` | `copyTo` | missing | `google_raw_request` only | Missing sheet copy wrapper. |

## Slides API v1

| Resource | Official methods | Status | Curated MCP coverage | Notes |
|---|---|---|---|---|
| `presentations` | `get`, `batchUpdate`, `create` | partial | `slides_create_presentation`, `slides_get_presentation`, `slides_replace_text` | Endpoint covered for create/get/text replacement only. Full batchUpdate request parity is not implemented. |
| `presentations.pages` | `get`, `getThumbnail` | missing | `google_raw_request` only | Add read-only page and thumbnail wrappers for visual review workflows. |

## Gmail API v1

| Resource | Official methods | Status | Curated MCP coverage | Notes |
|---|---|---|---|---|
| `users` | `getProfile`, `watch`, `stop` | partial | `mcp_health_check` uses `getProfile` internally | Watch/stop require Pub/Sub/channel lifecycle and are intentionally excluded until callback infrastructure exists. |
| `users.labels` | `create`, `list`, `patch`, `update`, `delete`, `get` | partial | `gmail_list_labels`, `gmail_create_label`, `gmail_delete_label` | `get`, `patch`, and `update` are missing. |
| `users.messages` | `delete`, `get`, `send`, `batchDelete`, `modify`, `import`, `trash`, `batchModify`, `untrash`, `insert`, `list` | partial | `gmail_list_messages`, `gmail_search_messages`, `gmail_get_message`, `gmail_get_message_headers`, `gmail_get_message_body`, `gmail_batch_get_metadata`, `gmail_send_message`, `gmail_send_raw_message`, `gmail_modify_message_labels`, `gmail_trash_message`, `gmail_untrash_message`, `gmail_delete_message` | Batch delete/modify, import, and insert are missing. Sends require explicit user approval in live tests. |
| `users.messages.attachments` | `get` | missing | `google_raw_request` only | Add a bounded attachment download wrapper before exposing by default. |
| `users.threads` | `delete`, `get`, `trash`, `untrash`, `modify`, `list` | partial | `gmail_list_threads`, `gmail_get_thread` | Thread mutation/delete/trash/untrash are missing and require confirmation semantics. |
| `users.drafts` | `send`, `delete`, `get`, `create`, `list`, `update` | partial | `gmail_create_draft`, `gmail_send_draft` | Draft list/get/update/delete are missing. |
| `users.history` | `list` | blocked_scope | `google_raw_request` only | Requires history workflow design and startHistoryId management. |
| `users.settings` | `updateLanguage`, `getAutoForwarding`, `getVacation`, `getLanguage`, `updateAutoForwarding`, `getPop`, `updateImap`, `updatePop`, `updateVacation`, `getImap` | blocked_scope | `google_raw_request` only | Provider settings changes require explicit approval and additional Gmail settings scopes. |
| `users.settings.filters` | `list`, `create`, `get`, `delete` | blocked_scope | `google_raw_request` only | Mail automation mutations are intentionally excluded until scoped and gated. |
| `users.settings.forwardingAddresses` | `get`, `delete`, `list`, `create` | blocked_scope | `google_raw_request` only | Forwarding setup is high-risk and can require verification. |
| `users.settings.delegates` | `list`, `create`, `get`, `delete` | blocked_scope | `google_raw_request` only | Mailbox delegation requires admin/user approval. |
| `users.settings.sendAs` | `get`, `delete`, `verify`, `update`, `list`, `create`, `patch` | blocked_scope | `google_raw_request` only | Send-as identity changes require explicit approval and verification. |
| `users.settings.sendAs.smimeInfo` | `get`, `delete`, `setDefault`, `list`, `insert` | blocked_scope | `google_raw_request` only | S/MIME key material and identity configuration are out of current scope. |
| `users.settings.cse.identities` | `delete`, `get`, `create`, `list`, `patch` | blocked_scope | `google_raw_request` only | Client-side encryption settings are out of current hosted BYOK scope. |
| `users.settings.cse.keypairs` | `get`, `disable`, `obliterate`, `enable`, `create`, `list` | blocked_scope | `google_raw_request` only | CSE keypair operations are high-risk and require dedicated approval gates. |

## Calendar API v3

| Resource | Official methods | Status | Curated MCP coverage | Notes |
|---|---|---|---|---|
| `calendarList` | `delete`, `get`, `insert`, `update`, `list`, `watch`, `patch` | partial | `calendar_list_calendars` | CalendarList mutation and watch endpoints are missing. |
| `calendars` | `insert`, `clear`, `delete`, `get`, `update`, `patch` | partial | `calendar_create_calendar`, `calendar_delete_calendar`, `calendar_get_calendar` | Clear/update/patch are missing and require destructive/update gates. |
| `events` | `patch`, `quickAdd`, `import`, `instances`, `update`, `insert`, `move`, `watch`, `list`, `delete`, `get` | partial | `calendar_list_events`, `calendar_search_events`, `calendar_batch_get_events`, `calendar_get_event`, `calendar_create_event`, `calendar_update_event`, `calendar_delete_event`, `calendar_quick_add` | Import, instances, full update, move, and watch are missing. |
| `acl` | `update`, `insert`, `delete`, `get`, `patch`, `watch`, `list` | missing | `google_raw_request` only | Calendar sharing mutations require explicit permission gates. |
| `freebusy` | `query` | missing | `google_raw_request` only | Good candidate for next read-only wrapper. |
| `settings` | `watch`, `get`, `list` | missing | `google_raw_request` only | Read-only settings wrappers can be added; watch requires callback lifecycle. |
| `colors` | `get` | missing | `google_raw_request` only | Low-risk read-only wrapper candidate. |
| `channels` | `stop` | intentionally_excluded | `google_raw_request` only | Channel stop needs tracked watch/channel creation first. |
