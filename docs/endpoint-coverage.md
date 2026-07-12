# Google MCP Endpoint Coverage

Retrieved: 2026-04-29  
Service: Google MCP (`google-mcp`)  
Implementation: Python FastMCP, Google API client library plus bounded raw REST escape hatch

ToolManifest conformance verified: 2026-07-12. The provider-owned catalog maps
all 150 registered native tools to a versioned descriptor. Endpoint inventory
source retrieval remains 2026-04-29 because this hardening pass did not expand
or claim new provider endpoint coverage.

Official source set consulted during this audit:

- Google Drive API v3 REST reference and discovery document: <https://developers.google.com/workspace/drive/api/reference/rest/v3>, <https://www.googleapis.com/discovery/v1/apis/drive/v3/rest>
- Google Docs API v1 REST reference and discovery document: <https://developers.google.com/docs/api/reference/rest>, <https://docs.googleapis.com/$discovery/rest?version=v1>
- Google Sheets API v4 REST reference and discovery document: <https://developers.google.com/workspace/sheets/api/reference/rest/v4/spreadsheets>, <https://sheets.googleapis.com/$discovery/rest?version=v4>
- Google Slides API v1 REST reference and discovery document: <https://developers.google.com/workspace/slides/api/reference/rest>, <https://slides.googleapis.com/$discovery/rest?version=v1>
- Gmail API v1 REST reference and discovery document: <https://developers.google.com/gmail/api/reference/rest>, <https://gmail.googleapis.com/$discovery/rest?version=v1>
- Google Calendar API v3 REST reference and discovery document: <https://developers.google.com/workspace/calendar/api/v3/reference>, <https://www.googleapis.com/discovery/v1/apis/calendar/v3/rest>
- YouTube Data API v3 reference and discovery document: <https://developers.google.com/youtube/v3/docs>, <https://youtube.googleapis.com/$discovery/rest?version=v3>
- Google Analytics Data API reference and discovery document: <https://developers.google.com/analytics/devguides/reporting/data/v1>, <https://analyticsdata.googleapis.com/$discovery/rest?version=v1beta>
- Search Console API reference and discovery document: <https://developers.google.com/webmaster-tools/v1>, <https://searchconsole.googleapis.com/$discovery/rest?version=v1>
- Google Business Profile APIs: <https://developers.google.com/my-business>
- Google Maps Platform APIs: <https://developers.google.com/maps/documentation>
- Content API for Shopping / Merchant API docs: <https://developers.google.com/shopping-content>, <https://developers.google.com/merchant/api>
- AdSense Management API docs: <https://developers.google.com/adsense/management>

Legend:

- `implemented`: covered by a curated MCP tool.
- `missing`: stable endpoint exists but no curated MCP tool currently wraps it.
- `intentionally_excluded`: excluded for safety, callback infrastructure, or legacy provider surface.
- `blocked_scope`: requires OAuth scopes or account privileges outside the current default hosted BYOK scope set.
- `raw_escape_hatch`: available through `google_raw_request`, but not counted as curated endpoint coverage.

Agent discovery uses `find_tools`, full references use `get_tool_usage`, and
bounded coverage reads use `get_endpoint_coverage`. These calls are local and
grant-only; they do not contact Google or require Google OAuth headers.

## Summary

| API | Discovery methods | Curated status | Notes |
|---|---:|---|---|
| Drive v3 | 58 | partial | Core file list/get/create/upload/download/delete covered. Added about, permissions, comments, revisions, shared drives, copy, and metadata update wrappers. Changes, labels, watches, replies, and app resources remain partial/missing. |
| Docs v1 | 3 | partial | Documents create/get covered. `documents.batchUpdate` is covered for insert and replace text only, not every request type. |
| Sheets v4 | 17 | partial | Spreadsheet create/get, values get/batchGet/update/append/clear/batchUpdate/batchClear, data-filter get, and spreadsheet batchUpdate covered. Developer metadata and sheet copy remain missing. |
| Slides v1 | 5 | partial | Presentation create/get, text replacement, generic batchUpdate, page get, and page thumbnail covered. |
| Gmail v1 | 79 | partial | Labels/messages/threads/drafts common workflows plus inbox-scale overview/clustering/plans, batch modify/delete, attachments, thread mutation, draft management, and history are covered. Settings, CSE, forwarding, delegates, send-as, filters, and watches remain blocked or intentionally excluded. |
| Calendar v3 | 37 | partial | Calendars, calendar list, events, freebusy, colors, settings, ACL, event instances/move/import/full update, and CalendarList updates covered. Watch/channel lifecycle remains excluded until callback infrastructure exists. |
| YouTube Data v3 | 86 | partial | Search, channels, videos, playlists, playlist items, and comment threads covered. Uploads and mutations remain gated. |
| Analytics Data v1beta | 11 | partial | Metadata, runReport, batchRunReports, and realtime reports covered. Pivot/audience export endpoints remain missing. |
| Search Console v1 | 11 | partial | Sites list, Search Analytics query, URL inspection, and sitemap list covered. Mutations are intentionally excluded by default. |
| Business Profile APIs | 37+ | partial | Accounts, locations, location get, and performance daily metrics covered. Verification, notifications, Q&A, place actions, and location mutation are gated or missing. |
| Maps Platform | varies | partial | Geocoding, reverse geocoding, Places text search/details, and Routes compute covered. Requires Maps API key and usage-cost controls. |
| Merchant/Shopping + AdSense | 158+ | partial/blocked | Merchant product list/get and AdSense account/report read tools covered. Merchant mutations and Google Ads are blocked pending account prerequisites and approval gates. |

## Drive API v3

| Resource | Official methods | Status | Curated MCP coverage | Notes |
|---|---|---|---|---|
| `files` | `generateIds`, `create`, `generateCseToken`, `emptyTrash`, `update`, `copy`, `export`, `list`, `modifyLabels`, `delete`, `listLabels`, `watch`, `download`, `get` | partial | `drive_list_files`, `drive_search_files`, `drive_get_file`, `drive_batch_get_metadata`, `drive_create_folder`, `drive_upload_file`, `drive_download_file`, `drive_delete_file`, `drive_empty_trash`, `drive_purge_trash` | `drive_upload_file` covers direct small uploads and metadata-only resumable session initiation for large uploads. `drive_download_file` covers media/export URL/content flows, not the newer long-running `files.download` operation. General `files.update`, copy, labels, watches, CSE token, and generated IDs are missing curated wrappers. |
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
| `users.messages` | `delete`, `get`, `send`, `batchDelete`, `modify`, `import`, `trash`, `batchModify`, `untrash`, `insert`, `list` | partial | `gmail_list_messages`, `gmail_search_messages`, `gmail_get_message`, `gmail_get_message_headers`, `gmail_get_message_body`, `gmail_batch_get_metadata`, `gmail_send_message`, `gmail_send_raw_message`, `gmail_modify_message_labels`, `gmail_batch_modify_messages`, `gmail_batch_delete_messages`, `gmail_trash_message`, `gmail_untrash_message`, `gmail_delete_message` | Import and insert remain missing. Sends require explicit user approval in live tests. |
| `users.messages.attachments` | `get` | implemented | `gmail_get_attachment` | Bounded content return with max byte guard. |
| `users.threads` | `delete`, `get`, `trash`, `untrash`, `modify`, `list` | implemented | `gmail_list_threads`, `gmail_get_thread`, `gmail_modify_thread_labels`, `gmail_trash_thread`, `gmail_untrash_thread`, `gmail_delete_thread` | Destructive thread operations require confirmation. |
| `users.drafts` | `send`, `delete`, `get`, `create`, `list`, `update` | implemented | `gmail_create_draft`, `gmail_send_draft`, `gmail_list_drafts`, `gmail_get_draft`, `gmail_update_draft`, `gmail_delete_draft` | Send still requires explicit approval. |
| `users.history` | `list` | implemented | `gmail_list_history` | Requires caller-provided `start_history_id`. |
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

## Business API Expansion

| API | Resource | Status | Curated MCP coverage | Notes |
|---|---|---|---|---|
| YouTube Data v3 | `search`, `channels`, `videos`, `playlists`, `playlistItems`, `commentThreads` | partial | `youtube_search`, `youtube_list_channels`, `youtube_list_videos`, `youtube_list_playlists`, `youtube_list_playlist_items`, `youtube_list_comment_threads` | Uploads, captions mutation, comment moderation, subscriptions, and channel updates are not default agent tools. |
| Analytics Data v1beta | `properties` reports | partial | `analytics_get_metadata`, `analytics_run_report`, `analytics_batch_run_reports`, `analytics_run_realtime_report` | Pivot reports, compatibility checks, and audience exports remain missing. |
| Search Console v1 | Sites, sitemaps, Search Analytics, URL Inspection | partial | `searchconsole_list_sites`, `searchconsole_list_sitemaps`, `searchconsole_query_search_analytics`, `searchconsole_inspect_url` | Site/sitemap mutations are intentionally excluded by default. |
| Google Business Profile | Account Management, Business Information, Performance | partial | `business_profile_list_accounts`, `business_profile_list_locations`, `business_profile_get_location`, `business_profile_fetch_performance` | Requires `business.manage`; profile mutations, verification, Q&A, notifications, and place actions need additional approval gates. |
| Maps Platform | Geocoding, Places, Routes | partial | `maps_geocode`, `maps_reverse_geocode`, `maps_place_text_search`, `maps_place_details`, `maps_compute_routes` | Requires `x-google-maps-api-key` or `GOOGLE_MAPS_API_KEY`; billing-impacting tests need explicit approval. |
| Merchant/Shopping | Content API products | partial | `merchant_list_products`, `merchant_get_product` | Product mutations remain blocked pending account-specific approval. |
| AdSense | Accounts and reports | partial | `adsense_list_accounts`, `adsense_generate_report` | Read-only reporting only. |
| Google Ads | Campaign/customer reporting and mutation | blocked_scope | None | Requires developer token, login customer ID, and a dedicated approval model before exposing curated tools. |
