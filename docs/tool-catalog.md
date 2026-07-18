# Complete tool catalog

This catalog is generated from the **151 registered native tools** and the
tier/category rules in `fastmcp/tool_manifest.py`. The runtime ToolManifest
remains the machine-readable source of truth.

## Contract summary

- Native tools: **151**
- Standard navigation tools: **5**
- Other operational and compatibility tools: **146**
- Tiers: **144 agent-ready**, **5 legacy**, **2 hidden**
- Manifest risk: **84 read**, **18 write**, **49 destructive or billable**

Manifest risk classes are mutually exclusive for routing. The last class
contains 44 state-destructive tools plus five Maps reads that set
`destructiveHint=true`, require explicit confirmation, and disable automatic
retry because they can consume billable quota.

Six additional non-destructive write tools also expose a `confirm` parameter
and publish required confirmation metadata. The full catalog therefore has
**55 confirmation-required tools**.

A read classification does not mean output is public, free of personal data,
or free of quota impact. State-destructive includes sends, overwrites, sharing
changes, moves, and bulk mutation—not only deletion.

## Compatibility and hidden tools

| Legacy tool | Replacement |
|---|---|
| `drive_purge_trash` | `drive_empty_trash` |
| `google_mcp_welcome` | `list_capabilities` |
| `google_mcp_list_capabilities` | `list_capabilities` |
| `google_mcp_get_endpoint_coverage` | `get_endpoint_coverage` |
| `google_mcp_get_tool_usage` | `get_tool_usage` |

Two tools are hidden from default discovery:

- `gmail_signature_preflight` returns the selected send-as alias and SHA-256
  signature/message fingerprints for authorized send binding; it never returns
  signature HTML.
- `google_raw_request` permits bounded GET/HEAD reads only, rejects bodies and
  caller headers, filters credential-bearing query keys, never automatically
  retries, and is restricted to approved Google API hosts.

`find_tools` never returns hidden tools and omits legacy tools unless
`include_legacy=true`.

## Local navigation (9)

Five standard local discovery tools and four retained compatibility entries.

| Tool | Tier | Manifest risk |
|---|---|---|
| `check_configuration` | `agent_ready` | `read` |
| `find_tools` | `agent_ready` | `read` |
| `get_endpoint_coverage` | `agent_ready` | `read` |
| `get_tool_usage` | `agent_ready` | `read` |
| `google_mcp_get_endpoint_coverage` | `legacy` | `read` |
| `google_mcp_get_tool_usage` | `legacy` | `read` |
| `google_mcp_list_capabilities` | `legacy` | `read` |
| `google_mcp_welcome` | `legacy` | `read` |
| `list_capabilities` | `agent_ready` | `read` |

## Configuration diagnostic (1)

Presence-only configuration and optional provider diagnostics.

| Tool | Tier | Manifest risk |
|---|---|---|
| `mcp_health_check` | `agent_ready` | `read` |

## Google Drive (24)

Files, folders, downloads, resumable sessions, permissions, comments, revisions, and shared drives.

| Tool | Tier | Manifest risk |
|---|---|---|
| `drive_about_get` | `agent_ready` | `read` |
| `drive_batch_get_metadata` | `agent_ready` | `read` |
| `drive_copy_file` | `agent_ready` | `write` |
| `drive_create_comment` | `agent_ready` | `write` |
| `drive_create_folder` | `agent_ready` | `write` |
| `drive_create_permission` | `agent_ready` | `destructive/billable` |
| `drive_delete_comment` | `agent_ready` | `destructive/billable` |
| `drive_delete_file` | `agent_ready` | `destructive/billable` |
| `drive_delete_permission` | `agent_ready` | `destructive/billable` |
| `drive_download_file` | `agent_ready` | `read` |
| `drive_empty_trash` | `agent_ready` | `destructive/billable` |
| `drive_get_file` | `agent_ready` | `read` |
| `drive_get_permission` | `agent_ready` | `read` |
| `drive_list_comments` | `agent_ready` | `read` |
| `drive_list_files` | `agent_ready` | `read` |
| `drive_list_permissions` | `agent_ready` | `read` |
| `drive_list_revisions` | `agent_ready` | `read` |
| `drive_list_shared_drives` | `agent_ready` | `read` |
| `drive_purge_trash` | `legacy` | `destructive/billable` |
| `drive_search_files` | `agent_ready` | `read` |
| `drive_update_comment` | `agent_ready` | `destructive/billable` |
| `drive_update_file_metadata` | `agent_ready` | `destructive/billable` |
| `drive_update_permission` | `agent_ready` | `destructive/billable` |
| `drive_upload_file` | `agent_ready` | `write` |

## Google Docs (5)

Document creation, reads, text edits, and bounded batch updates.

| Tool | Tier | Manifest risk |
|---|---|---|
| `docs_batch_update` | `agent_ready` | `destructive/billable` |
| `docs_create_document` | `agent_ready` | `write` |
| `docs_get_document` | `agent_ready` | `read` |
| `docs_insert_text` | `agent_ready` | `write` |
| `docs_replace_text` | `agent_ready` | `destructive/billable` |

## Google Sheets (11)

Spreadsheet metadata, values, append/clear, batch updates, and data filters.

| Tool | Tier | Manifest risk |
|---|---|---|
| `sheets_append_values` | `agent_ready` | `destructive/billable` |
| `sheets_batch_clear_values` | `agent_ready` | `destructive/billable` |
| `sheets_batch_get_values` | `agent_ready` | `read` |
| `sheets_batch_update` | `agent_ready` | `destructive/billable` |
| `sheets_batch_update_values` | `agent_ready` | `destructive/billable` |
| `sheets_clear_values` | `agent_ready` | `destructive/billable` |
| `sheets_create_spreadsheet` | `agent_ready` | `write` |
| `sheets_get_by_data_filter` | `agent_ready` | `read` |
| `sheets_get_spreadsheet` | `agent_ready` | `read` |
| `sheets_get_values` | `agent_ready` | `read` |
| `sheets_update_values` | `agent_ready` | `destructive/billable` |

## Google Slides (6)

Presentation creation, reads, text replacement, batch updates, pages, and thumbnails.

| Tool | Tier | Manifest risk |
|---|---|---|
| `slides_batch_update` | `agent_ready` | `destructive/billable` |
| `slides_create_presentation` | `agent_ready` | `write` |
| `slides_get_page` | `agent_ready` | `read` |
| `slides_get_page_thumbnail` | `agent_ready` | `read` |
| `slides_get_presentation` | `agent_ready` | `read` |
| `slides_replace_text` | `agent_ready` | `destructive/billable` |

## Gmail (38)

Messages, threads, drafts, labels, attachments, history, cleanup workflows, and signature-safe send preflight.

| Tool | Tier | Manifest risk |
|---|---|---|
| `gmail_apply_cleanup_plan` | `agent_ready` | `destructive/billable` |
| `gmail_batch_delete_messages` | `agent_ready` | `destructive/billable` |
| `gmail_batch_get_metadata` | `agent_ready` | `read` |
| `gmail_batch_modify_messages` | `agent_ready` | `destructive/billable` |
| `gmail_cleanup_plan` | `agent_ready` | `read` |
| `gmail_create_draft` | `agent_ready` | `write` |
| `gmail_create_label` | `agent_ready` | `write` |
| `gmail_delete_draft` | `agent_ready` | `destructive/billable` |
| `gmail_delete_label` | `agent_ready` | `destructive/billable` |
| `gmail_delete_message` | `agent_ready` | `destructive/billable` |
| `gmail_delete_thread` | `agent_ready` | `destructive/billable` |
| `gmail_get_attachment` | `agent_ready` | `read` |
| `gmail_get_draft` | `agent_ready` | `read` |
| `gmail_get_label` | `agent_ready` | `read` |
| `gmail_get_message` | `agent_ready` | `read` |
| `gmail_get_message_body` | `agent_ready` | `read` |
| `gmail_get_message_headers` | `agent_ready` | `read` |
| `gmail_get_thread` | `agent_ready` | `read` |
| `gmail_list_drafts` | `agent_ready` | `read` |
| `gmail_list_history` | `agent_ready` | `read` |
| `gmail_list_labels` | `agent_ready` | `read` |
| `gmail_list_messages` | `agent_ready` | `read` |
| `gmail_list_threads` | `agent_ready` | `read` |
| `gmail_mailbox_overview` | `agent_ready` | `read` |
| `gmail_modify_message_labels` | `agent_ready` | `destructive/billable` |
| `gmail_modify_thread_labels` | `agent_ready` | `write` |
| `gmail_search_messages` | `agent_ready` | `read` |
| `gmail_send_draft` | `agent_ready` | `destructive/billable` |
| `gmail_send_message` | `agent_ready` | `destructive/billable` |
| `gmail_send_raw_message` | `agent_ready` | `destructive/billable` |
| `gmail_sender_clusters` | `agent_ready` | `read` |
| `gmail_signature_preflight` | `hidden` | `read` |
| `gmail_trash_message` | `agent_ready` | `destructive/billable` |
| `gmail_trash_thread` | `agent_ready` | `destructive/billable` |
| `gmail_untrash_message` | `agent_ready` | `write` |
| `gmail_untrash_thread` | `agent_ready` | `destructive/billable` |
| `gmail_update_draft` | `agent_ready` | `destructive/billable` |
| `gmail_update_label` | `agent_ready` | `write` |

## Google Calendar (29)

Calendars, events, free/busy, settings, instances, moves/imports, and ACLs.

| Tool | Tier | Manifest risk |
|---|---|---|
| `calendar_batch_get_events` | `agent_ready` | `read` |
| `calendar_clear_calendar` | `agent_ready` | `destructive/billable` |
| `calendar_create_calendar` | `agent_ready` | `write` |
| `calendar_create_event` | `agent_ready` | `write` |
| `calendar_delete_acl` | `agent_ready` | `destructive/billable` |
| `calendar_delete_calendar` | `agent_ready` | `destructive/billable` |
| `calendar_delete_calendar_list_entry` | `agent_ready` | `destructive/billable` |
| `calendar_delete_event` | `agent_ready` | `destructive/billable` |
| `calendar_freebusy_query` | `agent_ready` | `read` |
| `calendar_get_acl` | `agent_ready` | `read` |
| `calendar_get_calendar` | `agent_ready` | `read` |
| `calendar_get_calendar_list_entry` | `agent_ready` | `read` |
| `calendar_get_colors` | `agent_ready` | `read` |
| `calendar_get_event` | `agent_ready` | `read` |
| `calendar_get_setting` | `agent_ready` | `read` |
| `calendar_import_event` | `agent_ready` | `destructive/billable` |
| `calendar_list_acl` | `agent_ready` | `read` |
| `calendar_list_calendars` | `agent_ready` | `read` |
| `calendar_list_event_instances` | `agent_ready` | `read` |
| `calendar_list_events` | `agent_ready` | `read` |
| `calendar_list_settings` | `agent_ready` | `read` |
| `calendar_move_event` | `agent_ready` | `destructive/billable` |
| `calendar_quick_add` | `agent_ready` | `write` |
| `calendar_replace_event` | `agent_ready` | `destructive/billable` |
| `calendar_search_events` | `agent_ready` | `read` |
| `calendar_update_calendar` | `agent_ready` | `write` |
| `calendar_update_calendar_list_entry` | `agent_ready` | `write` |
| `calendar_update_event` | `agent_ready` | `destructive/billable` |
| `calendar_upsert_acl` | `agent_ready` | `destructive/billable` |

## YouTube Data (6)

Read-only search and list wrappers for six YouTube Data resources.

| Tool | Tier | Manifest risk |
|---|---|---|
| `youtube_list_channels` | `agent_ready` | `read` |
| `youtube_list_comment_threads` | `agent_ready` | `read` |
| `youtube_list_playlist_items` | `agent_ready` | `read` |
| `youtube_list_playlists` | `agent_ready` | `read` |
| `youtube_list_videos` | `agent_ready` | `read` |
| `youtube_search` | `agent_ready` | `read` |

## Google Analytics Data (4)

Read-only GA4 metadata and standard, batch, and realtime reports.

| Tool | Tier | Manifest risk |
|---|---|---|
| `analytics_batch_run_reports` | `agent_ready` | `read` |
| `analytics_get_metadata` | `agent_ready` | `read` |
| `analytics_run_realtime_report` | `agent_ready` | `read` |
| `analytics_run_report` | `agent_ready` | `read` |

## Search Console (4)

Read-only sites, sitemaps, performance, and URL inspection.

| Tool | Tier | Manifest risk |
|---|---|---|
| `searchconsole_inspect_url` | `agent_ready` | `read` |
| `searchconsole_list_sitemaps` | `agent_ready` | `read` |
| `searchconsole_list_sites` | `agent_ready` | `read` |
| `searchconsole_query_search_analytics` | `agent_ready` | `read` |

## Google Business Profile (4)

Read-only accounts, locations, location details, and performance.

| Tool | Tier | Manifest risk |
|---|---|---|
| `business_profile_fetch_performance` | `agent_ready` | `read` |
| `business_profile_get_location` | `agent_ready` | `read` |
| `business_profile_list_accounts` | `agent_ready` | `read` |
| `business_profile_list_locations` | `agent_ready` | `read` |

## Google Maps Platform (5)

Confirmed, non-retried geocoding, Places, and route reads that may incur provider charges.

| Tool | Tier | Manifest risk |
|---|---|---|
| `maps_compute_routes` | `agent_ready` | `destructive/billable` |
| `maps_geocode` | `agent_ready` | `destructive/billable` |
| `maps_place_details` | `agent_ready` | `destructive/billable` |
| `maps_place_text_search` | `agent_ready` | `destructive/billable` |
| `maps_reverse_geocode` | `agent_ready` | `destructive/billable` |

## Merchant Center (2)

Read-only product listing and retrieval.

| Tool | Tier | Manifest risk |
|---|---|---|
| `merchant_get_product` | `agent_ready` | `read` |
| `merchant_list_products` | `agent_ready` | `read` |

## AdSense (2)

Read-only account listing and report generation.

| Tool | Tier | Manifest risk |
|---|---|---|
| `adsense_generate_report` | `agent_ready` | `read` |
| `adsense_list_accounts` | `agent_ready` | `read` |

## Advanced raw request (1)

Hidden, non-retried GET/HEAD-only escape hatch for approved Google API hosts.

| Tool | Tier | Manifest risk |
|---|---|---|
| `google_raw_request` | `hidden` | `read` |

## Reading one exact descriptor

Call `get_tool_usage` with a native name, canonical `google.<name>` identity, or documented alias. The response contains the complete input/output schemas, annotations, tier, deprecation state, confirmation metadata, catalog version, and descriptor hash.

For provider endpoint accountability, use the [endpoint coverage ledger](endpoint-coverage.md).
