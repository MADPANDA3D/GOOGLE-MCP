# Google MCP Endpoint Coverage

- Reviewed: 2026-07-18
- Catalog: `google-2026.07.18.2`
- Native MCP tools: **151**

This ledger describes the curated public wrappers in v1.0.0. It does not claim
complete parity with every Google API endpoint. Google changes its APIs
independently of this project; confirm scopes, quotas, availability, and terms
in the current official documentation before production use.

## Status language

| Status | Meaning |
|---|---|
| `curated` | A dedicated, documented MCP wrapper exists. |
| `partial` | Common operations are curated; the provider offers more endpoints or request variants. |
| `excluded` | The surface is intentionally absent because it needs callbacks, elevated scopes, stronger approval design, or a separate product decision. |
| `advanced-read` | The hidden read-only escape hatch may cover an allowlisted Google REST GET/HEAD request, but it is not a curated contract. |

The hidden `google_raw_request` tool is never counted as provider-family
coverage. In v1.0.0 it accepts only `GET` and `HEAD`, rejects request bodies and
caller-supplied HTTP headers, and restricts destinations to approved Google API
hosts. It does not automatically retry.

## Coverage summary

| Provider family | Curated tools | Status | Covered in v1.0.0 | Important remaining surface |
|---|---:|---|---|---|
| Google Drive | 24 | partial | Files, folders, metadata, upload/download, copy, permissions, comments, revisions, shared drives, trash | Changes/watch, labels, replies, apps, approvals, access proposals |
| Google Docs | 5 | partial | Create/get documents, insert/replace text, bounded batch update | Exhaustive parity with every batch request type |
| Google Sheets | 11 | partial | Spreadsheet create/get/batch update; values get, append, update, clear, batch and data-filter operations | Developer metadata and sheet `copyTo` |
| Google Slides | 6 | partial | Create/get presentations, replace text, batch update, page reads and thumbnails | Exhaustive parity with every batch request type |
| Gmail | 38 | partial | Messages, threads, signature-safe sends/drafts, fingerprint-only send preflight, labels, attachments, history, bulk planning and bounded mutations | Watch/Pub/Sub, general settings mutation, CSE, forwarding, delegates, filters |
| Google Calendar | 29 | partial | Calendars, calendar list, events, instances, import/move, free/busy, settings, colors, ACLs | Watch/channel lifecycle and complete recurrence/notification parity |
| YouTube Data | 6 | partial, read-only | Search, channels, videos, playlists, playlist items, comment threads | Uploads, channel mutation, captions, subscriptions, moderation |
| Google Analytics Data | 4 | partial, read-only | Metadata, standard reports, batch reports, realtime reports | Pivot reports, compatibility checks, audience exports |
| Search Console | 4 | partial, read-only | Sites, sitemaps, search analytics, URL inspection | Site/sitemap mutation |
| Google Business Profile | 4 | partial, read-only | Accounts, locations, location reads, performance | Verification, mutations, Q&A, notifications, place actions |
| Google Maps Platform | 5 | partial, read-only | Geocoding, reverse geocoding, Places search/details, routes | The wider Maps Platform surface; all calls remain quota/billing sensitive |
| Merchant Center | 2 | partial, read-only | Product list and get | Product/account mutations and broader Merchant API resources |
| AdSense | 2 | partial, read-only | Account list and report generation | Payments, alerts, ad clients, sites, units, and other management resources |
| **Provider subtotal** | **140** | | | |

The remaining 11 native tools are nine local navigation/compatibility tools,
one configuration diagnostic, and the hidden advanced-read tool. Together the
surface totals 151.

## Provider notes

### Workspace APIs

- **Drive:** curated sharing and irreversible operations carry write or
  destructive annotations and may require `confirm=true`. Resumable upload
  support is bounded; it is not an unbounded file-transfer service.
- **Docs, Sheets, and Slides:** generic batch-update wrappers expose more
  request shapes than the named convenience tools, but they do not constitute
  a promise of future Google request-type compatibility.
- **Gmail:** sending, permanent deletion, bulk mutation, and cleanup execution
  are intentionally gated. Send and draft flows enforce the selected
  configured send-as signature exactly once; the hidden preflight returns
  only alias and SHA-256 fingerprints. Cleanup-oriented tools default to
  planning or dry runs where supported. General mailbox settings mutation and
  callback infrastructure remain outside the curated v1.0.0 surface.
- **Calendar:** common event, calendar, free/busy, settings, and ACL workflows
  are curated. Watch channels remain excluded because safe delivery requires
  external callback registration, ownership, renewal, and teardown.

### Public presence and business APIs

- **YouTube, Analytics, Search Console, Business Profile, Merchant, and
  AdSense** are intentionally read-only in this release.
- **Maps** requires the request-scoped `X-Google-Maps-Api-Key` header in
  addition to OAuth. A successful read can still consume quota or incur cost.
- Google Ads is not part of the 151-tool contract. It needs a developer token,
  account hierarchy, and a dedicated authorization and approval model.

## Official references

- [Google Drive API v3](https://developers.google.com/workspace/drive/api/reference/rest/v3)
- [Google Docs API v1](https://developers.google.com/docs/api/reference/rest)
- [Google Sheets API v4](https://developers.google.com/workspace/sheets/api/reference/rest/v4/spreadsheets)
- [Google Slides API v1](https://developers.google.com/workspace/slides/api/reference/rest)
- [Gmail API v1](https://developers.google.com/gmail/api/reference/rest)
- [Google Calendar API v3](https://developers.google.com/workspace/calendar/api/v3/reference)
- [YouTube Data API v3](https://developers.google.com/youtube/v3/docs)
- [Google Analytics Data API](https://developers.google.com/analytics/devguides/reporting/data/v1)
- [Search Console API](https://developers.google.com/webmaster-tools/v1/api_reference_index)
- [Google Business Profile APIs](https://developers.google.com/my-business)
- [Google Maps Platform](https://developers.google.com/maps/documentation)
- [Merchant API](https://developers.google.com/merchant/api)
- [AdSense Management API](https://developers.google.com/adsense/management)

## How to verify the live contract

After authenticating to the MCP service, use local calls in this order:

1. `check_configuration` confirms mode and presence-only readiness.
2. `find_tools` searches agent-ready curated wrappers.
3. `get_tool_usage` returns the exact input/output descriptor for one tool.
4. `get_endpoint_coverage` filters the runtime coverage ledger.
5. `list_capabilities(include_descriptors=true)` returns the complete ordered
   contract and deterministic descriptor hash.

These navigation calls do not contact Google. A provider call is a separate,
explicit step requiring request-scoped credentials and the relevant Google
permissions.
