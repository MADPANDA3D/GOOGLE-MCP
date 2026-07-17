# Google MCP ToolManifest

Catalog version: `google-2026.07.17.1`

Schema version: `1.0.0`

Canonical service ID: `google`

`list_capabilities(include_descriptors=true)` is the provider-owned source of
truth for the Google MCP catalog. It returns all registered native tools with:

- native and canonical identities plus compatibility aliases;
- title, full agent-facing description, category, and deprecation metadata;
- complete input and output JSON schemas;
- explicit read-only, destructive, open-world, and idempotency annotations;
- Portal v2 confirmation requirements for destructive execution;
- documentation URL, navigation role, catalog version, and descriptor hash.

## Contract tiers

| Tier | Count | Discovery behavior |
|---|---:|---|
| `agent_ready` | 144 | Returned by default discovery. |
| `legacy` | 5 | Available only when advanced discovery explicitly includes legacy tools. |
| `hidden` | 2 | `google_raw_request` plus the Portal-only `gmail_signature_preflight`; unavailable in normal discovery. |

The legacy set contains the four prefixed navigation tools and
`drive_purge_trash`. Existing callers remain compatible. New agents should use
the five standard navigation tools and `drive_empty_trash`.

`gmail_signature_preflight` returns only the selected send-as alias and
SHA-256 signature/draft fingerprints. Portal uses those safe values to bind the
current signature and, for existing drafts, MIME content into preview approval.
It never returns signature HTML.

## Safe discovery flow

1. Call `check_configuration` to see whether the Portal gate and current BYOK
   request are ready without revealing values.
2. Call `find_tools` with a plain-language task. Search is deterministic,
   punctuation-normalized, multi-token, tier-aware, and locally computed.
3. Call `get_tool_usage` with the returned native or canonical name to receive
   the complete lossless descriptor.
4. Use `get_endpoint_coverage` when deciding whether a curated Google wrapper
   exists.

Only these five standard navigation calls can run with a valid Portal grant and
without Google OAuth headers. Protocol initialization, `tools/list`, legacy
navigation, and provider tools require per-request Google BYOK credentials.

## Build identity

Production images set `MCP_BUILD_SHA` to the full source commit. `/health` and
the ToolManifest return that SHA. Descriptor hashes intentionally describe the
contract and do not change merely because the same catalog is rebuilt from a
different commit.
