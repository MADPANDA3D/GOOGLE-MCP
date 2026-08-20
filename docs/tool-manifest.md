# Google MCP ToolManifest

## Identity

| Field | Value |
|---|---|
| Schema version | `1.0.0` |
| Service ID | `google` |
| Catalog version | `google-2026.07.18.2` |
| Native tools | 151 |
| Descriptor hash | `2c777ccf9f5528e8a3fcaea8de69535ca8a8aae8f85fa622fa55e7d76ffc76d0` |

`list_capabilities(include_descriptors=true)` is the provider-owned source of
truth for the runtime contract. It returns the complete ordered descriptor
catalog plus a deterministic SHA-256 descriptor hash.

## Tier contract

| Tier | Count | Discovery behavior |
|---|---:|---|
| `agent_ready` | 144 | Returned by default task discovery |
| `legacy` | 5 | Returned only when advanced discovery includes legacy tools |
| `hidden` | 2 | Excluded from task discovery |
| **Total** | **151** | |

The five legacy entries are:

- `drive_purge_trash` → use `drive_empty_trash`
- `google_mcp_welcome` → use `list_capabilities`
- `google_mcp_list_capabilities` → use `list_capabilities`
- `google_mcp_get_endpoint_coverage` → use `get_endpoint_coverage`
- `google_mcp_get_tool_usage` → use `get_tool_usage`

The two hidden entries are:

- `gmail_signature_preflight`, a Portal binding helper that returns the
  selected send-as alias plus SHA-256 signature/message fingerprints and never
  returns signature HTML
- `google_raw_request`, a bounded GET/HEAD-only read escape hatch that rejects
  bodies, caller headers, credential-bearing query keys, and non-allowlisted
  destinations and never automatically retries

Both are deliberately absent from default discovery.

## Standard navigation

Exactly five agent-ready tools form the local navigation surface:

| Tool | Role |
|---|---|
| `check_configuration` | Reports mode and request readiness without credential values |
| `find_tools` | Ranks agent-ready tools by task, category, and risk |
| `get_endpoint_coverage` | Filters the bounded endpoint ledger |
| `get_tool_usage` | Resolves one native, canonical, or alias identity |
| `list_capabilities` | Returns counts or the complete descriptor catalog |

These calls require only the mode-specific service credential. They do not
contact Google and do not require Google OAuth headers.

## Risk contract

The registered tools have mutually exclusive manifest risk classes used for
confirmation and routing:

| Manifest risk | Count | Meaning |
|---|---:|---|
| Read | 84 | No intended provider mutation; may still access sensitive data or consume quota |
| Write | 18 | Creates or updates provider state without a destructive/billable routing classification |
| Destructive/billable | 49 | 44 state-destructive sends, overwrites, moves, shares, removes, deletes, or batch mutations plus five confirmed Maps reads |
| **Total** | **151** | |

Annotations add behavioral detail. All five Maps tools set both
`readOnlyHint=true` and `destructiveHint=true`; the manifest places them in the
destructive/billable routing class, requires explicit confirmation, and
disables automatic retry. Across the catalog, 89 tools set
`readOnlyHint=true`, while 49 set `destructiveHint=true`.

Confirmation is schema-driven as well as annotation-driven. All 49
destructive/billable tools require confirmation, and six non-destructive write
tools with an explicit `confirm` parameter do too, for **55 confirmed tools**
in total.

Every descriptor contains:

- native and canonical identities plus bounded compatibility aliases
- title, complete description, category, tier, and deprecation metadata
- complete input and output JSON schemas
- `readOnlyHint`, `destructiveHint`, `openWorldHint`, and
  `idempotentHint` where true
- confirmation metadata for every tool whose schema requires explicit approval
- documentation URL, navigation role, catalog version, and descriptor hash

Annotations describe behavior; they are not authorization. Service
authentication, Google OAuth scopes, explicit confirmations, dry runs, and
provider permissions remain separate enforcement boundaries.

## Safe discovery sequence

1. Call `check_configuration`.
2. Call `find_tools` with a plain-language task.
3. Call `get_tool_usage` for the selected native or canonical name.
4. Call `get_endpoint_coverage` when deciding whether a curated wrapper
   exists.
5. Use `list_capabilities(include_descriptors=true)` only for full contract
   publication or audit.

Search is deterministic, punctuation-normalized, multi-token, tier-aware, and
locally computed. Hidden tools never appear in `find_tools`.

## Build and descriptor identity

Release builds set `MCP_BUILD_SHA` to the full source commit. `/health` and
ToolManifest responses report that build SHA, the catalog version, descriptor
hash, and tier counts without revealing credentials.

The descriptor hash identifies the ordered tool contract. It does not change
merely because the same contract is rebuilt from another commit. An immutable
container digest is separate release metadata and is recorded only after a
release image exists.

## Change control

A tool addition, removal, rename, schema change, annotation change, alias
change, or tier change requires synchronized updates to:

- `fastmcp/tool_manifest.py`
- provider-free conformance tests
- [the complete tool catalog](tool-catalog.md)
- [the endpoint ledger](endpoint-coverage.md)
- README counts and release notes

The runtime, `/health`, broker catalog, and public documentation must agree
before a contract change is released.
