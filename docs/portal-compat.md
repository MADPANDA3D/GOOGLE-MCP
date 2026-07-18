# MAD MCP Portal Compatibility

Portal support is optional. The same Google MCP runtime can operate as an
independent standalone server; Portal mode changes only the service-to-service
authentication boundary and network placement. Provider authorization remains
request-scoped BYOK in both modes.

## Required runtime contract

```dotenv
MCP_MODE=portal
MCP_PORTAL_GRANT_TOKEN=<UNIQUE_SERVICE_GRANT>
MCP_PORTAL_GRANT_HEADER=x-madpanda-portal-grant
MCP_DISABLE_DEFAULT_GOOGLE_FALLBACK=true
MCP_ALLOW_REQUEST_OVERRIDES=true
MCP_BYOK_CLIENT_CACHE_SIZE=0
MCP_BYOK_CLIENT_CACHE_TTL_SECONDS=0
```

Startup fails if Portal mode has no grant token. The grant authenticates the
broker to this MCP server; it is not a Google credential and must be unique to
this service/environment.

## Broker-to-provider headers

| Header | Required | Source |
|---|---|---|
| `X-MADPANDA-PORTAL-GRANT` | Yes | Portal secret store; fixed for this service/environment |
| `X-Google-Client-Id` | Provider calls | Authorized owner's request-scoped secret context |
| `X-Google-Client-Secret` | Provider calls | Authorized owner's request-scoped secret context |
| `X-Google-Refresh-Token` | Provider calls | Authorized owner's request-scoped secret context |
| `X-Google-Maps-Api-Key` | Maps calls only | Authorized owner's request-scoped secret context |

The OAuth trio is all-or-nothing. The broker must not place these values in a
registry record, URL, job payload, ticket, trace, or log. It should inject them
only after authenticating the owner and authorizing the specific connection.

The five standard navigation tools need the Portal grant but not the Google
headers. Provider tools need both service authentication and the complete
request-scoped tuple.

## Registry example

Field names vary by broker. A safe logical record looks like:

```yaml
service_id: google
transport: streamable-http
mcp_url: http://<GOOGLE_MCP_SERVICE>:8086/mcp
health_url: http://<GOOGLE_MCP_SERVICE>:8086/health
auth:
  kind: static-header
  header: X-MADPANDA-PORTAL-GRANT
  secret_ref: <PORTAL_SECRET_REFERENCE>
provider_credentials:
  kind: request-scoped-headers
  required:
    - X-Google-Client-Id
    - X-Google-Client-Secret
    - X-Google-Refresh-Token
  optional:
    - X-Google-Maps-Api-Key
catalog:
  source: list_capabilities
  pin: <RELEASE_CATALOG_VERSION_AND_DESCRIPTOR_HASH>
```

Use internal service discovery or a private, TLS-protected endpoint. The public
repository intentionally contains no production hostname, network address,
secret reference, or tenant identifier.

## Descriptor and routing behavior

The provider-owned ToolManifest is authoritative. On admission or upgrade, the
Portal should:

1. Authenticate with the service grant.
2. Call `list_capabilities(include_descriptors=true)` without Google headers.
3. Verify the release catalog version, native count, tier counts, and
   deterministic descriptor hash against the approved deployment record.
4. Store descriptors, not provider credentials.
5. Use `find_tools` for default agent discovery and `get_tool_usage` for the
   exact selected descriptor.
6. Exclude hidden tools from ordinary agent selection.
7. Preserve native names when invoking the service; treat canonical names and
   aliases only as discovery compatibility.

Do not hard-code a stale tool list in the broker. Reject an unreviewed
descriptor-hash change even if the HTTP service is healthy.

## Confirmation and policy

- Enforce descriptor confirmation metadata before dispatch.
- Preserve explicit `confirm=true` only after the caller has approved the exact
  operation and target.
- Keep dry-run or planning defaults for cleanup and bulk workflows.
- Treat all Maps tools as billable side-effect operations: require confirmation
  and do not automatically retry them.
- Treat sends, sharing changes, ACL changes, permanent deletions, and broad
  batch mutations as human-approval boundaries.
- Use hidden `gmail_signature_preflight` only inside the authorized Gmail send
  binding. Preserve its alias and SHA-256 fingerprint checks through delivery;
  never request or store signature HTML in Portal workflow state.
- Keep the hidden raw tool behind an advanced, explicit policy even though it
  is GET/HEAD-only.

Portal policy may be stricter than the server descriptor. It must never weaken
server-side authentication, scope, confirmation, or URL validation.

## Error handling

The broker may relay normalized provider errors needed for recovery, but must
sanitize:

- service grants and bearer tokens
- OAuth client secrets, refresh tokens, and access tokens
- Maps keys
- credential-bearing URLs or query values
- provider response bodies containing personal or confidential data
- private topology and secret-store references

Authentication failures should be distinguishable from missing BYOK,
insufficient Google scope, provider permission, quota, and confirmation errors
without revealing the rejected value.

## Admission checklist

- [ ] Exact release source, package, or image identity is verified.
- [ ] Catalog version, descriptor hash, native count, and tier counts match.
- [ ] Portal mode refuses startup without its unique grant.
- [ ] Missing and incorrect grants are rejected before MCP parsing.
- [ ] Browser-origin traffic is rejected by default.
- [ ] Host allowlist contains only the private service/proxy names in use.
- [ ] Local navigation works without Google credentials.
- [ ] Provider calls fail when the OAuth tuple is missing or partial.
- [ ] No provider credential fallback or persistent cache is enabled.
- [ ] Raw non-Google URLs and mutation methods fail.
- [ ] Maps confirmation and key requirements fail closed.
- [ ] Request/response limits and sanitized observability are active.
- [ ] A provider-free smoke passes before any Google call.

After admission, use a dedicated test account for a separately authorized,
narrow read. Do not certify first with a send, mutation, deletion, sharing
change, or Maps request.
